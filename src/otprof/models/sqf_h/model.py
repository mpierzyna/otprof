"""SQF-H network: the height-aware SqueezeFormer (plain `nn.Module`, no Lightning).

The backbone is the LEAP Squeezeformer, based on https://www.kaggle.com/code/shlomoron/leap-training-1#Layers by
shlomoron, converted to PyTorch. `SqueezeformerHModel` adds the underground mask, the interpolation in physical height
and the query height embedding. Trained with `otprof.models.sqf_h.lit`; saved and loaded without Lightning by
`save_pretrained` / `from_pretrained` (safetensors + `config.json`) and `from_lightning_checkpoint`.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from otprof import schema
from otprof.models import check_versions, get_bundle_meta
from otprof.models.sqf_h.inputs import FEATURES_VERSION

logger = logging.getLogger(__name__)

ARCH = "sqf_h"
FORMAT_VERSION = 1  # bump on a rename of submodules / parameters, with a key mapping in `from_pretrained`


class GLU(nn.Module):
    """
    Gated Linear Unit activation.

    Splits input in half along the feature dimension and applies gating:
    output = x * swish(gate)

    Notes
    -----
    Input shape: (batch_size, seq_len, 2 * features)
    Output shape: (batch_size, seq_len, features)

    The input feature dimension must be even.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)  # SiLU is equivalent to Swish


class GLUMlp(nn.Module):
    """
    Two-layer feedforward network with GLU activation.

    Architecture: Linear -> GLU -> Linear

    Parameters
    ----------
    dim : int
        Input feature dimension
    dim_expand : int
        Hidden layer dimension (gets halved after GLU)
    dim_out : int, optional
        Output feature dimension. If None, defaults to `dim`.

    Notes
    -----
    Input shape: (batch_size, seq_len, dim)
    Output shape: (batch_size, seq_len, dim)
    """

    def __init__(self, dim: int, dim_expand: int, dim_out: int | None = None):
        super().__init__()
        if dim_out is None:
            dim_out = dim

        self.fc1 = nn.Linear(dim, dim_expand)
        self.glu = GLU()
        self.fc2 = nn.Linear(dim_expand // 2, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.glu(x)
        x = self.fc2(x)
        return x


class ScaleBias(nn.Module):
    """
    Learnable per-feature affine transformation.

    Applies: output = input * scale + bias
    where scale and bias are learnable parameters.

    Parameters
    ----------
    dim : int
        Feature dimension

    Notes
    -----
    Input shape: (batch_size, seq_len, dim)
    Output shape: (batch_size, seq_len, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) module.

    Computes channel-wise attention weights using global average pooling
    followed by 1D convolution.

    Parameters
    ----------
    kernel_size : int, default=5
        Kernel size for 1D convolution

    Notes
    -----
    Input shape: (batch_size, seq_len, channels)
    Output shape: (batch_size, seq_len, channels)

    References
    ----------
    Wang et al., "ECA-Net: Efficient Channel Attention for Deep
    Convolutional Neural Networks", CVPR 2020
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # (B, L, C) -> (B, C), mean over valid levels only if mask (B, L) is given
        if mask is None:
            y = x.mean(dim=1)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)
            y = (x * m).sum(dim=1) / m.sum(dim=1)

        # (B, C) -> (B, 1, C) for Conv1d
        y = y.unsqueeze(1)

        # Conv1d: (B, 1, C) -> (B, 1, C)
        y = self.conv(y)

        # (B, 1, C) -> (B, C) -> (B, 1, C)
        y = torch.sigmoid(y.squeeze(1)).unsqueeze(1)

        # Broadcast multiply: (B, L, C) * (B, 1, C)
        return x * y


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with post-norm residual connections.

    Architecture:
    - Multi-head self-attention with residual & norm
    - GLU-based feedforward network with residual & norm
    - Learnable scale-bias after each sub-layer

    Parameters
    ----------
    embed_dim : int
        Embedding/model dimension
    num_heads : int
        Number of attention heads
    ffn_dim : int
        Feedforward network hidden dimension

    Notes
    -----
    Input shape: (batch_size, seq_len, embed_dim)
    Output shape: (batch_size, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = GLUMlp(embed_dim, ffn_dim)

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.scale_bias1 = ScaleBias(embed_dim)
        self.scale_bias2 = ScaleBias(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`mask` (B, L) bool, True = valid token. Invalid tokens are excluded as attention keys."""
        # Self-attention with residual
        residual = x
        attn_mask = None
        if mask is not None:
            # Additive mask materialized per head instead of key_padding_mask: under torch.compile
            # the broadcast key_padding_mask bias breaks the mem-efficient SDPA backward
            # ("(*bias): last dimension must be contiguous", torch 2.8).
            b, n_levels, _ = x.shape
            n_heads = self.attention.num_heads
            bias = torch.zeros(mask.shape, dtype=x.dtype, device=x.device).masked_fill(~mask, float("-inf"))
            attn_mask = (
                bias[:, None, None, :].expand(b, n_heads, n_levels, n_levels).reshape(b * n_heads, n_levels, n_levels)
            )
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(residual + self.scale_bias1(attn_out))

        # Feedforward with residual
        residual = x
        x = self.norm2(residual + self.scale_bias2(self.ffn(x)))

        return x


class SqueezeformerConvBlock(nn.Module):
    """
    Squeezeformer convolutional block.

    Combines:
    1. Expansion with GLU
    2. Depthwise separable convolution
    3. Efficient channel attention
    4. Projection back to original dimension
    5. GLU-based feedforward network

    All with residual connections and normalization.

    Parameters
    ----------
    channels : int
        Number of input/output channels
    kernel_size : int
        Depthwise convolution kernel size
    dilation_rate : int, default=1
        Dilation rate for convolution
    expand_ratio : int, default=4
        Channel expansion ratio

    Notes
    -----
    Input shape: (batch_size, seq_len, channels)
    Output shape: (batch_size, seq_len, channels)

    References
    ----------
    Kim et al., "Squeezeformer: An Efficient Transformer for Automatic
    Speech Recognition", NeurIPS 2022
    """

    def __init__(self, channels: int, kernel_size: int, dilation_rate: int = 1, expand_ratio: int = 4):
        super().__init__()
        hidden_channels = channels * expand_ratio

        # Point-wise expansion with GLU
        self.expand = nn.Linear(channels, hidden_channels)
        self.glu = GLU()

        # Depthwise convolution (channels are halved after GLU)
        self.dwconv = nn.Conv1d(
            hidden_channels // 2,
            hidden_channels // 2,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            dilation=dilation_rate,
            groups=hidden_channels // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(hidden_channels // 2, momentum=0.05)
        self.activation = nn.SiLU()

        # Channel attention
        self.eca = EfficientChannelAttention()

        # Point-wise projection
        self.project = nn.Linear(hidden_channels // 2, channels)
        self.scale_bias1 = ScaleBias(channels)

        # Feedforward network
        self.ffn = GLUMlp(channels, channels * 4)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.scale_bias2 = ScaleBias(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """`mask` (B, L) bool, True = valid token. Invalid tokens are zeroed before the depthwise
        conv (the stencil sees them like zero padding) and excluded from the ECA mean."""
        # Convolutional path with residual
        residual = x

        # Expand and gate
        h = self.expand(x)
        h = self.glu(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)

        # Depthwise convolution (requires channel-first format)
        h = h.transpose(1, 2)  # (B, C, L)
        h = self.dwconv(h)
        h = self.bn(h)
        h = h.transpose(1, 2)  # (B, L, C)
        h = self.activation(h)

        # Channel attention and projection
        h = self.eca(h, mask)
        h = self.project(h)
        x = residual + self.scale_bias1(h)

        # Feedforward path with residual
        residual = x
        x = self.norm(residual + self.scale_bias2(self.ffn(x)))

        return x


class HeightInterpolation(nn.Module):
    """Per-sample linear interpolation in physical height, without learnable parameters.

    Replaces the index-space `LevelInterpolation` for inputs with masked (underground) tokens: only
    valid tokens are used as neighbours, and queries are placed by their height `lz_q`, not by
    their index. Queries outside the valid input range are clamped to the edge tokens.
    """

    def forward(self, x: torch.Tensor, lz_in: torch.Tensor, valid: torch.Tensor, lz_q: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C) token features, lz_in: (B, L) token heights, increasing over the valid tokens,
        valid: (B, L) bool with invalid tokens contiguous at the bottom, lz_q: (B, Q) query heights.
        Returns (B, Q, C).
        """
        _, n_levels, n_channels = x.shape

        # Invalid tokens at -inf keep rows sorted and are never bracketing neighbours
        lz_in = lz_in.masked_fill(~valid, float("-inf"))
        n_invalid = (~valid).sum(dim=1, keepdim=True)  # index of the lowest valid token

        # Upper neighbour, restricted to [lowest valid + 1, top]
        hi = torch.searchsorted(lz_in.contiguous(), lz_q.contiguous())
        hi = torch.maximum(hi, n_invalid + 1).clamp(max=n_levels - 1)
        lo = hi - 1

        lz_lo = lz_in.gather(1, lo)
        lz_hi = lz_in.gather(1, hi)
        w = ((lz_q - lz_lo) / (lz_hi - lz_lo).clamp_min(1e-6)).clamp(0, 1).unsqueeze(-1)  # clamp = edge values

        x_lo = x.gather(1, lo.unsqueeze(-1).expand(-1, -1, n_channels))
        x_hi = x.gather(1, hi.unsqueeze(-1).expand(-1, -1, n_channels))
        return x_lo + w * (x_hi - x_lo)


class FourierEmbedding(nn.Module):
    """Linear projection of Fourier features sin/cos(pi * f_k * z) of a ~[0, 1] coordinate z.

    `n_freq` frequencies, log-spaced between 1 and `f_max`, give 2 * n_freq features.
    """

    def __init__(self, n_freq: int, dim_out: int, f_max: float = 64.0):
        super().__init__()
        self.register_buffer("freqs", torch.pi * torch.logspace(0, math.log10(f_max), n_freq))
        self.proj = nn.Linear(2 * n_freq, dim_out)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, Q) -> (B, Q, dim_out)"""
        arg = z.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1))


class SqueezeformerHModel(nn.Module):
    """Height-aware Squeezeformer (SQF-H).

    Squeezeformer backbone (alternating conv and transformer blocks) with a prediction and a confidence head, where
    1. underground input tokens (channel `idx_valid` < 0.5) are masked in every block,
    2. backbone features are interpolated in physical height (input channel `idx_lz`) to the
       query heights `lz_q` instead of in index space, and
    3. a Fourier embedding of the query height is added before the prediction head, so the head
       is a function of (state at height, height).

    `lz` in inputs and queries must be the same coordinate (unscaled `lz` feature/forcing).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_conf_head: bool,
        embed_dim: int,
        head_dim: int,
        num_blocks: int,
        num_heads: int,
        conv_kernel_size: int,
        idx_valid: int,
        idx_lz: int,
        n_freq: int = 16,
    ):
        super().__init__()
        # Architecture hparams as given, written to `config.json` by `save_pretrained`
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.idx_valid = idx_valid
        self.idx_lz = idx_lz

        # Input processing
        self.input_proj = nn.Linear(input_dim, embed_dim, bias=False)
        self.input_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Backbone: alternating conv and transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(SqueezeformerConvBlock(embed_dim, conv_kernel_size))
            self.blocks.append(TransformerEncoderBlock(embed_dim, num_heads, embed_dim * 4))

        # Height resampling and query height embedding
        self.height_interp = HeightInterpolation()
        self.z_embed = FourierEmbedding(n_freq, embed_dim)

        # Prediction head
        self.head_proj = nn.Linear(embed_dim, head_dim)
        self.head_ffn = GLUMlp(head_dim, head_dim * 2)
        self.pred_head = nn.Linear(head_dim, output_dim)
        if use_conf_head:
            self.conf_head = nn.Linear(head_dim, output_dim)
        else:
            self.conf_head = None

    def forward(self, x: torch.Tensor, lz_q: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, levels_in, features), lz_q: (batch, levels_out) query heights.
        Returns predictions (and confidence) of shape (batch, levels_out, output_dim).
        """
        valid = x[..., self.idx_valid] > 0.5  # (batch, levels_in)
        lz_in = x[..., self.idx_lz]
        m = valid.unsqueeze(-1).to(x.dtype)

        # Input embedding
        x = self.input_norm(self.input_proj(x)) * m  # out: (batch, levels_in, embed_dim)

        # Backbone processing, re-zeroing invalid tokens after every block
        for block in self.blocks:
            x = block(x, valid) * m

        # Resample to query heights and add query height embedding
        x = self.height_interp(x, lz_in, valid, lz_q)  # out: (batch, levels_out, embed_dim)
        x = x + self.z_embed(lz_q)

        # Prediction head
        x = F.silu(self.head_proj(x))  # out: (batch, levels_out, head_dim)
        x = self.head_ffn(x)  # out: (batch, levels_out, head_dim)
        predictions = self.pred_head(x)  # out: (batch, levels_out, output_dim)

        if self.conf_head is None:
            return predictions
        else:
            confidence = self.conf_head(x)  # out: (batch, levels_out, output_dim)
            return predictions, confidence

    @classmethod
    def from_lightning_checkpoint(cls, path: str | pathlib.Path, map_location: str = "cpu") -> SqueezeformerHModel:
        """The network of a `LitSqueezeFormerH` checkpoint, read from the checkpoint dict (no Lightning needed):
        architecture hparams from `hyper_parameters` (training ones like `lr` dropped), weights from `state_dict`
        without the Lit (`model.`) and `torch.compile` (`_orig_mod.`) prefixes."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hparams = {k: v for k, v in ckpt["hyper_parameters"].items() if k in ARCH_HPARAMS}
        model = cls(**hparams)
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if not k.startswith("model."):
                continue  # not part of the network
            k = k.removeprefix("model.").removeprefix("_orig_mod.")
            state_dict[k] = v
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, out_dir: str | pathlib.Path, **meta) -> None:
        """`model.safetensors` (weights) and `config.json` (architecture, format and otprof versions, hparams, and
        `meta`, e.g. the variable order of inputs, forcings and targets)."""
        from safetensors.torch import save_file  # otprof[torch]

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "arch": ARCH,
            "format_version": FORMAT_VERSION,
            **get_bundle_meta(FEATURES_VERSION),
            "hparams": self.config,
            **meta,
        }
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, out_dir / "model.safetensors")

    @classmethod
    def from_pretrained(cls, model_dir: str | pathlib.Path, map_location: str = "cpu") -> SqueezeformerHModel:
        """Load a model saved by `save_pretrained`. Raises if the architecture or format differ, or if the MAJOR
        version of the features or the input schema the model was trained with differs from the installed otprof.
        The full config is available as `model.bundle_config`."""
        from safetensors.torch import load_file  # otprof[torch]

        model_dir = pathlib.Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        if config.get("arch") != ARCH:
            raise ValueError(f"{model_dir}: architecture '{config.get('arch')}', expected '{ARCH}'")
        if config.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{model_dir}: format_version {config.get('format_version')}, expected {FORMAT_VERSION}")
        installed = {"features_version": FEATURES_VERSION, "schema_version": schema.SCHEMA_VERSION}
        check_versions(config, installed, source=str(model_dir))
        model = cls(**config["hparams"])
        model.load_state_dict(load_file(model_dir / "model.safetensors", device=map_location))
        model.bundle_config = config
        return model


# Constructor arguments of `SqueezeformerHModel`: the architecture hparams kept from a Lightning checkpoint
ARCH_HPARAMS = [
    "input_dim", "output_dim", "use_conf_head", "embed_dim", "head_dim", "num_blocks", "num_heads",
    "conv_kernel_size", "idx_valid", "idx_lz", "n_freq",
]  # fmt: skip

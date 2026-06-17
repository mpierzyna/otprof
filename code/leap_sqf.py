"""LEAP Squeezeformer based on https://www.kaggle.com/code/shlomoron/leap-training-1#Layers by shlomoron.
Converted to Pytorch and refactored with Sonnet 4.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging


logger = logging.getLogger(__name__)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C) -> (B, C)
        y = x.mean(dim=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        attn_out, _ = self.attention(x, x, x, need_weights=False)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional path with residual
        residual = x

        # Expand and gate
        h = self.expand(x)
        h = self.glu(h)

        # Depthwise convolution (requires channel-first format)
        h = h.transpose(1, 2)  # (B, C, L)
        h = self.dwconv(h)
        h = self.bn(h)
        h = h.transpose(1, 2)  # (B, L, C)
        h = self.activation(h)

        # Channel attention and projection
        h = self.eca(h)
        h = self.project(h)
        x = residual + self.scale_bias1(h)

        # Feedforward path with residual
        residual = x
        x = self.norm(residual + self.scale_bias2(self.ffn(x)))

        return x


class VariableLevelEmbedding(nn.Module):
    """Variable level embedding layer.
    Given input (batch, levels_in, input_dim), outputs (batch, levels_out, embed_dim),
    so projects levels_in to levels_out and then embeds input_dim to embed_dim.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        levels_in: int | None,
        levels_out: int | None,
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        # If specified, we need both levels
        if levels_in is not None:
            assert levels_out is not None, "levels_out must be specified if levels_in is given"
        if levels_out is not None:
            assert levels_in is not None, "levels_in must be specified if levels_out is given"

        # Linear layer to adjust levels if they differ
        if (levels_in is not None and levels_out is not None) and (levels_in != levels_out):
            logger.info(f"Adding level adjustment layer: {levels_in} -> {levels_out}.")
            self.lin_levels = nn.Linear(levels_in, levels_out)
        else:
            self.lin_levels = None

        self.lin_embed = nn.Linear(input_dim, output_dim, bias=bias)  # bias=False in original input layer, but why?

        if normalize:
            self.layer_norm = nn.LayerNorm(output_dim, eps=1e-6)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take input with (batch, levels_in, input_dim) and output (batch, levels_out, embed_dim)."""
        if self.lin_levels is not None:
            # Levels are dim 1, so roll, apply, and roll back
            x = x.transpose(1, 2)  # (B, input_dim, levels_in)
            x = self.lin_levels(x)
            x = x.transpose(1, 2)  # (B, levels_out, input_dim)
        x = self.lin_embed(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class InterpolationLevelEmbedding(nn.Module):
    """Interpolation level embedding layer.
    Similar to VariableLevelEmbedding but instead of projecting inputs using linear layer, simply interpolate.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        levels_in: int | None,
        levels_out: int | None,
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        # If specified, we need both levels
        if levels_in is not None:
            assert levels_out is not None, "levels_out must be specified if levels_in is given"
        if levels_out is not None:
            assert levels_in is not None, "levels_in must be specified if levels_out is given"

        # Linear layer to adjust levels if they differ
        self.levels_out = None
        if (levels_in is not None and levels_out is not None) and (levels_in != levels_out):
            logger.info(f"Interpolating levels before embedding: {levels_in} -> {levels_out}.")
            self.levels_out = levels_out

        self.lin_embed = nn.Linear(input_dim, output_dim, bias=bias)  # bias=False in original input layer, but why?

        if normalize:
            self.layer_norm = nn.LayerNorm(output_dim, eps=1e-6)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take input with (batch, levels_in, input_dim) and output (batch, levels_out, embed_dim)."""
        if self.levels_out is not None:
            # Levels are dim 1, so roll, apply, and roll back
            x = x.transpose(1, 2)  # (B, input_dim, levels_in)
            x = nn.functional.interpolate(x, size=self.levels_out, mode="linear", align_corners=True)
            x = x.transpose(1, 2)  # (B, levels_out, input_dim)
        x = self.lin_embed(x)  # (B, levels_out, embed_dim)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class LevelInterpolation(nn.Module):
    """Simple level interpolation without any learnable parameters."""

    def __init__(self, levels_out: int | None, mode: str = "linear"):
        super().__init__()
        self.levels_out = levels_out
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take input with (batch, levels_in, input_dim) and output (batch, levels_out, input_dim)."""
        # Don't do anything if levels_out is None
        if self.levels_out is None:
            return x

        # Don't interpolate if levels are the same
        _, levels_in, _ = x.shape
        if levels_in == self.levels_out:
            return x

        # Levels are dim 1, so roll, apply, and roll back
        x = x.transpose(1, 2)  # (B, input_dim, levels_in)
        x = nn.functional.interpolate(x, size=self.levels_out, mode=self.mode, align_corners=True)
        x = x.transpose(1, 2)  # (B, levels_out, input_dim)
        return x


class SqueezeformerModel(nn.Module):
    """
    Squeezeformer model for sequence prediction with confidence estimation.

    Hybrid architecture alternating between:
    - Squeezeformer convolutional blocks (local patterns)
    - Transformer encoder blocks (long-range dependencies)

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    embed_dim : int, default=384
        Model embedding dimension
    head_dim : int, default=2048
        Hidden dimension in prediction head
    num_blocks : int, default=12
        Number of conv-transformer block pairs
    num_heads : int, default=4
        Number of attention heads in transformer
    conv_kernel_size : int, default=15
        Kernel size for convolutional blocks
    output_dim : int, default=20
        Output feature dimension

    Notes
    -----
    Input shape: (batch_size, seq_len + static_len)
    Output shape: Tuple[(batch_size, levels, output_dim), (batch_size, levels, output_dim)]

    The model outputs both predictions and confidence scores,
    concatenated into a single flattened vector.
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
        levels_in: int | None = None,
        levels_out: int | None = None,
    ):
        super().__init__()

        # Input finetuning placeholder
        self.input_finetuning = nn.Identity()

        # Input processing
        self.input_proj = nn.Linear(input_dim, embed_dim, bias=False)
        self.input_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # self.input_emb = VariableLevelEmbedding(input_dim, embed_dim, levels_in, levels_out)
        # self.input_emb = InterpolationLevelEmbedding(input_dim, embed_dim, levels_in, levels_out)
        # self.input_emb = VariableLevelEmbedding(input_dim, embed_dim, None, None, bias=False, normalize=True)

        # Backbone: alternating conv and transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(SqueezeformerConvBlock(embed_dim, conv_kernel_size))
            self.blocks.append(TransformerEncoderBlock(embed_dim, num_heads, embed_dim * 4))

        # Prediction head
        self.level_interp = LevelInterpolation(levels_out=levels_out)
        self.head_proj = nn.Linear(embed_dim, head_dim)
        # self.head_proj = VariableLevelEmbedding(embed_dim, head_dim, levels_in, levels_out, bias=True, normalize=False)
        self.head_ffn = GLUMlp(head_dim, head_dim * 2)
        self.pred_head = nn.Linear(head_dim, output_dim)
        if use_conf_head:
            self.conf_head = nn.Linear(head_dim, output_dim)
        else:
            self.conf_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, levels, features)

        Returns
        -------
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
            Predictions and optional confidence scores.
        """
        # Input embedding
        x = self.input_finetuning(x)
        x = self.input_proj(x)  # out: (batch, levels, embed_dim)
        x = self.input_norm(x)  # out: (batch, levels, embed_dim)

        # Backbone processing
        for block in self.blocks:
            x = block(x)  # out: (batch, levels, embed_dim)

        # Prediction head
        x = F.silu(self.head_proj(x))  # out: (batch, levels, head_dim)
        x = self.head_ffn(x)  # out: (batch, levels, head_dim)

        # Interpolate prediction to desired output levels before heads
        x = self.level_interp(x)  # out: (batch, levels_out, output_dim)
        predictions = self.pred_head(x)  # out: (batch, levels_out, output_dim)

        if self.conf_head is None:
            return predictions
        else:
            confidence = self.conf_head(x)  # out: (batch, levels_out, output_dim)
            return predictions, confidence


if __name__ == "__main__":
    # Simple test
    X = torch.rand(10, 30, 5)  # batch, levels, features
    model = SqueezeformerModel(
        input_dim=X.shape[2],
        levels_in=X.shape[1],
        output_dim=2,
        levels_out=100,
        use_conf_head=True,
        embed_dim=32,
        head_dim=64,
        num_blocks=2,
        num_heads=4,
    )
    y_hat, y_conf = model(X)
    assert y_hat.shape == (10, 100, 2)
    assert y_conf.shape == (10, 100, 2)

"""SQF-H network: underground tokens are masked, features are interpolated in physical height, and a column shifted
up by underground tokens gives the same output (the failure of the non-height-aware SqueezeFormer)."""

import pytest
import torch

from otprof.models.sqf_h.model import HeightInterpolation, SqueezeformerHModel

N_FEAT = 8
IDX_VALID, IDX_LZ = 0, 1
N_LEVELS = 30


def make_model(seed: int = 0) -> SqueezeformerHModel:
    torch.manual_seed(seed)
    return SqueezeformerHModel(
        input_dim=N_FEAT,
        output_dim=3,
        use_conf_head=True,
        embed_dim=16,
        head_dim=32,
        num_blocks=2,
        num_heads=2,
        conv_kernel_size=3,
        idx_valid=IDX_VALID,
        idx_lz=IDX_LZ,
    )


def make_input(n_invalid: list[int], n_levels: int = N_LEVELS, seed: int = 1) -> torch.Tensor:
    """Random profiles with `n_invalid[b]` masked bottom tokens and increasing lz in [0.1, 0.9]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(n_invalid), n_levels, N_FEAT, generator=g)
    x[..., IDX_LZ] = torch.linspace(0.1, 0.9, n_levels)
    for b, k in enumerate(n_invalid):
        x[b, :, IDX_VALID] = (torch.arange(n_levels) >= k).float()
    return x


def query_forcings(batch: int, n: int = 99) -> torch.Tensor:
    """Forcings [z_agl, lz] of `n` query heights, lz evenly in [0, 1] (z_agl unused by the network)."""
    lz = torch.linspace(0.0, 1.0, n).expand(batch, n)
    return torch.stack([torch.zeros_like(lz), lz], dim=-1)


@pytest.mark.parametrize("train", [False, True])
def test_mask_invariance(train: bool):
    """Masked tokens' contents must not influence the output at all."""
    model = make_model().train(train)
    x = make_input([0, 3, 7, 1])
    x_pert = x.clone()
    masked = x[..., IDX_VALID] < 0.5
    noise = 100 * torch.randn(x.shape, generator=torch.Generator().manual_seed(2))
    noise[..., IDX_VALID] = 0  # keep the mask itself
    x_pert[masked] += noise[masked]

    frc = query_forcings(x.shape[0])
    with torch.no_grad():
        y, c = model(x, frc)
        y_p, c_p = model(x_pert, frc)
    assert torch.equal(y, y_p)
    assert torch.equal(c, c_p)


def test_interpolation_identity():
    """Queries at valid token heights return those tokens' features; outside the range they clamp."""
    x = make_input([0, 4])
    h = torch.randn(2, N_LEVELS, 5)
    valid = x[..., IDX_VALID] > 0.5
    lz_in = x[..., IDX_LZ]
    interp = HeightInterpolation()

    for b, k in enumerate([0, 4]):
        out = interp(h[b : b + 1, k:].clone(), lz_in[b : b + 1, k:], valid[b : b + 1, k:], lz_in[b : b + 1, k:])
        torch.testing.assert_close(out, h[b : b + 1, k:])
        out = interp(h[b : b + 1], lz_in[b : b + 1], valid[b : b + 1], lz_in[b : b + 1, k:])
        torch.testing.assert_close(out, h[b : b + 1, k:])

    # Midpoint between two valid tokens, and clamping below / above the valid range
    lz_q = torch.stack([(lz_in[:, 10] + lz_in[:, 11]) / 2, torch.zeros(2), torch.ones(2)], dim=1)
    out = interp(h, lz_in, valid, lz_q)
    torch.testing.assert_close(out[:, 0], (h[:, 10] + h[:, 11]) / 2)
    torch.testing.assert_close(out[:, 1], h[torch.arange(2), torch.tensor([0, 4])])  # lowest valid token
    torch.testing.assert_close(out[:, 2], h[:, -1])


def test_shift_invariance():
    """The v183 failure: k underground tokens below an otherwise identical column (same lz) must not
    change the output. Prepending masked tokens shifts every valid token up by k indices."""
    model = make_model().eval()
    k = 5
    x = make_input([0, 0], n_levels=20)
    junk = torch.randn(2, k, N_FEAT, generator=torch.Generator().manual_seed(3))
    junk[..., IDX_VALID] = 0
    junk[..., IDX_LZ] = 0.0  # underground copies of the lowest level in the real data
    x_shift = torch.cat([junk, x], dim=1)

    frc = query_forcings(2)
    with torch.no_grad():
        y, c = model(x, frc)
        y_s, c_s = model(x_shift, frc)
    torch.testing.assert_close(y, y_s, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c, c_s, rtol=1e-5, atol=1e-5)

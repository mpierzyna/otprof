import numpy as np
import pytest
import xarray as xr

from otprof.physics import ot

H = 10_000.0  # column height [m]
CN2 = 1e-16  # constant Cn2 [m^-2/3]


@pytest.fixture
def column() -> tuple[xr.DataArray, xr.DataArray]:
    """Constant Cn2 on a fine non-uniform grid from 0 to H, so trapezoid integrals of powers of z are near exact."""
    z = H * np.linspace(0, 1, 20_001) ** 1.5
    return xr.DataArray(np.full_like(z, CN2), dims="z"), xr.DataArray(z, dims="z")


def test_trapz_matches_numpy_and_exact():
    x = np.linspace(0, 5, 100) ** 2  # non-uniform spacing
    y = x**2 + 3 * x + 2
    exact = (x[-1] ** 3 / 3 + 3 / 2 * x[-1] ** 2 + 2 * x[-1]) - (x[0] ** 3 / 3 + 3 / 2 * x[0] ** 2 + 2 * x[0])

    res = ot._trapz_non_uniform(xr.DataArray(y, dims="z"), xr.DataArray(x, dims="z"), dim="z").item()
    assert res == pytest.approx(np.trapezoid(y, x))
    assert res == pytest.approx(exact, rel=1e-3)


def test_piecewise_integration_with_extended_mask():
    x = xr.DataArray(np.linspace(0, 5, 100) ** 2, dims="z")
    y = x**2 + 3 * x + 2
    full = ot._trapz_non_uniform(y, x, dim="z").item()
    mask_1, mask_2 = x < 10, x >= 10

    # Without extension, the segment between the two masks is lost
    seg = [ot._trapz_non_uniform(y.where(m), x.where(m), dim="z").item() for m in (mask_1, mask_2)]
    assert sum(seg) < full

    mask_1_ext = ot.extend_mask_right(mask_1, dim="z")
    seg = [ot._trapz_non_uniform(y.where(m), x.where(m), dim="z").item() for m in (mask_1_ext, mask_2)]
    assert sum(seg) == pytest.approx(full)


def test_r0_constant_cn2(column):
    cn2, z = column
    expected = (0.423 * ot.K**2 * CN2 * H) ** (-3 / 5)
    assert ot.fried_r0_xr(cn2, z, dim="z").item() == pytest.approx(expected, rel=1e-6)


def test_isoplanatic_angle_constant_cn2(column):
    cn2, z = column
    expected = 0.057 * ot.LAMBDA ** (6 / 5) * (CN2 * 3 / 8 * H ** (8 / 3)) ** (-3 / 5)
    assert ot.isoplanatic_angle_xr(cn2, z, dim="z").item() == pytest.approx(expected, rel=1e-4)


def test_wave_coherence_time_constant_cn2_and_wind(column):
    cn2, z = column
    m = 10.0
    expected = 0.057 * ot.LAMBDA ** (6 / 5) * (CN2 * m ** (5 / 3) * H) ** (-3 / 5)
    assert ot.wave_coherence_time_xr(cn2, xr.full_like(z, m), z, dim="z").item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("mode, c", [("plane", 1.23), ("spherical", 0.5)])
def test_rytov_variance_constant_cn2(column, mode, c):
    cn2, z = column
    expected = c * ot.K ** (7 / 6) * CN2 * 6 / 11 * H ** (11 / 6)
    assert ot._rytov_var(cn2, z, "z", ot._trapz_non_uniform, mode).item() == pytest.approx(expected, rel=1e-4)


def test_scint_index_weak_turbulence_limit(column):
    """For a small Rytov variance, the scintillation index approaches the Rytov variance."""
    cn2, z = column
    cn2 = cn2 * 1e-4
    sr2 = ot._rytov_var(cn2, z, "z", ot._trapz_non_uniform, "plane").item()
    assert sr2 < 1e-2
    assert ot.scint_index_xr(cn2, z, dim="z", mode="plane").item() == pytest.approx(sr2, rel=1e-2)


def test_seeing_from_r0(column):
    cn2, z = column
    r0 = ot.fried_r0_xr(cn2, z, dim="z").item()
    assert ot.seeing_xr(cn2, z, dim="z").item() == pytest.approx(0.98 * ot.LAMBDA / r0)


def test_numpy_matches_xarray():
    rng = np.random.default_rng(0)
    z = np.sort(rng.uniform(0, 20_000, (5, 40)), axis=1)
    cn2 = 10 ** rng.uniform(-17, -14, (5, 40))
    m = rng.uniform(0, 30, (5, 40))
    z_xr, cn2_xr, m_xr = (xr.DataArray(a, dims=("s", "z")) for a in (z, cn2, m))

    np.testing.assert_allclose(ot.fried_r0_np(cn2, z, axis=1), ot.fried_r0_xr(cn2_xr, z_xr, "z"))
    np.testing.assert_allclose(ot.isoplanatic_angle_np(cn2, z, axis=1), ot.isoplanatic_angle_xr(cn2_xr, z_xr, "z"))
    np.testing.assert_allclose(
        ot.wave_coherence_time_np(cn2, m, z, axis=1), ot.wave_coherence_time_xr(cn2_xr, m_xr, z_xr, "z")
    )
    np.testing.assert_allclose(
        ot.scint_index_np(cn2, z, axis=1, mode="plane"), ot.scint_index_xr(cn2_xr, z_xr, "z", mode="plane")
    )


def test_invalid_mode(column):
    cn2, z = column
    with pytest.raises(ValueError):
        ot.scint_index_xr(cn2, z, dim="z", mode="cylindrical")

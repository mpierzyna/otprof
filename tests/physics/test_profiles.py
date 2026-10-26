import numpy as np
import pytest
import xarray as xr

from otprof.physics.profiles import hufnagel_valley, hufnagel_valley_xr


def test_ground_value():
    z = np.linspace(0, 25_000, 101)
    cn2 = hufnagel_valley(cn2_0=1e-14, m=np.full_like(z, 20.0), z=z)
    assert cn2[0] == pytest.approx(1e-14)


def test_hv57():
    """With the HV5/7 parameters (W = 21 m/s, A = 1.7e-14) the profile matches the closed form."""
    z = np.linspace(0, 30_000, 3001)  # 10 m spacing, the 5-20 km wind integral is near exact
    h = z / 1000
    cn2_0 = 1.7e-14 + 2.7e-16
    m = np.full_like(z, 21.0 * np.sqrt(15 / 14.99))  # the wind integral excludes the 20 km level
    expected = 8.2e-26 * 21.0**2 * h**10 * np.exp(-h) + 2.7e-16 * np.exp(-h / 1.5) + 1.7e-14 * np.exp(-h / 0.1)
    np.testing.assert_allclose(hufnagel_valley(cn2_0=cn2_0, m=m, z=z), expected, rtol=1e-9)


def test_xarray_matches_numpy():
    rng = np.random.default_rng(0)
    z = np.linspace(0, 25_000, 60)
    z_eval = np.linspace(0, 25_000, 120)
    cn2_0 = 10 ** rng.uniform(-16, -13, 4)
    m = rng.uniform(0, 40, (4, 60))

    out = hufnagel_valley_xr(
        cn2_0=xr.DataArray(cn2_0, dims="sample"),
        m=xr.DataArray(m, dims=("sample", "level")),
        z=xr.DataArray(z, dims="level"),
        z_eval=xr.DataArray(z_eval, dims="level"),
        dim="level",
    )
    assert out.dims == ("sample", "level")
    assert out.sizes["level"] == 120
    for i in range(4):
        np.testing.assert_allclose(out.isel(sample=i), hufnagel_valley(cn2_0[i], m[i], z, z_eval))

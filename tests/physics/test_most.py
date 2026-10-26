import numpy as np
import pytest
import xarray as xr

from otprof.physics.most import w71_flux_ct2


def test_neutral_limit():
    """zeta -> 0+ gives g3 = 4.9."""
    L = np.array([1e12, 1e12])
    ct2 = w71_flux_ct2(T_st=0.1, z=2.0, L=L, L_bounds=(-np.inf, np.inf))
    np.testing.assert_allclose(ct2, 0.1**2 * 2.0 ** (-2 / 3) * 4.9)


@pytest.mark.parametrize(
    "zeta, g3",
    [
        (-1.0, 4.9 * 7.1 ** (-2 / 3)),  # unstable
        (1.0, 4.9 * 3.2),  # stable
    ],
)
def test_stability_branches(zeta, g3):
    z = 2.0
    ct2 = w71_flux_ct2(T_st=1.0, z=z, L=np.array([z / zeta]), L_bounds=(-np.inf, np.inf))
    np.testing.assert_allclose(ct2, z ** (-2 / 3) * g3)


def test_default_clips_to_percentiles():
    L = np.linspace(-100, 100, 400)  # even count: L = 0 not included
    lo, hi = np.percentile(L, 2.5), np.percentile(L, 97.5)
    np.testing.assert_allclose(
        w71_flux_ct2(T_st=1.0, z=2.0, L=L),
        w71_flux_ct2(T_st=1.0, z=2.0, L=L, L_bounds=(lo, hi)),
    )


def test_fixed_bounds_independent_of_subset():
    L = np.linspace(-100, 100, 400)  # even count: L = 0 not included
    bounds = (-50.0, 50.0)
    full = w71_flux_ct2(T_st=1.0, z=2.0, L=L, L_bounds=bounds)
    single = w71_flux_ct2(T_st=1.0, z=2.0, L=L[:1], L_bounds=bounds)
    np.testing.assert_allclose(single, full[:1])


def test_xarray_in_xarray_out():
    L = xr.DataArray(np.linspace(-100, 100, 11), dims="sample", coords={"sample": np.arange(11)})
    ct2 = w71_flux_ct2(T_st=xr.ones_like(L), z=2.0, L=L)
    assert isinstance(ct2, xr.DataArray)
    assert ct2.dims == ("sample",)
    np.testing.assert_array_equal(ct2["sample"], L["sample"])

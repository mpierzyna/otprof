import numpy as np
import pytest
import xarray as xr

from otprof.physics.most import w71_flux_ct2


def test_neutral_limit():
    """zeta = 0 (neutral, e.g. no heat flux) gives g3 = 4.9."""
    ct2 = w71_flux_ct2(T_st=0.1, zeta=np.array([0.0, 0.0]), z=2.0)
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
    ct2 = w71_flux_ct2(T_st=1.0, zeta=np.array([zeta]), z=z)
    np.testing.assert_allclose(ct2, z ** (-2 / 3) * g3)


def test_xarray_in_xarray_out():
    zeta = xr.DataArray(np.linspace(-1, 1, 11), dims="sample", coords={"sample": np.arange(11)})
    ct2 = w71_flux_ct2(T_st=xr.ones_like(zeta), zeta=zeta, z=2.0)
    assert isinstance(ct2, xr.DataArray)
    assert ct2.dims == ("sample",)
    np.testing.assert_array_equal(ct2["sample"], zeta["sample"])

import numpy as np
import xarray as xr

from otprof.physics.vertical import get_half_levels, interp_vert


def _profiles() -> tuple[xr.Dataset, xr.DataArray]:
    rng = np.random.default_rng(0)
    z = np.sort(rng.uniform(0, 10_000, (3, 30)), axis=1)
    z = xr.DataArray(z, dims=("sample", "bottom_top"))
    ds = xr.Dataset(
        {
            "a": 2.0 * z + 1.0,  # linear in z: interpolation is exact
            "b": -0.5 * z,
            "sfc": xr.DataArray(np.arange(3.0), dims="sample"),  # no vertical dim: dropped
        }
    )
    ds["a"].attrs["units"] = "K"
    return ds, z


def test_interp_vert_linear_profiles_exact():
    ds, z = _profiles()
    z_target = (z.isel(bottom_top=slice(None, -1)) + z.isel(bottom_top=slice(1, None)).values) / 2
    out = interp_vert(ds, z=z, z_target=z_target, dim="bottom_top")

    np.testing.assert_allclose(out["a"], 2.0 * z_target + 1.0)
    np.testing.assert_allclose(out["b"], -0.5 * z_target)
    assert "sfc" not in out
    assert out["a"].attrs["units"] == "K"
    np.testing.assert_array_equal(out["bottom_top"], z_target)


def test_interp_vert_outside_range_is_nan():
    ds, z = _profiles()
    z_target = xr.concat(
        [z.isel(bottom_top=[0]) - 10.0, z.isel(bottom_top=[1]), z.isel(bottom_top=[-1]) + 10.0], dim="bottom_top"
    )
    out = interp_vert(ds, z=z, z_target=z_target, dim="bottom_top")
    assert out["a"].isel(bottom_top=0).isnull().all()
    np.testing.assert_allclose(out["a"].isel(bottom_top=1), ds["a"].isel(bottom_top=1))
    assert out["a"].isel(bottom_top=2).isnull().all()


def test_get_half_levels():
    v = xr.DataArray([0.0, 2.0, 6.0, 12.0], dims="bottom_top", coords={"bottom_top": np.arange(4)})
    v_h = get_half_levels(v, "bottom_top")
    assert v_h.dims == ("bottom_top_h",)
    assert "bottom_top" not in v_h.coords
    np.testing.assert_allclose(v_h, [1.0, 4.0, 9.0])

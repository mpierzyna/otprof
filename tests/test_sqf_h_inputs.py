import numpy as np
import pytest
import xarray as xr

from otprof import schema
from otprof.features import potential_temperature
from otprof.models.sqf_h.inputs import FEATURES, FEATURES_VERSION, INPUTS, make_inputs, make_targets


def test_outputs_and_levels(canonical):
    X = make_inputs(canonical)
    assert list(X.data_vars) == list(FEATURES)
    assert X["p_level"].values.tolist() == list(schema.P_LEVELS_HPA[:-1])  # 20 hPa only enters the top gradient
    assert X.sizes["bottom_top"] == 30
    for f, (kind, _) in INPUTS.items():  # the declared kind matches what make_inputs computes
        assert ("bottom_top" in X[f].dims) == (kind == "profile"), f
    assert X.attrs["otprof_features_version"] == FEATURES_VERSION
    assert X.attrs["otprof_schema_version"] == schema.SCHEMA_VERSION


def test_gradients_to_level_above(canonical):
    ds = canonical.copy()
    dz = ds["z"].isel(bottom_top=slice(1, None)).values - ds["z"].isel(bottom_top=slice(0, -1)).values
    ds["u"] = (2e-3 * ds["z"]).assign_attrs(units="m s-1")  # constant shear 2e-3 1/s
    ds["v"] = xr.zeros_like(ds["u"])
    X = make_inputs(ds)
    np.testing.assert_allclose(X["S"], 2e-3, rtol=1e-4)
    th = potential_temperature(ds["tk"], ds["p"]).values
    expected = (th[:, 1:] - th[:, :-1]) / dz
    np.testing.assert_allclose(X["dth_dz"].transpose("time", "bottom_top", "loc"), expected, rtol=1e-4)


def test_th_from_source_is_kept(canonical):
    ds = canonical.assign(th=canonical["tk"] + 10).drop_vars("tk")
    ds["th"].attrs["units"] = "K"
    X = make_inputs(ds)
    xr.testing.assert_equal(
        X["th"],
        (canonical["tk"] + 10).isel(bottom_top=slice(0, -1)).drop_vars("p_level").assign_coords(p_level=X["p_level"]),
    )


def test_height_features(canonical):
    X = make_inputs(canonical)
    z_agl = canonical["z"] - canonical["z_sfc_src"]
    np.testing.assert_allclose(X["z_agl"], z_agl.isel(bottom_top=slice(0, -1)).transpose(*X["z_agl"].dims))
    p_nominal = np.array(schema.P_LEVELS_HPA[:-1]) * 100.0
    expected_valid = p_nominal[None, :, None] < canonical["p_sfc"].values[:, None, :]
    np.testing.assert_array_equal(X["valid"].transpose("time", "bottom_top", "loc"), expected_valid)
    assert float(X["lz"].min()) >= 0
    np.testing.assert_allclose(X["lp_sfc"], np.log(canonical["p_sfc"]))
    xr.testing.assert_equal(X["z_sfc"].drop_attrs(), canonical["z_sfc_src"].drop_attrs())


def test_z_sfc_target(canonical):
    """Training data carry the terrain of the target column; z_agl and valid stay with the source's surface."""
    target = xr.full_like(canonical["z_sfc_src"], 42.0).assign_attrs(units="m")
    X = make_inputs(canonical.assign(z_sfc_target=target))
    assert (X["z_sfc"] == 42.0).all()
    np.testing.assert_allclose(X["z_agl"], make_inputs(canonical)["z_agl"])  # z_agl stays with the source surface


def test_model_grid(canonical):
    """Without the pressure level coordinate, the levels are model levels (grid detected)."""
    ds = canonical.isel(bottom_top=slice(0, 12)).drop_vars("p_level")
    X = make_inputs(ds)
    assert X.sizes["bottom_top"] == 11 and X.attrs["otprof_grid"] == "model"
    assert (X["valid"] == 1).all()
    xr.testing.assert_identical(X, make_inputs(ds, grid="model"))


def test_make_targets():
    """lz of the target heights is the lz of the inputs; lcn2 only if Cn2 is given."""
    z_agl = xr.DataArray([[0.0, 10.0, 1_000.0]], dims=("sample", "bottom_top"))
    y = make_targets(xr.Dataset({"z_agl": z_agl, "cn2": 10 ** (-15 + 0 * z_agl)}))
    np.testing.assert_allclose(y["lz"], np.log1p(z_agl.clip(min=0) / 10.0) / np.log1p(3_000.0))
    np.testing.assert_allclose(y["lcn2"], -15.0)
    assert "lcn2" not in make_targets(xr.Dataset({"z_agl": z_agl}))


def test_repeated_levels_get_nearest_gradient(canonical):
    ds = canonical.copy(deep=True)
    for v in ["z", "u", "v", "tk"]:  # lowest two levels repeat the third, as WRF filled below ground
        ds[v][:, 0:2, :] = ds[v][:, 2:3, :].values
    X = make_inputs(ds)
    assert not X["S"].isnull().any() and not X["dth_dz"].isnull().any()
    np.testing.assert_array_equal(X["S"].isel(bottom_top=0), X["S"].isel(bottom_top=2))


def test_w71_bounds_fixed_vs_percentile(canonical):
    fixed = (-50.0, 50.0)
    full = make_inputs(canonical, w71_L_bounds=fixed)["lct2_w71f"]
    single = make_inputs(canonical.isel(time=[0]), w71_L_bounds=fixed)["lct2_w71f"]
    xr.testing.assert_allclose(single, full.isel(time=[0]))


def test_invalid_input_rejected(canonical):
    with pytest.raises(schema.SchemaError):
        make_inputs(canonical.drop_vars("blh"))

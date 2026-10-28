import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otprof import features, schema


def make_canonical(n_time: int = 3, n_loc: int = 2, seed: int = 0) -> xr.Dataset:
    """Small canonical dataset on the pressure grid with plausible, smooth profiles."""
    rng = np.random.default_rng(seed)
    p = np.array(schema.P_LEVELS_HPA, dtype=float) * 100
    z = 7_000.0 * np.log(101_325.0 / p)  # isothermal-ish scale height, increasing with level
    shape = (n_time, len(p), n_loc)
    dims3, dims2 = ("time", "bottom_top", "loc"), ("time", "loc")
    ds = xr.Dataset(
        {
            "z": (dims3, np.broadcast_to(z[None, :, None], shape) + rng.normal(0, 5, shape)),
            "p": (("bottom_top",), (p).astype(int)),
            "u": (dims3, 5 + z[None, :, None] / 1_000 + rng.normal(0, 1, shape)),
            "v": (dims3, rng.normal(0, 3, shape)),
            "tk": (dims3, np.broadcast_to(288 - 0.0065 * np.minimum(z, 11_000)[None, :, None], shape).copy()),
            "p_sfc": (dims2, rng.uniform(95_000, 102_000, (n_time, n_loc))),
            "z_sfc_src": (("loc",), rng.uniform(0, 300, n_loc)),
            "ust": (dims2, rng.uniform(0.1, 0.6, (n_time, n_loc))),
            "hfx": (dims2, rng.uniform(-50, 200, (n_time, n_loc))),
            "lh": (dims2, rng.uniform(0, 300, (n_time, n_loc))),
            "blh": (dims2, rng.uniform(200, 2_000, (n_time, n_loc))),
            "u10": (dims2, rng.normal(0, 5, (n_time, n_loc))),
            "v10": (dims2, rng.normal(0, 5, (n_time, n_loc))),
            "tk2": (dims2, rng.uniform(270, 300, (n_time, n_loc))),
            "slp": (dims2, rng.uniform(990, 1_030, (n_time, n_loc))),
            "lsm": (dims2, np.ones((n_time, n_loc))),
        },
        coords={
            "time": pd.date_range("2017-06-01", periods=n_time, freq="6h"),
            "p_level": ("bottom_top", list(schema.P_LEVELS_HPA)),
            "lat": ("loc", np.linspace(50, 52, n_loc)),
            "lon": ("loc", np.linspace(4, 6, n_loc)),
        },
    )
    return schema.set_attrs(ds)


@pytest.fixture
def canonical() -> xr.Dataset:
    return make_canonical()


@pytest.fixture
def domain(tmp_path) -> pathlib.Path:
    """Synthetic domain (layout of examples/new_domain): X_wrf_pl.nc (all features, 4 levels, random except `valid`
    and `lz`) and y_wrf.nc (5 levels), 2017-2020, one invalid profile."""
    rng = np.random.default_rng(0)
    time = pd.date_range("2017-01-01", "2020-12-31", freq="30D")
    n_t, n_loc, n_x, n_y = len(time), 3, 4, 5
    coords = {"time": time, "loc": np.arange(n_loc), "lat": ("loc", [50.0, 51, 52]), "lon": ("loc", [4.0, 5, 6])}
    profile = {"z_agl", "p", "u", "v", "th", "S", "dth_dz", "valid", "lz"}
    X = xr.Dataset(
        {
            f: (("time", "bottom_top", "loc"), rng.normal(size=(n_t, n_x, n_loc)).astype("float32"))
            if f in profile
            else (("time", "loc"), rng.normal(size=(n_t, n_loc)).astype("float32"))
            for f in features.FEATURES
        },
        coords=coords,
    )
    # Physical mask and height coordinate (SQF-H reads them unscaled): first level underground at the first location
    X["valid"][:] = 1.0
    X["valid"][:, 0, 0] = 0.0
    X["lz"][:] = np.linspace(0.1, 0.9, n_x, dtype="float32")[None, :, None]
    z_agl = np.broadcast_to(np.array([10.0, 50, 200, 1_000, 5_000])[None, :, None], (n_t, n_y, n_loc))
    y = xr.Dataset(
        {
            "cn2": (("time", "bottom_top", "loc"), 10 ** rng.uniform(-17, -14, (n_t, n_y, n_loc))),
            "QKE": (("time", "bottom_top", "loc"), rng.uniform(0, 2, (n_t, n_y, n_loc))),
            "TSQ": (("time", "bottom_top", "loc"), rng.uniform(0, 1, (n_t, n_y, n_loc))),
            "EL_PBL": (("time", "bottom_top", "loc"), rng.uniform(1, 100, (n_t, n_y, n_loc))),
            "z_agl": (("time", "bottom_top", "loc"), z_agl.copy()),
            "z": (("time", "bottom_top", "loc"), z_agl + 20),
        },
        coords=coords,
    )
    y["cn2"][0, 2, 1] = np.nan  # filtered profile
    (tmp_path / "4_features_interp").mkdir()
    (tmp_path / "3_features").mkdir()
    X.to_netcdf(tmp_path / "4_features_interp" / "X_wrf_pl.nc")
    y.to_netcdf(tmp_path / "3_features" / "y_wrf.nc")
    return tmp_path

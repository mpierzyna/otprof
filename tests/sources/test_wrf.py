import numpy as np
import pandas as pd
import xarray as xr

from otprof import schema
from otprof.sources import wrf


def make_wrf(n_lev=60, n_time=2, n_loc=3) -> xr.Dataset:
    """Native WRF Cn2 output (after alignment: dims time, bottom_top, loc) with fields linear in log-pressure."""
    p_sfc = np.array([101_000.0, 95_500.0, 99_000.0])[:n_loc]  # second column: elevated terrain
    eta = np.linspace(0, 1, n_lev)
    p = p_sfc[None, None, :] * (1500.0 / p_sfc[None, None, :]) ** eta[None, :, None]  # p_sfc .. 15 hPa
    p = np.broadcast_to(p, (n_time, n_lev, n_loc)).astype("float32")
    lp = np.log(p / 100)
    dims3, dims2 = ("time", "bottom_top", "loc"), ("time", "loc")
    full2 = lambda val: (dims2, np.full((n_time, n_loc), val, dtype="float32"))  # noqa: E731
    return xr.Dataset(
        {
            "p": (dims3, p),
            "z": (dims3, (7_000.0 * (np.log(1013.25) - lp)).astype("float32")),
            "th": (dims3, (300 + 10 * (np.log(1013.25) - lp)).astype("float32")),
            "u_met": (dims3, (2 * lp).astype("float32")),
            "v_met": (dims3, np.zeros_like(p)),
            "QKE": (dims3, np.ones_like(p)),
            "HGT": (("loc",), np.array([10.0, 500.0, 50.0][:n_loc], dtype="float32")),
            "PBLH": full2(700.0), "LANDMASK": full2(1.0), "U10": full2(2.0), "V10": full2(1.0), "T2": full2(285.0),
            "UST": full2(0.3), "HFX": full2(50.0), "LH": full2(80.0), "slp": full2(1013.0),
        },
        coords={"time": pd.date_range("2017-01-01", periods=n_time, freq="6h"),
                "XLAT": ("loc", np.linspace(50, 52, n_loc)), "XLONG": ("loc", np.linspace(4, 6, n_loc))},
    )  # fmt: skip


def test_model_grid_renames_only():
    raw = make_wrf()
    ds = wrf.standardize(raw, grid="model")
    assert ds.sizes["bottom_top"] == raw.sizes["bottom_top"]
    np.testing.assert_array_equal(ds["p_sfc"], raw["p"].isel(bottom_top=0))
    assert {"u", "v", "blh", "lsm", "tk2", "z_sfc_src", "lat", "lon", "QKE"} <= set(ds.variables)


def test_pressure_grid_interpolation():
    raw = make_wrf()
    ds = wrf.standardize(raw, grid="pressure")
    assert ds["p_level"].values.tolist() == list(schema.P_LEVELS_HPA)
    assert "bottom_top" not in ds.coords  # positional levels, no leftover interpolation coordinate
    lp = np.log(np.array(schema.P_LEVELS_HPA, dtype=float))
    above = np.array(schema.P_LEVELS_HPA)[None, :] * 100 < raw["p"].isel(bottom_top=0).values[:, :, None].min(0)
    # fields linear in log-pressure are reproduced exactly above the lowest model level
    u = ds["u"].isel(time=0).transpose("loc", "bottom_top").values
    np.testing.assert_allclose(u[above], np.broadcast_to(2 * lp, u.shape)[above], rtol=1e-4)
    # below the lowest model level: nearest level above
    i_first = above[1].argmax()
    np.testing.assert_allclose(u[1, :i_first], u[1, i_first])
    assert not ds["u"].isnull().any()

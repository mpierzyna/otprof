import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otprof import dataset

N_LEV, NY, NX = 30, 12, 10


def write_wrf_files(tmp_path, n_files=3, steps=2):
    """Small WRF Cn2 output files (native names) on a regular 0.1 deg grid around 51N, 5E."""
    rng = np.random.default_rng(0)
    lat2d, lon2d = np.meshgrid(51 + 0.1 * np.arange(NY), 5 + 0.1 * np.arange(NX), indexing="ij")
    files = []
    for i in range(n_files):
        times = pd.date_range("2017-01-01", periods=steps, freq="6h") + pd.Timedelta(hours=6 * steps * i)
        shp = (steps, N_LEV, NY, NX)
        eta = np.linspace(0, 1, N_LEV)[None, :, None, None]
        p = 100_000 * (1500 / 100_000) ** eta * np.ones(shp)
        z = 7_000 * np.log(101_325 / p)
        dims3, dims2 = ("time", "bottom_top", "south_north", "west_east"), ("time", "south_north", "west_east")
        sfc = lambda val: (dims2, np.full((steps, NY, NX), val, "float32"))  # noqa: E731
        ds = xr.Dataset(
            {
                "z": (dims3, z.astype("float32")),
                "p": (dims3, p.astype("float32")),
                "th": (dims3, (290 + 0.004 * z).astype("float32")),
                "u_met": (dims3, (5 + z / 2000).astype("float32")),
                "v_met": (dims3, np.ones(shp, "float32")),
                "cn2": (dims3, (1e-15 * np.exp(-z / 2000) + rng.uniform(0, 1e-17, shp)).astype("float32")),
                "QKE": (dims3, np.ones(shp, "float32")),
                "HGT": (dims2, np.full((steps, NY, NX), 20.0, "float32")),
                "PBLH": sfc(800.0), "LANDMASK": sfc(1.0), "U10": sfc(3.0), "V10": sfc(1.0), "T2": sfc(285.0),
                "UST": sfc(0.3), "HFX": sfc(50.0), "LH": sfc(80.0), "slp": sfc(1013.0),
            },
            coords={"time": times, "XLAT": (("south_north", "west_east"), lat2d.astype("float32")),
                    "XLONG": (("south_north", "west_east"), lon2d.astype("float32"))},
        )  # fmt: skip
        f = tmp_path / f"wrfout_aux_d01_{i}_cn2.nc"
        ds.to_netcdf(f)
        files.append(f)
    return files


def make_era5(times):
    """ARCO-like ERA5 on a 0.25 deg grid, longitudes 0..360."""
    lat = np.array([51.75, 51.5, 51.25, 51.0])
    lon = np.array([5.0, 5.25, 5.5, 5.75])
    t = len(times)
    return xr.Dataset(
        {"surface_pressure": (("time", "latitude", "longitude"), np.full((t, 4, 4), 1e5, "float32"))},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


def test_align_selects_and_keeps_raw(tmp_path):
    files = write_wrf_files(tmp_path)
    times = pd.date_range("2016-12-31T18", periods=10, freq="6h")  # starts before and ends after WRF
    era5, wrf, diag = dataset.align(make_era5(times), files, crop=1, dist_th=0.02)
    # only ERA5 points inside the cropped WRF grid and on a WRF cell centre are kept
    assert diag["n_points"] == era5.sizes["loc"] == wrf.sizes["loc"] > 0
    np.testing.assert_allclose(era5["latitude"], wrf["XLAT"], atol=0.02)
    np.testing.assert_allclose(era5["longitude"], wrf["XLONG"], atol=0.02)
    # common time steps only
    assert era5.sizes["time"] == wrf.sizes["time"] == 6
    np.testing.assert_array_equal(era5["time"], wrf["time"])
    # nothing converted
    assert "u_met" in wrf and "HGT" in wrf and "surface_pressure" in era5


def test_align_parallel_same_as_sequential(tmp_path):
    files = write_wrf_files(tmp_path)
    era5 = make_era5(pd.date_range("2017-01-01", periods=6, freq="6h"))
    _, seq, _ = dataset.align(era5, files, crop=1, dist_th=0.02)
    _, par, _ = dataset.align(era5, files, crop=1, dist_th=0.02, n_workers=2)
    xr.testing.assert_identical(seq, par)


def test_wrf_targets_and_filter(tmp_path):
    files = write_wrf_files(tmp_path, n_files=1)
    era5 = make_era5(pd.date_range("2017-01-01", periods=2, freq="6h"))
    _, wrf, _ = dataset.align(era5, files, crop=1, dist_th=0.02)
    y = dataset.wrf_targets(wrf)
    for v in ["z_agl", "m", "d_sin", "r0_full", "r0_bl", "tau0_fa", "th0_sl", "lsi2_full"]:
        assert v in y, v
    assert float(y["r0_full"].min()) > 0
    np.testing.assert_allclose(y["z_agl"], y["z"] - 20.0, rtol=1e-6)

    y["cn2"][0, :, 0] = 0.0  # one invalid profile
    y_f, diag = dataset.filter_cn2(y)
    assert diag["n_cn2_zero"] == 1
    assert y_f["cn2"].isel(time=0, loc=0).isnull().all()
    assert diag["n_valid_profiles"] < diag["n_profiles"]


@pytest.mark.parametrize("missing", ["HGT", "PBLH"])
def test_wrf_targets_rejects_incomplete_output(tmp_path, missing):
    files = write_wrf_files(tmp_path, n_files=1)
    _, wrf, _ = dataset.align(make_era5(pd.date_range("2017-01-01", periods=2, freq="6h")), files, crop=1, dist_th=0.02)
    with pytest.raises(ValueError):
        dataset.wrf_targets(wrf.drop_vars(missing))

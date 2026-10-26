import numpy as np
import pandas as pd
import xarray as xr

from otprof import schema
from otprof.sources import era5

ARCO_LEVELS = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550,
               600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]  # fmt: skip


def make_arco(n_time=2) -> xr.Dataset:
    """ERA5 as downloaded from ARCO: top-to-bottom levels in hPa, lon 0..360, ERA5 units and signs."""
    lev = np.array(ARCO_LEVELS)
    z = 7_000.0 * np.log(1013.25 / lev) * 9.81  # geopotential
    t, lat, lon = n_time, 2, 2

    def prof(a):
        return ("time", "level", "latitude", "longitude"), np.broadcast_to(
            a[None, :, None, None], (t, len(lev), lat, lon)
        ).astype("float32")

    def sfc(val):
        return ("time", "latitude", "longitude"), np.full((t, lat, lon), val, dtype="float32")

    return xr.Dataset(
        {
            "geopotential": prof(z),
            "temperature": prof(288 - 0.0065 * np.minimum(z / 9.81, 11_000)),
            "u_component_of_wind": prof(np.full(len(lev), 5.0)),
            "v_component_of_wind": prof(np.zeros(len(lev))),
            "potential_vorticity": prof(np.zeros(len(lev))),
            "geopotential_at_surface": sfc(981.0),
            "surface_pressure": sfc(100_000.0),
            "friction_velocity": sfc(0.3),
            "instantaneous_surface_sensible_heat_flux": sfc(-100.0),  # downward positive: 100 W m-2 upward
            "surface_latent_heat_flux": sfc(-360_000.0),  # J m-2 over 1 h: 100 W m-2 upward
            "boundary_layer_height": sfc(800.0),
            "10m_u_component_of_wind": sfc(3.0),
            "10m_v_component_of_wind": sfc(1.0),
            "2m_temperature": sfc(290.0),
            "mean_sea_level_pressure": sfc(101_300.0),
            "land_sea_mask": sfc(0.7),
        },
        coords={
            "time": pd.date_range("2017-01-01", periods=t, freq="h"),
            "level": lev,
            "latitude": np.array([52.0, 51.75], dtype="float32"),
            "longitude": np.array([4.0, 350.0], dtype="float32"),
        },
    )


def test_standardize_conventions():
    ds = era5.standardize(make_arco())
    assert ds["level"].values.tolist() == list(schema.P_LEVELS_HPA)
    assert ds["p"].values.tolist() == [p * 100 for p in schema.P_LEVELS_HPA]
    np.testing.assert_allclose(ds["z"].isel(level=0), 7_000.0 * np.log(1013.25 / 1000), rtol=1e-5)
    np.testing.assert_allclose(ds["z_sfc_src"], 100.0, rtol=1e-6)
    assert "time" not in ds["z_sfc_src"].dims
    np.testing.assert_allclose(ds["hfx"], 100.0)
    np.testing.assert_allclose(ds["lh"], 100.0)
    np.testing.assert_allclose(ds["slp"], 1013.0)
    np.testing.assert_allclose(ds["lon"], [4.0, -10.0])
    assert "pv" in ds  # extra fields pass through


def test_standardize_output_feeds_make_inputs():
    from otprof.features import make_inputs

    X = make_inputs(era5.standardize(make_arco()))
    assert X.sizes["level"] == 30

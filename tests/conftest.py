import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otprof import schema


def make_canonical(n_time: int = 3, n_loc: int = 2, seed: int = 0) -> xr.Dataset:
    """Small canonical dataset on the pressure grid with plausible, smooth profiles."""
    rng = np.random.default_rng(seed)
    p = np.array(schema.P_LEVELS_HPA, dtype=float) * 100
    z = 7_000.0 * np.log(101_325.0 / p)  # isothermal-ish scale height, increasing with level
    shape = (n_time, len(p), n_loc)
    dims3, dims2 = ("time", "level", "loc"), ("time", "loc")
    ds = xr.Dataset(
        {
            "z": (dims3, np.broadcast_to(z[None, :, None], shape) + rng.normal(0, 5, shape)),
            "p": (("level",), (p).astype(int)),
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
            "level": list(schema.P_LEVELS_HPA),
            "lat": ("loc", np.linspace(50, 52, n_loc)),
            "lon": ("loc", np.linspace(4, 6, n_loc)),
        },
    )
    return schema.set_attrs(ds)


@pytest.fixture
def canonical() -> xr.Dataset:
    return make_canonical()

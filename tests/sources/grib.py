"""Small GRIB2 messages for the GFS tests: real GRIB encoding (eccodes samples) on tiny regular lat-lon grids."""

from __future__ import annotations

import numpy as np

eccodes = __import__("pytest").importorskip("eccodes")


def make_message(values: np.ndarray, *, lat0: float, lon0: float, step: float, keys: dict) -> bytes:
    """One GRIB2 message with `values` (Nj, Ni) on a regular grid from (lat0, lon0) north to south, east to west
    spacing `step`; `keys`: e.g. discipline, parameterCategory, parameterNumber, typeOfLevel, level."""
    nj, ni = values.shape
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        eccodes.codes_set_long(h, "Ni", ni)
        eccodes.codes_set_long(h, "Nj", nj)
        for k, v in {
            "latitudeOfFirstGridPointInDegrees": lat0,
            "longitudeOfFirstGridPointInDegrees": lon0,
            "latitudeOfLastGridPointInDegrees": lat0 - (nj - 1) * step,
            "longitudeOfLastGridPointInDegrees": lon0 + (ni - 1) * step,
            "iDirectionIncrementInDegrees": step,
            "jDirectionIncrementInDegrees": step,
        }.items():
            eccodes.codes_set_double(h, k, v)
        for k, v in keys.items():
            (eccodes.codes_set_string if isinstance(v, str) else eccodes.codes_set_long)(h, k, v)
        eccodes.codes_set_values(h, values.ravel().astype(float))
        return eccodes.codes_get_message(h)
    finally:
        eccodes.codes_release(h)

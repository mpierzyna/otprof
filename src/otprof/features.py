"""Building blocks for model inputs: physical quantities derived from the canonical input dataset (`otprof.schema`).

Each model combines them into its own inputs (e.g. `otprof.models.sqf_h.inputs.make_inputs`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from otprof.physics import most

K_VON_KARMAN = 0.4
G = 9.81  # m s-2
RHO_CP = 1.216e3  # J m-3 K-1, converts a heat flux W m-2 -> K m s-1


def potential_temperature(tk: xr.DataArray, p: xr.DataArray) -> xr.DataArray:
    """Potential temperature [K] from temperature [K] and pressure [Pa] (reference 1000 hPa, dry air)."""
    p0 = 100000  # Pa
    gamma = 1.4
    th = tk * (p0 / p) ** ((gamma - 1) / gamma)
    return th.astype(np.float32).assign_attrs(long_name="Potential temperature", units="K")


def sfc_ct2_w71(
    ust: xr.DataArray,
    hfx: xr.DataArray,
    tk2: xr.DataArray,
    L_bounds: Literal["percentile"] | tuple[float, float] = "percentile",
) -> xr.DataArray:
    """Surface CT2 [K2 m-2/3] at 2 m from the W71 flux model (`otprof.physics.most`), clipped at 1e-8.

    `ust` friction velocity [m s-1], `hfx` sensible heat flux [W m-2, upward positive], `tk2` 2 m temperature [K].
    """
    shfx = hfx / RHO_CP  # W m-2 -> K m s-1
    L = -(ust**3) * tk2 / (K_VON_KARMAN * shfx * G)  # Obukhov length
    theta_st = -shfx / ust  # temperature scale
    ct2 = most.w71_flux_ct2(T_st=theta_st, z=2.0, L=L, L_bounds=None if L_bounds == "percentile" else L_bounds)
    ct2 = ct2.clip(1e-8, None)  # avoid zero or negative CT2
    return ct2.assign_attrs(long_name="Surface CT2 at 2 m (W71 flux model)", units="K2 m-2/3")


def get_wind_mag_dir(u: xr.DataArray, v: xr.DataArray) -> dict[str, xr.DataArray]:
    """Wind speed `m` [m s-1] and the direction the wind comes from `d` [deg], also as `d_sin`, `d_cos`."""
    m = np.sqrt(u**2 + v**2).astype(np.float32).assign_attrs(long_name="Wind speed", units="m s-1")
    d = np.rad2deg(np.arctan2(-u, -v)).astype(np.float32).assign_attrs(long_name="Wind direction", units="deg")
    d_sin = np.sin(np.deg2rad(d)).astype(np.float32).assign_attrs(long_name="Sine of wind direction", units="1")
    d_cos = np.cos(np.deg2rad(d)).astype(np.float32).assign_attrs(long_name="Cosine of wind direction", units="1")
    return {"m": m, "d": d, "d_sin": d_sin, "d_cos": d_cos}


def get_time_features(time: xr.DataArray) -> dict[str, xr.DataArray]:
    """Hour of day and day of year as sine / cosine."""
    t = time.dt
    return {
        "hr_sin": np.sin(2 * np.pi * t.hour / 24),
        "hr_cos": np.cos(2 * np.pi * t.hour / 24),
        "doy_sin": np.sin(2 * np.pi * t.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * t.dayofyear / 365),
    }


def fill_nearest(da: xr.DataArray) -> xr.DataArray:
    """Fill NaN along `bottom_top` with the value of the nearest non-NaN level (ties: the lower level)."""
    if not da.isnull().any():
        return da

    def _fill(a):
        idx = np.arange(a.shape[-1])
        ok = ~np.isnan(a)
        # index of the last valid level at or below / the first valid level at or above each level
        below = np.where(ok, idx, -1)
        below = np.maximum.accumulate(below, axis=-1)
        above = np.where(ok, idx, a.shape[-1])
        above = np.flip(np.minimum.accumulate(np.flip(above, -1), axis=-1), -1)
        use_above = (below < 0) | ((above < a.shape[-1]) & (above - idx < idx - below))
        src = np.where(use_above, above, below).clip(0, a.shape[-1] - 1)
        return np.take_along_axis(a, src, axis=-1)

    filled = xr.apply_ufunc(
        _fill, da, input_core_dims=[["bottom_top"]], output_core_dims=[["bottom_top"]], keep_attrs=True
    )
    return filled.transpose(*da.dims)

"""Model inputs (unscaled X) from the canonical input dataset (`otprof.schema`).

The same function builds the training inputs (from WRF or ERA5) and the inputs at inference (ERA5, later GFS).
Bump `FEATURES_VERSION` whenever the output changes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from otprof import schema
from otprof.physics import most

FEATURES_VERSION = 1

# All inputs of SQF-H, in model order
FEATURES = [
    # Profiles
    "z_agl", "p", "u", "v", "th",
    # Gradients to the level above
    "S", "dth_dz",
    # Surface
    "ust", "hfx", "lh", "blh", "u10", "v10", "tk2", "slp", "lsm",
    # Time
    "hr_sin", "hr_cos", "doy_sin", "doy_cos",
    # Surface CT2 estimate (W71)
    "lct2_w71f",
    # Height
    "valid", "lz", "z_sfc", "lp_sfc",
]  # fmt: skip

K_VON_KARMAN = 0.4
G = 9.81  # m s-2
RHO_CP = 1.216e3  # J m-3 K-1, converts a heat flux W m-2 -> K m s-1


def make_inputs(
    ds: xr.Dataset,
    *,
    grid: schema.TGrid = "pressure",
    z_sfc_target: xr.DataArray | None = None,
    w71_L_bounds: Literal["percentile"] | tuple[float, float] = "percentile",
) -> xr.Dataset:
    """Unscaled model inputs `FEATURES` on the lowest N-1 of the N levels of `ds` (canonical inputs).

    - Gradients (`S`, `dth_dz`) are taken between each level and the level above; the top level of `ds` only enters
      these gradients (on the pressure grid: 31 levels 1000..20 hPa → 30 levels 1000..30 hPa).
    - `z_sfc_target`: terrain height of the column to predict (the target), used as the `z_sfc` input. Defaults to
      the source's own terrain `z_sfc_src`. `z_agl` and `valid` always refer to the source's own surface.
    - `w71_L_bounds`: clipping of the Obukhov length for the W71 surface CT2. `"percentile"` clips to the 2.5 / 97.5
      percentiles of the data passed in (as the training data was built), a tuple to fixed bounds (needed to get the
      same result for single forecasts).
    """
    schema.validate(ds, grid=grid)
    ds = ds.drop_vars("level", errors="ignore")  # positional levels; the coordinate is restored at the end
    full = {"level": slice(0, -1)}

    # Profiles
    z_agl = (ds["z"] - ds["z_sfc_src"]).assign_attrs(long_name="Height above ground", units="m")
    th = ds["th"] if "th" in ds else potential_temperature(tk=ds["tk"], p=ds["p"])

    # Gradients between consecutive levels, assigned to the lower level
    dz = ds["z"].diff("level", label="lower")
    du_dz = ds["u"].diff("level", label="lower") / dz
    dv_dz = ds["v"].diff("level", label="lower") / dz
    S = np.sqrt(du_dz**2 + dv_dz**2).astype(np.float32).assign_attrs(long_name="Wind shear", units="s-1")
    dth_dz = (th.diff("level", label="lower") / dz).astype(np.float32)
    dth_dz = dth_dz.assign_attrs(long_name="Vertical gradient of potential temperature", units="K m-1")
    # Levels repeated by the source (dz = 0, e.g. WRF levels filled below ground) -> gradient of the nearest level
    S, dth_dz = _fill_nearest(S), _fill_nearest(dth_dz)

    # Height features
    if grid == "pressure":
        p_nominal = xr.DataArray(np.array(schema.P_LEVELS_HPA, dtype=float) * 100, dims="level")
        valid = p_nominal < ds["p_sfc"]  # level above the source's surface
    else:
        valid = xr.ones_like(z_agl, dtype=bool)
    valid = valid.astype(np.float32).assign_attrs(long_name="Level above the surface", units="1")
    lz = (np.log1p(z_agl.clip(min=0) / schema.LZ_REF) / schema.LZ_NORM).assign_attrs(
        long_name="Warped height above ground", units="1"
    )
    z_sfc = ds["z_sfc_src"] if z_sfc_target is None else z_sfc_target
    lp_sfc = np.log(ds["p_sfc"]).assign_attrs(long_name="Log surface pressure", units="log(Pa)")

    X = {
        "z_agl": z_agl.isel(full),
        "p": ds["p"].isel(full),
        "u": ds["u"].isel(full),
        "v": ds["v"].isel(full),
        "th": th.isel(full),
        "S": S,
        "dth_dz": dth_dz,
        **{v: ds[v] for v in ["ust", "hfx", "lh", "blh", "u10", "v10", "tk2", "slp", "lsm"]},
        **time_features(ds["time"]),
        "lct2_w71f": np.log10(sfc_ct2_w71(ust=ds["ust"], hfx=ds["hfx"], tk2=ds["tk2"], L_bounds=w71_L_bounds)),
        "valid": valid.isel(full),
        "lz": lz.isel(full),
        "z_sfc": z_sfc.assign_attrs(long_name="Terrain height of the target column", units="m"),
        "lp_sfc": lp_sfc,
    }
    X = xr.Dataset({k: X[k] for k in FEATURES})
    if grid == "pressure":
        X = X.assign_coords(level=list(schema.P_LEVELS_HPA[:-1]))
    return X.assign_attrs(otprof_features_version=FEATURES_VERSION, otprof_grid=grid)


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


def wind(u: xr.DataArray, v: xr.DataArray) -> dict[str, xr.DataArray]:
    """Wind speed `m` [m s-1] and the direction the wind comes from `d` [deg], also as `d_sin`, `d_cos`."""
    m = np.sqrt(u**2 + v**2).astype(np.float32).assign_attrs(long_name="Wind speed", units="m s-1")
    d = np.rad2deg(np.arctan2(-u, -v)).astype(np.float32).assign_attrs(long_name="Wind direction", units="deg")
    d_sin = np.sin(np.deg2rad(d)).astype(np.float32).assign_attrs(long_name="Sine of wind direction", units="1")
    d_cos = np.cos(np.deg2rad(d)).astype(np.float32).assign_attrs(long_name="Cosine of wind direction", units="1")
    return {"m": m, "d": d, "d_sin": d_sin, "d_cos": d_cos}


def time_features(time: xr.DataArray) -> dict[str, xr.DataArray]:
    """Hour of day and day of year as sine / cosine."""
    t = time.dt
    return {
        "hr_sin": np.sin(2 * np.pi * t.hour / 24),
        "hr_cos": np.cos(2 * np.pi * t.hour / 24),
        "doy_sin": np.sin(2 * np.pi * t.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * t.dayofyear / 365),
    }


def _fill_nearest(da: xr.DataArray) -> xr.DataArray:
    """Fill NaN along `level` with the value of the nearest non-NaN level (ties: the lower level)."""
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

    filled = xr.apply_ufunc(_fill, da, input_core_dims=[["level"]], output_core_dims=[["level"]], keep_attrs=True)
    return filled.transpose(*da.dims)

"""WRF Cn2 output of otprof-wrf (`wrfout_aux_*_cn2.nc`, native variable names) → canonical inputs (`otprof.schema`).

Used to build training data: the model learns from WRF on pressure levels and is applied to ERA5 (or GFS) inputs.
Needs scipy (`otprof[data]`) for the pressure grid.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from otprof import schema
from otprof.physics.vertical import interp_vert

# Native WRF names → canonical names. Fields not listed keep their name (z, p, th, w, QKE, cn2, ...).
RENAME = {
    "HGT": "z_sfc_src",
    "u_met": "u",
    "v_met": "v",
    "PBLH": "blh",
    "LANDMASK": "lsm",
    "U10": "u10",
    "V10": "v10",
    "T2": "tk2",
    "UST": "ust",
    "HFX": "hfx",
    "LH": "lh",
    "XLAT": "lat",
    "XLONG": "lon",
}


def standardize(ds: xr.Dataset, grid: schema.TGrid = "pressure") -> xr.Dataset:
    """Convert WRF output to the canonical inputs.

    `ds` holds the native WRF fields (dim `bottom_top`, bottom to top). WRF already follows the canonical conventions
    (fluxes upward positive, sea level pressure in hPa); WRF's own potential temperature `th` is kept.
    - `p_sfc` is the pressure of the lowest model level.
    - `grid="model"`: native levels, renames only.
    - `grid="pressure"`: all profile variables are interpolated linearly in log-pressure to `schema.P_LEVELS_HPA`.
      Levels below the lowest model level (p > p_sfc) are filled with the nearest level above.
    """
    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables or k in ds.dims})
    ds["p_sfc"] = ds["p"].isel(bottom_top=0)

    if grid == "pressure":
        ds = _to_pressure_levels(ds)
    ds = schema.set_attrs(ds)
    return schema.validate(ds, grid=grid)


def _to_pressure_levels(ds: xr.Dataset) -> xr.Dataset:
    # Interpolation needs increasing log-pressure, i.e. levels from top to bottom
    ds = ds.isel(bottom_top=slice(None, None, -1))
    p_hPa_target = xr.DataArray(list(schema.P_LEVELS_HPA[::-1]), dims="bottom_top")

    # Interpolate on log-pressure scale like the ECMWF example
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#heading-Step2Interpolatevariablesonmodellevelstocustompressurelevels
    p_hPa = ds["p"] / 100.0
    ds_interp = interp_vert(ds, np.log(p_hPa), np.log(p_hPa_target), "bottom_top")
    ds_interp = ds_interp.drop_vars("bottom_top", errors="ignore")  # the log-pressure targets; positional levels

    # Levels below the lowest model level are NaN (WRF surface pressure < level pressure) -> nearest level
    ds_interp = ds_interp.interpolate_na(dim="bottom_top", method="nearest", fill_value="extrapolate")

    # Add the surface variables back, levels bottom to top with the nominal pressure as coordinate
    v_sfc = [v for v in ds.data_vars if v not in ds_interp.data_vars]
    ds_interp = ds_interp.merge(ds[v_sfc])
    ds_interp = ds_interp.isel(bottom_top=slice(None, None, -1))
    return ds_interp.assign_coords(p_level=("bottom_top", list(schema.P_LEVELS_HPA)))

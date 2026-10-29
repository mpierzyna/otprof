"""ERA5 (pressure levels + surface) from the ARCO-ERA5 zarr store → canonical inputs (`otprof.schema`).

- `fetch(times, where)`: ERA5 at the times in an `Area` or at `Points`, in ARCO names and units, lazily (needs
  `otprof[data]`: gcsfs, zarr)
- `standardize(raw)`: → canonical inputs; `load(times, where)` = `standardize(fetch(...))`
- `get_download_size(times)`: what a fetch downloads (the store keeps each time step of the globe in one piece)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from otprof import schema
from otprof.physics.constants import G
from otprof.sources import Where, select, to_180

# ARCO-ERA5 names → canonical (and short research) names. Fields not listed keep their ARCO name.
RENAME = {
    "geopotential": "z",
    "geopotential_at_surface": "z_sfc_src",
    "potential_vorticity": "pv",
    "vertical_velocity": "w",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "tk",
    "specific_humidity": "q",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "tk2",
    "boundary_layer_height": "blh",
    "surface_pressure": "p_sfc",
    "surface_latent_heat_flux": "lh",
    "mean_surface_net_short_wave_radiation_flux": "msnsrf",
    "low_cloud_cover": "lcc",
    "skin_temperature": "tsk",
    "land_sea_mask": "lsm",
    "mean_sea_level_pressure": "slp",
    "friction_velocity": "ust",
    "instantaneous_surface_sensible_heat_flux": "hfx",
    "latitude": "lat",
    "longitude": "lon",
}

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Fields used by otprof (and the research pipelines), in ARCO names
VARIABLES = [
    # Profiles
    "geopotential",
    "potential_vorticity",
    "vertical_velocity",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    # Surface
    "soil_type",
    "land_sea_mask",
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "geopotential_at_surface",
    "boundary_layer_height",
    "boundary_layer_dissipation",
    "surface_pressure",
    "friction_velocity",
    "instantaneous_surface_sensible_heat_flux",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_latent_heat_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "low_cloud_cover",
    "skin_temperature",
]


def open_store(times, variables: list[str] = VARIABLES) -> xr.Dataset:
    """The ARCO-ERA5 store at `times`, lazily (global; needs `otprof[data]`: gcsfs, zarr)."""
    full = xr.open_zarr(ARCO_URL, chunks=None, storage_options=dict(token="anon"))
    return full.sel(time=pd.DatetimeIndex(times))[variables]


def fetch(times, where: Where, variables: list[str] = VARIABLES) -> xr.Dataset:
    """ERA5 at `times` in an `Area` or at `Points` (`otprof.sources`), in ARCO names and units, opened lazily: nothing
    is downloaded until the values are used (`.load()`, `.to_netcdf(...)`). The store keeps every time step of the
    whole globe in one piece, so each time step downloads about 1 GB whatever the area (`get_download_size`)."""
    return select(open_store(times, variables), where, lat="latitude", lon="longitude")


def load(times, where: Where, variables: list[str] = VARIABLES) -> xr.Dataset:
    """ERA5 at `times` in an `Area` or at `Points` as canonical inputs: `standardize(fetch(...))`, downloaded."""
    return standardize(fetch(times, where, variables).load())


def get_download_size(times, variables: list[str] = VARIABLES) -> dict[str, int]:
    """What `fetch` downloads for `times`, whatever the area: the storage chunks it touches (uncompressed, global)."""
    full = xr.open_zarr(ARCO_URL, chunks=None, storage_options=dict(token="anon"))
    touched = get_chunks_touched(open_store(times, variables), full)
    return {"download": sum(nbytes for _, nbytes in touched.values()), "n_chunks": sum(n for n, _ in touched.values())}


def get_chunks_touched(subset: xr.Dataset, full: xr.Dataset) -> dict[str, tuple[int, int]]:
    """Per variable of `subset`: number of storage chunks of `full` it touches and their uncompressed size [bytes].

    `subset` must be a selection of `full` (e.g. `full.sel(...)`). Reading a subset reads every chunk it touches
    completely, so this is the amount of data transferred (before compression), not `subset.nbytes`. The chunk shape
    comes from the variable encoding of `full` (zarr or netCDF chunks); unchunked dims count as one chunk. Edge chunks
    are counted at full size.
    """
    out = {}
    for name, da in subset.data_vars.items():
        enc = full[name].encoding
        chunks = enc.get("preferred_chunks") or dict(zip(full[name].dims, enc.get("chunks") or full[name].shape))
        n_chunks = 1
        for dim in da.dims:
            size = chunks.get(dim, full.sizes[dim])
            if dim in full.indexes:
                pos = full.indexes[dim].get_indexer(subset.indexes[dim])
                assert (pos >= 0).all(), f"{name}: `{dim}` values of the subset not found in the full dataset"
            else:
                pos = np.arange(da.sizes[dim])
            n_chunks *= len(np.unique(pos // size))
        chunk_bytes = int(np.prod([chunks.get(d, full.sizes[d]) for d in full[name].dims])) * da.dtype.itemsize
        out[name] = (n_chunks, n_chunks * chunk_bytes)
    return out


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Convert ERA5 in ARCO names and units to the canonical inputs on the pressure grid.

    `ds` holds the pressure-level fields (dim `level` in hPa, any order, must include `schema.P_LEVELS_HPA`) and the
    surface fields of the same times and points. Conventions applied:
    - geopotential → height [m]; the surface geopotential is averaged over the times in `ds` (it is static in ERA5)
    - fluxes: ERA5 is downward positive → upward positive; the latent heat flux is accumulated over 1 h [J m-2] → W m-2
    - sea level pressure Pa → hPa; longitude 0..360 → -180..180
    - the 31 levels of the pressure grid, bottom to top (dim `bottom_top`, nominal pressure as coordinate `p_level`
      [hPa] and as `p` [Pa])
    """
    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables})
    ds = ds.assign_coords(lon=(ds["lon"].dims, to_180(ds["lon"].values)))

    # Fluxes to upward positive (WRF convention); lh accumulated over 1 h
    ds["hfx"] = -ds["hfx"]
    ds["lh"] = -ds["lh"] / 3600
    ds["slp"] = ds["slp"] / 100

    # Geopotential to height
    ds["z"] = ds["z"] / G
    ds["z_sfc_src"] = ds["z_sfc_src"].mean("time") / G

    # Pressure grid: select and order the levels bottom to top; nominal pressure in Pa (int, as in the training data)
    ds = ds.sel(level=list(schema.P_LEVELS_HPA))
    ds["p"] = (ds["level"] * 100).astype(int)
    ds = ds.rename(level="p_level").swap_dims(p_level="bottom_top")
    ds = ds.transpose(*[d for d in ds.dims if d != "bottom_top"], "bottom_top")

    ds = schema.set_attrs(ds)
    return schema.validate(ds, grid="pressure")

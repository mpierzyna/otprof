"""ERA5 (pressure levels + surface) from the ARCO-ERA5 zarr store → canonical inputs (`otprof.schema`).

- `fetch`: open ERA5 for given times and area (needs `otprof[data]`: gcsfs, zarr)
- `standardize`: ERA5 in ARCO names and units → canonical inputs
"""

from __future__ import annotations

import xarray as xr

from otprof import schema

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

G = 9.81  # m s-2, as in the training data

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


def fetch(
    times,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    variables: list[str] = VARIABLES,
) -> xr.Dataset:
    """ERA5 at `times` for the area (ARCO longitudes run 0..360), opened lazily from the ARCO-ERA5 store.

    Nothing is downloaded until the values are used (e.g. `.to_netcdf(...)`). The store has no spatial chunks, so
    every time step is downloaded for the whole globe and cut to the area afterwards.
    """
    ds = xr.open_zarr(ARCO_URL, chunks=None, storage_options=dict(token="anon"))
    ds = ds.sel(time=times)[variables]
    # ERA5 latitude is stored in descending order
    return ds.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Convert ERA5 in ARCO names and units to the canonical inputs on the pressure grid.

    `ds` holds the pressure-level fields (dim `level` in hPa, any order, must include `schema.P_LEVELS_HPA`) and the
    surface fields of the same times and points. Conventions applied:
    - geopotential → height [m]; the surface geopotential is averaged over the times in `ds` (it is static in ERA5)
    - fluxes: ERA5 is downward positive → upward positive; the latent heat flux is accumulated over 1 h [J m-2] → W m-2
    - sea level pressure Pa → hPa; longitude 0..360 → -180..180
    - the 31 levels of the pressure grid, bottom to top, with the nominal pressure as `p` [Pa]
    """
    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables})
    ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180))

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

    ds = schema.set_attrs(ds)
    return schema.validate(ds, grid="pressure")

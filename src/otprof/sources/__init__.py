"""Source adapters: convert data of one source to the canonical inputs (`otprof.schema`).

- `era5`: ERA5 from the ARCO-ERA5 store
- `gfs`: GFS forecasts (v16 and later) from AWS; `gfs_archive`: GFS of earlier years from NCAR GDEX
- `wrf`: WRF Cn2 simulations (training data)

Each module has `standardize(raw) -> canonical`: renames, units, sign conventions, vertical levels; `era5`, `gfs` and
`gfs_archive` also `fetch` the raw data.
"""

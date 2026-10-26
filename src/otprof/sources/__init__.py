"""Source adapters: convert data of one source (ERA5, WRF, later GFS) to the canonical inputs (`otprof.schema`).

Each module has `standardize(raw) -> canonical`: renames, units, sign conventions, vertical levels.
"""

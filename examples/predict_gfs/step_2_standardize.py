"""Step 2: GFS in the canonical form of otprof → `data/gfs_pl.nc`.

`gfs.standardize` converts names and units to the input schema of otprof (`otprof.schema`) on the 31 pressure levels
1000 to 20 hPa. Dims: time, loc, bottom_top (bottom_top 0 = 1000 hPa).
"""

import pathlib

import xarray as xr

from otprof.sources import gfs

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    ds = gfs.standardize(xr.load_dataset(HERE / "data" / "gfs.nc"))
    ds.to_netcdf(HERE / "data" / "gfs_pl.nc")
    print(ds)
    print(f"Wrote {HERE / 'data' / 'gfs_pl.nc'}")

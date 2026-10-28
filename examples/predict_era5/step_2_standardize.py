"""Step 2: ERA5 in the canonical form of otprof → `data/era5_pl.nc`.

`era5.standardize` converts names and units to the input schema of otprof (`otprof.schema`) and selects the 31
pressure levels 1000 to 20 hPa. Dims: time, lat, lon, bottom_top (bottom_top 0 = 1000 hPa).
"""

import pathlib

import xarray as xr

from otprof.sources import era5

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    ds = era5.standardize(xr.load_dataset(HERE / "data" / "era5.nc"))
    ds.to_netcdf(HERE / "data" / "era5_pl.nc")
    print(ds)
    print(f"Wrote {HERE / 'data' / 'era5_pl.nc'}")

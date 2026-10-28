"""Step 2: ERA5 → model inputs `data/X.nc`.

`era5.standardize` converts names and units to the input schema of otprof and selects the 31 pressure levels
(1000 to 20 hPa); `make_inputs` computes the model inputs from them (gradients, surface CT2 estimate, heights, time
features) on the 30 levels 1000 to 30 hPa. Dims: time, lat, lon, bottom_top (bottom_top 0 = 1000 hPa).
"""

import pathlib

import xarray as xr

from otprof.features import make_inputs
from otprof.sources import era5

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    ds = xr.load_dataset(HERE / "data" / "era5.nc")
    X = make_inputs(era5.standardize(ds))
    X.to_netcdf(HERE / "data" / "X.nc")
    print(X)
    print(f"Wrote {HERE / 'data' / 'X.nc'}")

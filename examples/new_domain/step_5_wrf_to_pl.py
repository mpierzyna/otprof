"""Step 5: model inputs from WRF interpolated to the ERA5 pressure levels → `data/<domain>/4_features_interp/X_wrf_pl.nc`.

These are the training inputs of the decoupled SQF-H: the model learns on WRF inputs that look like ERA5 and is
applied to ERA5 (or GFS) at inference.
"""

import xarray as xr
from otprof.features import make_inputs
from otprof.sources import wrf

import domain

if __name__ == "__main__":
    root, _ = domain.parse(__doc__)
    out = root / "4_features_interp"
    out.mkdir(exist_ok=True)
    X = make_inputs(wrf.standardize(xr.load_dataset(root / "2_aligned" / "wrf.nc"), grid="pressure"))
    X.to_netcdf(out / "X_wrf_pl.nc")
    print(f"Wrote {out / 'X_wrf_pl.nc'}")

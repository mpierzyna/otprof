"""Step 5: WRF interpolated to the ERA5 pressure levels → `data/<domain>/4_inputs_interp/wrf_pl.nc`.

These are the training inputs of the decoupled SQF-H: the model learns on WRF inputs that look like ERA5 and is
applied to ERA5 (or GFS) at inference. Canonical form (`otprof.schema`), like `3_inputs/era5_pl.nc`.
"""

import xarray as xr
from otprof.sources import wrf

import domain

if __name__ == "__main__":
    root, _ = domain.parse(__doc__)
    out = root / "4_inputs_interp"
    out.mkdir(exist_ok=True)
    ds = wrf.standardize(xr.load_dataset(root / "2_aligned" / "wrf.nc"), grid="pressure")
    ds.to_netcdf(out / "wrf_pl.nc")
    print(f"Wrote {out / 'wrf_pl.nc'}")

"""Step 2: pair each ERA5 grid point with the nearest WRF column, keep the common time steps
→ `data/<domain>/2_aligned/{era5,wrf}.nc`.

Pure selection, nothing converted: check here that the points and times match what you expect.
"""

import numpy as np
import xarray as xr
from otprof import dataset

import domain

N_WORKERS = 8  # WRF files read in parallel

if __name__ == "__main__":
    root, params = domain.parse(__doc__)
    out = root / "2_aligned"
    out.mkdir(exist_ok=True)

    era5 = xr.open_dataset(root / "1_raw" / "era5.nc")
    wrf_files = sorted((domain.HERE / params["wrf_output"]).glob("**/wrfout_aux_*_cn2.nc"))
    era5, wrf, diag = dataset.align(
        era5, wrf_files, crop=params["crop"], dist_th=params["dist_th"], n_workers=N_WORKERS
    )
    print(f"{diag['n_points']} of {len(diag['dist'])} ERA5 points within {diag['dist_th']} deg of a WRF column")
    print(f"median distance of the kept points: {np.median(diag['dist'][diag['dist'] < diag['dist_th']]):.4f} deg")
    print(f"{diag['n_times']} common time steps")

    era5.to_netcdf(out / "era5.nc")
    wrf.to_netcdf(out / "wrf.nc")
    print(f"Wrote {out}")

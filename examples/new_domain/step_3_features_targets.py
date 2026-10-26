"""Step 3: model inputs from ERA5 and the WRF targets → `data/<domain>/3_features/{X_era5_pl,y_wrf_all}.nc`.

- targets: WRF Cn2 and turbulence profiles on the model levels plus astroclimate parameters (r0, tau0, th0, lsi2)
- inputs: ERA5 on the 30 pressure levels 1000..30 hPa (`otprof.features.FEATURES`), with the WRF terrain as the
  terrain input
"""

import xarray as xr
from otprof import dataset
from otprof.features import make_inputs
from otprof.sources import era5

import domain

if __name__ == "__main__":
    root, _ = domain.parse(__doc__)
    out = root / "3_features"
    out.mkdir(exist_ok=True)

    y = dataset.wrf_targets(xr.load_dataset(root / "2_aligned" / "wrf.nc"))
    y.to_netcdf(out / "y_wrf_all.nc")

    era5_canonical = era5.standardize(xr.load_dataset(root / "2_aligned" / "era5.nc"))
    X = make_inputs(era5_canonical, z_sfc_target=y["z_sfc_src"].reset_coords(drop=True))
    X.to_netcdf(out / "X_era5_pl.nc")
    print(f"Wrote {out}")

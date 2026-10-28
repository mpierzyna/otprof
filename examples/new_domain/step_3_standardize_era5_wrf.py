"""Step 3: standardize WRF (targets) and ERA5 (inputs) → `data/<domain>/3_targets/y_wrf_all.nc`, `3_inputs/era5_pl.nc`.

- targets: WRF Cn2 and turbulence profiles on the model levels plus astroclimate parameters (r0, tau0, th0, lsi2)
- inputs: ERA5 in the canonical form (`otprof.schema`: names, units, the 31 pressure levels 1000..20 hPa), with the
  terrain height of the paired WRF column as `z_sfc_target`. The model computes its own inputs from it when training.
"""

import xarray as xr
from otprof import dataset
from otprof.sources import era5

import domain

if __name__ == "__main__":
    root, _ = domain.parse(__doc__)
    (root / "3_targets").mkdir(exist_ok=True)
    (root / "3_inputs").mkdir(exist_ok=True)

    y = dataset.make_wrf_targets(xr.load_dataset(root / "2_aligned" / "wrf.nc"))
    y.to_netcdf(root / "3_targets" / "y_wrf_all.nc")

    ds = era5.standardize(xr.load_dataset(root / "2_aligned" / "era5.nc"))
    ds["z_sfc_target"] = (
        y["z_sfc_src"].reset_coords(drop=True).assign_attrs(long_name="Terrain height of the target column", units="m")
    )
    ds.to_netcdf(root / "3_inputs" / "era5_pl.nc")
    print(f"Wrote {root / '3_targets'}, {root / '3_inputs'}")

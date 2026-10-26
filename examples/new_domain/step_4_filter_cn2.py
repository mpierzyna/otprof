"""Step 4: mask implausible Cn2 profiles and astroclimate values → `data/<domain>/3_features/y_wrf.nc`."""

import xarray as xr
from otprof import dataset

import domain

if __name__ == "__main__":
    root, _ = domain.parse(__doc__)
    y, diag = dataset.filter_cn2(xr.open_dataset(root / "3_features" / "y_wrf_all.nc"))
    print(f"r0 threshold {diag['r0_th']:.3g} m ({diag['n_r0_invalid']} above)")
    print(f"log10 scintillation index threshold {diag['lsi2_th']:.3g} ({diag['n_lsi2_invalid']} above)")
    print(f"{diag['n_cn2_zero']} profiles with Cn2 = 0 at some level")
    print(f"{diag['n_valid_profiles']} of {diag['n_profiles']} profiles kept")
    y.to_netcdf(root / "3_features" / "y_wrf.nc")

"""Regenerate the golden model inputs after a deliberate change of the values (bump
`otprof.models.sqf_h.inputs.FEATURES_VERSION` first: MAJOR, or MINOR for a small change shown not to need retraining):

    uv run python tests/data/make_golden.py

`era5_arco_sample.nc`: 3 time steps, 2 x 2 grid points (Canary Islands, sea and land), all 37 levels of ARCO-ERA5.
Contains modified Copernicus Climate Change Service information 2026.
"""

import pathlib

import xarray as xr

from otprof.models.sqf_h.inputs import make_inputs
from otprof.sources import era5

DATA = pathlib.Path(__file__).parent

if __name__ == "__main__":
    X = make_inputs(era5.standardize(xr.open_dataset(DATA / "era5_arco_sample.nc")))
    X.to_netcdf(DATA / "era5_inputs_golden.nc")
    print(f"Wrote {DATA / 'era5_inputs_golden.nc'} (features version {X.attrs['otprof_features_version']})")

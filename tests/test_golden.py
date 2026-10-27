"""Model inputs from a real ERA5 sample must not change unnoticed (see tests/data/make_golden.py)."""

import pathlib

import xarray as xr

from otprof.features import FEATURES_VERSION, make_inputs
from otprof.sources import era5

DATA = pathlib.Path(__file__).parent / "data"


def test_era5_inputs_golden():
    X = make_inputs(era5.standardize(xr.open_dataset(DATA / "era5_arco_sample.nc")))
    golden = xr.open_dataset(DATA / "era5_inputs_golden.nc")
    major = golden.attrs["otprof_features_version"].split(".")[0]
    assert major == FEATURES_VERSION.split(".")[0], "FEATURES_VERSION major bumped: regenerate golden"
    # MINOR / PATCH changes keep the values of the existing outputs
    xr.testing.assert_allclose(X[list(golden.data_vars)], golden, rtol=1e-6)

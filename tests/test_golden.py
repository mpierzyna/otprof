"""Model inputs from a real ERA5 sample must not change unnoticed (see tests/data/make_golden.py)."""

import pathlib

import xarray as xr

from otprof.features import FEATURES_VERSION, make_inputs
from otprof.sources import era5

DATA = pathlib.Path(__file__).parent / "data"


def test_era5_inputs_golden():
    X = make_inputs(era5.standardize(xr.open_dataset(DATA / "era5_arco_sample.nc")))
    golden = xr.open_dataset(DATA / "era5_inputs_golden.nc")
    assert golden.attrs["otprof_features_version"] == FEATURES_VERSION, "FEATURES_VERSION bumped: regenerate golden"
    xr.testing.assert_allclose(X, golden, rtol=1e-6)

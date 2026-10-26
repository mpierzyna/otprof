"""Smoke test: the new-domain example steps 2-5 run end to end on a tiny synthetic domain."""

import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otprof import features, schema

from .sources.test_era5 import make_arco
from .test_dataset import write_wrf_files

EXAMPLES = __import__("pathlib").Path(__file__).parents[1] / "examples" / "new_domain"


@pytest.fixture
def example_dir(tmp_path):
    d = tmp_path / "new_domain"
    shutil.copytree(EXAMPLES, d, ignore=shutil.ignore_patterns("data", "__pycache__"))
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    write_wrf_files(wrf_dir, n_files=2, steps=3)
    (d / "params.yaml").write_text(
        "domains:\n  test:\n    wrf_output: ../wrf\n"
        "    area: {lon_min: 5, lon_max: 6, lat_min: 51, lat_max: 52}\n    crop: 1\n    dist_th: 0.02\n"
    )
    # Stand-in for step 1 (download): ERA5 on a grid overlapping the WRF grid, covering the WRF time steps
    era5 = make_arco(n_time=8)
    era5 = era5.isel(latitude=[0, 0, 0, 0], longitude=[0, 0, 0, 0])  # 4 x 4 grid
    era5 = era5.assign_coords(
        latitude=[51.75, 51.5, 51.25, 51.0],
        longitude=[5.0, 5.25, 5.5, 5.75],
        time=pd.date_range("2017-01-01", periods=8, freq="6h"),
    )
    (d / "data" / "test" / "1_raw").mkdir(parents=True)
    era5.to_netcdf(d / "data" / "test" / "1_raw" / "era5.nc")
    return d


def test_steps_2_to_5(example_dir):
    for step in ["step_2_align.py", "step_3_features_targets.py", "step_4_filter_cn2.py", "step_5_wrf_to_pl.py"]:
        res = subprocess.run([sys.executable, step, "test"], cwd=example_dir, capture_output=True, text=True)
        assert res.returncode == 0, f"{step} failed:\n{res.stdout}\n{res.stderr}"

    root = example_dir / "data" / "test"
    X_era5 = xr.open_dataset(root / "3_features" / "X_era5_pl.nc")
    X_wrf = xr.open_dataset(root / "4_features_interp" / "X_wrf_pl.nc")
    y = xr.open_dataset(root / "3_features" / "y_wrf.nc")
    for X in (X_era5, X_wrf):
        assert list(X.data_vars) == features.FEATURES
        assert X["level"].values.tolist() == list(schema.P_LEVELS_HPA[:-1])
    # same samples on both sides
    assert X_era5.sizes["loc"] == X_wrf.sizes["loc"] == y.sizes["loc"] > 0
    np.testing.assert_array_equal(X_era5["time"], y["time"])
    assert "r0_full" in y

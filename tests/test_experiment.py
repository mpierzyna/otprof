import pathlib

import numpy as np
import pytest
import xarray as xr

from otprof.models.sqf_h import inputs as sqf_h_inputs
from otprof.experiment import Experiment, ExperimentConfig, get_pipeline_from_config
from otprof.models.sqf_h.data import SqfhPipeline

REPO = pathlib.Path(__file__).parents[1]
CONFIGS = sorted([*(REPO / "examples" / "configs").glob("*.yaml"), *(REPO / "paper" / "configs").glob("*.yaml")])


def land_months(p):
    """Named selector: months 1, 4, 7, 10 on land."""
    return p.select_time({"month": [1, 4, 7, 10]}) & (p.ds_X["lsm"] > 0.5)


def _config(root, top: dict | None = None, **data) -> ExperimentConfig:
    """The example config on `root`, with `data` entries and top-level entries replaced (validated, as from YAML)."""
    d = ExperimentConfig.from_yaml(REPO / "examples" / "configs" / "wrf_pl_h.yaml").model_dump(mode="json")
    return ExperimentConfig.model_validate({**d, **(top or {}), "data": {**d["data"], "root": str(root), **data}})


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_resolve(path, tmp_path):
    """Every shipped config validates and resolves to a pipeline with its variables (no data needed)."""
    config = ExperimentConfig.from_yaml(path)
    exp = Experiment(config, get_pipeline_from_config(config, tmp_path))
    assert exp.features == sqf_h_inputs.FEATURES
    assert exp.pipeline.inputs == config.data.inputs


def test_unknown_keys_and_variables_fail(tmp_path):
    with pytest.raises(ValueError, match="extra"):
        ExperimentConfig.model_validate({**_config(tmp_path).model_dump(), "batch": 1})
    config = _config(tmp_path, top={"targets": ["lcn2", "nope"]})
    with pytest.raises(ValueError, match="target `nope`"):
        Experiment(config, get_pipeline_from_config(config, tmp_path))


def test_profile_pipeline(domain):
    exp = Experiment.from_config(_config(domain)).setup()
    dp = exp.pipeline
    assert isinstance(dp, SqfhPipeline)
    assert dp.ds_X.sizes == {"sample": 49 * 3 - 1, "bottom_top": 30}  # one invalid profile dropped; 30 input levels
    assert dp.ds_y.sizes["bottom_top"] == 4  # top level dropped
    cn2 = (
        xr.load_dataset(domain / "3_targets" / "y_wrf.nc")["cn2"]
        .isel(bottom_top=slice(0, -1))
        .stack(sample=("time", "loc"))
    )
    np.testing.assert_allclose(dp.ds_y["lcn2"], np.log10(cn2.sel(sample=dp.ds_y["sample"])).T)
    years = {
        s: set(np.unique(dp.split(s, transform=False)[0]["time"].dt.year.values)) for s in ["train", "val", "test"]
    }
    assert years == {"train": {2019, 2020}, "val": {2018}, "test": {2017}}


def test_selectors_and_scaler_overrides_from_config(domain):
    config = _config(
        domain,
        top={"scalers_X": {"z_agl": [None, "quantile", [0.0, 0.99]]}},
        sel_train="land_months",
        sel_val={"month": [2, 5, 8, 11]},
        sel_test={"month": [3, 6, 9, 12]},
    )
    exp = Experiment.from_config(config, functions={"land_months": land_months}).setup()
    X_train, _ = exp.pipeline.split("train", transform=False)
    assert set(X_train["time"].dt.month.values) <= {1, 4, 7, 10} and (X_train["lsm"] > 0.5).all()
    scaler = exp.pipeline.pre_X[0]
    assert scaler.config["z_agl"] == (None, "quantile", (0.0, 0.99)) and scaler.config["p"] == (None, "minmax", None)


def test_run_dir_roundtrip(domain, tmp_path):
    config = _config(domain, sel_train="land_months")
    exp = Experiment.from_config(config, functions={"land_months": land_months}).setup()
    exp.to_dir(tmp_path / "run")
    loaded = Experiment.from_dir(tmp_path / "run", functions={"land_months": land_months})
    assert loaded.config == exp.config and loaded.pipeline.ds_X is None
    loaded.setup()
    for stage in ["train", "val", "test"]:
        for a, b in zip(exp.pipeline.split(stage, transform=True), loaded.pipeline.split(stage, transform=True)):
            xr.testing.assert_identical(a, b)


def test_missing_variable_in_file(domain):
    """An inputs file the model cannot compute its inputs from fails before any data is loaded."""
    ds = xr.load_dataset(domain / "4_inputs_interp" / "wrf_pl.nc")
    (domain / "3_inputs").mkdir()
    ds.drop_vars("ust").to_netcdf(domain / "3_inputs" / "era5_pl.nc")
    with pytest.raises(ValueError, match="ust"):
        Experiment.from_config(_config(domain, inputs="3_inputs/era5_pl.nc"))

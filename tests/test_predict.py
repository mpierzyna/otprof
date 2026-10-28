import json
import pathlib

import lightning as L
import numpy as np
import pytest
import torch
import xarray as xr

from otprof.experiment import Experiment, ExperimentConfig
from otprof.export import export_model
from otprof.predict import fit_oqm, get_best_ckpt, load_model, predict
from otprof.training import get_torch_dataset, get_torch_loader, train

REPO = pathlib.Path(__file__).parents[1]
TINY = {"embed_dim": 8, "num_blocks": 1, "num_heads": 2, "head_dim": 16, "conv_kernel_size": 3}


@pytest.fixture(autouse=True)
def no_compile(monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "disable", True)


@pytest.fixture
def run(domain, tmp_path):
    """A tiny model trained for 3 epochs on the synthetic domain: (experiment, run dir)."""
    d = ExperimentConfig.from_yaml(REPO / "examples" / "configs" / "wrf_pl_h.yaml").model_dump(mode="json")
    d["data"]["root"] = str(domain)
    d["model"].update(TINY)
    d["training"] = {"batch_size": 8, "max_epochs": 3, "lr": 1e-2}
    exp = Experiment.from_config(ExperimentConfig.model_validate(d)).setup()
    L.seed_everything(0)
    run_dir = train(exp, save_dir=tmp_path, version="run", accelerator="cpu", devices=1, num_workers=0,
                    compile=False, enable_progress_bar=False)  # fmt: skip
    return exp, run_dir


def test_sort_surface_first_matches_profile_dataset(run):
    exp, _ = run
    assert exp.pipeline.get_input_order(exp.features) == get_torch_dataset(exp, "test").vars_X


def test_predict_matches_network(run):
    """predict on the test split (target heights as queries) == network on the loader batches, scaled back. Batch
    size and memory layout as in training: the CPU kernels differ by 1 ulp for other layouts of `lz`."""
    exp, run_dir = run
    predictor = load_model(run_dir)
    X_test, y_test = exp.pipeline.split("test", transform=False)
    pred = predict(predictor, X_test, z_agl_query=y_test["z_agl"], device="cpu", batch_size=5)

    loader = get_torch_loader(exp, "test", num_workers=0)  # batches as in training and the research predictions
    with torch.no_grad():
        y_hat = torch.cat([predictor.model(x, frc[:, :, 1])[0] for x, frc, _ in loader])
    ref = loader.dataset.unstack_y(y_hat, sample_coord="self").rename(level="bottom_top")
    ref = exp.pipeline.backward_y(ref)
    for v in ["lcn2", "QKE", "TSQ", "EL_PBL"]:
        np.testing.assert_array_equal(pred[v].values, ref[v].values)
    assert {"r0", "lsi2", "z_agl"} <= set(pred) and pred.attrs["otprof_oqm"] == 0
    assert np.isfinite(pred["r0"]).all() and (pred["r0"] > 0).all()


def test_bundle_roundtrip_oqm_and_unstacked_inputs(run, tmp_path):
    exp, run_dir = run
    predictor = load_model(run_dir)
    X_test, y_test = exp.pipeline.split("test", transform=False)
    pred = predict(predictor, X_test, z_agl_query=y_test["z_agl"], device="cpu")

    # oQM fitted on the test predictions, appended to post_y and exported with the bundle
    pipeline = exp.pipeline.model_copy(update={"post_y": [fit_oqm(pred, y_test, n_quantiles=10)]})
    z_default = y_test["z_agl"].mean("sample").values
    bundle = export_model(get_best_ckpt(run_dir), pipeline, exp.config, tmp_path / "bundle", z_agl_default=z_default)
    assert sorted(p.name for p in bundle.iterdir()) == [
        "config.json", "experiment.yaml", "model.safetensors", "pipeline.json", "steps.json"
    ]  # fmt: skip
    config = json.loads((bundle / "config.json").read_text())
    assert config["vars_X"] == predictor.vars_X and config["targets"] == ["lcn2", "QKE", "TSQ", "EL_PBL"]
    assert json.loads((bundle / "pipeline.json").read_text())["data_root"] == exp.config.data.root

    loaded = load_model(bundle)
    no_oqm = predict(loaded, X_test, z_agl_query=y_test["z_agl"], oqm=False, device="cpu")
    xr.testing.assert_identical(no_oqm, pred)
    with_oqm = predict(loaded, X_test, z_agl_query=y_test["z_agl"], device="cpu")
    assert with_oqm.attrs["otprof_oqm"] == 1 and not np.array_equal(with_oqm["lcn2"], pred["lcn2"])
    np.testing.assert_array_equal(with_oqm["QKE"], pred["QKE"])  # oQM maps lcn2 only

    # Inputs computed by the model from canonical data (time, loc, bottom_top), default query heights
    X = loaded.make_inputs(xr.load_dataset(exp.pipeline.data_root / exp.pipeline.inputs))
    out = predict(loaded, X, device="cpu")
    assert out["lcn2"].dims == ("time", "loc", "bottom_top") and out.sizes["bottom_top"] == len(z_default)
    # Samples are independent: one sample alone == in the batch (float tolerance: BLAS differs by batch size)
    one = predict(loaded, X, oqm=False, device="cpu").isel(time=5, loc=1)
    single = predict(loaded, X.isel(time=[5], loc=[1]), oqm=False, device="cpu").isel(time=0, loc=0)
    np.testing.assert_allclose(one["lcn2"], single["lcn2"], rtol=1e-6)

    # Constant profiles as on pressure levels (`p` only over bottom_top) are broadcast over the samples
    assert X["p"].dims == ("bottom_top",)
    xr.testing.assert_identical(
        predict(loaded, X, oqm=False, device="cpu"),
        predict(loaded, X.assign(p=X["p"].broadcast_like(X["z_agl"])), oqm=False, device="cpu"),
    )

    # Query heights not covering the training column
    with pytest.warns(UserWarning, match="do not cover"):
        predict(loaded, X.isel(time=[0]), z_agl_query=z_default[:2], device="cpu")


def test_predict_checks_input_versions(run):
    exp, run_dir = run
    predictor = load_model(run_dir)
    predictor.config = {"features_version": "1.0.0", "schema_version": "1.0.0"}
    X_test, y_test = exp.pipeline.split("test", transform=False)
    X_test.attrs["otprof_features_version"] = "2.0.0"
    with pytest.raises(ValueError, match="different MAJOR"):
        predict(predictor, X_test, z_agl_query=y_test["z_agl"], device="cpu")

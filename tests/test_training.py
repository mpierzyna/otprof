import json
import pathlib

import lightning as L
import pytest
import torch

from otprof.experiment import Experiment, ExperimentConfig
from otprof.models import sqf_h
from otprof.models.bundle import load_network, load_network_from_checkpoint, save_network
from otprof.models.sqf_h.inputs import FEATURES_VERSION
from otprof.models.sqf_h.model import SqueezeformerHModel
from otprof.training import get_torch_dataset, get_torch_loader, train

REPO = pathlib.Path(__file__).parents[1]

TINY = {"embed_dim": 8, "num_blocks": 1, "num_heads": 2, "head_dim": 16, "conv_kernel_size": 3}


@pytest.fixture(autouse=True)
def no_compile(monkeypatch):
    """No torch.compile on CPU in tests (also the compiled metrics in `otprof.losses`)."""
    monkeypatch.setattr(torch._dynamo.config, "disable", True)


@pytest.fixture
def exp(domain) -> Experiment:
    d = ExperimentConfig.from_yaml(REPO / "examples" / "configs" / "wrf_pl_h.yaml").model_dump(mode="json")
    d["data"]["root"] = str(domain)
    d["model"].update(TINY)
    d["training"] = {"batch_size": 8, "max_epochs": 15, "lr": 1e-2}
    return Experiment.from_config(ExperimentConfig.model_validate(d)).setup()


def _eval_loss(lit, loader) -> float:
    """Training loss (RMSE + confidence error) on all levels, eval mode."""
    lit.eval()
    with torch.no_grad():
        return float(torch.stack([lit._shared_step(b, 0)[0] for b in loader]).mean())


def test_profile_dataset_order(exp):
    """Surface variables first, then profiles, each in feature order; forcings and targets as configured."""
    data = get_torch_dataset(exp, "train")
    sfc = [f for f in exp.features if "bottom_top" not in exp.pipeline.ds_X[f].dims]
    assert data.vars_X == sfc + [f for f in exp.features if f not in sfc]
    assert data.vars_y_frc == ["z_agl", "lz"] and data.vars_y == ["lcn2", "QKE", "TSQ", "EL_PBL"]
    x, frc, y = data[0]
    assert x.shape == (30, 25) and frc.shape == (4, 2) and y.shape == (4, 4)


def test_train_and_reload(exp, tmp_path):
    L.seed_everything(0)
    run_dir = train(exp, save_dir=tmp_path, version="smoke", accelerator="cpu", devices=1, num_workers=0,
                    compile=False, enable_progress_bar=False)  # fmt: skip
    assert (run_dir / "config.yaml").exists() and (run_dir / "data_pipeline" / "steps.json").exists()
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    assert 0 < len(ckpts) <= 5

    # Loss decreases: untrained (same seed) vs the last checkpoint, on the training data
    from otprof.models.sqf_h.lit import LitSqueezeFormerH, get_lit_model

    loader = get_torch_loader(exp, "train", num_workers=0)
    L.seed_everything(0)
    untrained = get_lit_model(exp.config, loader.dataset, compile=False)
    ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
    lit = LitSqueezeFormerH.load_from_checkpoint(ckpt, map_location="cpu")
    assert _eval_loss(lit, loader) < _eval_loss(untrained, loader)
    assert lit.hparams.vars_X == loader.dataset.vars_X

    # ckpt -> network -> save_network -> load_network: same forward output
    x, frc, _ = next(iter(loader))
    with torch.no_grad():
        ref = lit(x, frc)
        model = load_network_from_checkpoint(ckpt, SqueezeformerHModel).eval()
        out = model(x, frc)
        save_network(model, tmp_path / "bundle", "sqf_h", FEATURES_VERSION, vars_X=lit.hparams.vars_X)
        loaded, config = load_network(tmp_path / "bundle", sqf_h)
        out_loaded = loaded.eval()(x, frc)
    for a, b, c in zip(ref, out, out_loaded):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    assert config["vars_X"] == loader.dataset.vars_X and config["model"] == "sqf_h"


def test_load_network_from_checkpoint_strips_compile_prefix(tmp_path):
    """Checkpoints of compiled models keep the weights under `model._orig_mod.`; training hparams are dropped."""
    hparams = {"input_dim": 5, "output_dim": 2, "use_conf_head": True, "idx_valid": 3, "idx_lz": 4, **TINY}
    model = SqueezeformerHModel(**hparams)
    ckpt = {
        "hyper_parameters": {**hparams, "lr": 1e-3, "n_query_train": 32, "vars_X": None},
        "state_dict": {f"model._orig_mod.{k}": v for k, v in model.state_dict().items()},
    }
    torch.save(ckpt, tmp_path / "x.ckpt")
    loaded = load_network_from_checkpoint(tmp_path / "x.ckpt", SqueezeformerHModel)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v, rtol=0, atol=0)


@pytest.mark.parametrize("key", ["features_version", "schema_version"])
def test_load_network_version_check(tmp_path, key):
    hparams = {"input_dim": 5, "output_dim": 2, "use_conf_head": True, "idx_valid": 3, "idx_lz": 4, **TINY}
    save_network(SqueezeformerHModel(**hparams), tmp_path, "sqf_h", FEATURES_VERSION)
    config = json.loads((tmp_path / "config.json").read_text())
    major, minor, patch = config[key].split(".")
    for version, ok in [(f"{major}.{int(minor) + 1}.{patch}", True), (f"{int(major) + 1}.0.0", False)]:
        (tmp_path / "config.json").write_text(json.dumps({**config, key: version}))
        if ok:
            load_network(tmp_path, sqf_h)
        else:
            with pytest.raises(ValueError, match="different MAJOR"):
                load_network(tmp_path, sqf_h)

"""A model outside otprof, added only through `otprof.models.MODELS`: configured, trained, exported, loaded and run
by the generic code without changes to otprof."""

import lightning as L
import numpy as np
import pydantic
import pytest

from otprof import models
from otprof.experiment import Experiment, ExperimentConfig
from otprof.export import export_model
from otprof.predict import get_best_ckpt, load_model, predict
from otprof.training import train


@pytest.fixture
def dummy(monkeypatch):
    monkeypatch.setitem(models.MODELS, "dummy", "tests.dummy_model")


def get_config(domain, **model) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "name": "dummy_wrfpl",
        "data": {"root": str(domain)},
        "forcings": ["z_agl"],
        "targets": ["lcn2"],
        "model": {"name": "dummy", **model},
        "training": {"batch_size": 8, "max_epochs": 2, "lr": 1e-2},
    })  # fmt: skip


def test_model_config_validated_by_the_model(dummy, domain):
    config = get_config(domain, hidden=4)
    assert type(config.model).__name__ == "Config" and config.model.hidden == 4
    assert config.model_dump()["model"] == {"name": "dummy", "hidden": 4}  # subclass fields serialized
    with pytest.raises(pydantic.ValidationError, match="embed_dim"):
        get_config(domain, embed_dim=8)  # an SQF-H hyperparameter
    with pytest.raises(pydantic.ValidationError, match="Unknown model"):
        ExperimentConfig.model_validate({**get_config(domain).model_dump(), "model": {"name": "nope"}})


def test_train_export_load_predict(dummy, domain, tmp_path):
    exp = Experiment.from_config(get_config(domain, hidden=4)).setup()
    L.seed_everything(0)
    run_dir = train(exp, save_dir=tmp_path, version="run", accelerator="cpu", devices=1, num_workers=0,
                    compile=False, enable_progress_bar=False)  # fmt: skip
    assert Experiment.from_dir(run_dir).config == exp.config

    X, y = exp.pipeline.split("test", transform=False)
    from_run = predict(load_model(run_dir), X, z_agl_query=y["z_agl"], device="cpu")
    bundle = export_model(get_best_ckpt(run_dir), exp.pipeline, exp.config, tmp_path / "bundle")
    from_bundle = predict(load_model(bundle), X, z_agl_query=y["z_agl"], device="cpu")
    assert set(from_run) == {"lcn2", "z_agl", "r0", "lsi2"}
    np.testing.assert_array_equal(from_run["lcn2"].values, from_bundle["lcn2"].values)

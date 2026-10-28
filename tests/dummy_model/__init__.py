"""A second model outside otprof, as research would write one (`otprof.models.ModelPackage`): own inputs (a subset of
the SQF-H inputs), own pipeline, a tiny network and its Lightning module. Used by `tests/test_models.py`."""

from __future__ import annotations

import pathlib
from typing import Literal

import lightning as L
import torch
import xarray as xr

from otprof.models import ModelConfig
from otprof.models.sqf_h import inputs as sqf_h_inputs
from otprof.pipeline import ProfilePipeline
from otprof.transforms import DatasetScaler

FEATURES_VERSION = "0.1.0"
FEATURES = ["lsm", "z_agl", "u", "v", "th"]


class Config(ModelConfig):
    name: Literal["dummy"] = "dummy"
    hidden: int = 8


class DummyPipeline(ProfilePipeline):
    def make_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        X = sqf_h_inputs.make_inputs(ds)[FEATURES]
        return X.assign_attrs(otprof_features_version=FEATURES_VERSION)  # the version of this model's inputs

    def make_targets(self, ds_y: xr.Dataset) -> xr.Dataset:
        return sqf_h_inputs.make_targets(ds_y)

    def get_input_order(self, features: list[str]) -> list[str]:
        return [f for f in features if f == "lsm"] + [f for f in features if f != "lsm"]


PIPELINE_CLASSES = {"DummyPipeline": DummyPipeline}


def get_pipeline(data_root: pathlib.Path) -> DummyPipeline:
    return DummyPipeline(
        data_root=data_root,
        sel_train={"year": [2019, 2020]},
        sel_val={"year": [2018]},
        sel_test={"year": [2017]},
        vars_X=FEATURES,
        vars_y=["lcn2", "z_agl"],
        pre_X=[DatasetScaler({f: (None, "minmax", None) for f in FEATURES})],
        pre_y=[DatasetScaler({"lcn2": (None, "minmax", None), "z_agl": (None, "minmax", None)})],
    )


class DummyNet(torch.nn.Module):
    """Column mean of the inputs and the query height -> targets at the query height."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.config = {"input_dim": input_dim, "output_dim": output_dim, "hidden": hidden}
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, output_dim)
        )

    def forward(self, x: torch.Tensor, frc: torch.Tensor) -> torch.Tensor:
        col = x.mean(1, keepdim=True).expand(-1, frc.shape[1], -1)
        return self.net(torch.cat([col, frc[..., :1]], dim=-1))


def get_network_class():
    return DummyNet


class LitDummy(L.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, lr: float):
        super().__init__()
        self.model = DummyNet(input_dim, output_dim, hidden)
        self.lr = lr
        self.save_hyperparameters()

    def _loss(self, batch):
        x, frc, y = batch
        return torch.nn.functional.mse_loss(self.model(x, frc), y)

    def training_step(self, batch, batch_idx):
        return self._loss(batch)

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", self._loss(batch))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def get_lit_model(config, data_train, compile: bool = True) -> LitDummy:
    return LitDummy(data_train.n_vars_X, data_train.n_vars_y, config.model.hidden, config.training.lr)

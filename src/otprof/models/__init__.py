"""Models: one subpackage per model, holding everything that defines it. The generic code (experiments, training,
prediction, export) finds a model by its name in `MODELS` and only uses the interface `ModelPackage`:

- `Config`: the model's hyperparameters (the `model:` block of an experiment config, a `ModelConfig`)
- `FEATURES_VERSION`, `get_pipeline`, `PIPELINE_CLASSES`: its inputs (computed from the canonical input dataset) and
  data processing
- `get_network_class`: the network (`Network`), saved and loaded by `otprof.models.bundle`
- `get_lit_model`: its training (Lightning module)

A new model is a new subpackage with this interface plus one entry in `MODELS`. Research develops models the same way
in its own package and adds them to `MODELS` (e.g. `MODELS["sqf_x"] = "otprof_research.models.sqf_x"`); promoting a
model to otprof means moving the subpackage here and changing its entry.
"""

from __future__ import annotations

import importlib
import pathlib
from typing import TYPE_CHECKING, Any, Protocol, cast

import pydantic

if TYPE_CHECKING:
    import lightning as L
    import torch

    from otprof.experiment import ExperimentConfig
    from otprof.pipeline import BaseDataPipeline
    from otprof.training import ProfileDataset

# Model name (`model.name` of an experiment config, `model` of a bundle's `config.json`) -> its subpackage. Imported
# when used, so that a model's dependencies (torch, Lightning) are only needed where it runs.
MODELS: dict[str, str] = {"sqf_h": "otprof.models.sqf_h"}


class ModelConfig(pydantic.BaseModel):
    """Base of a model's hyperparameters, the `model:` block of an experiment config. Each model defines its own
    subclass with its fields; unknown keys are refused."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str


class Network(Protocol):
    """A model's network (`torch.nn.Module`). Called with the inputs `x` (batch, input levels, `vars_X`) and the
    forcings `frc` of the query heights (batch, output levels, forcings in the order of the experiment config); returns
    the targets (batch, output levels, targets), or a tuple starting with them (e.g. with a confidence). `config` holds
    its constructor arguments, saved with the weights."""

    config: dict[str, Any]

    def __call__(self, x: torch.Tensor, frc: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]: ...


class ModelPackage(Protocol):
    """What a model subpackage provides (module attributes, see the module docstring)."""

    Config: type[ModelConfig]
    FEATURES_VERSION: str
    PIPELINE_CLASSES: dict[str, type[BaseDataPipeline]]

    def get_pipeline(self, data_root: pathlib.Path) -> BaseDataPipeline: ...

    def get_network_class(self) -> type[Network]: ...

    def get_lit_model(
        self, config: ExperimentConfig, data_train: ProfileDataset, compile: bool = True
    ) -> L.LightningModule: ...


def get_model(name: str) -> ModelPackage:
    """The subpackage of model `name` (`MODELS`)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Known: {sorted(MODELS)}")
    return cast(ModelPackage, importlib.import_module(MODELS[name]))

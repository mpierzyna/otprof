"""Experiments: one YAML file per experiment (`ExperimentConfig`) and the pipeline built from it (`Experiment`).

The config only holds what belongs to the experiment: the pipeline by name (`PIPELINES`: the model's data processing,
e.g. `otprof.models.sqf_h.data.get_pipeline`), the data (root, inputs file, optionally other selectors than the pipeline's default
split), which variables the model uses, scaler overrides, model and training settings. Example:
`examples/configs/wrf_pl_h.yaml`.

A run directory holds `config.yaml` (the config as given) and `data_pipeline/` (the resolved pipeline with its fitted
steps), next to the checkpoints. `Experiment.from_dir` restores both; the pipeline comes from the run, not from the
factory, so a later change of a factory does not change an old run.
"""

from __future__ import annotations

import pathlib
from typing import Callable, Self

import pydantic
import xarray as xr

from otprof.models.sqf_h import data as sqf_h_data
from otprof.pipeline import BaseDataPipeline, ProfilePipeline, TimeSel
from otprof.transforms import DatasetScaler, TScalerConfig

# Name in an experiment config -> function (data_root) -> pipeline. Callers add their own (e.g. research).
PIPELINES: dict[str, Callable[[pathlib.Path], BaseDataPipeline]] = {"sqf_h": sqf_h_data.get_pipeline}

# Saved class name -> class, for loading saved pipelines (callers add their own classes)
PIPELINE_CLASSES: dict[str, type[BaseDataPipeline]] = {
    "ProfilePipeline": ProfilePipeline,
    "SqfhPipeline": sqf_h_data.SqfhPipeline,
}


class _Strict(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")


class DataConfig(_Strict):
    root: str  # data directory of the domain (or a name the caller resolves, e.g. a research domain)
    inputs: str = "4_inputs_interp/wrf_pl.nc"  # relative to root, layout of examples/new_domain
    targets: str = "3_targets/y_wrf.nc"
    # Other selectors than the pipeline's default split: a time selection ({month: [1, 4, 7, 10]}) or the name of a
    # selector function (passed to `Experiment.from_config(functions=...)`)
    sel_train: TimeSel | str | None = None
    sel_val: TimeSel | str | None = None
    sel_test: TimeSel | str | None = None


class ModelConfig(_Strict):
    arch: str = "sqf_h"
    embed_dim: int  # capacity of squeezeformer and attention layers
    head_dim: int  # dimensionality of the final prediction head
    num_blocks: int  # depth, i.e. number of squeezeformer blocks
    num_heads: int  # number of attention heads
    conv_kernel_size: int  # kernel size of the convolutional layers
    use_conf_head: bool = True


class TrainingConfig(_Strict):
    batch_size: int
    max_epochs: int
    lr: float = 1e-3


class ExperimentConfig(_Strict):
    name: str
    pipeline: str = "sqf_h"
    data: DataConfig
    features: list[str] | None = None  # inputs of the model; default: all of the pipeline's vars_X
    forcings: list[str]  # from y, passed to the model (SQF-H: z_agl first, then lz)
    targets: list[str]
    # Overrides of single entries of the pipeline's scaler configs, e.g. {z_agl: [null, quantile, [0.0, 0.99]]}
    scalers_X: dict[str, TScalerConfig] = {}
    scalers_y: dict[str, TScalerConfig] = {}
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> Self:
        import yaml  # otprof[data]

        return cls.model_validate(yaml.safe_load(pathlib.Path(path).read_text()))

    def to_yaml(self, path: str | pathlib.Path) -> None:
        import yaml  # otprof[data]

        pathlib.Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))


def get_pipeline_from_config(
    config: ExperimentConfig,
    data_root: str | pathlib.Path,
    pipelines: dict[str, Callable] | None = None,
    functions: dict[str, Callable] | None = None,
) -> BaseDataPipeline:
    """The pipeline of `config`: built by the function `config.pipeline` (looked up in `pipelines`, default
    `PIPELINES`) on `data_root`, with the experiment's inputs file, selectors and scaler overrides. `functions`: named
    selector functions the selectors refer to. Not set up (no data loaded)."""
    pipelines = PIPELINES if pipelines is None else pipelines
    if config.pipeline not in pipelines:
        raise ValueError(f"Unknown pipeline '{config.pipeline}'. Known: {sorted(pipelines)}")
    dp = pipelines[config.pipeline](pathlib.Path(data_root))
    dp.functions = {**dp.functions, **(functions or {})}

    data = config.data
    for field in ["inputs", "targets"]:
        if hasattr(dp, field):
            setattr(dp, field, getattr(data, field))
    for stage in ["train", "val", "test"]:
        sel = getattr(data, f"sel_{stage}")
        if sel is not None:
            setattr(dp, f"sel_{stage}", sel)

    for steps, overrides in [(dp.pre_X, config.scalers_X), (dp.pre_y, config.scalers_y)]:
        if overrides:
            scalers = [s for s in steps if isinstance(s, DatasetScaler)]
            if len(scalers) != 1:
                raise ValueError(f"Scaler overrides need exactly one DatasetScaler in the steps, found {len(scalers)}")
            scalers[0].config = {**scalers[0].config, **overrides}
    return dp


class Experiment:
    """An experiment config and the pipeline built from it (training, evaluation)."""

    def __init__(self, config: ExperimentConfig, pipeline: BaseDataPipeline):
        self.config = config
        self.pipeline = pipeline
        self.validate()

    @property
    def features(self) -> list[str]:
        return self.pipeline.vars_X if self.config.features is None else self.config.features

    def validate(self) -> None:
        """Features, forcings and targets must be variables of the pipeline. Raises listing all problems."""
        problems = [
            f"feature `{f}` not in the pipeline's vars_X" for f in self.features if f not in self.pipeline.vars_X
        ]
        for kind in ["forcings", "targets"]:
            problems += [f"{kind[:-1]} `{v}` not in the pipeline's vars_y" for v in getattr(self.config, kind)
                         if v not in self.pipeline.vars_y]  # fmt: skip
        if problems:
            raise ValueError(f"Experiment '{self.config.name}':\n- " + "\n- ".join(problems))

    def check_files(self) -> None:
        """The model inputs can be computed from the pipeline's inputs file (checked on its first sample, before
        loading any data)."""
        path = pathlib.Path(self.pipeline.data_root) / self.pipeline.inputs
        with xr.open_dataset(path) as ds:
            first = ds.isel({d: slice(0, 1) for d in ds.dims if d != "bottom_top"})
            try:
                X = self.pipeline.make_inputs(first.load())
            except ValueError as e:
                raise ValueError(f"Experiment '{self.config.name}': {path}: {e}") from None
        missing = [v for v in self.pipeline.vars_X if v not in X]
        if missing:
            raise ValueError(f"Experiment '{self.config.name}': the inputs from {path} lack {missing}")

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig | str | pathlib.Path,
        data_root: str | pathlib.Path | None = None,
        pipelines: dict[str, Callable] | None = None,
        functions: dict[str, Callable] | None = None,
    ) -> Self:
        """Build the experiment of a config (or a YAML path) on `data_root` (default: `config.data.root`). Not set up."""
        if not isinstance(config, ExperimentConfig):
            config = ExperimentConfig.from_yaml(config)
        data_root = config.data.root if data_root is None else data_root
        exp = cls(config, get_pipeline_from_config(config, data_root, pipelines=pipelines, functions=functions))
        if hasattr(exp.pipeline, "inputs"):
            exp.check_files()
        return exp

    def setup(self) -> Self:
        """Load the data and fit the pipeline's steps (skipped for steps already fitted, e.g. loaded from a run)."""
        self.pipeline.setup()
        return self

    def to_dir(self, run_dir: str | pathlib.Path) -> None:
        """`config.yaml` and `data_pipeline/` (pipeline config and fitted steps), without data."""
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(run_dir / "config.yaml")
        self.pipeline.to_disk(run_dir / "data_pipeline")

    @classmethod
    def from_dir(
        cls,
        run_dir: str | pathlib.Path,
        data_root: str | pathlib.Path | None = None,
        pipeline_classes: dict[str, type[BaseDataPipeline]] | None = None,
        functions: dict[str, Callable] | None = None,
    ) -> Self:
        """Restore a run: its config and its saved pipeline with the fitted steps, without data (`setup()` loads it).
        `data_root` replaces the saved one (e.g. another machine or domain)."""
        run_dir = pathlib.Path(run_dir)
        config = ExperimentConfig.from_yaml(run_dir / "config.yaml")
        pipeline_classes = PIPELINE_CLASSES if pipeline_classes is None else pipeline_classes
        dp = BaseDataPipeline.from_disk(run_dir / "data_pipeline", pipelines=pipeline_classes, functions=functions)
        if data_root is not None:
            dp.data_root = pathlib.Path(data_root)
        return cls(config, dp)

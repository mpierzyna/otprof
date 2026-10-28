"""Export a trained model to a bundle: one directory, safetensors + JSON only (no pickle), loadable with
`otprof.predict.load_model` and publishable as a Hugging Face model repo.

    config.json         architecture, format and otprof / features / schema versions, hparams, variable order,
                        default query heights
    model.safetensors   weights
    pipeline.json       the pipeline config (variables, selectors, class name)
    steps.json          the processing chain: pre_X, pre_y (transforms, fitted scalers), post_y (e.g. fitted oQM)
    experiment.yaml     the experiment config the model was trained with
"""

from __future__ import annotations

import pathlib

import numpy as np

from otprof.experiment import ExperimentConfig
from otprof.pipeline import BaseDataPipeline
from otprof.predict import MODELS


def export_model(
    ckpt: str | pathlib.Path,
    pipeline: BaseDataPipeline,
    config: ExperimentConfig,
    out_dir: str | pathlib.Path,
    z_agl_default: np.ndarray | None = None,
) -> pathlib.Path:
    """Write the bundle of the Lightning checkpoint `ckpt` trained with experiment `config` and the (fitted) steps of
    `pipeline`. `z_agl_default`: default query heights [m] of `predict` (e.g. the mean training target heights).
    Pipelines with prototype features (`fe_fns`) are refused: inference has no targets to compute them from."""
    if pipeline.fe_fns:
        raise ValueError(
            "Pipelines with fe_fns cannot be exported: move the features into the model's make_inputs first"
        )
    if not pipeline.is_fitted:
        raise ValueError("The pipeline's steps are not fitted")

    out_dir = pathlib.Path(out_dir)
    model = MODELS[config.model.arch].from_lightning_checkpoint(ckpt)
    features = pipeline.vars_X if config.features is None else config.features
    model.save_pretrained(
        out_dir,
        vars_X=pipeline.get_input_order(features),
        forcings=config.forcings,
        targets=config.targets,
        z_agl_default=None if z_agl_default is None else np.asarray(z_agl_default, dtype=float).tolist(),
    )
    if hasattr(pipeline, "data_root"):  # the config's (relative) data root instead of a local path
        pipeline = pipeline.model_copy(update={"data_root": pathlib.Path(config.data.root)})
    pipeline.to_disk(out_dir)
    config.to_yaml(out_dir / "experiment.yaml")
    return out_dir

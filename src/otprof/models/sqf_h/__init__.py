"""SQF-H: the height-aware SqueezeFormer of the paper (`inputs`: its inputs computed from the canonical input dataset,
`data`: what it makes of the dataset files, `model`: the network, `lit`: its training). This module is its interface
for the generic code (`otprof.models.ModelPackage`).

The published SQF-H (decoupled, trained on the nl domain, with its output quantile mapping) is a Hugging Face model
repo of its own; other models get their own repos:

    from otprof.models import sqf_h
    from otprof.predict import load_model

    predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)
"""

from __future__ import annotations

from typing import Literal

from otprof.models import ModelConfig
from otprof.models.sqf_h.data import PIPELINE_CLASSES, get_pipeline
from otprof.models.sqf_h.inputs import FEATURES_VERSION

HF_REPO_ID = "maxiimilian/otprof-sqfh"
# Commit of the repo this otprof version uses, so a model update never changes the results of an installed version.
HF_REVISION: str | None = "8c8af10e1e2cac0d4101742e132d1bef93b93975"


class Config(ModelConfig):
    """Hyperparameters of SQF-H (the `model:` block of an experiment config)."""

    name: Literal["sqf_h"] = "sqf_h"
    embed_dim: int  # capacity of the squeezeformer and attention layers
    head_dim: int  # dimensionality of the final prediction head
    num_blocks: int  # depth, i.e. number of squeezeformer blocks
    num_heads: int  # number of attention heads
    conv_kernel_size: int  # kernel size of the convolutional layers
    use_conf_head: bool = True
    n_query_train: int = 32  # output levels predicted per training batch (random subset)


def get_network_class():
    """`SqueezeformerHModel` (needs `otprof[torch]`)."""
    from otprof.models.sqf_h.model import SqueezeformerHModel

    return SqueezeformerHModel


def get_lit_model(config, data_train, compile: bool = True):
    """The Lightning module of an experiment config for the training `ProfileDataset` (needs `otprof[train]`)."""
    from otprof.models.sqf_h import lit

    return lit.get_lit_model(config, data_train, compile=compile)


__all__ = ["Config", "FEATURES_VERSION", "PIPELINE_CLASSES", "get_pipeline", "get_network_class", "get_lit_model"]

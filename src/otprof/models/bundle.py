"""Saving and loading networks without Lightning, for every model: `model.safetensors` (weights) and `config.json`
(model name, format and versions, hyperparameters, variable order). Networks also load from Lightning checkpoints.
Needs `otprof[torch]`.

A saved network records the otprof version, the version of its model's inputs (`FEATURES_VERSION`) and of the
canonical input schema; loading checks them against the installed otprof.
"""

from __future__ import annotations

import inspect
import json
import pathlib
from importlib.metadata import version
from typing import TYPE_CHECKING

import torch

from otprof import schema

if TYPE_CHECKING:
    from otprof.models import ModelPackage, Network

FORMAT_VERSION = 1  # bump on a rename of a network's submodules / parameters, with a key mapping in `load_network`


def get_bundle_meta(features_version: str) -> dict[str, str]:
    """Versions recorded in a saved network's `config.json`: otprof, the model's inputs (`features_version` of its
    `make_inputs`) and the canonical input schema."""
    return {
        "otprof_version": version("otprof"),
        "features_version": features_version,
        "schema_version": schema.SCHEMA_VERSION,
    }


def check_versions(config: dict, current: dict[str, str], source: str = "model") -> None:
    """Raise if the MAJOR of `features_version` / `schema_version` in `config` (a saved model) differs from `current`
    (the installed model inputs and schema, or the versions of an inputs dataset): the inputs would no longer be
    computed as in training. MINOR / PATCH differences pass."""
    for key, now in current.items():
        saved = config.get(key)
        if saved is None:
            raise ValueError(f"{source}: no {key} recorded")
        if saved.split(".")[0] != now.split(".")[0]:
            raise ValueError(
                f"{source}: trained with {key} {saved}, but the inputs have {now} (different MAJOR: they are no longer "
                f"computed as in training). Use an otprof version with {key} {saved.split('.')[0]}.x"
            )


def save_network(network: Network, out_dir: str | pathlib.Path, model: str, features_version: str, **meta) -> None:
    """`model.safetensors` and `config.json` of `network` (of model `model`); `meta` is added to the config (e.g. the
    variable order of inputs, forcings and targets)."""
    from safetensors.torch import save_file  # otprof[torch]

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model,
        "format_version": FORMAT_VERSION,
        **get_bundle_meta(features_version),
        "hparams": network.config,
        **meta,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    state_dict = {k: v.contiguous() for k, v in network.state_dict().items()}
    save_file(state_dict, out_dir / "model.safetensors")


def load_network(
    model_dir: str | pathlib.Path, package: ModelPackage, map_location: str = "cpu"
) -> tuple[Network, dict]:
    """The network saved by `save_network` in `model_dir` and its `config.json`. Raises if the model or format differ,
    or if the MAJOR version of the features or the input schema it was trained with differs from the installed ones."""
    from safetensors.torch import load_file  # otprof[torch]

    model_dir = pathlib.Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{model_dir}: format_version {config.get('format_version')}, expected {FORMAT_VERSION}")
    installed = {"features_version": package.FEATURES_VERSION, "schema_version": schema.SCHEMA_VERSION}
    check_versions(config, installed, source=str(model_dir))
    network = package.get_network_class()(**config["hparams"])
    network.load_state_dict(load_file(model_dir / "model.safetensors", device=map_location))
    return network, config


def load_network_from_checkpoint(
    path: str | pathlib.Path, network_class: type[Network], map_location: str = "cpu"
) -> Network:
    """The network of a Lightning checkpoint, read from the checkpoint dict (no Lightning needed). The Lightning module
    must hold the network as `self.model` and save the network's constructor arguments as hparams: these are taken
    from `hyper_parameters` (training ones like `lr` dropped), the weights from `state_dict` without the `model.` and
    `torch.compile` (`_orig_mod.`) prefixes."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    arguments = inspect.signature(network_class).parameters
    network = network_class(**{k: v for k, v in ckpt["hyper_parameters"].items() if k in arguments})
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if not k.startswith("model."):
            continue  # not part of the network
        state_dict[k.removeprefix("model.").removeprefix("_orig_mod.")] = v
    network.load_state_dict(state_dict)
    return network

"""Models: one subpackage per model, holding everything that defines it (`inputs`: its inputs computed from the
canonical input dataset, `data`: data processing, `model`: the network, `lit`: its training). A new model is a new
subpackage with entries in `otprof.experiment.PIPELINES` / `PIPELINE_CLASSES` and `otprof.training.ARCHITECTURES`.

The version helpers here are shared by all models: a saved model records the otprof, input and schema versions it
was trained with, and loading it checks them against the installed otprof.
"""

from __future__ import annotations

from importlib.metadata import version

from otprof import schema


def get_bundle_meta(features_version: str) -> dict[str, str]:
    """Versions recorded in a saved model's `config.json`: otprof, the model's inputs (`features_version` of its
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

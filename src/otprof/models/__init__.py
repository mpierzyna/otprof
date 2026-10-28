"""Models: one subpackage per model, holding everything that defines it (`data`: data processing, `model`: the
network, `lit`: its training). A new model is a new subpackage with an entry in `otprof.experiment.PIPELINES` and
`otprof.training.ARCHITECTURES`.

The version helpers here are shared by all models: a saved model records the otprof, features and schema versions it
was trained with, and loading it checks them against the installed otprof.
"""

from __future__ import annotations

from importlib.metadata import version

from otprof import features, schema


def get_bundle_meta() -> dict[str, str]:
    """Versions recorded in a saved model's `config.json`."""
    return {
        "otprof_version": version("otprof"),
        "features_version": features.FEATURES_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
    }


def check_versions(config: dict, source: str = "model") -> None:
    """Raise if the MAJOR of `features_version` / `schema_version` in `config` differs from the installed otprof:
    `make_inputs` would no longer compute the inputs the model was trained on. MINOR / PATCH differences pass."""
    installed = {"features_version": features.FEATURES_VERSION, "schema_version": schema.SCHEMA_VERSION}
    for key, current in installed.items():
        saved = config.get(key)
        if saved is None:
            raise ValueError(f"{source}: no {key} recorded")
        if saved.split(".")[0] != current.split(".")[0]:
            raise ValueError(
                f"{source}: trained with {key} {saved}, installed otprof has {current} (different MAJOR: the inputs "
                f"are no longer computed as in training). Use an otprof version with {key} {saved.split('.')[0]}.x"
            )

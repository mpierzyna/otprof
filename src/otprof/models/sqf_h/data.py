"""SQF-H data processing: inputs and their scalers, targets, target transforms and the default split.

`get_pipeline(data_root)` builds the pipeline on the dataset files of a domain; experiments refer to it by name
(`pipeline: sqf_h`) and choose the inputs file (WRF-PL for the production model; native WRF or ERA5-PL for the paper's
comparison models) and, if needed, other selectors. A saved run restores its resolved pipeline, not this function.
"""

from __future__ import annotations

import pathlib

from otprof.pipeline import ProfilePipeline
from otprof.transforms import DatasetScaler, Transforms, TScalerConfig

# Paper split of the nl domain by year
SEL_TRAIN = {"year": [2019, 2020]}
SEL_VAL = {"year": [2018]}
SEL_TEST = {"year": [2017]}

# Inputs of SQF-H, in model order, each with its scaler (`otprof.transforms.DatasetScaler`: dims to aggregate over,
# kind, extra arguments; quantile ranges where outliers would squeeze the bulk of the values). Choose from the features
# `make_inputs` computes (`otprof.features.FEATURES`). Unscaled (None): `valid` (exactly 0/1) and `lz` (the common
# height coordinate of inputs and targets; X and y scalers are fitted separately and would put the two sides into
# different affine spaces).
INPUTS: dict[str, TScalerConfig | None] = {
    # Profiles
    "z_agl": (None, "minmax", None),
    "p": (None, "minmax", None),
    "u": (None, "quantile", (0.01, 0.99)),
    "v": (None, "quantile", (0.01, 0.99)),
    "th": (None, "minmax", None),
    # Gradients to the level above
    "S": (None, "quantile", (0, 0.99)),
    "dth_dz": (None, "quantile", (0.01, 0.99)),
    # Surface
    "ust": (None, "quantile", (0, 0.99)),
    "hfx": (None, "quantile", (0.01, 0.99)),
    "lh": (None, "quantile", (0.01, 0.99)),
    "blh": (None, "quantile", (0.00, 0.99)),
    "u10": (None, "quantile", (0.01, 0.99)),
    "v10": (None, "quantile", (0.01, 0.99)),
    "tk2": (None, "quantile", (0.01, 0.99)),
    "slp": (None, "minmax", None),
    "lsm": (None, "minmax", None),
    # Time
    "hr_sin": (None, "minmax", None),
    "hr_cos": (None, "minmax", None),
    "doy_sin": (None, "minmax", None),
    "doy_cos": (None, "minmax", None),
    # Surface CT2 estimate (W71)
    "lct2_w71f": (None, "quantile", (0.00, 0.995)),  # already clipped on left
    # Height
    "valid": None,
    "lz": None,
    "z_sfc": (None, "minmax", None),
    "lp_sfc": (None, "minmax", None),
}
SCALER_X = {f: scaler for f, scaler in INPUTS.items() if scaler is not None}
# Scalers of the targets (after the target transforms) and of the forcing `z_agl`
SCALER_Y = {
    "lcn2": (None, "quantile", (0.005, 0.995)),
    "QKE": (None, "minmax", None),  # already clipped on lower end
    "TSQ": (None, "quantile", (0.005, 0.995)),
    "EL_PBL": (None, "quantile", (0.005, 0.995)),
    "z": (None, "minmax", None),
    "z_agl": (None, "minmax", None),
}
# Target transforms by name (`otprof.transforms.TRANSFORMS`)
TRANSFORMS_Y = {
    "QKE": ["log10_clip", {"min": 1e-8}],
    "TSQ": ["log10_clip", {"min": 1e-8}],
    "EL_PBL": ["log10_clip", {"min": 1e-1}],
}


def get_pipeline(data_root: pathlib.Path) -> ProfilePipeline:
    """SQF-H: the `INPUTS`, the WRF turbulence targets and heights; log10 of QKE, TSQ and
    EL_PBL before scaling; paper split. The inputs file (WRF-PL, ERA5-PL, native WRF) is chosen by the experiment."""
    return ProfilePipeline(
        data_root=data_root,
        sel_train=SEL_TRAIN,
        sel_val=SEL_VAL,
        sel_test=SEL_TEST,
        vars_X=list(INPUTS),
        vars_y=["lcn2", "QKE", "EL_PBL", "TSQ", "z", "z_agl", "lz"],
        pre_X=[DatasetScaler(SCALER_X)],
        pre_y=[Transforms(TRANSFORMS_Y), DatasetScaler(SCALER_Y)],
    )

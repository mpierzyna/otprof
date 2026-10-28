"""SQF-H data processing: the pipeline on the dataset files, target transforms and scalers, and the default split.

`get_pipeline(data_root)` builds the pipeline on the dataset files of a domain; experiments refer to it by name
(`pipeline: sqf_h`) and choose the inputs file (WRF-PL for the production model; native WRF or ERA5-PL for the paper's
comparison models) and, if needed, other selectors. A saved run restores its resolved pipeline, not this function.
"""

from __future__ import annotations

import pathlib

import numpy as np
import xarray as xr

from otprof.pipeline import ProfilePipeline
from otprof.models.sqf_h import inputs
from otprof.transforms import DatasetScaler, Transforms

# Paper split of the nl domain by year
SEL_TRAIN = {"year": [2019, 2020]}
SEL_VAL = {"year": [2018]}
SEL_TEST = {"year": [2017]}

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


class SqfhPipeline(ProfilePipeline):
    """`ProfilePipeline` that computes the SQF-H inputs (`inputs.make_inputs`) from the canonical inputs file and adds
    `lz` and `lcn2` to the targets (`inputs.make_targets`)."""

    def make_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        return inputs.make_inputs(ds)

    def make_targets(self, ds_y: xr.Dataset) -> xr.Dataset:
        return inputs.make_targets(ds_y)

    def get_input_order(self, features: list[str]) -> list[str]:
        return inputs.sort_surface_first(features)

    def load_data(self) -> None:
        super().load_data()
        assert not self.ds_y["lcn2"].isnull().any() and not np.isinf(self.ds_y["lcn2"]).any()


def get_pipeline(data_root: pathlib.Path) -> SqfhPipeline:
    """SQF-H: the `inputs.INPUTS` (computed from the canonical inputs file, scaled with `inputs.SCALER_X`), the WRF
    turbulence targets and heights; log10 of QKE, TSQ and EL_PBL before scaling; paper split. The inputs file (WRF-PL,
    ERA5-PL, native WRF) is chosen by the experiment."""
    return SqfhPipeline(
        data_root=data_root,
        sel_train=SEL_TRAIN,
        sel_val=SEL_VAL,
        sel_test=SEL_TEST,
        vars_X=list(inputs.FEATURES),
        vars_y=["lcn2", "QKE", "EL_PBL", "TSQ", "z", "z_agl", "lz"],
        pre_X=[DatasetScaler(inputs.SCALER_X)],
        pre_y=[Transforms(TRANSFORMS_Y), DatasetScaler(SCALER_Y)],
    )

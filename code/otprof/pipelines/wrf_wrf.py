from typing import Literal
import pathlib
import warnings

import numpy as np
import xarray as xr
from otprof import PKG_ROOT
from otprof.pipelines.base import BaseDataPipeline
from otprof.pipelines.shared import (
    WRF_SCALER_CONFIG,
    WRF_TF_CONFIG,
    add_X_time_features,
)


class WRForigPipeline(BaseDataPipeline):
    """WRF data without modification as input and output"""

    data_root: pathlib.Path = PKG_ROOT / "../data"

    def load_data(self):
        # Load
        ds_wrf = xr.load_dataset(self.data_root / "wrf_highres.nc")

        # Poor person's solution to match full and half levels. Don't interpolate because I want to avoid smoothing
        bt_full_vars = [v for v in ds_wrf.data_vars if "bottom_top" in ds_wrf[v].dims]
        ds_wrf[bt_full_vars] = ds_wrf[bt_full_vars].isel(bottom_top=slice(0, -1)).rename(bottom_top="bottom_top_h")

        # Compute before flattening to avoid MultiIndex discard error
        with warnings.catch_warnings():
            # Catch divide by zero warnings because I will discard later
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds_wrf["lcn2"] = np.log10(ds_wrf["cn2"])

        # Flatten
        ds_wrf = ds_wrf.stack(sample=("time", "loc")).transpose("sample", ...)

        # Discard invalid Cn2 profiles (NaN from dataset generation)
        cn2_invalid = ds_wrf["cn2"].isnull().any("bottom_top_h")
        ds_wrf = ds_wrf.sel(sample=~cn2_invalid)

        # Make sure above condition caught all NaNs
        assert (ds_wrf["lcn2"].isnull().sum() == 0) & (np.isinf(ds_wrf["lcn2"]).sum() == 0)

        self.ds_X = ds_wrf
        self.ds_y = ds_wrf


# Map vertical-grid mode to the published low-resolution input file.
# Only the ERA5 pressure-level grid ("era5pl") is shipped with this extract.
_MODE_TO_FILE = {"era5pl": "wrf_lowres.nc"}


class WRFWRFPipeline(BaseDataPipeline):
    """WRF interpolated to ERA5 PL as input, original WRF as output"""

    data_root: pathlib.Path = PKG_ROOT / "../data"
    mode: Literal["era5pl"]

    def load_data(self):
        # Load
        if self.mode not in _MODE_TO_FILE:
            raise FileNotFoundError(
                f"Low-resolution input for mode '{self.mode}' is not part of this published extract. "
                f"Available modes: {sorted(_MODE_TO_FILE)}."
            )
        ds_wrf_X = xr.load_dataset(self.data_root / _MODE_TO_FILE[self.mode])
        ds_wrf_X = ds_wrf_X.drop_vars(["bottom_top", "bottom_top_h"])  # drop coordinates
        ds_wrf_y = xr.load_dataset(self.data_root / "wrf_highres.nc")  # standard bottom_top

        # For compatability with WRFonly pipeline, remove highest y levels
        ds_wrf_y = ds_wrf_y.isel(bottom_top=slice(0, -1), bottom_top_h=slice(0, -1))

        # Poor person's solution to match full and half levels. Don't interpolate because I want to avoid smoothing
        full_vars = [v for v in ds_wrf_X.data_vars if "bottom_top" in ds_wrf_X[v].dims]
        ds_wrf_X[full_vars] = ds_wrf_X[full_vars].isel(bottom_top=slice(0, -1)).rename(bottom_top="bottom_top_h")

        # Compute before flattening to avoid MultiIndex discard error
        with warnings.catch_warnings():
            # Catch divide by zero warnings because I will discard later
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ds_wrf_y["lcn2"] = np.log10(ds_wrf_y["cn2"])

        # Flatten
        ds_wrf_X = ds_wrf_X.stack(sample=("time", "loc")).transpose("sample", ...)
        ds_wrf_y = ds_wrf_y.stack(sample=("time", "loc")).transpose("sample", ...)

        # Discard invalid Cn2 profiles (NaN from dataset generation)
        cn2_invalid = ds_wrf_y["cn2"].isnull().any("bottom_top")
        ds_wrf_X = ds_wrf_X.sel(sample=~cn2_invalid)
        ds_wrf_y = ds_wrf_y.sel(sample=~cn2_invalid)

        # Make sure above condition caught all NaNs
        assert (ds_wrf_y["lcn2"].isnull().sum() == 0) & (np.isinf(ds_wrf_y["lcn2"]).sum() == 0)
        self.ds_X = ds_wrf_X
        self.ds_y = ds_wrf_y


# WRF original in and out
p_wrf_orig = WRForigPipeline(
    sel_train=lambda p: {"sample": p.ds_X["time"].dt.year.isin([2019, 2020])},  # 2019-2020
    sel_val=lambda p: {"sample": p.ds_X["time"].dt.year == 2018},  # 2018
    sel_test=lambda p: {"sample": p.ds_X["time"].dt.year == 2017},  # 2017
    vars_X=[
        # "z_rel",
        "z",
        "z_agl",
        "m",
        "d_sin",
        "d_cos",
        "p",
        "u",
        "v",
        "w",
        "th",
        "S",
        "dth_dz",
        # Surface
        "ust",
        "hfx",
        "lh",
        "blh",
        "u10",
        "v10",
        "tk2",
        "slp",
        "lsm",
        "lct2_w71f",
    ],
    vars_y=[
        "lcn2",
        "QKE",
        "EL_PBL",
        "TSQ",
        "z",
        "z_agl",
    ],
    fe_fns=[add_X_time_features],
    scaler_config_X=WRF_SCALER_CONFIG,
    scaler_config_y=WRF_SCALER_CONFIG,
    tf_config_y=WRF_TF_CONFIG,
)

# WRF input to ERA5 PL, same settings as original WRF
p_wrfpl_wrf = WRFWRFPipeline(mode="era5pl", **p_wrf_orig.model_dump())

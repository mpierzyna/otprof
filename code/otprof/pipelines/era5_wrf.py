import itertools
import pathlib
import warnings
from typing import List, Literal

import numpy as np
import xarray as xr

from otprof import PKG_ROOT
from otprof.logging import get_logger
from otprof.transform import QuantileMapper
from otprof.pipelines.base import BaseDataPipeline
from otprof.pipelines.shared import (
    ERA5_SCALER_CONFIG,
    ERA5_VARS_ML,
    ERA5_VARS_PL,
    WRF_SCALER_CONFIG,
    WRF_TF_CONFIG,
    add_X_time_features,
)

logger = get_logger()


class ERA5WRFPipeline(BaseDataPipeline):
    """Original ERA5 (PL or ML) as input, original WRF as output (profiles and astroclim variables)."""

    data_root: pathlib.Path = PKG_ROOT / "../data"
    mode: Literal["ml", "pl"]

    @property
    def ac_vars(self) -> List[str]:
        """Astroclimatic variables in WRF output."""
        ac_vars = itertools.product(
            ["r0", "tau0", "th0", "lsi2"],
            ["full", "bl", "fa"],
        )
        ac_vars = [f"{v}_{r}" for v, r in ac_vars]
        return ac_vars

    def load_data(self) -> None:
        # Load profiles and surface data
        logger.info(f"Using mode {self.mode.upper()} for vertical levels.")
        ds_era5_vert = xr.load_dataset(self.data_root / f"era5_{self.mode}.nc")
        ds_era5_sfc = xr.load_dataset(self.data_root / "era5_sfc.nc")
        ds_era5 = xr.merge([ds_era5_vert, ds_era5_sfc], compat="override")

        # Discard very high levels not present in WRF using precomputed mask
        ds_era5 = ds_era5.sel({"bottom_top": ds_era5["_wrf_mask"], f"bottom_top_h": ds_era5["_wrf_mask_h"]})
        ds_era5 = ds_era5.drop_vars(["_wrf_mask", "_wrf_mask_h"])  # drop masks
        ds_era5 = ds_era5.isel(bottom_top=slice(0, -1))  # Remove highest level to match half-levels

        # Rename half-levels to full-levels for consistency
        bt_h_vars = [v for v in ds_era5.data_vars if "bottom_top_h" in ds_era5[v].dims]
        ds_era5[bt_h_vars] = ds_era5[bt_h_vars].rename(bottom_top_h="bottom_top")
        logger.info(f"Using {ds_era5.sizes['bottom_top']} vertical levels from ERA5 after masking.")

        # Load WRF
        # Poor person's solution to match full and half levels. Don't interpolate because I want to avoid smoothing
        # ERA5 alignment done above when discarding too high levels.
        ds_wrf = xr.load_dataset(self.data_root / "wrf_highres.nc")
        ds_wrf = ds_wrf.isel(bottom_top=slice(0, -1))

        # Rename half-levels for consistency
        bt_h_vars = [v for v in ds_wrf.data_vars if "bottom_top_h" in ds_wrf[v].dims]
        ds_wrf[bt_h_vars] = ds_wrf[bt_h_vars].rename(bottom_top_h="bottom_top")

        if "lcn2" in self.vars_y:
            # Compute before flattening to avoid MultiIndex discard error
            with warnings.catch_warnings():
                # Catch divide by zero warnings because I will discard later
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ds_wrf["lcn2"] = np.log10(ds_wrf["cn2"])

        # Flatten
        ds_era5 = ds_era5.stack(sample=("time", "loc")).transpose("sample", ...)
        ds_wrf = ds_wrf.stack(sample=("time", "loc")).transpose("sample", ...)

        self.ds_X = ds_era5
        self.ds_y = ds_wrf

    def fill_gaps(self):
        """Filter invalid Cn2 if Cn2 is target."""
        ds_X = self.ds_X
        ds_y = self.ds_y

        if "lcn2" in self.vars_y:
            # Discard invalid Cn2 profiles (NaN from dataset generation)
            cn2_invalid = ds_y["cn2"].isnull().any("bottom_top")
            ds_X = ds_X.sel(sample=~cn2_invalid)
            ds_y = ds_y.sel(sample=~cn2_invalid)

            # Make sure above condition caught all NaNs
            assert (ds_y["lcn2"].isnull().sum() == 0) & (np.isinf(ds_y["lcn2"]).sum() == 0)

        if any([v in self.ac_vars for v in self.vars_y]):
            # Discard invalid astroclimatic samples (NaN from dataset generation)
            invalid = ds_y[self.ac_vars].to_array().isnull().any("variable")
            ds_X = ds_X.sel(sample=~invalid)
            ds_y = ds_y.sel(sample=~invalid)

        # Save only selected variables
        self.ds_X = ds_X[self.vars_X]
        self.ds_y = ds_y[self.vars_y]


def era5_match_wrf(
    dp_era5: BaseDataPipeline,
    dp_wrf: BaseDataPipeline,
    features: List[str],
    quantile_mapping: bool,
    features_qm: List[str] | None = None,
) -> BaseDataPipeline:
    """Match ERA5 data pipeline to WRF data pipeline.
    - Rename variables
    - Overwrite scaler
    - Optionally, map quantiles

    We may not want to apply QM to all features (eg LSM or sin/cos hr or day).
    Provide list in `features_qm`, which should be aligned. If None, apply to all `features`.
    """
    # Make a copy
    dp_era5 = dp_era5.model_copy(deep=True)

    # Ensure that X variables match WRF
    features_set = set(features)
    vars_X_era5 = set(dp_era5.vars_X)
    vars_X_wrf = set(dp_wrf.vars_X)
    if not features_set.issubset(vars_X_era5):
        missing = features_set - vars_X_era5
        raise ValueError(f"ERA5 is missing features: {missing}")
    if not features_set.issubset(vars_X_wrf):
        missing = features_set - vars_X_wrf
        raise ValueError(f"WRF is missing features: {missing}")
    dp_era5.vars_X = features
    dp_era5.ds_X = dp_era5.ds_X[features].isel(bottom_top=slice(0, -1))  # Remove highest level

    # Optionally, apply quantile mapping to ERA5 inputs to match WRF
    if quantile_mapping:
        qm = QuantileMapper(
            n_quantiles=500,
            vars=features_qm or features,
        )

        logger.info("Fitting QuantileMapper on training data of ERA5 and WRF.")
        logger.info(f"Transforming variables: {qm.vars}")
        X_wrf, _ = dp_wrf.split("train", transform=False)
        X_era5, _ = dp_era5.split("train", transform=False)
        qm.fit(X_obs=X_wrf, X_model=X_era5)

        logger.info("Applying QuantileMapper to all ERA5 data.")
        dp_era5.ds_X = qm.transform(dp_era5.ds_X)

    # Overwrite scaler
    dp_era5.scaler_X = dp_wrf.scaler_X

    return dp_era5


# ERA5 PL/ML to WRF original
p_era5pl_wrf = ERA5WRFPipeline(
    mode="pl",
    sel_train=lambda p: {"sample": p.ds_X["time"].dt.year.isin([2019, 2020])},  # 2019-2020
    sel_val=lambda p: {"sample": p.ds_X["time"].dt.year == 2018},  # 2018
    sel_test=lambda p: {"sample": p.ds_X["time"].dt.year == 2017},  # 2017
    vars_X=ERA5_VARS_PL,
    vars_y=[
        "m",
        "u",
        "v",
        "w",
        "d_sin",
        "d_cos",
        "th",
        "p",
        "z_agl",
        "z",
        "lcn2",
        "TSQ",
        "EL_PBL",
        "QKE",
    ],
    fe_fns=[add_X_time_features],
    scaler_config_X=ERA5_SCALER_CONFIG,
    scaler_config_y=WRF_SCALER_CONFIG,
    tf_config_y=WRF_TF_CONFIG,
)
p_era5ml_wrf = p_era5pl_wrf.model_copy(
    update={
        "mode": "ml",
        "vars_X": ERA5_VARS_ML,
    }
)

# ERA5 PL/ML to WRF astroclim parameters
p_era5pl_wrf_ac = ERA5WRFPipeline(
    mode="pl",
    sel_train=lambda p: {"sample": p.ds_X["time"].dt.year.isin([2019, 2020])},  # 2019-2020
    sel_val=lambda p: {"sample": p.ds_X["time"].dt.year == 2018},  # 2018
    sel_test=lambda p: {"sample": p.ds_X["time"].dt.year == 2017},  # 2017
    vars_X=ERA5_VARS_PL,
    vars_y=[
        "r0_full",
        "r0_bl",
        "r0_fa",
        "th0_full",
        "lsi2_full",
        "lsi2_bl",
        "lsi2_fa",
        "tau0_full",
        "tau0_bl",
        "tau0_fa",
    ],
    fe_fns=[add_X_time_features],
    scaler_config_X=ERA5_SCALER_CONFIG,
    scaler_config_y=WRF_SCALER_CONFIG,
)
p_era5ml_wrf_ac = p_era5pl_wrf_ac.model_copy(
    update={
        "mode": "ml",
        "vars_X": ERA5_VARS_ML,
    }
)

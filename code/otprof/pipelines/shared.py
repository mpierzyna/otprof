from typing import Tuple, Dict

import numpy as np
import xarray as xr


def add_X_time_features(ds_X: xr.Dataset, _) -> Tuple[Dict[str, xr.DataArray], Dict]:
    """Add time-based features to input dataset."""
    t = ds_X["time"].dt
    X_fe = {
        "hr_sin": np.sin(2 * np.pi * t.hour / 24),
        "hr_cos": np.cos(2 * np.pi * t.hour / 24),
        "doy_sin": np.sin(2 * np.pi * t.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * t.dayofyear / 365),
    }
    return X_fe, {}


# Engineered features
FE_SCALER_CONFIG = {
    "hr_sin": (None, "minmax", None),
    "hr_cos": (None, "minmax", None),
    "doy_sin": (None, "minmax", None),
    "doy_cos": (None, "minmax", None),
}

# This is used for all ERA5 pipelines (PL and ML)
ERA5_SCALER_CONFIG = {
    # both PL and ML
    "z": (None, "minmax", None),
    "z_rel": (None, "quantile", (0, 0.99)),
    "z_agl": (None, "quantile", (0, 0.99)),
    "p": (None, "minmax", None),
    "u": (None, "quantile", (0.01, 0.99)),
    "v": (None, "quantile", (0.01, 0.99)),
    "m": (None, "minmax", None),
    "d_sin": (None, "minmax", None),
    "d_cos": (None, "minmax", None),
    "w": (None, "quantile", (0.01, 0.99)),
    "th": (None, "minmax", None),
    "dth_dz": (None, "quantile", (0.01, 0.99)),
    "S": (None, "quantile", (0, 0.99)),
    # sfc
    "ust": (None, "quantile", (0, 0.99)),
    "hfx": (None, "quantile", (0.01, 0.99)),
    "lh": (None, "quantile", (0.01, 0.99)),
    "blh": (None, "quantile", (0.00, 0.99)),
    "u10": (None, "quantile", (0.01, 0.99)),
    "v10": (None, "quantile", (0.01, 0.99)),
    "tk2": (None, "quantile", (0.01, 0.99)),
    "lsm": (None, "minmax", None),
    "boundary_layer_dissipation": (None, "quantile", (0.0, 0.99)),
    "angle_of_sub_gridscale_orography": (None, "minmax", None),
    "anisotropy_of_sub_gridscale_orography": (None, "minmax", None),
    "slp": (None, "minmax", None),  # sea level pressure
    "lct2_w71f": (None, "quantile", (0.00, 0.995)),  # already clipped on left
}
ERA5_SCALER_CONFIG_PL = {
    # PL only
    "pv": (None, "quantile", (0.01, 0.99)),
}
ERA5_SCALER_CONFIG_ML = {
    # ML only
    "omega": (None, "quantile", (0.01, 0.99)),
    "div": (None, "quantile", (0.01, 0.99)),
}
ERA5_VARS_PL = list(ERA5_SCALER_CONFIG.keys()) + list(ERA5_SCALER_CONFIG_PL.keys())
ERA5_VARS_ML = list(ERA5_SCALER_CONFIG.keys()) + list(ERA5_SCALER_CONFIG_ML.keys())

# Combine all configs here
ERA5_SCALER_CONFIG = {
    **ERA5_SCALER_CONFIG,
    **ERA5_SCALER_CONFIG_PL,
    **ERA5_SCALER_CONFIG_ML,
    **FE_SCALER_CONFIG,
}

WRF_SCALER_CONFIG = {
    # Turbulent profiles
    "lcn2": (None, "quantile", (0.005, 0.995)),
    "QKE": (None, "minmax", None),  # already clipped on lower end
    "TSQ": (None, "quantile", (0.005, 0.995)),
    "EL_PBL": (None, "quantile", (0.005, 0.995)),
    # Mean profiles
    "z": (None, "minmax", None),
    "z_agl": (None, "minmax", None),
    "p": (None, "minmax", None),
    "m": (None, "quantile", (0.0, 0.99)),
    "d_sin": (None, "minmax", None),
    "d_cos": (None, "minmax", None),
    "u": (None, "quantile", (0.01, 0.99)),
    "v": (None, "quantile", (0.01, 0.99)),
    "w": (None, "quantile", (0.01, 0.99)),
    "th": (None, "minmax", None),
    # Gradients
    "dth_dz": (None, "quantile", (0.01, 0.99)),
    "S": (None, "quantile", (0, 0.99)),
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
    # Integrated parameters
    "r0_full": (None, "quantile", (0.00, 0.99)),
    "r0_bl": (None, "quantile", (0.00, 0.99)),
    "r0_fa": (None, "quantile", (0.00, 0.99)),
    "th0_full": (None, "quantile", (0.00, 0.99)),
    "lsi2_full": (None, "quantile", (0.01, 0.99)),
    "lsi2_bl": (None, "quantile", (0.01, 0.99)),
    "lsi2_fa": (None, "quantile", (0.01, 0.99)),
    "tau0_full": (None, "quantile", (0.00, 0.99)),
    "tau0_bl": (None, "quantile", (0.00, 0.99)),
    "tau0_fa": (None, "quantile", (0.00, 0.99)),
    # Engineered features
    "lct2_w71f": (None, "quantile", (0.00, 0.995)),  # already clipped on left
    **FE_SCALER_CONFIG,
}

# Have lambda in individual lines for serialization to work.
WRF_TF_CONFIG = {
    "QKE": (
        lambda qke: np.log10(qke.clip(min=1e-8)),
        lambda x: 10**x,
    ),
    "TSQ": (
        lambda tsq: np.log10(tsq.clip(min=1e-8)),
        lambda x: 10**x,
    ),
    "EL_PBL": (
        lambda el_pbl: np.log10(el_pbl.clip(min=1e-1)),
        lambda x: 10**x,
    ),
}

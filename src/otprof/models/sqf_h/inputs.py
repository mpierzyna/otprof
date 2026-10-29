"""SQF-H inputs: the features the model takes, how they are computed from the canonical input dataset
(`otprof.schema`) and how they are scaled, and the target variables it adds (`lcn2`, `lz`).

`make_inputs` builds the training inputs (from the WRF or ERA5 dataset files, when the pipeline loads them) and the
inputs at inference (ERA5, later GFS) alike. Bump `FEATURES_VERSION` (semver, see there) whenever its output changes.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import xarray as xr

from otprof import schema
from otprof.features import fill_nearest, get_time_features, potential_temperature, sfc_ct2_w71
from otprof.transforms import TScalerConfig

# Version of the SQF-H inputs (semver). MAJOR: the values of a feature change or a feature is added or removed, so
# models trained on the old inputs need retraining. MINOR: compatible addition (e.g. new metadata) or a small value
# change shown not to need retraining. PATCH: no change to the output (refactoring, docs). A model records the version
# it was trained with; inputs fit if the MAJOR versions are equal.
# 1.1.0: W71 without clipping of the Obukhov length (changes `lct2_w71f` of near-neutral samples by < 0.02 dex)
FEATURES_VERSION = "1.1.0"

# The inputs of SQF-H in model order: name -> (kind, scaler). Kind "profile": on the levels (dim `bottom_top`);
# "surface": one value per column (surface fields, time, surface estimates). Scaler: `otprof.transforms.DatasetScaler`
# config (dims to aggregate over, kind, extra arguments; quantile ranges where outliers would squeeze the bulk of the
# values), None = unscaled: `valid` (exactly 0/1) and `lz` (the common height coordinate of inputs and targets; X and
# y scalers are fitted separately and would put the two sides into different affine spaces).
INPUTS: dict[str, tuple[Literal["profile", "surface"], TScalerConfig | None]] = {
    # Profiles
    "z_agl": ("profile", (None, "minmax", None)),
    "p": ("profile", (None, "minmax", None)),
    "u": ("profile", (None, "quantile", (0.01, 0.99))),
    "v": ("profile", (None, "quantile", (0.01, 0.99))),
    "th": ("profile", (None, "minmax", None)),
    # Gradients to the level above
    "S": ("profile", (None, "quantile", (0, 0.99))),
    "dth_dz": ("profile", (None, "quantile", (0.01, 0.99))),
    # Surface
    "ust": ("surface", (None, "quantile", (0, 0.99))),
    "hfx": ("surface", (None, "quantile", (0.01, 0.99))),
    "lh": ("surface", (None, "quantile", (0.01, 0.99))),
    "blh": ("surface", (None, "quantile", (0.00, 0.99))),
    "u10": ("surface", (None, "quantile", (0.01, 0.99))),
    "v10": ("surface", (None, "quantile", (0.01, 0.99))),
    "tk2": ("surface", (None, "quantile", (0.01, 0.99))),
    "slp": ("surface", (None, "minmax", None)),
    "lsm": ("surface", (None, "minmax", None)),
    # Time
    "hr_sin": ("surface", (None, "minmax", None)),
    "hr_cos": ("surface", (None, "minmax", None)),
    "doy_sin": ("surface", (None, "minmax", None)),
    "doy_cos": ("surface", (None, "minmax", None)),
    # Surface CT2 estimate (W71)
    "lct2_w71f": ("surface", (None, "quantile", (0.00, 0.995))),  # already clipped on left
    # Height
    "valid": ("profile", None),
    "lz": ("profile", None),
    "z_sfc": ("surface", (None, "minmax", None)),
    "lp_sfc": ("surface", (None, "minmax", None)),
}
FEATURES = list(INPUTS)
PROFILE_FEATURES = [f for f, (kind, _) in INPUTS.items() if kind == "profile"]
SCALER_X = {f: scaler for f, (_, scaler) in INPUTS.items() if scaler is not None}

# Warped height lz = log1p(z_agl / LZ_REF) / LZ_NORM, ~[0, 1] over 0..30 km: the common height coordinate of the
# model's inputs and outputs. Fixed constants, never fitted (X and y scalers would put the two sides into different
# spaces).
LZ_REF = 10.0  # m
LZ_NORM = float(np.log1p(30_000.0 / LZ_REF))


def compute_warped_height(z_agl: xr.DataArray) -> xr.DataArray:
    """lz = log1p(z_agl / LZ_REF) / LZ_NORM; below ground clipped to 0."""
    return np.log1p(z_agl.clip(min=0) / LZ_REF) / LZ_NORM


def sort_surface_first(features: list[str]) -> list[str]:
    """Order of the variables in the model's input tensor: surface features first, then profiles, each in the given
    order (as `otprof.training.ProfileDataset` stacks them)."""
    return [f for f in features if f not in PROFILE_FEATURES] + [f for f in features if f in PROFILE_FEATURES]


def make_inputs(
    ds: xr.Dataset,
    *,
    grid: schema.TGrid | None = None,
) -> xr.Dataset:
    """The SQF-H inputs `FEATURES` (unscaled) on the lowest N-1 of the N levels of `ds` (canonical inputs).

    - `grid`: "pressure" or "model"; by default "pressure" if `ds` has the coordinate `p_level`.
    - Gradients (`S`, `dth_dz`) are taken between each level and the level above; the top level of `ds` only enters
      these gradients (on the pressure grid: 31 levels 1000..20 hPa → 30 levels 1000..30 hPa).
    - `z_sfc`: the terrain height of the column to predict, `z_sfc_target` if `ds` has it (training data: the WRF
      terrain of the target column), else the source's own terrain `z_sfc_src`. `z_agl` and `valid` always refer to
      the source's own surface.
    - Every input depends on its own column and time only, not on the other samples passed in.
    """
    grid = schema.get_grid(ds) if grid is None else grid
    schema.validate(ds, grid=grid)
    ds = ds.drop_vars("p_level", errors="ignore")  # positional levels; the coordinate is restored at the end
    full = {"bottom_top": slice(0, -1)}

    # Profiles
    z_agl = (ds["z"] - ds["z_sfc_src"]).assign_attrs(long_name="Height above ground", units="m")
    th = ds["th"] if "th" in ds else potential_temperature(tk=ds["tk"], p=ds["p"])

    # Gradients between consecutive levels, assigned to the lower level
    dz = ds["z"].diff("bottom_top", label="lower")
    du_dz = ds["u"].diff("bottom_top", label="lower") / dz
    dv_dz = ds["v"].diff("bottom_top", label="lower") / dz
    S = np.sqrt(du_dz**2 + dv_dz**2).astype(np.float32).assign_attrs(long_name="Wind shear", units="s-1")
    dth_dz = (th.diff("bottom_top", label="lower") / dz).astype(np.float32)
    dth_dz = dth_dz.assign_attrs(long_name="Vertical gradient of potential temperature", units="K m-1")
    # Levels repeated by the source (dz = 0, e.g. WRF levels filled below ground) -> gradient of the nearest level
    S, dth_dz = fill_nearest(S), fill_nearest(dth_dz)

    # Height features
    if grid == "pressure":
        p_nominal = xr.DataArray(np.array(schema.P_LEVELS_HPA, dtype=float) * 100, dims="bottom_top")
        valid = p_nominal < ds["p_sfc"]  # level above the source's surface
    else:
        valid = xr.ones_like(z_agl, dtype=bool)
    valid = valid.astype(np.float32).assign_attrs(long_name="Level above the surface", units="1")
    lz = compute_warped_height(z_agl).assign_attrs(long_name="Warped height above ground", units="1")
    z_sfc = ds["z_sfc_target"] if "z_sfc_target" in ds else ds["z_sfc_src"]
    lp_sfc = np.log(ds["p_sfc"]).assign_attrs(long_name="Log surface pressure", units="log(Pa)")

    X = {
        "z_agl": z_agl.isel(full),
        "p": ds["p"].isel(full),
        "u": ds["u"].isel(full),
        "v": ds["v"].isel(full),
        "th": th.isel(full),
        "S": S,
        "dth_dz": dth_dz,
        **{v: ds[v] for v in ["ust", "hfx", "lh", "blh", "u10", "v10", "tk2", "slp", "lsm"]},
        **get_time_features(ds["time"]),
        "lct2_w71f": np.log10(sfc_ct2_w71(ust=ds["ust"], hfx=ds["hfx"], tk2=ds["tk2"])),
        "valid": valid.isel(full),
        "lz": lz.isel(full),
        "z_sfc": z_sfc.assign_attrs(long_name="Terrain height of the target column", units="m"),
        "lp_sfc": lp_sfc,
    }
    X = xr.Dataset({k: X[k] for k in FEATURES})
    if grid == "pressure":
        X = X.assign_coords(p_level=("bottom_top", list(schema.P_LEVELS_HPA[:-1])))
    return X.assign_attrs(
        otprof_features_version=FEATURES_VERSION, otprof_schema_version=schema.SCHEMA_VERSION, otprof_grid=grid
    )


def make_targets(ds_y: xr.Dataset) -> xr.Dataset:
    """The target variables SQF-H adds to the WRF targets (or to query heights `z_agl`): the warped height `lz` of
    `z_agl` and, if `cn2` is present, `lcn2` = log10 Cn2 (NaN and -inf for masked or zero Cn2)."""
    ds_y = ds_y.copy()
    ds_y["lz"] = compute_warped_height(ds_y["z_agl"]).assign_attrs(long_name="Warped height above ground", units="1")
    if "cn2" in ds_y:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # log10(0) and NaN: dropped by the pipeline
            ds_y["lcn2"] = np.log10(ds_y["cn2"])
    return ds_y

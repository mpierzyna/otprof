"""Canonical input dataset: what a source adapter (`otprof.sources`) delivers and what `otprof.features.make_inputs`
relies on.

- Vertical dim `bottom_top`: index 0 is the lowest level, the last the top. On the pressure grid (all published
  models) exactly the levels `P_LEVELS_HPA`, with the nominal pressure [hPa] as coordinate `p_level`; on the model grid
  (native WRF levels) any number of levels.
- Profile variables have the `bottom_top` dim, surface variables do not. All other dims are free: `(time, loc)` for points,
  `(time, lat, lon)` for grids.
- Variables not listed in `VARIABLES` pass through untouched (e.g. extra ERA5 fields used in research).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pydantic
import xarray as xr

# Version of this input contract (semver). MAJOR: what an adapter must deliver changes incompatibly (variable removed or
# renamed, units, sign conventions, level set). MINOR: compatible addition (new optional variable). PATCH: no change to
# what is accepted (descriptions, plausible ranges).
SCHEMA_VERSION = "1.0.0"

TGrid = Literal["pressure", "model"]

# Pressure levels [hPa] of the pressure grid, bottom to top. The model sees the first 30 (1000..30 hPa); 20 hPa only
# enters the gradients of the top level.
P_LEVELS_HPA = (
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500,
    450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20,
)  # fmt: skip

# Warped height lz = log1p(z_agl / LZ_REF) / LZ_NORM, ~[0, 1] over 0..30 km: the common height coordinate of the
# model's inputs and outputs. Fixed constants, never fitted (X and y scalers would put the two sides into different
# spaces).
LZ_REF = 10.0  # m
LZ_NORM = float(np.log1p(30_000.0 / LZ_REF))


class Var(pydantic.BaseModel):
    """One variable of the canonical input dataset."""

    model_config = pydantic.ConfigDict(frozen=True)

    long_name: str
    units: str
    kind: Literal["profile", "surface"]
    required: bool = True
    range: tuple[float, float]  # plausible values; catches wrong units / conventions, not a physical limit


VARIABLES: dict[str, Var] = {
    # Profiles
    "z": Var(long_name="Height above mean sea level", units="m", kind="profile", range=(-1_000, 45_000)),
    "p": Var(long_name="Pressure", units="Pa", kind="profile", range=(100, 115_000)),
    "u": Var(long_name="Eastward wind", units="m s-1", kind="profile", range=(-200, 200)),
    "v": Var(long_name="Northward wind", units="m s-1", kind="profile", range=(-200, 200)),
    # One of tk / th is required (checked in `validate`); th is derived from tk and p if missing
    "tk": Var(long_name="Temperature", units="K", kind="profile", required=False, range=(150, 350)),
    "th": Var(long_name="Potential temperature", units="K", kind="profile", required=False, range=(150, 2_000)),
    # Surface
    "p_sfc": Var(long_name="Surface pressure", units="Pa", kind="surface", range=(30_000, 115_000)),
    "z_sfc_src": Var(long_name="Terrain height of the source", units="m", kind="surface", range=(-500, 9_000)),
    "ust": Var(long_name="Friction velocity", units="m s-1", kind="surface", range=(0, 10)),
    "hfx": Var(long_name="Sensible heat flux, upward positive", units="W m-2", kind="surface", range=(-2_000, 2_000)),
    "lh": Var(long_name="Latent heat flux, upward positive", units="W m-2", kind="surface", range=(-2_000, 3_000)),
    "blh": Var(long_name="Boundary layer height", units="m", kind="surface", range=(0, 10_000)),
    "u10": Var(long_name="Eastward wind at 10 m", units="m s-1", kind="surface", range=(-150, 150)),
    "v10": Var(long_name="Northward wind at 10 m", units="m s-1", kind="surface", range=(-150, 150)),
    "tk2": Var(long_name="Temperature at 2 m", units="K", kind="surface", range=(170, 350)),
    "slp": Var(long_name="Sea level pressure", units="hPa", kind="surface", range=(850, 1_100)),
    "lsm": Var(long_name="Land-sea mask (land = 1)", units="1", kind="surface", range=(0, 1)),
}


class SchemaError(ValueError):
    """The dataset does not follow the canonical input schema. The message lists every problem found."""


def set_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Set `long_name` and `units` of all canonical variables in `ds` (for adapters)."""
    for name, var in VARIABLES.items():
        if name in ds:
            ds[name].attrs.update(long_name=var.long_name, units=var.units)
    return ds


def validate(ds: xr.Dataset, grid: TGrid = "pressure", check_ranges: bool = True) -> xr.Dataset:
    """Check `ds` against the canonical schema and return it unchanged. Raises `SchemaError` listing all problems."""
    problems = []

    # Vertical grid
    if "bottom_top" not in ds.dims:
        problems.append("no vertical dim `bottom_top`")
    elif grid == "pressure":
        if "p_level" not in ds.coords or not np.array_equal(ds["p_level"].values, P_LEVELS_HPA):
            found = ds["p_level"].values.tolist() if "p_level" in ds.coords else "no coordinate"
            problems.append(
                f"`p_level` must be the pressure levels {list(P_LEVELS_HPA)} hPa (bottom to top), got {found}"
            )
    elif ds.sizes["bottom_top"] < 2:
        problems.append("the model grid needs at least 2 levels")

    # Variables
    for name, var in VARIABLES.items():
        if name not in ds:
            if var.required:
                problems.append(f"missing `{name}` ({var.long_name} [{var.units}])")
            continue
        da = ds[name]
        has_level = "bottom_top" in da.dims
        if var.kind == "profile" and not has_level:
            problems.append(f"`{name}` is a profile variable but has no `bottom_top` dim (dims {da.dims})")
        if var.kind == "surface" and has_level:
            problems.append(f"`{name}` is a surface variable but has a `bottom_top` dim")
        if da.attrs.get("units") != var.units:
            problems.append(f"`{name}` must have units `{var.units}`, attrs say `{da.attrs.get('units')}`")
        if check_ranges and da.size:
            lo, hi = float(da.min()), float(da.max())
            if lo < var.range[0] or hi > var.range[1]:
                problems.append(f"`{name}` outside the plausible range {var.range} [{var.units}]: {lo:.4g}..{hi:.4g}")
    if "tk" not in ds and "th" not in ds:
        problems.append("one of `tk` (temperature) or `th` (potential temperature) is required")

    if "lon" in ds.coords and ds["lon"].size and (float(ds["lon"].min()) < -180 or float(ds["lon"].max()) > 180):
        problems.append("`lon` must be in -180..180")

    if problems:
        raise SchemaError("Dataset does not follow the otprof input schema:\n- " + "\n- ".join(problems))
    return ds

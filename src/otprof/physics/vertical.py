"""Vertical grid utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr


def interp_vert(ds: xr.Dataset, z: xr.DataArray, z_target: xr.DataArray, dim: str) -> xr.Dataset:
    """Linearly interpolate all variables of `ds` along `dim` from heights `z` to `z_target`.

    ATTENTION! z and z_target must increase monotonically! Targets outside the range of `z` give NaN (both
    bracketing levels collapse onto the outermost level).
    """

    def _searchsorted(a, v):
        # Break out function for debugging apply_ufunc
        res = np.searchsorted(a, v, side="left")
        return res

    # Compute indices of bracketing bottom_top levels
    bt_j = xr.apply_ufunc(
        _searchsorted,
        z,
        z_target,
        input_core_dims=[[dim], [dim]],
        exclude_dims={dim},
        output_core_dims=[[dim]],
        vectorize=True,
    )
    bt_i = (bt_j - 1).clip(min=0)
    bt_j = bt_j.clip(max=ds.sizes[dim] - 1)

    # Get values for bracketing levels
    z_i = z.sel({dim: bt_i})
    z_j = z.sel({dim: bt_j})

    # Get data for bracketing levels
    vars_interp = [v for v in ds if dim in ds[v].dims]
    ds_i = ds[vars_interp].sel({dim: bt_i})
    ds_j = ds[vars_interp].sel({dim: bt_j})

    # Interpolate
    ds_interp = ds_i + (z_target - z_i) * (ds_j - ds_i) / (z_j - z_i)
    for v in vars_interp:
        # Attributes are lost in interpolation. Restore them.
        ds_interp[v].attrs.update(ds[v].attrs)

    # Assign interpolation target values as coordinate
    ds_interp = ds_interp.assign_coords({dim: z_target})
    return ds_interp


def get_half_levels(v: xr.DataArray, dim: str) -> xr.DataArray:
    """Get half-levels of a DataArray along given dimension."""
    v_h = v.rolling({dim: 2}, center=True).mean()
    v_h = v_h.isel({dim: slice(1, None)})  # Drop NaNs at edge
    if dim in v_h.coords:
        v_h = v_h.drop_vars(dim)
    v_h = v_h.rename({dim: f"{dim}_h"})
    return v_h

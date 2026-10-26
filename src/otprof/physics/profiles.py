"""Empirical Cn2 reference profiles."""

from __future__ import annotations

import numpy as np
import xarray as xr


def hufnagel_valley(
    cn2_0: float,
    m: np.ndarray,
    z: np.ndarray,
    z_eval: np.ndarray | None = None,
) -> np.ndarray:
    """Hufnagel-Valley Cn2 profile.

    Parameters
    ----------
    cn2_0 : float
        Ground level Cn2 value (m^(-2/3)).
    m : np.ndarray
        Profile of wind magnitude (m/s).
    z : np.ndarray
        Altitude profile (m) matching `m` for computing integrated wind speed.
    z_eval : np.ndarray | None
        Altitude profile (m) where to evaluate Cn2. If None, use `z`.

    Returns
    -------
    np.ndarray
        Cn2 profile (m^(-2/3)).

    References
    ----------
    Smith 1993
    """
    c1 = 8.2e-26  # constant
    c2 = 2.7e-16  # constant
    A = cn2_0 - c2  # compute A to match ground level Cn2

    # convert to km
    h = z / 1000
    if z_eval is None:
        h_eval = h
    else:
        h_eval = z_eval / 1000

    # Integrate wind between 5km and 20km
    mask = (h >= 5) & (h < 20)
    W = np.sqrt(np.trapezoid(m[mask] ** 2, h[mask]) / 15)  # m/s

    cn2 = c1 * W**2 * h_eval**10 * np.exp(-h_eval) + c2 * np.exp(-h_eval / 1.5) + A * np.exp(-h_eval / 0.1)
    return cn2


def hufnagel_valley_xr(
    cn2_0: xr.DataArray,
    m: xr.DataArray,
    z: xr.DataArray,
    dim: str,
    z_eval: xr.DataArray | None = None,
) -> xr.DataArray:
    """Hufnagel-Valley Cn2 profile using xarray"""
    assert dim in m.dims, f"Dimension {dim} not found in m"
    assert dim in z.dims, f"Dimension {dim} not found in z"
    if z_eval is not None:
        assert dim in z_eval.dims, f"Dimension {dim} not found in z_eval"

    if z_eval is None:
        dim_ = None
        args = (cn2_0, m, z)
        input_core_dims = [[], [dim], [dim]]
        out_core_dims = [[dim]]
    else:
        # Evaluation height needs is own dim
        dim_ = f"{dim}_"
        z_eval = z_eval.rename({dim: dim_})
        args = (cn2_0, m, z, z_eval)
        input_core_dims = [[], [dim], [dim], [dim_]]
        out_core_dims = [[dim_]]

    ds = xr.apply_ufunc(
        hufnagel_valley,
        *args,
        input_core_dims=input_core_dims,
        output_core_dims=out_core_dims,
        vectorize=True,
    )
    if dim_ is not None:
        ds = ds.rename({dim_: dim})
    return ds

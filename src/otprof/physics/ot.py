"""Integrated optical turbulence parameters from Cn2 profiles (numpy and xarray versions).

All parameters are computed for a wavelength of 1550 nm.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypeVar

import numpy as np
import xarray as xr

T = TypeVar("T")
TWaveMode = Literal["plane", "spherical"]

LAMBDA = 1550e-9  # wavelength [m]
K = (2 * np.pi) / LAMBDA  # wavenumber [1/m]


class TrapezoidFn(Protocol):
    def __call__(self, y: T, x: T, dim: int | str) -> T:
        """Trapezoid function protocol.
        Types are determined by implementation, but function must have y, x, and dim as arguments.
        """


# Base implementations are agnostic of the array framework
def _fried_r0(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Fried parameter r0 along a given dimension. Result in m."""
    return (0.423 * K**2 * trapezoid_fn(y=cn2, x=z, dim=dim)) ** (-3 / 5)


def _rytov_var(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn, mode: TWaveMode) -> T:
    """Rytov variance along a path of length l. [1, p. 50]

    References
    ----------
    .. [1] Andrews, Larry C. Field Guide to Atmospheric Optics. Second Edition, vol. FG41, SPIE Press, 2014. SPIE Field Guides.

    """
    if mode == "plane":
        c = 1.23
    elif mode == "spherical":
        c = 0.5
    else:
        raise ValueError("mode must be 'plane' or 'spherical'")
    return c * K ** (7 / 6) * trapezoid_fn(y=cn2 * (z ** (5 / 6)), x=z, dim=dim)


def _scint_index(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn, mode: TWaveMode) -> T:
    """Scintillation index in weak and strong turbulence regimes."""
    sr2 = _rytov_var(cn2=cn2, z=z, dim=dim, trapezoid_fn=trapezoid_fn, mode=mode)
    si2 = (0.49 * sr2 / (1 + 1.11 * sr2 ** (6 / 5)) ** (7 / 6)) + (0.51 * sr2 / (1 + 0.69 * sr2 ** (6 / 5)) ** (5 / 6))
    return np.exp(si2) - 1


def _isoplanatic_angle(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Isoplanatic angle. Result in rad."""
    return 0.057 * LAMBDA ** (6 / 5) * (trapezoid_fn(y=cn2 * (z ** (5 / 3)), x=z, dim=dim) ** (-3 / 5))


def _wave_coherence_time(cn2: T, m: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Wave coherence time. m is the vertical wind profile. Result in s."""
    return 0.057 * LAMBDA ** (6 / 5) * (trapezoid_fn(y=cn2 * (m ** (5 / 3)), x=z, dim=dim) ** (-3 / 5))


def _seeing(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Seeing. Result in rad."""
    r0 = _fried_r0(cn2, z, dim, trapezoid_fn)
    return 0.98 * LAMBDA / r0


# Numpy implementations
def _np_trapezoid(y: np.ndarray, x: np.ndarray, dim: int) -> np.ndarray:
    """Rename axis to dim because base implementation requires dim as kwarg."""
    return np.trapezoid(y, x, axis=dim)


def fried_r0_np(cn2: np.ndarray, z: np.ndarray, axis: int) -> np.ndarray:
    """Fried parameter r0 along a given axis."""
    return _fried_r0(cn2, z, axis, _np_trapezoid)


def scint_index_np(cn2: np.ndarray, z: np.ndarray, axis: int, mode: TWaveMode) -> np.ndarray:
    """Scintillation index along a given axis."""
    return _scint_index(cn2=cn2, z=z, dim=axis, trapezoid_fn=_np_trapezoid, mode=mode)


def isoplanatic_angle_np(cn2: np.ndarray, z: np.ndarray, axis: int) -> np.ndarray:
    """Isoplanatic angle along a given axis."""
    return _isoplanatic_angle(cn2, z, axis, _np_trapezoid)


def wave_coherence_time_np(cn2: np.ndarray, m: np.ndarray, z: np.ndarray, axis: int) -> np.ndarray:
    """Wave coherence time along a given axis. m is the vertical wind profile."""
    return _wave_coherence_time(cn2, m, z, axis, _np_trapezoid)


# Xarray implementations
def _trapz_non_uniform(y: xr.DataArray, x: xr.DataArray, dim: str) -> xr.DataArray:
    """Trapezoidal integration for non-uniform x. NaN segments are skipped, which allows piecewise integration of
    masked profiles (see `extend_mask_right`)."""
    # Coordinates will result in automatic alignment, which we don't want here.
    # Drop coordinates from dimensions if needed.
    assert dim not in y.coords, "Dimension must not be a coordinate of y for alignment to work"
    assert dim not in x.coords, "Dimension must not be a coordinate of x for alignment to work"

    yi = y.isel({dim: slice(None, -1)})
    yj = y.isel({dim: slice(1, None)})
    xi = x.isel({dim: slice(None, -1)})
    xj = x.isel({dim: slice(1, None)})
    return (0.5 * (yi + yj) * (xj - xi)).sum(dim=dim)


def extend_mask_right(mask: xr.DataArray, dim: str) -> xr.DataArray:
    """Extend mask to the right by one element along given dimension.
    This is needed for piecewise integration to include the segment between two masks.
    """
    # Find indices where mask changes from True to False
    diff = mask.astype(int).diff(dim)
    idxs = np.where(diff == -1)

    # Increment index by one to include the next element along specified dimension
    i_dim = mask.get_axis_num(dim)
    idxs = tuple(idx + 1 if i == i_dim else idx for i, idx in enumerate(idxs))

    # Create extended mask
    mask_ext = mask.copy()
    mask_ext.values[idxs] = True  # Use numpy indexing because xarray wouldn't just update at coordinates
    return mask_ext


def fried_r0_xr(cn2: xr.DataArray, z: xr.DataArray, dim: str) -> xr.DataArray:
    """Fried parameter r0 along a given dimension."""
    return _fried_r0(cn2, z, dim, _trapz_non_uniform)


def scint_index_xr(cn2: xr.DataArray, z: xr.DataArray, dim: str, mode: TWaveMode) -> xr.DataArray:
    """Scintillation index along a given dimension."""
    return _scint_index(cn2=cn2, z=z, dim=dim, trapezoid_fn=_trapz_non_uniform, mode=mode)


def isoplanatic_angle_xr(cn2: xr.DataArray, z: xr.DataArray, dim: str) -> xr.DataArray:
    """Isoplanatic angle along a given dimension."""
    return _isoplanatic_angle(cn2, z, dim, _trapz_non_uniform)


def wave_coherence_time_xr(cn2: xr.DataArray, m: xr.DataArray, z: xr.DataArray, dim: str) -> xr.DataArray:
    """Wave coherence time along a given dimension. m is the vertical wind profile."""
    return _wave_coherence_time(cn2, m, z, dim, _trapz_non_uniform)


def seeing_xr(cn2: xr.DataArray, z: xr.DataArray, dim: str) -> xr.DataArray:
    """Seeing along a given dimension."""
    return _seeing(cn2, z, dim, _trapz_non_uniform)

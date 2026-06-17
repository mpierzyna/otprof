"""Optical turbulence calculations (torch and xarray versions)."""

from __future__ import annotations

from typing import Protocol, TypeVar, Literal
import numpy as np
import xarray as xr


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if hasattr(np, "trapezoid"):
    # numpy > 2.0
    NP_TRAPZ = np.trapezoid
else:
    # numpy <= 1.26
    NP_TRAPZ = np.trapz

# Define types
T = TypeVar("T")
TWaveMode = Literal["plane", "spherical"]


class TrapezoidFn(Protocol):
    def __call__(self, y: T, x: T, dim: int | str) -> T:
        """Trapezoid function protocol.
        Types are determined by implementation, but function must have y, x, and dim as arguments.
        """


# Base implementations should be agnostic of framework used
def _fried_r0(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Calculate Fried parameter r0 along a given dimension."""
    lam = 1550e-9  # nm
    k = (2 * np.pi) / lam  # wavenumber
    return (0.423 * k**2 * trapezoid_fn(y=cn2, x=z, dim=dim)) ** (-3 / 5)


def _rytov_var(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn, mode: TWaveMode) -> T:
    """Rytov variance along a path of length l. [1, p. 50]

    References
    ----------
    .. [1] Andrews, Larry C. Field Guide to Atmospheric Optics. Second Edition, vol. FG41, SPIE Press, 2014. SPIE Field Guides.

    """
    lam = 1550e-9  # nm
    k = (2 * np.pi) / lam  # wavenumber
    if mode == "plane":
        c = 1.23
    elif mode == "spherical":
        c = 0.5
    else:
        raise ValueError("mode must be 'plane' or 'spherical'")
    return c * k ** (7 / 6) * trapezoid_fn(y=cn2 * (z ** (5 / 6)), x=z, dim=dim)


def _scint_index(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn, mode: TWaveMode) -> T:
    """Scintillation index for plane wave in weak and strong turbulence regimes."""
    sr2 = _rytov_var(cn2=cn2, z=z, dim=dim, trapezoid_fn=trapezoid_fn, mode=mode)
    si2 = (0.49 * sr2 / (1 + 1.11 * sr2 ** (6 / 5)) ** (7 / 6)) + (0.51 * sr2 / (1 + 0.69 * sr2 ** (6 / 5)) ** (5 / 6))
    si2 = np.exp(si2) - 1  # todo: likely won't work for torch
    return si2


def _isoplanatic_angle(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Calculate isoplanatic angle. Result in radians."""
    lam = 1550e-9  # nm
    return 0.057 * lam ** (6 / 5) * (trapezoid_fn(y=cn2 * (z ** (5 / 3)), x=z, dim=dim) ** (-3 / 5))


def _wave_coherence_time(cn2: T, m: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Calculate wave coherence time. m is the vertical wind profile. Result in seconds."""
    lam = 1550e-9  # nm
    return 0.057 * lam ** (6 / 5) * (trapezoid_fn(y=cn2 * (m ** (5 / 3)), x=z, dim=dim) ** (-3 / 5))


def _seeing(cn2: T, z: T, dim, trapezoid_fn: TrapezoidFn) -> T:
    """Calculate seeing. Result in radians."""
    r0 = _fried_r0(cn2, z, dim, trapezoid_fn)
    lam = 1550e-9  # nm
    return 0.98 * lam / r0  #


# Numpy implementations
def _np_trapezoid(y: np.ndarray, x: np.ndarray, dim: int) -> np.ndarray:
    """Rename axis to dim because base implementation requires dim as kwarg."""
    return NP_TRAPZ(y, x, axis=dim)


def fried_r0_np(cn2: np.ndarray, z: np.ndarray, axis: int) -> np.ndarray:
    """Calculate Fried parameter r0 along a given dimension."""
    return _fried_r0(cn2, z, axis, _np_trapezoid)


def scint_index_np(cn2: np.ndarray, z: np.ndarray, axis: int, mode: TWaveMode) -> np.ndarray:
    """Calculate scintillation index along a given dimension."""
    return _scint_index(cn2=cn2, z=z, dim=axis, trapezoid_fn=_np_trapezoid, mode=mode)


def isoplanatic_angle_np(cn2: np.ndarray, z: np.ndarray, axis: int) -> np.ndarray:
    """Calculate isoplanatic angle."""
    return _isoplanatic_angle(cn2, z, axis, _np_trapezoid)


def wave_coherence_time_np(cn2: np.ndarray, m: np.ndarray, z: np.ndarray, axis: int) -> np.ndarray:
    """Calculate wave coherence time. m is the vertical wind profile."""
    return _wave_coherence_time(cn2, m, z, axis, _np_trapezoid)


# Torch implementations
if TORCH_AVAILABLE:

    @torch.compile
    def fried_r0_pt(cn2: torch.Tensor, z: torch.Tensor, dim: int) -> torch.Tensor:
        return _fried_r0(cn2, z, dim, torch.trapezoid)

    @torch.compile
    def scint_index_pt(cn2: torch.Tensor, z: torch.Tensor, dim: int, mode: TWaveMode) -> torch.Tensor:
        """Calculate scintillation index along a given dimension."""
        return _scint_index(cn2=cn2, z=z, dim=dim, trapezoid_fn=torch.trapezoid, mode=mode)

    @torch.compile
    def isoplanatic_angle_pt(cn2: torch.Tensor, z: torch.Tensor, dim: int) -> torch.Tensor:
        """Calculate isoplanatic angle."""
        return _isoplanatic_angle(cn2, z, dim, torch.trapezoid)

    @torch.compile
    def wave_coherence_time_pt(cn2: torch.Tensor, m: torch.Tensor, z: torch.Tensor, dim: int) -> torch.Tensor:
        """Calculate wave coherence time. m is the vertical wind profile."""
        return _wave_coherence_time(cn2, m, z, dim, torch.trapezoid)


# Xarray implementations
def _trapz_non_uniform(y: xr.DataArray, x: xr.DataArray, dim: str) -> xr.DataArray:
    """Trapezoidal integration for non-uniform x."""
    # Coordinates will result in automatic alignment, which we don't want here.
    # Drop coordinates from dimensions if needed.
    assert dim not in y.coords, "Dimension must not be a coordinate of y for alignment to work"
    assert dim not in x.coords, "Dimension must not be a coordinate of x for alignment to work"

    yi = y.isel({dim: slice(None, -1)})
    yj = y.isel({dim: slice(1, None)})
    xi = x.isel({dim: slice(None, -1)})
    xj = x.isel({dim: slice(1, None)})
    return (0.5 * (yi + yj) * (xj - xi)).sum(dim=dim)


def _extend_mask_right(mask: xr.DataArray, dim: str):
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
    """Calculate Fried parameter r0 along a given dimension."""
    return _fried_r0(cn2, z, dim, _trapz_non_uniform)


def scint_index_xr(cn2: xr.DataArray, z: xr.DataArray, dim: str, mode: TWaveMode) -> xr.DataArray:
    """Calculate scintillation index along a given dimension."""
    return _scint_index(cn2=cn2, z=z, dim=dim, trapezoid_fn=_trapz_non_uniform, mode=mode)


def isoplanatic_angle_xr(cn2: xr.DataArray, z: xr.DataArray, dim: str) -> xr.DataArray:
    """Calculate isoplanatic angle."""
    return _isoplanatic_angle(cn2, z, dim, _trapz_non_uniform)


def wave_coherence_time_xr(cn2: xr.DataArray, m: xr.DataArray, z: xr.DataArray, dim: str) -> xr.DataArray:
    """Calculate wave coherence time. m is the vertical wind profile."""
    return _wave_coherence_time(cn2, m, z, dim, _trapz_non_uniform)


def seeing_xr(cn2: xr.DataArray, z: xr.DataArray, dim: str) -> xr.DataArray:
    """Calculate seeing."""
    return _seeing(cn2, z, dim, _trapz_non_uniform)


if __name__ == "__main__":
    ## Validate that xarray implementation and numpy implementation give same results
    def exact(a, b):
        def _fn_int(x):
            return 1 / 3 * x**3 + 3 / 2 * x**2 + 2 * x

        return _fn_int(b) - _fn_int(a)

    x = np.linspace(0, 5, 100) ** 2  # non-uniform spacing
    y = x**2 + 3 * x + 2

    xa = xr.DataArray(x, dims="z")
    ya = xr.DataArray(y, dims="z")

    np_result = np.trapezoid(y, x, axis=0)
    xr_result = _trapz_non_uniform(ya, xa, dim="z").item()
    exact_result = exact(x[0], x[-1])
    print("Numpy trapz:", np_result)
    print("Xarray trapz:", xr_result)
    print("Exact integral:", exact_result)
    print("-----")

    ## Demonstrate piecewise integration
    # Standard
    mask_1 = xa < 10
    mask_2 = xa >= 10
    seg_1 = _trapz_non_uniform(ya.where(mask_1), xa.where(mask_1), dim="z").item()
    seg_2 = _trapz_non_uniform(ya.where(mask_2), xa.where(mask_2), dim="z").item()
    full = _trapz_non_uniform(ya, xa, dim="z").item()
    print(f"Piecewise trapz:\t{seg_1 + seg_2}")
    print(f"Full trapz:\t\t{full}")
    print("NOT THE SAME")
    print("-----")

    # Now extend first mask to the right
    mask_1_ext = _extend_mask_right(mask_1, dim="z")
    seg_1_ext = _trapz_non_uniform(ya.where(mask_1_ext), xa.where(mask_1_ext), dim="z").item()
    seg_2_ext = _trapz_non_uniform(ya.where(mask_2), xa.where(mask_2), dim="z").item()
    print(f"Piecewise extended trapz:\t{seg_1_ext + seg_2_ext}")
    print(f"Full trapz:\t\t\t{full}")
    print("SAME NOW")

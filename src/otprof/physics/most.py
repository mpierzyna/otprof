"""Surface-layer CT2 from Monin-Obukhov similarity theory."""

from __future__ import annotations

import numpy as np
import xarray as xr


def w71_flux_ct2(*, T_st, zeta, z) -> np.ndarray:
    """Flux-based Wyngaard et al. (1971) CT2 with updated constants by Andreas (1989).
    `zeta` is the dimensionless height (z / L), where L is the Obukhov length.
    """
    zeta_neg = np.where(zeta < 0, zeta, np.nan)  # unstable
    zeta_pos = np.where(zeta >= 0, zeta, np.nan)  # stable
    # from numpy docs: Where True, yield x, otherwise yield y.
    g3 = np.where(
        zeta < 0,  # unstable
        4.9 * np.power(1 - 6.1 * zeta_neg, -2 / 3),  # unstable
        4.9 * (1 + 2.2 * np.power(zeta_pos, 2 / 3)),  # stable
    )

    # If zeta is an xarray DataArray, convert g to xarray DataArray for proper broadcasting
    if isinstance(zeta, xr.DataArray):
        g3 = xr.DataArray(data=g3, dims=zeta.dims, coords=zeta.coords)

    return T_st**2 * z ** (-2 / 3) * g3

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


def osborn_sarazin(th, Gamma, S, p_hPa, k=6.0):
    """Cn2 profile following Osborn and Sarazin (2019).
    Only stable conditions as dth_dz <= 0  leads to invalid/infinite length scale.

    Parameters
    ----------
    th : np.ndarray
        Potential temperature profile (K).
    Gamma : np.ndarray
        Potential temperature gradient profile (K/m).
    S : np.ndarray
        Wind shear profile (1/s).
    p_hPa : np.ndarray
        Pressure profile (hPa).
    k : float, optional
        OS19 set this constant to 6.0, by default 6.0.

    Returns
    -------
    np.ndarray
        Cn2 profile (m^(-2/3)).
    """
    g = 9.81  # m/s^2

    # Compute temperature from potential temperature
    p0 = 1000  # hPa
    tk = th / ((p0 / p_hPa) ** 0.286)  # eq 3

    # Compute length scale and tke
    E = S**2  # THIS IS WRONG! E needs m^2/s^2 but S^2 only yields 1/s^2!
    L = np.sqrt(2 * E / ((g / th) * Gamma))  # eq 5

    ct2 = k * L ** (4 / 3) * Gamma**2  # eq 4
    cn2 = (80e-6 * p_hPa / (th * tk)) ** 2 * ct2  # eq 2

    return cn2


def osborn_sarazin_xr(
    *,
    th: xr.DataArray,
    Gamma: xr.DataArray,
    S: xr.DataArray,
    p_hPa: xr.DataArray,
    dim: str,
    k: float = 6.0,
) -> xr.DataArray:
    """Cn2 profile following Osborn and Sarazin (2019) using xarray"""
    assert dim in th.dims, f"Dimension {dim} not found in th"
    assert dim in Gamma.dims, f"Dimension {dim} not found in Gamma"
    assert dim in S.dims, f"Dimension {dim} not found in S"
    assert dim in p_hPa.dims, f"Dimension {dim} not found in p_hPa"

    return xr.apply_ufunc(
        osborn_sarazin,
        th,
        Gamma,
        S,
        p_hPa,
        input_core_dims=[[dim], [dim], [dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        kwargs={"k": k},
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test HV xarray function
    n_samples, n_levels = 10, 20
    cn2_0 = xr.DataArray(np.linspace(1e-16, 1e-14, n_samples), dims=["sample"])
    m = xr.DataArray(np.random.rand(n_samples, n_levels), dims=["sample", "level"])
    z = xr.DataArray(np.linspace(0, 20000, n_levels), dims=["level"])
    z_eval = xr.DataArray(np.linspace(0, 20000, n_levels * 2), dims=["level"])

    cn2 = hufnagel_valley_xr(cn2_0=cn2_0, m=m, z=z, dim="level")
    np.log10(cn2).plot()
    plt.show()

    cn2 = hufnagel_valley_xr(cn2_0=cn2_0, m=m, z=z, z_eval=z_eval, dim="level")
    np.log10(cn2).plot()
    plt.show()

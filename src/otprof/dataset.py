"""Build a training dataset for a domain from WRF Cn2 output (otprof-wrf) and ERA5. Needs `otprof[data]`.

Steps, as run by `examples/new_domain/` (each writes NetCDF that can be inspected):
1. `get_wrf_times` → `otprof.sources.era5.fetch`: download ERA5 for the WRF time steps
2. `align`: pair each ERA5 grid point with the nearest WRF column and keep the common time steps. Pure selection:
   variables, units and levels stay as delivered.
3. `make_wrf_targets`: WRF Cn2 profiles and astroclimate parameters (targets); `make_inputs` for the inputs
4. `filter_cn2`: mask implausible Cn2 profiles and astroclimate values
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import pathlib
import warnings
from typing import Sequence

import numpy as np
import xarray as xr

from otprof import features
from otprof.physics import ot
from otprof.sources import wrf


def get_wrf_times(wrf_files: Sequence[pathlib.Path]) -> xr.DataArray:
    """All valid time steps in the WRF files (for the ERA5 download)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "variable 'p' has multiple fill values"
        times = xr.concat([xr.open_dataset(f)["time"] for f in wrf_files], dim="time")
    return times[~times.isnull()]


def align(
    era5: xr.Dataset,
    wrf_files: Sequence[pathlib.Path],
    *,
    crop: int,
    dist_th: float,
    n_workers: int = 1,
) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Pair ERA5 grid points with the nearest WRF column and keep the time steps both have.

    - `era5`: ERA5 as fetched (`latitude`, `longitude` dims, ARCO names and units).
    - `wrf_files`: WRF Cn2 output files, sorted by time; the WRF grid must be the same in all of them.
    - `crop`: outer WRF cells dropped on each side (edge effects of the WRF boundary).
    - `dist_th`: an ERA5 point is kept if the nearest WRF column is closer than this [deg].
    - `n_workers`: WRF files read in parallel (reading is limited by decompression, one core per file).

    Returns ERA5 and WRF on the common dims `(time, [level | bottom_top], loc)` and diagnostics for a report. Nothing
    is converted: names, units, signs, levels and ERA5's 0..360 longitudes stay as delivered.
    """
    era5 = era5.stack(loc=("latitude", "longitude"))
    # ERA5 (ARCO) longitudes run 0..360, WRF's -180..180: match on wrapped longitudes, keep the original ones
    era5_lon = ((era5["longitude"] + 180) % 360) - 180

    # The WRF grid does not move in time: match on the first file and reuse it for all
    wrf_first = _crop_stack(_load(wrf_files[0]), crop)
    dist, idx = _nearest(
        wrf_lat=wrf_first["XLAT"].values,
        wrf_lon=wrf_first["XLONG"].values,
        lat=era5["latitude"].values,
        lon=era5_lon.values,
    )
    keep = dist < dist_th

    # Read the WRF columns of the kept ERA5 points, file by file (in parallel if n_workers > 1)
    args = [(f, crop, idx[keep]) for f in wrf_files]
    if n_workers > 1:
        ctx = multiprocessing.get_context("spawn")  # no fork after netCDF/HDF5 use in the parent
        with concurrent.futures.ProcessPoolExecutor(n_workers, mp_context=ctx) as ex:
            parts = list(ex.map(_load_columns, *zip(*args)))  # keeps the file order
    else:
        parts = [_load_columns(*a) for a in args]
    ds_wrf = xr.concat(parts, dim="time")
    era5 = era5.isel(loc=keep)

    # Common time steps
    era5 = era5.sel(time=era5.time.isin(ds_wrf.time))
    ds_wrf = ds_wrf.sel(time=~ds_wrf.time.isnull())  # some WRF files have NaT time steps
    ds_wrf = ds_wrf.sel(time=era5.time)

    diagnostics = {
        "wrf_files": [str(f) for f in wrf_files],
        "dist": dist,
        "dist_th": dist_th,
        "era5_grid": (era5["latitude"].values, era5_lon.values[keep]),
        "era5_grid_all": (era5_lon.coords["latitude"].values, era5_lon.values),
        "wrf_grid": (wrf_first["XLAT"].values, wrf_first["XLONG"].values),
        "wrf_shape": dict(wrf_first.sizes),
        "n_points": int(keep.sum()),
        "n_times": int(era5.sizes["time"]),
    }
    # Stacked indices cannot be written to NetCDF
    return era5.reset_index("loc"), ds_wrf.reset_index("loc"), diagnostics


def _load(f) -> xr.Dataset:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=xr.SerializationWarning)  # "p" has two fill values
        return xr.load_dataset(f)


def _crop_stack(ds: xr.Dataset, crop: int) -> xr.Dataset:
    ds = ds.isel(south_north=slice(crop, -crop), west_east=slice(crop, -crop))
    return ds.stack(loc=("south_north", "west_east"))


def _load_columns(f, crop: int, idx: np.ndarray) -> xr.Dataset:
    return _crop_stack(_load(f), crop).isel(loc=idx)


def _nearest(wrf_lat, wrf_lon, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Distance [deg] and index of the nearest WRF column for each point."""
    from scipy.spatial import KDTree

    tree = KDTree(np.column_stack((wrf_lat, wrf_lon)))
    return tree.query(np.column_stack((lat, lon)), k=1)


def make_wrf_targets(ds_wrf: xr.Dataset) -> xr.Dataset:
    """WRF on its model levels with canonical names, plus height above ground, wind and the astroclimate targets.

    `ds_wrf`: aligned WRF output (native names). Profile targets are `cn2`, `QKE`, `TSQ`, `EL_PBL` on
    `bottom_top`.
    """
    ds = wrf.standardize(ds_wrf, grid="model")
    ds["z_agl"] = (ds["z"] - ds["z_sfc_src"]).assign_attrs(long_name="Height above ground", units="m")
    ds = ds.assign(features.get_wind_mag_dir(ds["u"], ds["v"]))
    ds = ds.assign(compute_astroclim_targets(cn2=ds["cn2"], m=ds["m"], z_agl=ds["z_agl"], blh=ds["blh"]))
    return ds


def compute_astroclim_targets(cn2: xr.DataArray, m: xr.DataArray, z_agl: xr.DataArray, blh: xr.DataArray) -> dict:
    """r0, tau0, th0 and log10 scintillation index (plane wave) per region: surface layer (`sl`, below 0.1 BLH),
    boundary layer (`bl`), free atmosphere (`fa`) and the full column. Integrals over `bottom_top`."""
    dim = "bottom_top"
    # SL and BL masks are extended one level up so the piecewise integrals add up to the full column
    region_masks = {
        "sl": ot.extend_mask_right(z_agl <= (blh * 0.1), dim=dim),
        "bl": ot.extend_mask_right(z_agl <= blh, dim=dim),
        "fa": z_agl > blh,
        "full": np.ones_like(z_agl, dtype=bool),
    }
    regions_long = {"sl": "surface layer", "bl": "boundary layer", "fa": "free atmosphere", "full": "full column"}

    out = {}
    for r, mask in region_masks.items():
        cn2_r, z_r = cn2.where(mask), z_agl.where(mask)
        r0 = ot.fried_r0_xr(cn2=cn2_r, z=z_r, dim=dim)
        tau0 = ot.wave_coherence_time_xr(cn2=cn2_r, m=m.where(mask), z=z_r, dim=dim)
        th0 = ot.isoplanatic_angle_xr(cn2=cn2_r, z=z_r, dim=dim)
        lsi2 = np.log10(ot.scint_index_xr(cn2=cn2_r, z=z_r, dim=dim, mode="plane"))
        # Integration over too few levels can give inf
        for name, da, long_name, units in [
            ("r0", r0, "Fried parameter", "m"),
            ("tau0", tau0, "Wavefront coherence time", "s"),
            ("th0", th0, "Isoplanatic angle", "rad"),
            ("lsi2", lsi2, "Log10 scintillation index", "1"),
        ]:
            da = da.where(~np.isinf(da)).astype(np.float32)
            out[f"{name}_{r}"] = da.assign_attrs(long_name=f"{long_name} ({regions_long[r]})", units=units)
    return out


def filter_cn2(ds: xr.Dataset, quantile: float = 0.999) -> tuple[xr.Dataset, dict]:
    """Mask (NaN) Cn2 profiles that are zero at any level or whose full-column r0 or scintillation index exceeds its
    `quantile`; mask the astroclimate targets where r0 or the scintillation index exceed it. Returns the masked
    dataset and diagnostics (thresholds, counts) for a report."""
    r0, lsi2, cn2 = ds["r0_full"], ds["lsi2_full"], ds["cn2"]
    r0_th = r0.quantile(quantile).drop_vars("quantile")
    lsi2_th = lsi2.quantile(quantile).drop_vars("quantile")
    r0_valid = r0 < r0_th
    lsi2_valid = lsi2 < lsi2_th
    cn2_valid = ~(cn2 == 0).any("bottom_top")

    ds = ds.copy()
    valid_ac = r0_valid & lsi2_valid
    for v in ds.data_vars:
        if v.startswith(("r0_", "tau0_", "th0_", "lsi2_")):
            ds[v] = ds[v].where(valid_ac)
    valid_profiles = valid_ac & cn2_valid
    ds["cn2"] = ds["cn2"].where(valid_profiles)

    diagnostics = {
        "quantile": quantile,
        "r0": r0,
        "r0_th": float(r0_th),
        "lsi2": lsi2,
        "lsi2_th": float(lsi2_th),
        "n_profiles": int(valid_profiles.size),
        "n_r0_invalid": int((~r0_valid).sum()),
        "n_lsi2_invalid": int((~lsi2_valid).sum()),
        "n_cn2_zero": int((~cn2_valid).sum()),
        "n_valid_profiles": int(valid_profiles.sum()),
        "n_valid_astroclim": int(valid_ac.sum()),
    }
    return ds, diagnostics

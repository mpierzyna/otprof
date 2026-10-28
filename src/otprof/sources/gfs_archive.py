"""GFS forecasts of earlier years (before the AWS archive of `otprof.sources.gfs`, which starts in 2021) from the NCAR
GDEX archive ds084.1 (NCEP GFS 0.25 deg, from 2015) as canonical inputs (`otprof.schema`). Needs `otprof[gfs]`.

    ds = gfs_archive.fetch("2017-06-01 03:00", lat=[52.1, 51.9], lon=[4.4, 5.2])   # by valid time
    X = predictor.make_inputs(gfs_archive.standardize(ds))

The archive files have no `.idx` inventory: each file (about 230 MB in 2017, 550 MB in 2023) is downloaded whole, the
fields at the locations are extracted and the file is deleted. The GFS of 2017 (spectral v13, v14 from 2017-07-19
12 UTC) lacks fields of the current GFS: 6 of the 31 pressure levels (875, 825, 775, 225, 175, 125 hPa), the friction
velocity and instantaneous surface fluxes (only means since the cycle). The "archive recipe" of the research GFS
study makes the canonical inputs from what it has:
- a valid time (3-hourly) comes from the shortest lead time with fluxes: +3 h of the cycle 3 h earlier (03, 09, 15,
  21 UTC) or +6 h of the cycle 6 h earlier (00, 06, 12, 18 UTC). The fluxes are the means over the 3 h before the
  valid time: at +6 h de-averaged from the 0-6 h and 0-3 h means, so the +3 h file is read too
- friction velocity from the momentum flux, sqrt(|tau| / rho) with rho = p_sfc / (R_d T_2m)
- the six missing levels linear in ln p
- PBL height: HPBL of the archive file
"""

from __future__ import annotations

import pathlib
import tempfile
import time
import urllib.request
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from otprof import schema
from otprof.sources.gfs import UST_MIN, to_pressure_grid

GDEX = "https://osdf-director.osg-htc.org/ncar/gdex/d084001"

# The pressure levels [hPa] read from the archive files: the 25 the GFS of 2017 has
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100,
          70, 50, 30, 20)  # fmt: skip
# Fields by GRIB2 (discipline, category, number, typeOfLevel, level; None for the pressure levels) -> name.
# Geopotential height [gpm], fluxes upward positive and averaged since the cycle, `prmsl` in Pa, `lsm` 0 / 1.
FIELDS = {
    (0, 3, 5, "isobaricInhPa", None): "z",
    (0, 0, 0, "isobaricInhPa", None): "tk",
    (0, 2, 2, "isobaricInhPa", None): "u",
    (0, 2, 3, "isobaricInhPa", None): "v",
    (0, 3, 0, "surface", 0): "p_sfc",
    (0, 3, 5, "surface", 0): "z_sfc",  # orography
    (2, 0, 0, "surface", 0): "lsm",
    (0, 0, 0, "heightAboveGround", 2): "tk2",
    (0, 2, 2, "heightAboveGround", 10): "u10",
    (0, 2, 3, "heightAboveGround", 10): "v10",
    (0, 3, 1, "meanSea", 0): "prmsl",
    (0, 3, 196, "surface", 0): "blh",  # HPBL (NCEP local)
    (0, 0, 11, "surface", 0): "hfx",  # sensible heat flux, mean since the cycle
    (0, 0, 10, "surface", 0): "lh",  # latent heat flux, mean since the cycle
    (0, 2, 17, "surface", 0): "taux",  # momentum flux [N m-2], mean since the cycle
    (0, 2, 18, "surface", 0): "tauy",
}
MEANS = ["hfx", "lh", "taux", "tauy"]  # fields averaged since the cycle
R_D = 287.05  # J kg-1 K-1, gas constant of dry air


def get_url(cycle: pd.Timestamp, lead: int) -> str:
    """URL of the archive file of `cycle` at lead time `lead` [h]."""
    return f"{GDEX}/{cycle:%Y}/{cycle:%Y%m%d}/gfs.0p25.{cycle:%Y%m%d%H}.f{lead:03d}.grib2"


def get_cycle_lead(valid_time: pd.Timestamp) -> tuple[pd.Timestamp, int]:
    """Cycle and lead time [h] of a 3-hourly valid time: the shortest lead with fluxes (+3 h or +6 h)."""
    if valid_time.hour % 3 or valid_time.minute or valid_time.second:
        raise ValueError(f"{valid_time}: the archive recipe needs 3-hourly valid times")
    lead = 3 if valid_time.hour % 6 == 3 else 6
    return valid_time - pd.Timedelta(hours=lead), lead


def read_grib(path: str | pathlib.Path, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """Values of `FIELDS` at the grid points of the regular 0.25 deg grid (north to south) nearest to (lat, lon);
    profiles as (level, loc) on `LEVELS`."""
    import eccodes  # otprof[gfs]

    out, profiles, index = {}, {}, None
    with open(path, "rb") as f:
        while (h := eccodes.codes_grib_new_from_file(f)) is not None:
            key = tuple(eccodes.codes_get_long(h, n) for n in ["discipline", "parameterCategory", "parameterNumber"])
            type_of_level, level = eccodes.codes_get_string(h, "typeOfLevel"), eccodes.codes_get_long(h, "level")
            if type_of_level == "isobaricInhPa":
                name = FIELDS.get((*key, type_of_level, None)) if level in LEVELS else None
            else:
                name = FIELDS.get((*key, type_of_level, level))
            if name is not None:
                if index is None:  # grid indices of the locations, from the first message used
                    index = _get_index(h, lat, lon)
                values = eccodes.codes_get_values(h).reshape(eccodes.codes_get_long(h, "Nj"), -1)[index]
                if type_of_level == "isobaricInhPa":
                    profiles.setdefault(name, {})[level] = values
                else:
                    out[name] = values
            eccodes.codes_release(h)
    for name, by_level in profiles.items():
        out[name] = np.stack([by_level[p] for p in LEVELS])
    if missing := set(FIELDS.values()) - set(out):
        raise ValueError(f"{path}: missing fields {sorted(missing)}")
    return out


def _get_index(h: int, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, column) of the grid points nearest to (lat, lon) on the regular lat-lon grid of message `h`."""
    import eccodes  # otprof[gfs]

    if eccodes.codes_get_string(h, "gridType") != "regular_ll" or eccodes.codes_get_long(h, "jScansPositively"):
        raise ValueError("expected a regular lat-lon grid from north to south")
    lat0 = eccodes.codes_get_double(h, "latitudeOfFirstGridPointInDegrees")
    lon0 = eccodes.codes_get_double(h, "longitudeOfFirstGridPointInDegrees")
    dlat = eccodes.codes_get_double(h, "jDirectionIncrementInDegrees")
    dlon = eccodes.codes_get_double(h, "iDirectionIncrementInDegrees")
    return np.round((lat0 - lat) / dlat).astype(int), np.round(((lon - lon0) % 360) / dlon).astype(int)


def fetch_file(
    cycle: pd.Timestamp, lead: int, lat: np.ndarray, lon: np.ndarray, tmp_dir: str | pathlib.Path | None = None
) -> xr.Dataset:
    """The fields of one archive file at the locations (dims loc, level), downloaded to `tmp_dir` (default: the
    system's temporary directory) and deleted after reading; retried on network errors."""
    url = get_url(cycle, lead)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        path = pathlib.Path(tmp) / url.rsplit("/", 1)[-1]
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(url, path)
                values = read_grib(path, lat, lon)
                break
            except Exception:  # network or truncated file
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
    return xr.Dataset(
        {n: (("level", "loc") if v.ndim == 2 else ("loc",), v.astype(np.float32)) for n, v in values.items()},
        coords={"level": list(LEVELS), "loc": np.arange(len(lat)), "lat": ("loc", lat), "lon": ("loc", lon)},
    )


def fetch(
    valid_time: str | pd.Timestamp,
    lat: Sequence[float],
    lon: Sequence[float],
    tmp_dir: str | pathlib.Path | None = None,
) -> xr.Dataset:
    """GFS at the 3-hourly `valid_time` at the locations (lat, lon) [degrees north / east, dim `loc`], from the
    shortest lead time with fluxes, the fluxes as means over the 3 h before `valid_time` (see the module docstring).
    Returns the fields on dims (time, loc) and (time, level, loc), `level` the 25 levels of `LEVELS` [hPa]; coordinates
    `cycle` and `lead` along `time`, `lat` and `lon` of the locations. Several times: concatenate along `time`."""
    valid_time = pd.Timestamp(valid_time)
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    cycle, lead = get_cycle_lead(valid_time)
    ds = fetch_file(cycle, lead, lat, lon, tmp_dir)
    if lead == 6:  # mean over 3-6 h from the means over 0-6 h and 0-3 h
        ds[MEANS] = 2 * ds[MEANS] - fetch_file(cycle, 3, lat, lon, tmp_dir)[MEANS]
    ds = ds.expand_dims(time=[valid_time])
    return ds.assign_coords(cycle=("time", [cycle]), lead=("time", [lead]))


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Convert the fields of `fetch` (one or more times along `time`) to the canonical inputs on the pressure grid:
    friction velocity from the momentum flux (floored at `UST_MIN`), the six missing levels linear in ln p, orography
    -> `z_sfc_src` (averaged over the times), sea level pressure Pa -> hPa, longitude -> -180..180, dim `bottom_top`
    with the nominal pressure as coordinate `p_level` [hPa] and as `p` [Pa]."""
    rho = ds["p_sfc"] / (R_D * ds["tk2"])
    # GFS packs the momentum flux coarsely: 0 means ust below about 0.02 m/s
    ds["ust"] = np.sqrt(np.hypot(ds["taux"], ds["tauy"]) / rho).clip(min=UST_MIN)
    ds["slp"] = ds["prmsl"] / 100
    ds = ds.rename(z_sfc="z_sfc_src")
    ds["z_sfc_src"] = ds["z_sfc_src"].mean("time")
    profiles = interp_levels(ds[["z", "tk", "u", "v"]])
    ds = ds.drop_vars(["z", "tk", "u", "v", "level"]).assign(profiles.data_vars)
    return to_pressure_grid(ds)


def interp_levels(ds: xr.Dataset) -> xr.Dataset:
    """Profiles on `LEVELS` -> the 31 levels of the pressure grid, linear in ln p."""
    ln_p = np.log(np.array(LEVELS, dtype=float))
    out = ds.drop_vars("level").assign_coords(ln_p=("level", ln_p)).swap_dims(level="ln_p")
    out = out.interp(ln_p=np.log(np.array(schema.P_LEVELS_HPA, dtype=float)))
    return out.swap_dims(ln_p="level").drop_vars("ln_p").assign_coords(level=list(schema.P_LEVELS_HPA))

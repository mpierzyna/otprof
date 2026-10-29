"""GFS forecasts of earlier years (before the AWS archive of `otprof.sources.gfs`, which starts in 2021) from the NCAR
GDEX archive ds084.1 (NCEP GFS 0.25 deg, from 2015) as canonical inputs (`otprof.schema`), for hindcasts. Needs
`otprof[gfs]` (cfgrib, eccodes).

    X = gfs_archive.load(times, Area(lon_min=2, lon_max=9, lat_min=49, lat_max=55))   # 3-hourly valid times
    X = gfs_archive.load(times, Points.from_dict({"cabauw": (51.97, 4.93)}))

The archive files have no `.idx` inventory: each file (about 230 MB in 2017, 550 MB in 2023) is downloaded whole, the
fields in the area or at the points are read (cfgrib) and the file is deleted. The GFS of 2017 (spectral v13, v14 from 2017-07-19
12 UTC) lacks fields of the current GFS: 6 of the 31 pressure levels (875, 825, 775, 225, 175, 125 hPa), the friction
velocity and instantaneous surface fluxes (only means since the cycle). The canonical inputs are made from what it has:
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

import numpy as np
import pandas as pd
import xarray as xr

from otprof import schema
from otprof.sources import Where, select
from otprof.sources.gfs import UST_MIN, read_grib, to_pressure_grid

GDEX = "https://osdf-director.osg-htc.org/ncar/gdex/d084001"

# The pressure levels [hPa] read from the archive files: the 25 the GFS of 2017 has
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100,
          70, 50, 30, 20)  # fmt: skip
# The fields (`otprof.sources.gfs.FIELDS`) read; fluxes averaged since the cycle, momentum flux instead of ust
NAMES = ["z", "tk", "u", "v", "p_sfc", "z_sfc", "lsm", "tk2", "u10", "v10", "prmsl", "blh", "hfx", "lh", "taux", "tauy"]
MEANS = ("hfx", "lh", "taux", "tauy")  # fields averaged since the cycle
R_D = 287.05  # J kg-1 K-1, gas constant of dry air


def load(times, where: Where, tmp_dir: str | pathlib.Path | None = None) -> xr.Dataset:
    """GFS at the 3-hourly valid `times` in an `Area` or at `Points` as canonical inputs: `standardize(fetch(...))`."""
    return standardize(fetch(times, where, tmp_dir=tmp_dir))


def fetch(times, where: Where, tmp_dir: str | pathlib.Path | None = None) -> xr.Dataset:
    """GFS at the 3-hourly valid `times` in an `Area` or at `Points` (`otprof.sources`), each from the shortest lead
    time with fluxes, the fluxes as means over the 3 h before the time (see the module docstring). Returns the fields
    (`NAMES`) on dims (time, lat, lon) or (time, loc), profiles with `level` (the 25 `LEVELS` [hPa], bottom to top);
    coordinates `cycle` and `lead` [h] along `time`. Files go to `tmp_dir` (default: the system's) and are deleted."""
    parts = []
    for valid_time in pd.DatetimeIndex(times):
        cycle, lead = get_cycle_lead(valid_time)
        ds = fetch_file(cycle, lead, where, tmp_dir)
        if lead == 6:  # mean over 3-6 h from the means over 0-6 h and 0-3 h
            ds = ds.assign({v: 2 * ds[v] - fetch_file(cycle, 3, where, tmp_dir)[v] for v in MEANS})
        ds = ds.expand_dims(time=[valid_time])
        parts.append(ds.assign_coords(cycle=("time", [cycle]), lead=("time", [lead])))
    return xr.concat(parts, "time") if len(parts) > 1 else parts[0]


def fetch_file(cycle: pd.Timestamp, lead: int, where: Where, tmp_dir: str | pathlib.Path | None = None) -> xr.Dataset:
    """The fields of one archive file in an `Area` or at `Points`; retried on network errors."""
    url = get_url(cycle, lead)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp:
        path = pathlib.Path(tmp) / url.rsplit("/", 1)[-1]
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(url, path)
                ds = read_grib(path, means=MEANS)
                break
            except OSError:  # network or truncated file
                if attempt == 4:
                    raise
                time.sleep(10 * (attempt + 1))
    if missing := set(NAMES) - set(ds):
        raise ValueError(f"{url}: missing fields {sorted(missing)}")
    return select(ds[NAMES].sel(level=list(LEVELS)), where)


def get_url(cycle: pd.Timestamp, lead: int) -> str:
    """URL of the archive file of `cycle` at lead time `lead` [h]."""
    return f"{GDEX}/{cycle:%Y}/{cycle:%Y%m%d}/gfs.0p25.{cycle:%Y%m%d%H}.f{lead:03d}.grib2"


def get_cycle_lead(valid_time: pd.Timestamp) -> tuple[pd.Timestamp, int]:
    """Cycle and lead time [h] of a 3-hourly valid time: the shortest lead with fluxes (+3 h or +6 h)."""
    if valid_time.hour % 3 or valid_time.minute or valid_time.second:
        raise ValueError(f"{valid_time}: the archive needs 3-hourly valid times")
    lead = 3 if valid_time.hour % 6 == 3 else 6
    return valid_time - pd.Timedelta(hours=lead), lead


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Convert the fields of `fetch` to the canonical inputs on the pressure grid: friction velocity from the momentum
    flux (floored at `UST_MIN`), the six missing levels linear in ln p, orography -> `z_sfc_src` (averaged over the
    times), sea level pressure Pa -> hPa, dim `bottom_top` with the nominal pressure as coordinate `p_level` [hPa] and
    as `p` [Pa]."""
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

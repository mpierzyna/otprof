"""GFS forecasts (NCEP Global Forecast System, v16 and later, from 2021) from the NOAA archive on AWS
(`noaa-gfs-bdp-pds`) as canonical inputs (`otprof.schema`). Needs `otprof[gfs]` (cfgrib, eccodes).

    X = gfs.load("2025-06-01 00:00", times, Area(lon_min=2, lon_max=9, lat_min=49, lat_max=55))
    X = gfs.load("2025-06-01 00:00", times, Points.from_dict({"cabauw": (51.97, 4.93)}))

All times come from one forecast run, the `cycle` (00, 06, 12 or 18 UTC): the lead time of each is `time - cycle`
(0-120 h hourly, then 3-hourly up to 384 h). For a forecast, take the newest run that is complete on AWS (about 4-5 h
after the cycle time) and the times ahead; to compare runs, call `load` once per run. Hindcasts of 2015-2020 with the
shortest lead per time: `otprof.sources.gfs_archive`.

Per lead time, only the needed GRIB messages are downloaded (byte ranges from the `.idx` inventories) of three files:
- `pgrb2` (0.25 deg): geopotential height, temperature, wind on 25 of the 31 pressure levels; surface pressure,
  orography, land mask, 2 m temperature, 10 m wind, sea level pressure
- `pgrb2b` (0.25 deg): the six levels 875, 825, 775, 225, 175, 125 hPa that `pgrb2` lacks
- `sflux` (Gaussian grid, about 13 km): instantaneous sensible and latent heat flux, friction velocity, PBL height
Every message is a global field, so a lead time downloads about 120 MB, whatever `where`. An `Area` is the 0.25 deg
grid points inside it (the `sflux` fields from the nearest Gaussian grid point); `Points` take the nearest grid point
of each grid.
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import tempfile
import time
import urllib.error
import urllib.request
import warnings

import pandas as pd
import xarray as xr

from otprof import schema
from otprof.sources import Where, select, select_like, to_180

AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Pressure levels [hPa] by file: pgrb2 has 25 of the 31 levels of the pressure grid, pgrb2b the others
LEVELS_PGRB2B = (875, 825, 775, 225, 175, 125)
LEVELS_PGRB2 = tuple(p for p in schema.P_LEVELS_HPA if p not in LEVELS_PGRB2B)

# Fields by GRIB2 (discipline, category, number, typeOfLevel, height above ground or None) -> name, for both GFS
# sources. Geopotential height [gpm] for `z` and the orography `z_sfc`; fluxes upward positive; `prmsl` in Pa;
# `lsm` 0 / 1. Numbers 196 / 197 are NCEP's own (PBL height, friction velocity).
FIELDS = {
    (0, 3, 5, "isobaricInhPa", None): "z",
    (0, 0, 0, "isobaricInhPa", None): "tk",
    (0, 2, 2, "isobaricInhPa", None): "u",
    (0, 2, 3, "isobaricInhPa", None): "v",
    (0, 3, 0, "surface", None): "p_sfc",
    (0, 3, 5, "surface", None): "z_sfc",
    (2, 0, 0, "surface", None): "lsm",
    (0, 0, 0, "heightAboveGround", 2): "tk2",
    (0, 2, 2, "heightAboveGround", 10): "u10",
    (0, 2, 3, "heightAboveGround", 10): "v10",
    (0, 3, 1, "meanSea", None): "prmsl",
    (0, 3, 196, "surface", None): "blh",
    (0, 0, 11, "surface", None): "hfx",
    (0, 0, 10, "surface", None): "lh",
    (0, 2, 197, "surface", None): "ust",
    (0, 2, 17, "surface", None): "taux",  # momentum flux [N m-2] (archive)
    (0, 2, 18, "surface", None): "tauy",
}

# The messages by file in the `.idx` inventories: (variable, level) -> name
IDX_PROFILES = {"HGT": "z", "TMP": "tk", "UGRD": "u", "VGRD": "v"}
IDX_SURFACE = {
    ("pgrb2", "PRES", "surface"): "p_sfc",
    ("pgrb2", "HGT", "surface"): "z_sfc",
    ("pgrb2", "LAND", "surface"): "lsm",
    ("pgrb2", "TMP", "2 m above ground"): "tk2",
    ("pgrb2", "UGRD", "10 m above ground"): "u10",
    ("pgrb2", "VGRD", "10 m above ground"): "v10",
    ("pgrb2", "PRMSL", "mean sea level"): "prmsl",
    ("sflux", "SHTFL", "surface"): "hfx",
    ("sflux", "LHTFL", "surface"): "lh",
    ("sflux", "FRICV", "surface"): "ust",
    ("sflux", "HPBL", "surface"): "blh",
}

# Friction velocity floor [m s-1]: GRIB packing gives 0 for very small values, and the surface CT2 input needs ust > 0
UST_MIN = 0.01


def load(cycle: str | pd.Timestamp, times, where: Where, n_threads: int = 12) -> xr.Dataset:
    """GFS of the run `cycle` at `times` in an `Area` or at `Points` as canonical inputs: `standardize(fetch(...))`."""
    return standardize(fetch(cycle, times, where, n_threads=n_threads))


def fetch(cycle: str | pd.Timestamp, times, where: Where, n_threads: int = 12) -> xr.Dataset:
    """GFS of the run `cycle` (e.g. "2025-06-01 00:00") at the valid `times` (lead time `time - cycle`, see the module
    docstring) in an `Area` or at `Points` (`otprof.sources`). Returns the fields (`FIELDS`) on dims (time, lat, lon) or
    (time, loc), profiles with `level` [hPa] (the 31 levels of the pressure grid, bottom to top); coordinates `cycle`
    and `lead` [h] along `time`."""
    cycle = pd.Timestamp(cycle)
    leads = [(t - cycle) / pd.Timedelta(hours=1) for t in pd.DatetimeIndex(times)]
    if any(lead < 0 or lead != int(lead) for lead in leads):
        raise ValueError(f"times must be whole hours after the cycle {cycle}")
    parts = [fetch_lead(cycle, int(lead), where, n_threads) for lead in leads]
    return xr.concat(parts, "time") if len(parts) > 1 else parts[0]


def fetch_lead(cycle: pd.Timestamp, lead: int, where: Where, n_threads: int = 12) -> xr.Dataset:
    """One lead time of `fetch`."""
    messages = get_messages(cycle, lead)
    with concurrent.futures.ThreadPoolExecutor(n_threads) as pool:
        content = list(pool.map(lambda m: get_bytes(m[0], m[1]), messages))
    sflux = get_url(cycle, lead, "sflux")
    with tempfile.TemporaryDirectory() as tmp:
        paths = {grid: pathlib.Path(tmp) / f"{grid}.grib2" for grid in ["0p25", "sflux"]}
        for grid, path in paths.items():  # the messages by grid: 0.25 deg (pgrb2, pgrb2b), Gaussian (sflux)
            path.write_bytes(
                b"".join(c for (url, *_), c in zip(messages, content) if (url == sflux) == (grid == "sflux"))
            )
        ds = select(read_grib(paths["0p25"]), where)
        ds = ds.assign(select_like(read_grib(paths["sflux"]), ds).data_vars)
    ds = ds.sel(level=list(schema.P_LEVELS_HPA)).expand_dims(time=[cycle + pd.Timedelta(hours=lead)])
    return ds.assign_coords(cycle=("time", [cycle]), lead=("time", [lead]))


def read_grib(path: str | pathlib.Path, means: tuple[str, ...] = ()) -> xr.Dataset:
    """The fields of `FIELDS` in a GRIB file, on its grid: dims (lat, lon), profiles with `level` [hPa]. Fields named
    in `means` are taken as averages (GRIB `stepType` avg), all others as instantaneous values; fields the file lacks
    are left out. Each field is opened on its own with cfgrib (NCEP's own parameters have no short name and would
    collide); cfgrib's index is written next to the file."""
    out = {}
    for (discipline, category, number, type_of_level, height), name in FIELDS.items():
        keys = {"discipline": discipline, "parameterCategory": category, "parameterNumber": number,
                "typeOfLevel": type_of_level, "stepType": "avg" if name in means else "instant"}  # fmt: skip
        if height is not None:
            keys["level"] = height
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # cfgrib's internal xr.merge
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": keys, "indexpath": f"{path}.{{short_hash}}.idx"},
            )
        if not ds.data_vars:
            continue
        (field,) = ds.data_vars.values()
        field = field.rename(latitude="lat", longitude="lon")
        if "isobaricInhPa" in field.dims:
            field = field.rename(isobaricInhPa="level")
        out[name] = field.reset_coords(drop=True).load()
    ds = xr.Dataset(out)
    for v in ds.variables.values():
        v.attrs = {}
    return ds.assign_coords(level=ds["level"].astype(int)) if "level" in ds.coords else ds


def get_url(cycle: pd.Timestamp, lead: int, file: str) -> str:
    """URL of a GFS file (`pgrb2`, `pgrb2b` or `sflux`) of `cycle` at lead time `lead` [h]."""
    base = f"{AWS}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z"
    return f"{base}.sfluxgrbf{lead:03d}.grib2" if file == "sflux" else f"{base}.{file}.0p25.f{lead:03d}"


def is_cycle_available(cycle: str | pd.Timestamp, last_lead: int = 384) -> bool:
    """Whether the run `cycle` is on AWS up to lead time `last_lead` [h] (default: the whole run): the `.idx`
    inventories of all files `fetch` needs at `last_lead` exist. NCEP uploads the lead times in order, about 3.5-5 h
    after the cycle time, so a run complete up to `last_lead` can be fetched at all earlier lead times."""
    cycle = pd.Timestamp(cycle)
    return all(url_exists(get_url(cycle, last_lead, file) + ".idx") for file in ["pgrb2", "pgrb2b", "sflux"])


def url_exists(url: str) -> bool:
    """Whether `url` exists (HEAD request); other errors than a missing file raise."""
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30).close()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # S3 answers 403 for missing keys of a bucket without public listing
            return False
        raise
    return True


def get_bytes(url: str, byte_range: str | None = None, attempts: int = 5) -> bytes:
    """The content of `url`, or of the byte range "start-end" of it; retried on network errors."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={byte_range}"} if byte_range else {})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError:  # URLError, connection reset, timeout
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def get_messages(cycle: pd.Timestamp, lead: int) -> list[tuple[str, str, str, int | None]]:
    """The GRIB messages `fetch` needs: (url, byte range, name, pressure level [hPa] or None), from the `.idx`
    inventories ("n:offset:d=YYYYMMDDHH:VAR:LEVEL:TIME:"). Raises if a file lacks a field (e.g. not on AWS yet)."""
    when = "anl" if lead == 0 else f"{lead} hour fcst"
    wanted = {"pgrb2": {}, "pgrb2b": {}, "sflux": {}}  # file -> idx key "VAR:LEVEL:TIME" -> (name, level)
    for (file, var, level), name in IDX_SURFACE.items():
        wanted[file][f"{var}:{level}:{when}"] = (name, None)
    for var, name in IDX_PROFILES.items():
        for file, levels in [("pgrb2", LEVELS_PGRB2), ("pgrb2b", LEVELS_PGRB2B)]:
            for p in levels:
                wanted[file][f"{var}:{p} mb:{when}"] = (name, p)

    messages = []
    for file, keys in wanted.items():
        url = get_url(cycle, lead, file)
        lines = get_bytes(url + ".idx").decode().strip().splitlines()
        starts = [int(line.split(":")[1]) for line in lines]
        found = set()
        for i, line in enumerate(lines):
            key = ":".join(line.split(":")[3:6])
            if key in keys and key not in found:  # first occurrence (some variables appear twice)
                found.add(key)
                end = str(starts[i + 1] - 1) if i + 1 < len(lines) else ""
                messages.append((url, f"{starts[i]}-{end}", *keys[key]))
        if missing := set(keys) - found:
            raise ValueError(f"{url}: missing {sorted(missing)}")
    return messages


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Convert the fields of `fetch` to the canonical inputs on the pressure grid: orography -> `z_sfc_src` (averaged
    over the times), sea level pressure Pa -> hPa, friction velocity floored at `UST_MIN`, the 31 levels as dim
    `bottom_top` with the nominal pressure as coordinate `p_level` [hPa] and as `p` [Pa]. Heights are geopotential
    heights; fluxes are upward positive as delivered."""
    ds = ds.rename(z_sfc="z_sfc_src", prmsl="slp")
    ds["slp"] = ds["slp"] / 100
    ds["ust"] = ds["ust"].clip(min=UST_MIN)
    ds["z_sfc_src"] = ds["z_sfc_src"].mean("time")
    return to_pressure_grid(ds)


def to_pressure_grid(ds: xr.Dataset) -> xr.Dataset:
    """Common last step of the GFS adapters: canonical variables only, longitude -> -180..180, dim `level` [hPa] ->
    `bottom_top` with the coordinate `p_level` and `p` [Pa], attributes, validation."""
    ds = ds[[v for v in schema.VARIABLES if v in ds]]
    ds = ds.assign_coords(lon=(ds["lon"].dims, to_180(ds["lon"].values)))
    ds = ds.sel(level=list(schema.P_LEVELS_HPA))
    ds["p"] = (ds["level"] * 100).astype(int)
    ds = ds.rename(level="p_level").swap_dims(p_level="bottom_top")
    ds = ds.transpose(*[d for d in ds.dims if d != "bottom_top"], "bottom_top")
    return schema.validate(schema.set_attrs(ds), grid="pressure")

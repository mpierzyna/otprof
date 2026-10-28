"""GFS forecasts (NCEP Global Forecast System, v16 and later, from 2021) from the NOAA archive on AWS
(`noaa-gfs-bdp-pds`) as canonical inputs (`otprof.schema`). Needs `otprof[gfs]` (eccodes).

    ds = gfs.fetch(cycle, lead, lat=[52.1, 51.9], lon=[4.4, 5.2])   # GRIB fields at the nearest grid points
    X = predictor.make_inputs(gfs.standardize(ds))

`fetch` downloads only the needed messages (byte ranges from the `.idx` inventories) of three files per lead time:
- `pgrb2` (0.25 deg): HGT, TMP, UGRD, VGRD on 25 of the 31 pressure levels; surface pressure, orography, land mask,
  2 m temperature, 10 m wind, sea level pressure
- `pgrb2b` (0.25 deg): the six levels 875, 825, 775, 225, 175, 125 hPa that `pgrb2` lacks
- `sflux` (Gaussian grid, about 13 km): instantaneous sensible and latent heat flux, friction velocity, PBL height
Each location takes the value of the nearest grid point of each grid. This is the "current recipe" of the research
GFS study, which tracks ERA5 better than the fields the archive of earlier years offers (`otprof.sources.gfs_archive`).
"""

from __future__ import annotations

import concurrent.futures
import urllib.request
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from otprof import schema

AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Pressure levels [hPa] by file: pgrb2 has 25 of the 31 levels of the pressure grid, pgrb2b the others
LEVELS_PGRB2B = (875, 825, 775, 225, 175, 125)
LEVELS_PGRB2 = tuple(p for p in schema.P_LEVELS_HPA if p not in LEVELS_PGRB2B)

# Profile fields: idx variable -> name
PROFILES = {"HGT": "z", "TMP": "tk", "UGRD": "u", "VGRD": "v"}
# Surface fields: (file, idx variable, idx level) -> name. `z` and `z_sfc` are geopotential height [gpm] (= m to the
# precision that matters here), fluxes are upward positive, `prmsl` in Pa, `lsm` 0 / 1.
SURFACE = {
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


def get_url(cycle: pd.Timestamp, lead: int, file: str) -> str:
    """URL of a GFS file (`pgrb2`, `pgrb2b` or `sflux`) of `cycle` at lead time `lead` [h]."""
    base = f"{AWS}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z"
    return f"{base}.sfluxgrbf{lead:03d}.grib2" if file == "sflux" else f"{base}.{file}.0p25.f{lead:03d}"


def download(url: str, byte_range: str | None = None) -> bytes:
    """The content of `url`, or of the byte range "start-end" of it."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={byte_range}"} if byte_range else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def get_messages(cycle: pd.Timestamp, lead: int) -> list[tuple[str, str, str, int | None]]:
    """The GRIB messages `fetch` needs: (url, byte range, name, pressure level [hPa] or None), from the `.idx`
    inventories ("n:offset:d=YYYYMMDDHH:VAR:LEVEL:TIME:"). Raises if a file lacks a field."""
    time = f"{lead} hour fcst"
    wanted = {"pgrb2": {}, "pgrb2b": {}, "sflux": {}}  # file -> idx key "VAR:LEVEL:TIME" -> (name, level)
    for (file, var, level), name in SURFACE.items():
        wanted[file][f"{var}:{level}:{time}"] = (name, None)
    for var, name in PROFILES.items():
        for file, levels in [("pgrb2", LEVELS_PGRB2), ("pgrb2b", LEVELS_PGRB2B)]:
            for p in levels:
                wanted[file][f"{var}:{p} mb:{time}"] = (name, p)

    messages = []
    for file, keys in wanted.items():
        url = get_url(cycle, lead, file)
        lines = download(url + ".idx").decode().strip().splitlines()
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


def get_nearest(message: int, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Flat indices of the grid points of GRIB `message` (an eccodes handle) nearest to the locations (lat, lon),
    with the longitude distance scaled by cos(lat)."""
    import eccodes  # otprof[gfs]

    glat = eccodes.codes_get_array(message, "latitudes")
    glon = (eccodes.codes_get_array(message, "longitudes") + 180) % 360 - 180
    return np.array(
        [np.argmin((glat - la) ** 2 + ((glon - lo) * np.cos(np.deg2rad(la))) ** 2) for la, lo in zip(lat, lon)]
    )


def fetch(
    cycle: str | pd.Timestamp,
    lead: int,
    lat: Sequence[float],
    lon: Sequence[float],
    n_threads: int = 12,
) -> xr.Dataset:
    """GFS of `cycle` (e.g. "2024-06-01 00:00") at lead time `lead` [h] at the locations (lat, lon) [degrees north /
    east, dim `loc`], from the grid points nearest to them. Returns the fields (see the module docstring) on dims
    (time, loc) and (time, level, loc), `level` the 31 pressure levels [hPa] bottom to top; coordinates `cycle` and
    `lead` along `time`, `lat` and `lon` of the locations. Several lead times: concatenate along `time`."""
    import eccodes  # otprof[gfs]

    cycle = pd.Timestamp(cycle)
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    messages = get_messages(cycle, lead)

    # Nearest grid points per file (url): 0.25 deg grid of pgrb2 and pgrb2b, Gaussian grid of sflux
    nearest = {}
    for file in ["pgrb2", "sflux"]:
        url = get_url(cycle, lead, file)
        byte_range = next(m[1] for m in messages if m[0] == url)
        h = eccodes.codes_new_from_message(download(url, byte_range))
        nearest[url] = get_nearest(h, lat, lon)
        eccodes.codes_release(h)
    nearest[get_url(cycle, lead, "pgrb2b")] = nearest[get_url(cycle, lead, "pgrb2")]

    def decode(message):
        url, byte_range, name, level = message
        h = eccodes.codes_new_from_message(download(url, byte_range))
        try:
            return name, level, eccodes.codes_get_values(h)[nearest[url]]
        finally:
            eccodes.codes_release(h)

    with concurrent.futures.ThreadPoolExecutor(n_threads) as pool:
        results = list(pool.map(decode, messages))

    surface, profiles = {}, {}
    for name, level, values in results:
        if level is None:
            surface[name] = values
        else:
            profiles.setdefault(name, {})[level] = values
    levels = list(schema.P_LEVELS_HPA)
    ds = xr.Dataset(
        {
            **{n: (("loc",), v.astype(np.float32)) for n, v in surface.items()},
            **{n: (("level", "loc"), np.stack([by_level[p] for p in levels]).astype(np.float32))
               for n, by_level in profiles.items()},
        },
        coords={"level": levels, "loc": np.arange(len(lat)), "lat": ("loc", lat), "lon": ("loc", lon)},
    )  # fmt: skip
    ds = ds.expand_dims(time=[cycle + pd.Timedelta(hours=lead)])
    return ds.assign_coords(cycle=("time", [cycle]), lead=("time", [lead]))


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Convert the fields of `fetch` (one or more times along `time`) to the canonical inputs on the pressure grid:
    orography -> `z_sfc_src` (averaged over the times), sea level pressure Pa -> hPa, friction velocity floored at
    `UST_MIN`, longitude -> -180..180, the 31 levels as dim `bottom_top` with the nominal pressure as coordinate
    `p_level` [hPa] and as `p` [Pa]. Heights are geopotential heights; fluxes are upward positive as delivered."""
    ds = ds.rename(z_sfc="z_sfc_src", prmsl="slp")
    ds["slp"] = ds["slp"] / 100
    ds["ust"] = ds["ust"].clip(min=UST_MIN)
    ds["z_sfc_src"] = ds["z_sfc_src"].mean("time")
    return to_pressure_grid(ds)


def to_pressure_grid(ds: xr.Dataset) -> xr.Dataset:
    """Common last step of the GFS adapters: canonical variables only, longitude -> -180..180, dim `level` [hPa] ->
    `bottom_top` with the coordinate `p_level` and `p` [Pa], attributes, validation."""
    ds = ds[[v for v in schema.VARIABLES if v in ds]]
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=((ds["lon"] + 180) % 360) - 180)
    ds = ds.sel(level=list(schema.P_LEVELS_HPA))
    ds["p"] = (ds["level"] * 100).astype(int)
    ds = ds.rename(level="p_level").swap_dims(p_level="bottom_top")
    ds = ds.transpose(*[d for d in ds.dims if d != "bottom_top"], "bottom_top")
    return schema.validate(schema.set_attrs(ds), grid="pressure")

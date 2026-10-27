"""ERA5 (pressure levels + surface) from the Copernicus Climate Data Store (CDS) → the ARCO layout of `era5.fetch`.

- `make_jobs`: CDS requests (GRIB) for ERA5 over an area and a range of months, split to stay within the CDS cost limits
- `make_block_jobs`: the same for one block of years x months (e.g. one month or one quarter)
- `retrieve`: submit the requests and download the results, resumable (needs `otprof[data]`: ecmwf-datastores-client)
- `open_month`: one month of the downloaded GRIB files in ARCO names, units and layout, the input of
  `era5.standardize` (needs `otprof[data]`: cfgrib, eccodes)

The CDS client reads its URL and API key from `~/.ecmwfdatastoresrc` (or `ECMWF_DATASTORES_URL` / `_KEY`).
"""

from __future__ import annotations

import json
import pathlib
import time
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from otprof.sources import era5

PL_DATASET = "reanalysis-era5-pressure-levels"
SL_DATASET = "reanalysis-era5-single-levels"

# The 37 pressure levels of ERA5 [hPa], as in the ARCO store
LEVELS = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
)  # fmt: skip

# Short name in the GRIB files → ARCO name. CDS requests use the ARCO names, except the surface geopotential.
PL_NAMES = {
    "z": "geopotential",
    "pv": "potential_vorticity",
    "w": "vertical_velocity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "q": "specific_humidity",
}
SL_NAMES = {
    "slt": "soil_type",
    "lsm": "land_sea_mask",
    "anor": "angle_of_sub_gridscale_orography",
    "isor": "anisotropy_of_sub_gridscale_orography",
    "z": "geopotential_at_surface",
    "blh": "boundary_layer_height",
    "bld": "boundary_layer_dissipation",
    "sp": "surface_pressure",
    "zust": "friction_velocity",
    "ishf": "instantaneous_surface_sensible_heat_flux",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "slhf": "surface_latent_heat_flux",
    "avg_snswrf": "mean_surface_net_short_wave_radiation_flux",
    "lcc": "low_cloud_cover",
    "skt": "skin_temperature",
}
SL_REQUEST = {"geopotential_at_surface": "geopotential"}  # ARCO name → CDS request name, where they differ

# CDS cost limits [fields per request] (2026-10). The cost counts the dates that exist; requests are sized for 31-day
# months, so every request stays within the limit.
COST_LIMIT = {PL_DATASET: 60_000, SL_DATASET: 121_000}


def make_jobs(
    start: str,
    end: str,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    variables: list[str] = era5.VARIABLES,
    hours: Sequence[int] = range(24),
) -> dict[str, tuple[str, dict]]:
    """CDS requests for `hours` (UTC) of every day of the months `start` .. `end` (e.g. "2005-01", "2026-06").

    Returns {file name: (dataset, request)}. A CDS request is the cross product of its variables, levels, years,
    months, days and hours; each dataset gets the block of variables x years x months (`_choose_block`) that fills the cost
    limit best, i.e. the fewest requests: e.g. hourly 1 pressure-level variable x 2 months (55 056 of 60 000 fields)
    and 18 surface variables x 3 years x 3 months (120 528 of 121 000). Files: `<pl|sl>/<years>_<months>_<i>.grib`,
    `i` the variable group. The area takes ARCO longitudes (0..360), as `era5.fetch`.
    """
    area = {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max}
    months = set(pd.period_range(start, end, freq="M"))
    years = sorted({m.year for m in months})
    out = {}
    for kind, dataset, names, extra in _split_by_dataset(variables):
        n_levels = len(extra.get("pressure_level", [0]))
        _, ny, nm = _choose_block(len(names), n_levels * 31 * len(hours), COST_LIMIT[dataset], len(years))
        dataset_vars = [v for v in variables if v in (PL_NAMES if kind == "pl" else SL_NAMES).values()]
        for y0 in range(0, len(years), ny):
            for m0 in range(1, 13, nm):
                block_years, block_months = years[y0 : y0 + ny], range(m0, min(m0 + nm, 13))
                in_range = [(y, m) for y in block_years for m in block_months if pd.Period(f"{y}-{m}", "M") in months]
                if len(in_range) == len(block_years) * len(block_months):
                    products = [(block_years, list(block_months))]
                else:  # block cut by start / end: one request per year, with its months in range
                    products = [(ys, [m for y, m in in_range if y == ys[0]]) for ys in ([y] for y in block_years)]
                for ys, ms in products:
                    if not ms:
                        continue
                    span = f"{ys[0]}" if len(ys) == 1 else f"{ys[0]}-{ys[-1]}"
                    span += f"_{ms[0]:02d}" if len(ms) == 1 else f"_{ms[0]:02d}-{ms[-1]:02d}"
                    out |= make_block_jobs(span, ys, ms, **area, variables=dataset_vars, hours=hours)
    return out


def make_block_jobs(
    name: str,
    years: Sequence[int],
    months: Sequence[int],
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    variables: list[str] = era5.VARIABLES,
    hours: Sequence[int] = range(24),
) -> dict[str, tuple[str, dict]]:
    """CDS requests for `hours` (UTC) of every day of the block `years` x `months` (e.g. [2017], [1, 2, 3]).

    Per dataset, the variables are packed into as few requests as the cost limit allows, e.g. hourly for one month:
    4 pressure-level requests (2, 2, 2, 1 of the 7 variables) and 1 surface request. Files: `<pl|sl>/<name>_<i>.grib`,
    `i` the variable group. Returns {file name: (dataset, request)}, as `make_jobs`.
    """
    common = _make_common_keys(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, hours=hours)
    out = {}
    for kind, dataset, names, extra in _split_by_dataset(variables):
        n_levels = len(extra.get("pressure_level", [0]))
        nv = COST_LIMIT[dataset] // (n_levels * 31 * len(hours) * len(years) * len(months))
        assert nv >= 1, f"one variable of the block {name} exceeds the CDS limit of {COST_LIMIT[dataset]}"
        for i in range(0, len(names), nv):
            request = {
                **common,
                **extra,
                "variable": names[i : i + nv],
                "year": [str(y) for y in years],
                "month": [f"{m:02d}" for m in months],
            }
            out[f"{kind}/{name}_{i // nv}.grib"] = (dataset, request)
    return out


def _make_common_keys(*, lon_min: float, lon_max: float, lat_min: float, lat_max: float, hours: Sequence[int]) -> dict:
    """Request keys shared by all requests: every day, `hours`, the area (ARCO longitudes 0..360 → -180..180), GRIB."""
    lon = lambda x: ((x + 180) % 360) - 180
    return {
        "product_type": ["reanalysis"],
        "day": [f"{d:02d}" for d in range(1, 32)],  # days a month does not have are ignored
        "time": [f"{h:02d}:00" for h in hours],
        "area": [lat_max, lon(lon_min), lat_min, lon(lon_max)],  # N, W, S, E
        "data_format": "grib",
        "download_format": "unarchived",
    }


def _split_by_dataset(variables: list[str]) -> list[tuple[str, str, list[str], dict]]:
    """(kind, dataset, CDS variable names, extra request keys) of the datasets `variables` (ARCO names) come from."""
    pl = [v for v in variables if v in PL_NAMES.values()]
    sl = [v for v in variables if v in SL_NAMES.values()]
    unknown = set(variables) - set(pl) - set(sl)
    assert not unknown, f"no CDS name for {sorted(unknown)}: add them to `PL_NAMES` / `SL_NAMES`"
    out = [
        ("pl", PL_DATASET, pl, {"pressure_level": [str(p) for p in LEVELS]}),
        ("sl", SL_DATASET, [SL_REQUEST.get(v, v) for v in sl], {}),
    ]
    return [d for d in out if d[2]]


def _choose_block(n_vars: int, fields_per_var_month: int, limit: int, n_years: int) -> tuple[int, int, int]:
    """(variables, years, months) per request within `limit` fields with the fewest requests over `n_years` years."""
    best = None
    for nv in range(1, n_vars + 1):
        for ny in range(1, n_years + 1):
            for nm in range(1, 13):
                if nv * ny * nm * fields_per_var_month > limit:
                    continue
                n = -(-n_vars // nv) * -(-n_years // ny) * -(-12 // nm)
                key = (n, ny, -nv)  # ties: fewer years per request (shorter files), then fewer variable groups
                if best is None or key < best[0]:
                    best = (key, (nv, ny, nm))
    assert best, f"one variable-month ({fields_per_var_month} fields) exceeds the CDS limit of {limit}"
    return best[1]


def retrieve(jobs: dict[str, tuple[str, dict]], root: pathlib.Path, *, max_active: int = 8, poll: float = 30) -> None:
    """Submit the `jobs` (from `make_jobs`) to the CDS and download each result to `root / <file name>`.

    At most `max_active` requests are queued or running at a time. Files that exist are skipped. The request ids are
    kept in `root / requests.json`, so a restart downloads the requests already submitted instead of queueing them
    again. Failed requests are submitted again, up to 3 times. Progress: a bar over the requests, with the numbers
    running and queued at the CDS and the GB downloaded.
    """
    import requests
    from ecmwf.datastores import Client
    from tqdm import tqdm

    client = Client(progress=False, log_callback=lambda *args, **kwargs: None)
    state_file = root / "requests.json"
    ids = json.loads(state_file.read_text()) if state_file.exists() else {}
    failures: dict[str, int] = {}

    def save():
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(ids, indent=1))

    todo = [name for name in jobs if not (root / name).exists()]
    n_bytes = 0
    bar = tqdm(total=len(jobs), initial=len(jobs) - len(todo), unit="req", dynamic_ncols=True, smoothing=0.05)
    while todo:
        counts = {"accepted": 0, "running": 0}
        for name in list(todo):
            if name not in ids:
                continue
            try:
                remote = client.get_remote(ids[name])
                status = remote.status
            except requests.HTTPError as e:  # e.g. request deleted or results expired
                bar.write(f"{name}: request {ids[name]} not available ({e}), submitting again")
                status = "failed"
            if status == "successful":
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp")
                remote.download(str(tmp))
                tmp.rename(target)
                n_bytes += target.stat().st_size
                todo.remove(name)
                del ids[name]
                bar.update()
            elif status in counts:
                counts[status] += 1
            else:
                failures[name] = failures.get(name, 0) + 1
                assert failures[name] <= 3, f"{name}: request failed 3 times, last status `{status}`"
                bar.write(f"{name}: request {ids[name]} {status}, submitting again")
                del ids[name]
        for name in [n for n in todo if n not in ids][: max(max_active - sum(counts.values()), 0)]:
            dataset, request = jobs[name]
            ids[name] = client.submit(dataset, request).request_id
            counts["accepted"] += 1
        save()
        bar.set_postfix(running=counts["running"], queued=counts["accepted"], GB=f"{n_bytes / 1e9:.1f}")
        if todo:
            time.sleep(poll)
    bar.close()


def open_month(root: pathlib.Path, jobs: dict[str, tuple[str, dict]], month: str) -> xr.Dataset:
    """ERA5 of `month` (e.g. "2017-01") from the GRIB files of `jobs` in `root`, in the ARCO layout of `era5.fetch`.

    ARCO names and units, dims `(time, level, latitude, longitude)`, levels ascending, latitude descending, longitude
    0..360; the times are the hours of the requests. The surface fields CDS delivers as forecasts (`time` x `step`:
    accumulations, means, friction velocity) are put on their validity times.
    """
    m = pd.Period(month, freq="M")
    requests = {
        root / name: (dataset, request)
        for name, (dataset, request) in jobs.items()
        if str(m.year) in request["year"] and f"{m.month:02d}" in request["month"]
    }
    assert requests, f"no files for {month} in the jobs"
    (hours, *other) = {tuple(request["time"]) for _, request in requests.values()}
    assert not other, f"the requests for {month} differ in their hours"
    days = pd.date_range(m.start_time, m.end_time.floor("D"), freq="D")
    times = pd.DatetimeIndex([d + pd.Timedelta(f"{t}:00") for d in days for t in hours])
    return open_grib({path: dataset for path, (dataset, _) in requests.items()}, times)


def open_grib(files: dict[pathlib.Path, str], times) -> xr.Dataset:
    """GRIB files from the CDS ({path: dataset}, requests as in `make_jobs`) at `times`, in the ARCO layout."""
    parts = []
    for path, dataset in files.items():
        names = PL_NAMES if dataset == PL_DATASET else SL_NAMES
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # cfgrib's internal xr.merge
            for ds in _open_grib(path):
                ds = _move_to_valid_time(ds).sel(time=times)
                parts.append(ds.rename({k: v for k, v in names.items() if k in ds}))
    ds = xr.merge(parts, compat="equals", join="exact")
    ds = ds.rename({"isobaricInhPa": "level"}) if "isobaricInhPa" in ds.dims else ds
    ds = ds.assign_coords(
        longitude=(ds["longitude"] % 360).astype(np.float32), latitude=ds["latitude"].astype(np.float32)
    )
    ds = ds.sortby("longitude").sortby("latitude", ascending=False)
    if "level" in ds.dims:
        ds = ds.assign_coords(level=ds["level"].astype(np.int64)).sortby("level")
    for v in ds.variables.values():
        v.attrs = {k: a for k, a in v.attrs.items() if not k.startswith("GRIB_")}
    ds.attrs = {}
    return ds.transpose("time", ...)


def _open_grib(path: pathlib.Path) -> list[xr.Dataset]:
    import cfgrib

    # index next to the file (`<file>.<hash>.idx`): a file spans several months and is opened once per month
    return cfgrib.open_datasets(str(path))


def _move_to_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """Dims `(time, step)` (forecasts) → `time` = validity time; drop the scalar GRIB coordinates."""
    if "step" in ds.dims:
        valid = ds["valid_time"].values.ravel()
        ds = ds.drop_vars("valid_time").stack(vt=("time", "step"))
        ds = ds.drop_vars(["vt", "time", "step"]).assign_coords(vt=valid).rename(vt="time")
    else:
        ds = ds.assign_coords(time=ds["valid_time"].values)
    return ds.drop_vars([c for c in ("number", "step", "surface", "valid_time") if c in ds.coords])

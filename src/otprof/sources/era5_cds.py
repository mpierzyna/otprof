"""ERA5 (pressure levels + surface) from the Copernicus Climate Data Store (CDS), for long time series.

ARCO-ERA5 (`otprof.sources.era5`) stores one global chunk per time step, so long time series over a small area
download the whole globe (20 years hourly: about 217 TB). The CDS cuts the area on the server, but works with a queue:
requests wait and run for hours. Hence two steps:

    era5_cds.download(times, area, root)       # submit, wait, download; resumable (run it again after a stop)
    X = era5_cds.load(times, where, root)      # any times and `Area` / `Points` within the downloaded data

- `make_jobs`: the CDS requests (GRIB) for every day of the months of `times` at their hours of day, over `area`,
  packed to the CDS cost limits (as few requests as possible). `times` is one time range or a list of blocks
  (`Times`): each block gets its own requests, in block order, e.g. to download a sample of the months first
- `download`: `make_jobs`, submitted and downloaded to `root`; the plan is kept in `root/jobs.json`, the request ids
  in `root/requests.json` (needs `otprof[data]`: ecmwf-datastores-client)
- `fetch`: the downloaded data at `times` in an `Area` or at `Points`, in the ARCO layout of `era5.fetch` (needs
  `otprof[data]`: cfgrib, eccodes); `standardize` = `era5.standardize`; `load` = `standardize(fetch(...))`

The CDS client reads its URL and API key from `~/.ecmwfdatastoresrc` (or `ECMWF_DATASTORES_URL` / `_KEY`).
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import warnings
from collections.abc import Sequence
from typing import NamedTuple, NotRequired, TypedDict

import numpy as np
import pandas as pd
import xarray as xr

from otprof.sources import Area, Where, era5, select

standardize = era5.standardize

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


class Request(TypedDict):
    """One CDS request: the cross product of its lists."""

    product_type: list[str]
    variable: list[str]  # CDS names
    year: list[str]
    month: list[str]
    day: list[str]  # always "01" .. "31"; days a month does not have are ignored
    time: list[str]  # "HH:00", the hours of day of the block
    area: list[float]  # N, W, S, E
    data_format: str
    download_format: str
    pressure_level: NotRequired[list[str]]


class Job(NamedTuple):
    dataset: str  # `PL_DATASET` or `SL_DATASET`
    request: Request


Jobs = dict[str, Job]  # file name under `root` → job; the plan in `root/jobs.json`
Times = pd.DatetimeIndex | Sequence[pd.DatetimeIndex]  # one time range, or blocks in download order


def make_jobs(times: Times, area: Area, *, variables: Sequence[str] = era5.VARIABLES) -> Jobs:
    """The CDS requests for every day of the months of `times`, at their hours of day (UTC), over `area`.

    `times` is one `DatetimeIndex` or a list of them (blocks): each block gets its own requests (none spans two
    blocks), at the hours of day of that block, in block order. A CDS request is the cross product of its variables,
    levels, years, months, days and hours; each dataset gets the grouping of years x months (`_choose_groups`) with
    the fewest requests, e.g. hourly 1 pressure-level variable x 2 months (55 056 of 60 000 fields) and 9 surface
    variables x 3 years x 6 months (120 528 of 121 000). Files: `<pl|sl>/<years>_<months>_<i>.grib`, `i` the
    variable group, e.g. `pl/2017_01-03_0.grib`, `sl/2005-2007_04-06_0.grib`, `pl/2010_07+09_1.grib`.
    """
    out: Jobs = {}
    for block in _to_blocks(times):
        for name, job in _make_block_plan(block, area, variables).items():
            if out.get(name, job) != job:
                raise ValueError(f"two blocks plan {name} with other requests (other hours of day?)")
            out.setdefault(name, job)
    return out


def download(
    times: Times,
    area: Area,
    root: str | os.PathLike,
    *,
    variables: Sequence[str] = era5.VARIABLES,
    max_active: int = 8,
    poll: float = 30,
    retries: int = 3,
) -> Jobs:
    """Download ERA5 over `area` for every day of the months of `times`, at their hours of day (UTC), to `root`.

    The requests (`make_jobs`; blocks of `times` in their order) are added to the plan in `root/jobs.json`; a file of
    the plan with another request (e.g. another area) raises: one `root` per area and variables. At most `max_active`
    requests are queued or running at the CDS at a time, across blocks. Files that exist are skipped, so `download`
    resumes after a stop; failed requests and failed downloads are tried again up to `retries` times (`_retrieve`).
    Returns the jobs of this call.
    """
    root = pathlib.Path(root)
    jobs = make_jobs(times, area, variables=variables)
    stored = json.loads(json.dumps(jobs))  # as stored: lists
    plan_file = root / "jobs.json"
    plan = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    if conflict := [n for n in stored if n in plan and plan[n] != stored[n]]:
        raise ValueError(f"{root} holds other requests for {conflict[:3]}: use one root per area and variables")
    root.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps(plan | stored, indent=1))
    _retrieve(jobs, root, max_active=max_active, poll=poll, retries=retries)
    return jobs


def fetch(times: pd.DatetimeIndex, where: Where, root: str | os.PathLike) -> xr.Dataset:
    """ERA5 at `times` in an `Area` or at `Points` (`otprof.sources`) from the files `download` wrote to `root`, in
    ARCO names, units and layout (as `era5.fetch`). The forecast fields of the CDS (accumulations, means, friction
    velocity) are put on their validity times."""
    root = pathlib.Path(root)
    if not (root / "jobs.json").exists():
        raise ValueError(f"{root}: no jobs.json, download first")
    jobs = {name: Job(*job) for name, job in json.loads((root / "jobs.json").read_text()).items()}
    times = pd.DatetimeIndex(times)
    parts = []
    for month in sorted(set(times.to_period("M"))):
        requests = {
            root / name: job
            for name, job in jobs.items()
            if str(month.year) in job.request["year"] and f"{month.month:02d}" in job.request["month"]
        }
        if not requests:
            raise ValueError(f"{month} was not downloaded to {root}")
        if missing := [f for f in requests if not f.exists()]:
            raise ValueError(f"{len(missing)} files of {month} missing in {root} (e.g. {missing[0]}): download again")
        in_month = times[times.to_period("M") == month]
        for path, job in requests.items():
            if lacking := set(in_month.hour) - {int(t[:2]) for t in job.request["time"]}:
                raise ValueError(f"{path}: hours {sorted(lacking)} of {month} were not downloaded")
        parts.append(open_grib({path: job.dataset for path, job in requests.items()}, in_month))
    ds = xr.concat(parts, "time") if len(parts) > 1 else parts[0]
    return select(ds, where, lat="latitude", lon="longitude")


def load(times: pd.DatetimeIndex, where: Where, root: str | os.PathLike) -> xr.Dataset:
    """ERA5 at `times` in an `Area` or at `Points` from `root` as canonical inputs: `standardize(fetch(...))`."""
    return standardize(fetch(times, where, root).load())


def _to_blocks(times: Times) -> list[pd.DatetimeIndex]:
    """`times` as a list of blocks: one `DatetimeIndex` (or anything it takes) is one block."""
    if isinstance(times, pd.DatetimeIndex):
        return [times]
    if len(times) and all(isinstance(t, pd.DatetimeIndex) for t in times):
        return list(times)
    return [pd.DatetimeIndex(times)]


def _make_block_plan(times: pd.DatetimeIndex, area: Area, variables: Sequence[str]) -> Jobs:
    """The requests of one block (`make_jobs`): its months, at its hours of day."""
    hours = sorted(set(times.hour))
    months_of_year: dict[int, tuple[int, ...]] = {}
    for m in sorted(set(times.to_period("M"))):
        months_of_year[m.year] = (*months_of_year.get(m.year, ()), m.month)
    out: Jobs = {}
    for kind, dataset, names, extra in _split_by_dataset(variables):
        n_levels = len(extra.get("pressure_level", [0]))
        dataset_vars = [v for v in variables if v in (PL_NAMES if kind == "pl" else SL_NAMES).values()]
        groups = _choose_groups(len(names), n_levels * 31 * len(hours), COST_LIMIT[dataset], months_of_year)
        for years, months in groups:
            name = f"{_format_span(years, 4)}_{_format_span(months, 2)}"
            out |= _make_block_jobs(name, years, months, area, dataset_vars, hours)
    return out


def _make_block_jobs(
    name: str, years: Sequence[int], months: Sequence[int], area: Area, variables: Sequence[str], hours: Sequence[int]
) -> Jobs:
    """CDS requests for `hours` (UTC) of every day of `years` x `months` (e.g. [2017], [1, 2, 3]).

    Per dataset, the variables are packed into as few requests as the cost limit allows, e.g. hourly for one month:
    4 pressure-level requests (2, 2, 2, 1 of the 7 variables) and 1 surface request. Files: `<pl|sl>/<name>_<i>.grib`,
    `i` the variable group.
    """
    common = _make_common_keys(area, hours)
    out: Jobs = {}
    for kind, dataset, names, extra in _split_by_dataset(variables):
        nv = _count_vars_per_request(dataset, len(extra.get("pressure_level", [0])) * 31 * len(hours), years, months)
        if nv < 1:
            raise ValueError(f"one variable of {name} exceeds the CDS limit of {COST_LIMIT[dataset]}")
        for i in range(0, len(names), nv):
            request: Request = {
                **common,
                **extra,
                "variable": names[i : i + nv],
                "year": [str(y) for y in years],
                "month": [f"{m:02d}" for m in months],
            }
            out[f"{kind}/{name}_{i // nv}.grib"] = Job(dataset, request)
    return out


def _count_vars_per_request(dataset: str, fields_per_var_month: int, years: Sequence, months: Sequence) -> int:
    """The most variables of `dataset` per request of `years` x `months` within the cost limit."""
    return COST_LIMIT[dataset] // (fields_per_var_month * len(years) * len(months))


def _format_span(values: Sequence[int], width: int) -> str:
    """Years or months in a file name: `2017`, `2005-2007` (consecutive), `07+09` (not consecutive)."""
    text = [f"{v:0{width}d}" for v in values]
    if len(values) == 1:
        return text[0]
    if list(values) == list(range(values[0], values[-1] + 1)):
        return f"{text[0]}-{text[-1]}"
    return "+".join(text)


def _make_common_keys(area: Area, hours: Sequence[int]) -> dict:
    """Request keys shared by all requests: every day, `hours`, the area, GRIB."""
    return {
        "product_type": ["reanalysis"],
        "day": [f"{d:02d}" for d in range(1, 32)],  # days a month does not have are ignored
        "time": [f"{h:02d}:00" for h in hours],
        "area": [area.lat_max, area.lon_min, area.lat_min, area.lon_max],  # N, W, S, E
        "data_format": "grib",
        "download_format": "unarchived",
    }


def _split_by_dataset(variables: Sequence[str]) -> list[tuple[str, str, list[str], dict]]:
    """(kind, dataset, CDS variable names, extra request keys) of the datasets `variables` (ARCO names) come from."""
    pl = [v for v in variables if v in PL_NAMES.values()]
    sl = [v for v in variables if v in SL_NAMES.values()]
    unknown = set(variables) - set(pl) - set(sl)
    if unknown:
        raise ValueError(f"no CDS name for {sorted(unknown)}: add them to `PL_NAMES` / `SL_NAMES`")
    out = [
        ("pl", PL_DATASET, pl, {"pressure_level": [str(p) for p in LEVELS]}),
        ("sl", SL_DATASET, [SL_REQUEST.get(v, v) for v in sl], {}),
    ]
    return [d for d in out if d[2]]


def _choose_groups(
    n_vars: int, fields_per_var_month: int, limit: int, months_of_year: dict[int, tuple[int, ...]]
) -> list[tuple[list[int], list[int]]]:
    """The (years, months) of the requests of one dataset with the fewest requests within `limit` fields.

    Tries every number of years (`ny`) and months (`nm`) per request: years are grouped by `ny` in order, a group of
    years with the same months is split into runs of `nm` of these months, a group with other months per year into
    runs per year. Ties: fewer years per request (shorter files).
    """
    best = None
    for ny in range(1, len(months_of_year) + 1):
        for nm in range(1, 13):
            if fields_per_var_month * ny * nm > limit:
                continue
            groups = _group_months(months_of_year, ny, nm)
            n = sum(-(-n_vars // (limit // (fields_per_var_month * len(ys) * len(ms)))) for ys, ms in groups)
            if best is None or (n, ny) < best[0]:
                best = ((n, ny), groups)
    if best is None:
        raise ValueError(f"one variable-month ({fields_per_var_month} fields) exceeds the CDS limit of {limit}")
    return best[1]


def _group_months(months_of_year: dict[int, tuple[int, ...]], ny: int, nm: int) -> list[tuple[list[int], list[int]]]:
    """(years, months) of the requests: groups of `ny` years, runs of `nm` months (`_choose_groups`)."""
    years = sorted(months_of_year)
    out = []
    for i in range(0, len(years), ny):
        ys = years[i : i + ny]
        if len({months_of_year[y] for y in ys}) == 1:
            ms = months_of_year[ys[0]]
            out += [(ys, list(ms[k : k + nm])) for k in range(0, len(ms), nm)]
        else:  # the cross product would request months a year does not have: one request per year
            out += [
                ([y], list(months_of_year[y][k : k + nm])) for y in ys for k in range(0, len(months_of_year[y]), nm)
            ]
    return out


def _retrieve(jobs: Jobs, root: pathlib.Path, *, max_active: int = 8, poll: float = 30, retries: int = 3) -> None:
    """Submit the `jobs` to the CDS in their order and download each result to `root / <file name>`.

    At most `max_active` requests are queued or running at a time. Files that exist are skipped. The request ids are
    kept in `root / requests.json`, so a restart downloads the requests already submitted instead of queueing them
    again. Failed requests are submitted again and failed downloads (e.g. a truncated file) tried again, up to
    `retries` times each. Progress: a bar over the requests, with the numbers running and queued at the CDS and the
    GB downloaded.
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

    def count_failure(name: str, reason: str) -> None:
        failures[name] = failures.get(name, 0) + 1
        if failures[name] > retries:
            raise RuntimeError(f"{name}: failed {failures[name]} times, last: {reason}")
        bar.write(f"{name}: {reason}, trying again")

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
                try:
                    remote.download(str(tmp))
                except (requests.RequestException, OSError, AssertionError) as e:  # multiurl: size mismatch
                    tmp.unlink(missing_ok=True)
                    count_failure(name, f"download of {ids[name]} failed ({e})")  # the result stays on the CDS
                    continue
                tmp.rename(target)
                n_bytes += target.stat().st_size
                todo.remove(name)
                del ids[name]
                bar.update()
            elif status in counts:
                counts[status] += 1
            else:
                count_failure(name, f"request {ids[name]} {status}")
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


def open_grib(files: dict[pathlib.Path, str], times) -> xr.Dataset:
    """GRIB files from the CDS ({path: dataset}, requests as in `make_jobs`) at `times`, in the ARCO layout: ARCO
    names and units, dims (time, level, latitude, longitude), levels ascending, latitude descending, longitude 0..360."""
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

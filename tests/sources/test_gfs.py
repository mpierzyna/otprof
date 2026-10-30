"""otprof.sources.gfs on GFS-like files served from memory: small real GRIB2 messages and `.idx` inventories."""

import numpy as np
import pandas as pd
import pytest

from otprof import schema
from otprof.models.sqf_h.inputs import make_inputs
from otprof.sources import Area, Points, gfs

from tests.sources.grib import make_message

CYCLE = pd.Timestamp("2024-06-01 06:00")
TIMES = [CYCLE + pd.Timedelta(hours=h) for h in (3, 6)]
POINTS = Points(lat=[52.76, 52.98], lon=[3.49, 3.03], names=["a", "b"])
GRIDS = {"0p25": dict(lat0=53.0, lon0=3.0, step=0.25, shape=(3, 4)), "sflux": dict(lat0=53.25, lon0=2.75, step=0.5, shape=(3, 3))}  # fmt: skip
NEAREST = {"0p25": [6, 0], "sflux": [4, 4]}  # flat grid index of the point nearest to each of POINTS
BASE = {"tk": 250.0, "u": 5.0, "v": -3.0, "p_sfc": 101_000.0, "z_sfc": 10.0, "lsm": 0.0, "tk2": 285.0, "u10": 3.0,
        "v10": 2.0, "prmsl": 101_300.0, "hfx": 50.0, "lh": 100.0, "ust": 0.0, "blh": 500.0}  # fmt: skip
GRIB_KEYS = {name: key for key, name in gfs.FIELDS.items()}


def get_base(name: str, level: int | None, lead: int) -> float:
    """Plausible field value (heights of the standard atmosphere for `z`); grows with the lead time."""
    if name == "z":
        return 44_330.8 * (1 - (level / 1013.25) ** 0.1903) + lead
    return BASE[name] + (level or 0) / 1000 + lead / 100


def make_field(name: str, level: int | None, lead: int, grid: str) -> bytes:
    discipline, category, number, type_of_level, height = GRIB_KEYS[name]
    keys = {"discipline": discipline, "parameterCategory": category, "parameterNumber": number,
            "typeOfLevel": type_of_level, "level": level or height or 0}  # fmt: skip
    g = GRIDS[grid]
    values = get_base(name, level, lead) + 0.001 * np.arange(np.prod(g["shape"])).reshape(g["shape"])
    return make_message(values, lat0=g["lat0"], lon0=g["lon0"], step=g["step"], keys=keys)


def get_files(lead: int, drop: str | None = None) -> dict[str, tuple[str, bytes]]:
    """url -> (idx text, file content) of the three files of `lead`, with a duplicate and an unused message mixed in."""
    when = f"{lead} hour fcst"
    entries = {"pgrb2": [], "pgrb2b": [], "sflux": []}  # file -> [(idx key, name, level)]
    for (file, var, level), name in gfs.IDX_SURFACE.items():
        entries[file].append((f"{var}:{level}:{when}", name, None))
    for var, name in gfs.IDX_PROFILES.items():
        for file, levels in [("pgrb2", gfs.LEVELS_PGRB2), ("pgrb2b", gfs.LEVELS_PGRB2B)]:
            entries[file] += [(f"{var}:{p} mb:{when}", name, p) for p in levels]
    entries["pgrb2"] += [("RH:500 mb:" + when, "u", 400), ("TMP:500 mb:" + when, "u", 400)]  # unused, duplicate key
    files = {}
    for file, items in entries.items():
        idx, blob = [], b""
        for i, (key, name, level) in enumerate(items):
            if key.split(":")[0] == drop:
                continue
            idx.append(f"{i + 1}:{len(blob)}:d={CYCLE:%Y%m%d%H}:{key}:")
            blob += make_field(name, level, lead, "sflux" if file == "sflux" else "0p25")
        files[gfs.get_url(CYCLE, lead, file)] = ("\n".join(idx) + "\n", blob)
    return files


@pytest.fixture
def served(monkeypatch):
    def serve(drop=None):
        files = {url: f for lead in (3, 6) for url, f in get_files(lead, drop).items()}

        def get_bytes(url, byte_range=None):
            if url.endswith(".idx"):
                return files[url.removesuffix(".idx")][0].encode()
            start, end = byte_range.split("-")
            return files[url][1][int(start) : int(end) + 1 if end else None]

        monkeypatch.setattr(gfs, "get_bytes", get_bytes)

    return serve


def test_fetch_points(served):
    served()
    ds = gfs.fetch(CYCLE, TIMES, POINTS, n_threads=2)
    assert ds["lead"].values.tolist() == [3, 6] and (ds["cycle"] == CYCLE).all()
    assert ds["level"].values.tolist() == list(schema.P_LEVELS_HPA) and ds["z"].dims == ("time", "level", "loc")
    assert ds["name"].values.tolist() == ["a", "b"] and ds["lat"].values.tolist() == [52.76, 52.98]
    for (file, _, _), name in gfs.IDX_SURFACE.items():
        grid = "sflux" if file == "sflux" else "0p25"
        expected = get_base(name, None, 6) + 0.001 * np.array(NEAREST[grid])
        np.testing.assert_allclose(ds[name].isel(time=1), expected, atol=1e-4)
    for p in [1000, 875, 500, 125, 20]:  # pgrb2 and pgrb2b; 500: the first of two messages with the same key
        expected = get_base("tk", p, 3) + 0.001 * np.array(NEAREST["0p25"])
        np.testing.assert_allclose(ds["tk"].sel(level=p).isel(time=0), expected, atol=1e-4)


def test_fetch_area(served):
    served()
    ds = gfs.fetch(CYCLE, TIMES[:1], Area(lon_min=3.0, lon_max=3.5, lat_min=52.5, lat_max=53.0))
    assert ds["tk"].dims == ("time", "level", "lat", "lon") and ds["p_sfc"].dims == ("time", "lat", "lon")
    assert ds["lat"].values.tolist() == [53.0, 52.75, 52.5] and ds["lon"].values.tolist() == [3.0, 3.25, 3.5]
    point = dict(lat=52.75, lon=3.25)  # 0.25 deg grid: row 1, column 1; sflux: the nearest, row 1, column 1 of 3 x 3
    np.testing.assert_allclose(ds["p_sfc"].sel(point)[0], get_base("p_sfc", None, 3) + 0.005, atol=1e-4)
    np.testing.assert_allclose(ds["hfx"].sel(point)[0], get_base("hfx", None, 3) + 0.004, atol=1e-4)


def test_fetch_missing_field(served):
    served(drop="FRICV")
    with pytest.raises(ValueError, match="FRICV"):
        gfs.fetch(CYCLE, TIMES, POINTS)


@pytest.mark.parametrize("missing, available", [(None, True), ("sflux", False), ("pgrb2b", False)])
def test_is_cycle_available(monkeypatch, missing, available):
    asked = []

    def url_exists(url):
        asked.append(url)
        return not (missing and f".{missing}" in url)

    monkeypatch.setattr(gfs, "url_exists", url_exists)
    assert gfs.is_cycle_available(CYCLE, last_lead=120) is available
    assert all(url.endswith("f120.idx") or url.endswith("f120.grib2.idx") for url in asked)


def test_fetch_times_after_the_cycle():
    with pytest.raises(ValueError, match="after the cycle"):
        gfs.fetch(CYCLE, [CYCLE - pd.Timedelta(hours=3)], POINTS)


@pytest.mark.parametrize("where", [POINTS, Area(lon_min=3.0, lon_max=3.5, lat_min=52.5, lat_max=53.0)])
def test_load_to_model_inputs(served, where):
    served()
    X = gfs.load(CYCLE, TIMES, where)
    dims = ("time", "loc", "bottom_top") if isinstance(where, Points) else ("time", "lat", "lon", "bottom_top")
    assert X["z"].dims == dims and X["p_level"].values.tolist() == list(schema.P_LEVELS_HPA)
    assert (X["ust"] >= gfs.UST_MIN).all() and "time" not in X["z_sfc_src"].dims
    inputs = make_inputs(X)
    assert all(np.isfinite(inputs[v]).all() for v in inputs.data_vars)

"""otprof.sources.gfs on GFS-like files served from memory: small real GRIB2 messages and `.idx` inventories."""

import numpy as np
import pandas as pd
import pytest

from otprof import schema
from otprof.models.sqf_h.inputs import make_inputs
from otprof.sources import gfs

from tests.sources.grib import make_message

CYCLE, LEAD = pd.Timestamp("2024-06-01 06:00"), 3
LAT, LON = np.array([52.76, 52.98]), np.array([3.49, 3.03])
GRIDS = {
    "pgrb2": dict(lat0=53.0, lon0=3.0, step=0.25, shape=(3, 4)),
    "sflux": dict(lat0=53.25, lon0=2.75, step=0.5, shape=(3, 3)),
}
GRIDS["pgrb2b"] = GRIDS["pgrb2"]
NEAREST = {"pgrb2": [6, 0], "pgrb2b": [6, 0], "sflux": [4, 4]}  # flat grid index of the point nearest to each location
BASE = {"tk": 250.0, "u": 5.0, "v": -3.0, "p_sfc": 101_000.0, "z_sfc": 10.0, "lsm": 0.0, "tk2": 285.0, "u10": 3.0,
        "v10": 2.0, "prmsl": 101_300.0, "hfx": 50.0, "lh": 100.0, "ust": 0.0, "blh": 500.0}  # fmt: skip


def get_base(name: str, level: int | None) -> float:
    """Plausible field value; heights of the standard atmosphere for `z`."""
    return 44_330.8 * (1 - (level / 1013.25) ** 0.1903) if name == "z" else BASE[name] + (level or 0) / 1000


def get_files(drop: str | None = None) -> dict[str, tuple[str, bytes]]:
    """url -> (idx text, file content) of the three files, with a duplicate and an unused message mixed in."""
    entries = {f: [] for f in GRIDS}  # file -> [(idx key, name, level)]
    for (file, var, level), name in gfs.SURFACE.items():
        entries[file].append((f"{var}:{level}:{LEAD} hour fcst", name, None))
    for var, name in gfs.PROFILES.items():
        for file, levels in [("pgrb2", gfs.LEVELS_PGRB2), ("pgrb2b", gfs.LEVELS_PGRB2B)]:
            entries[file] += [(f"{var}:{p} mb:{LEAD} hour fcst", name, p) for p in levels]
    entries["pgrb2"] += [
        ("RH:500 mb:3 hour fcst", "tk", 500),
        ("TMP:500 mb:3 hour fcst", "u", 500),
    ]  # unused, duplicate
    files = {}
    for file, items in entries.items():
        grid = GRIDS[file]
        k = np.arange(np.prod(grid["shape"])).reshape(grid["shape"])
        idx, blob = [], b""
        for i, (key, name, level) in enumerate(items):
            if key.split(":")[0] == drop:
                continue
            values = get_base(name, level) + 0.001 * k
            msg = make_message(values, lat0=grid["lat0"], lon0=grid["lon0"], step=grid["step"], keys={"discipline": 0})
            idx.append(f"{i + 1}:{len(blob)}:d={CYCLE:%Y%m%d%H}:{key}:")
            blob += msg
        files[gfs.get_url(CYCLE, LEAD, file)] = ("\n".join(idx) + "\n", blob)
    return files


@pytest.fixture
def served(monkeypatch):
    def serve(drop=None):
        files = get_files(drop)

        def download(url, byte_range=None):
            if url.endswith(".idx"):
                return files[url.removesuffix(".idx")][0].encode()
            start, end = byte_range.split("-")
            blob = files[url][1]
            return blob[int(start) : int(end) + 1 if end else None]

        monkeypatch.setattr(gfs, "download", download)

    return serve


def test_fetch_values_at_the_nearest_grid_points(served):
    served()
    ds = gfs.fetch(CYCLE, LEAD, LAT, LON, n_threads=2)
    assert ds["time"].values[0] == CYCLE + pd.Timedelta(hours=LEAD) and int(ds["lead"]) == LEAD
    assert ds["level"].values.tolist() == list(schema.P_LEVELS_HPA) and ds["z"].dims == ("time", "level", "loc")
    for (file, _, _), name in gfs.SURFACE.items():
        np.testing.assert_allclose(ds[name][0], get_base(name, None) + 0.001 * np.array(NEAREST[file]), atol=1e-4)
    for p in [1000, 875, 125, 20]:  # pgrb2 and pgrb2b levels
        np.testing.assert_allclose(
            ds["tk"].sel(level=p)[0], get_base("tk", p) + 0.001 * np.array(NEAREST["pgrb2"]), atol=1e-4
        )
    # the first of two messages with the same key
    np.testing.assert_allclose(ds["tk"].sel(level=500)[0], get_base("tk", 500) + 0.001 * np.array([6, 0]), atol=1e-4)


def test_fetch_missing_field(served):
    served(drop="FRICV")
    with pytest.raises(ValueError, match="FRICV"):
        gfs.fetch(CYCLE, LEAD, LAT, LON)


def test_standardize_to_model_inputs(served):
    served()
    ds = gfs.fetch(CYCLE, LEAD, LAT, LON)
    X = gfs.standardize(ds)
    assert X["p_level"].values.tolist() == list(schema.P_LEVELS_HPA) and X["z"].dims == ("time", "loc", "bottom_top")
    assert X["z_sfc_src"].dims == ("loc",) and (X["ust"] >= gfs.UST_MIN).all()
    np.testing.assert_allclose(X["slp"], ds["prmsl"] / 100)
    inputs = make_inputs(X)
    assert all(np.isfinite(inputs[v]).all() for v in inputs.data_vars)

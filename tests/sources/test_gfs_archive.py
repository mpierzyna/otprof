"""otprof.sources.gfs_archive on a GFS-like archive file of small real GRIB2 messages."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otprof import schema
from otprof.sources import gfs_archive

from tests.sources.grib import make_message

LAT, LON = np.array([52.76, 52.98]), np.array([3.49, 3.03])
GRID = dict(lat0=53.0, lon0=3.0, step=0.25)
NEAREST = np.array([6, 0])  # flat index of the grid point (3 x 4 grid) nearest to each location
BASE = {"tk": 250.0, "u": 5.0, "v": -3.0, "p_sfc": 101_000.0, "z_sfc": 10.0, "lsm": 0.0, "tk2": 285.0, "u10": 3.0,
        "v10": 2.0, "prmsl": 101_300.0, "blh": 500.0, "hfx": 50.0, "lh": 100.0, "taux": 0.1, "tauy": -0.05}  # fmt: skip


def get_base(name: str, level: int | None) -> float:
    return 44_330.8 * (1 - (level / 1013.25) ** 0.1903) if name == "z" else BASE[name] + (level or 0) / 1000


def write_file(path) -> None:
    """All `FIELDS` plus messages the reader skips (an unused level and parameter)."""
    k = np.arange(12.0).reshape(3, 4)
    messages = []
    for (discipline, category, number, type_of_level, level), name in gfs_archive.FIELDS.items():
        for lev in gfs_archive.LEVELS + (10,) if level is None else [level]:
            keys = {"discipline": discipline, "parameterCategory": category, "parameterNumber": number,
                    "typeOfLevel": type_of_level, "level": lev}  # fmt: skip
            values = get_base(name, lev if level is None else None) + 0.001 * k
            messages.append(make_message(values, **GRID, keys=keys))
    keys = {"discipline": 0, "parameterCategory": 1, "parameterNumber": 1, "typeOfLevel": "surface", "level": 0}
    messages.append(make_message(np.zeros((3, 4)), **GRID, keys=keys))
    path.write_bytes(b"".join(messages))


def test_read_grib(tmp_path):
    write_file(tmp_path / "f.grib2")
    out = gfs_archive.read_grib(tmp_path / "f.grib2", LAT, LON)
    assert set(out) == set(gfs_archive.FIELDS.values()) and out["z"].shape == (len(gfs_archive.LEVELS), 2)
    np.testing.assert_allclose(out["p_sfc"], 101_000 + 0.001 * NEAREST, atol=1e-4)
    np.testing.assert_allclose(
        out["tk"][gfs_archive.LEVELS.index(850)], get_base("tk", 850) + 0.001 * NEAREST, atol=1e-4
    )


@pytest.mark.parametrize(
    "t, cycle, lead",
    [("2017-03-26 03:00", "2017-03-26 00:00", 3), ("2017-03-26 06:00", "2017-03-26 00:00", 6),
     ("2017-03-26 00:00", "2017-03-25 18:00", 6)],
)  # fmt: skip
def test_get_cycle_lead(t, cycle, lead):
    assert gfs_archive.get_cycle_lead(pd.Timestamp(t)) == (pd.Timestamp(cycle), lead)


def test_get_cycle_lead_needs_3_hourly_times():
    with pytest.raises(ValueError, match="3-hourly"):
        gfs_archive.get_cycle_lead(pd.Timestamp("2017-03-26 04:00"))


@pytest.fixture
def archive(monkeypatch, tmp_path):
    """fetch_file reads the test file; the means since the cycle grow with the lead time (+3 h: BASE, +6 h: 2 BASE)."""
    write_file(tmp_path / "f.grib2")

    def fetch_file(cycle, lead, lat, lon, tmp_dir=None):
        values = gfs_archive.read_grib(tmp_path / "f.grib2", lat, lon)
        for name in gfs_archive.MEANS:
            values[name] = values[name] * lead / 3
        return xr.Dataset(
            {n: (("level", "loc") if v.ndim == 2 else ("loc",), v.astype(np.float32)) for n, v in values.items()},
            coords={"level": list(gfs_archive.LEVELS), "loc": np.arange(len(lat)), "lat": ("loc", lat), "lon": ("loc", lon)},
        )  # fmt: skip

    monkeypatch.setattr(gfs_archive, "fetch_file", fetch_file)


def test_fetch_deaverages_the_fluxes(archive):
    at3 = gfs_archive.fetch("2017-03-26 03:00", LAT, LON)
    at6 = gfs_archive.fetch("2017-03-26 06:00", LAT, LON)
    assert int(at6["lead"]) == 6 and at6["time"].values[0] == pd.Timestamp("2017-03-26 06:00")
    for name in gfs_archive.MEANS:  # 3-6 h mean = 2 * (0-6 h mean) - (0-3 h mean) = 3 BASE here
        np.testing.assert_allclose(at6[name], 3 * at3[name], rtol=1e-5)


def test_standardize(archive):
    ds = xr.concat([gfs_archive.fetch(t, LAT, LON) for t in ["2017-03-26 03:00", "2017-03-26 06:00"]], "time")
    X = gfs_archive.standardize(ds)
    assert X["p_level"].values.tolist() == list(schema.P_LEVELS_HPA)
    native = X["z"].swap_dims(bottom_top="p_level")
    np.testing.assert_allclose(
        native.sel(p_level=list(gfs_archive.LEVELS)), ds["z"].transpose("time", "loc", "level"), rtol=1e-6
    )
    # a missing level: linear in ln p between its neighbours
    w = np.log(900 / 875) / np.log(900 / 850)
    expected = (1 - w) * native.sel(p_level=900) + w * native.sel(p_level=850)
    np.testing.assert_allclose(native.sel(p_level=875), expected, rtol=1e-6)
    rho = ds["p_sfc"] / (gfs_archive.R_D * ds["tk2"])
    np.testing.assert_allclose(X["ust"], np.sqrt(np.hypot(ds["taux"], ds["tauy"]) / rho), rtol=1e-6)

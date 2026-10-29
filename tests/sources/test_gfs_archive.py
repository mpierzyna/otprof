"""otprof.sources.gfs_archive on a GFS-like archive file of small real GRIB2 messages."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otprof import schema
from otprof.sources import Area, Points, gfs, gfs_archive, select

from tests.sources.grib import make_message

POINTS = Points(lat=[52.76, 52.98], lon=[3.49, 3.03])
GRID = dict(lat0=53.0, lon0=3.0, step=0.25)
NEAREST = np.array([6, 0])  # flat index of the grid point (3 x 4 grid) nearest to each of POINTS
BASE = {"tk": 250.0, "u": 5.0, "v": -3.0, "p_sfc": 101_000.0, "z_sfc": 10.0, "lsm": 0.0, "tk2": 285.0, "u10": 3.0,
        "v10": 2.0, "prmsl": 101_300.0, "blh": 500.0, "hfx": 50.0, "lh": 100.0, "taux": 0.1, "tauy": -0.05}  # fmt: skip
GRIB_KEYS = {name: key for key, name in gfs.FIELDS.items()}


def get_base(name: str, level: int | None) -> float:
    return 44_330.8 * (1 - (level / 1013.25) ** 0.1903) if name == "z" else BASE[name] + (level or 0) / 1000


def write_file(path) -> None:
    """The archive fields plus messages the reader must skip: another level, other NCEP parameters without short name
    (which collide in cfgrib when opened together), an instantaneous flux next to the mean."""
    k = np.arange(12.0).reshape(3, 4)
    messages = []
    for name in gfs_archive.NAMES:
        discipline, category, number, type_of_level, height = GRIB_KEYS[name]
        step_type = "avg" if name in gfs_archive.MEANS else "instant"
        for level in gfs_archive.LEVELS + (10,) if type_of_level == "isobaricInhPa" else [height or 0]:
            keys = {"discipline": discipline, "parameterCategory": category, "parameterNumber": number,
                    "typeOfLevel": type_of_level, "level": level, "stepType": step_type}  # fmt: skip
            base = get_base(name, level if type_of_level == "isobaricInhPa" else None)
            messages.append(make_message(base + 0.001 * k, **GRID, keys=keys))
    for category, number in [(3, 197), (1, 196), (0, 11)]:  # NCEP parameters; instantaneous sensible heat flux
        keys = {"discipline": 0, "parameterCategory": category, "parameterNumber": number, "typeOfLevel": "surface",
                "level": 0, "stepType": "instant"}  # fmt: skip
        messages.append(make_message(np.full((3, 4), -999.0), **GRID, keys=keys))
    path.write_bytes(b"".join(messages))


def test_read_grib(tmp_path):
    write_file(tmp_path / "f.grib2")
    ds = gfs.read_grib(tmp_path / "f.grib2", means=gfs_archive.MEANS)
    assert set(gfs_archive.NAMES) <= set(ds) and ds["z"].dims == ("level", "lat", "lon")
    assert set(gfs_archive.LEVELS) < set(ds["level"].values.tolist())  # 10 hPa too: fetch selects LEVELS
    np.testing.assert_allclose(ds["hfx"].values.ravel()[NEAREST], 50.0 + 0.001 * NEAREST, atol=1e-4)  # the mean
    np.testing.assert_allclose(ds["blh"].values.ravel()[NEAREST], 500.0 + 0.001 * NEAREST, atol=1e-4)


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

    def fetch_file(cycle, lead, where, tmp_dir=None):
        ds = gfs.read_grib(tmp_path / "f.grib2", means=gfs_archive.MEANS)[gfs_archive.NAMES]
        ds = ds.assign({v: ds[v] * lead / 3 for v in gfs_archive.MEANS})
        return select(ds.sel(level=list(gfs_archive.LEVELS)), where)

    monkeypatch.setattr(gfs_archive, "fetch_file", fetch_file)


def test_fetch_deaverages_the_fluxes(archive):
    at3 = gfs_archive.fetch(["2017-03-26 03:00"], POINTS)
    at6 = gfs_archive.fetch(["2017-03-26 06:00"], POINTS)
    assert at6["lead"].values.tolist() == [6] and at6["time"].values[0] == pd.Timestamp("2017-03-26 06:00")
    for name in gfs_archive.MEANS:  # 3-6 h mean = 2 * (0-6 h mean) - (0-3 h mean) = 3 BASE here
        np.testing.assert_allclose(at6[name], 3 * at3[name], rtol=1e-5)


@pytest.mark.parametrize("where", [POINTS, Area(lon_min=3.0, lon_max=3.5, lat_min=52.5, lat_max=53.0)])
def test_standardize(archive, where):
    ds = gfs_archive.fetch(["2017-03-26 03:00", "2017-03-26 06:00"], where)
    X = gfs_archive.standardize(ds)
    dims = ("time", "loc", "bottom_top") if isinstance(where, Points) else ("time", "lat", "lon", "bottom_top")
    assert X["z"].dims == dims and X["p_level"].values.tolist() == list(schema.P_LEVELS_HPA)
    native = X["z"].swap_dims(bottom_top="p_level")
    np.testing.assert_allclose(native.sel(p_level=list(gfs_archive.LEVELS)), ds["z"].transpose(*dims[:-1], "level"))
    # a missing level: linear in ln p between its neighbours
    w = np.log(900 / 875) / np.log(900 / 850)
    np.testing.assert_allclose(native.sel(p_level=875), (1 - w) * native.sel(p_level=900) + w * native.sel(p_level=850))
    rho = ds["p_sfc"] / (gfs_archive.R_D * ds["tk2"])
    np.testing.assert_allclose(X["ust"], np.sqrt(np.hypot(ds["taux"], ds["tauy"]) / rho), rtol=1e-6)


def test_load(archive):
    xr.testing.assert_identical(
        gfs_archive.load(["2017-03-26 03:00"], POINTS),
        gfs_archive.standardize(gfs_archive.fetch(["2017-03-26 03:00"], POINTS)),
    )

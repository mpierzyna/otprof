"""otprof.sources.era5_cds without the CDS: request planning, the plan in `root/jobs.json`, reading checks."""

import json
import pathlib

import pandas as pd
import pytest

from otprof.sources import Area, Points, era5_cds

NL = Area(lon_min=2, lon_max=9, lat_min=49, lat_max=55)


@pytest.fixture
def no_cds(monkeypatch):
    """`download` plans and records, but submits nothing."""
    submitted = []
    monkeypatch.setattr(era5_cds, "_retrieve", lambda jobs, root, **kw: submitted.append(jobs))
    return submitted


def test_requests_within_the_cost_limits(no_cds, tmp_path):
    times = pd.date_range("2017-01-01", "2017-03-31 23:00", freq="h")
    era5_cds.download(times, NL, tmp_path)
    (jobs,) = no_cds
    for dataset, request in jobs.values():
        n_levels = len(request.get("pressure_level", [0]))
        fields = len(request["variable"]) * n_levels * len(request["year"]) * len(request["month"]) * 31 * 24
        assert fields <= era5_cds.COST_LIMIT[dataset]
        assert request["area"] == [55, 2, 49, 9] and len(request["time"]) == 24
    months = {(y, m) for _, r in jobs.values() for y in r["year"] for m in r["month"]}
    assert months == {("2017", "01"), ("2017", "02"), ("2017", "03")}
    assert json.loads((tmp_path / "jobs.json").read_text()) == json.loads(json.dumps(jobs))


def test_hours_of_the_times(no_cds, tmp_path):
    era5_cds.download(pd.date_range("2017-01-01", periods=8, freq="6h"), NL, tmp_path)
    assert {tuple(r["time"]) for _, r in no_cds[0].values()} == {("00:00", "06:00", "12:00", "18:00")}


def test_one_root_per_area(no_cds, tmp_path):
    times = pd.date_range("2017-01-01", periods=4, freq="6h")
    era5_cds.download(times, NL, tmp_path)
    era5_cds.download(times, NL, tmp_path)  # resume: same plan
    with pytest.raises(ValueError, match="one root per area"):
        era5_cds.download(times, Area(lon_min=0, lon_max=1, lat_min=0, lat_max=1), tmp_path)


def test_fetch_checks(no_cds, tmp_path):
    with pytest.raises(ValueError, match="download first"):
        era5_cds.fetch(pd.DatetimeIndex(["2017-01-01"]), NL, tmp_path)
    era5_cds.download(pd.date_range("2017-01-01", periods=4, freq="6h"), NL, tmp_path)
    with pytest.raises(ValueError, match="missing"):  # planned, not downloaded
        era5_cds.fetch(pd.DatetimeIndex(["2017-01-01"]), Points(lat=[52.0], lon=[5.0]), tmp_path)
    with pytest.raises(ValueError, match="not downloaded"):
        era5_cds.fetch(pd.DatetimeIndex(["2017-05-01"]), NL, tmp_path)


def test_full_range_plan():
    jobs = era5_cds.make_jobs(pd.date_range("2005-01-01", "2026-06-30 23:00", freq="h"), NL)
    # 903 pressure-level requests (1 variable x 2 months), 29 surface (9 variables x 3 years x 6 months; 2026: 1)
    assert len(jobs) == 932
    assert jobs["sl/2005-2007_01-06_1.grib"].request["year"] == ["2005", "2006", "2007"]
    assert (
        jobs["sl/2026_01-06_0.grib"].request["variable"]
        == jobs["sl/2005-2007_01-06_0.grib"].request["variable"]
        + (jobs["sl/2005-2007_01-06_1.grib"].request["variable"])
    )
    assert all(isinstance(j, era5_cds.Job) for j in jobs.values())


def months_3h(*months: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([t for m in months for t in pd.date_range(m, pd.Period(m, "M").end_time, freq="3h")])


def test_blocks_in_order_and_packed_per_block():
    hourly = pd.date_range("2014-03-01", "2014-03-31 23:00", freq="h")
    blocks = [months_3h("2023-10", "2023-11", "2023-12"), hourly, months_3h("2014-01", "2014-02")]
    jobs = era5_cds.make_jobs(blocks, NL)
    assert list(jobs) == [
        *(f"pl/2023_10-12_{i}.grib" for i in range(4)),  # 2, 2, 2, 1 variables x 3 months
        "sl/2023_10-12_0.grib",
        *(f"pl/2014_03_{i}.grib" for i in range(4)),  # hourly: 2, 2, 2, 1 variables x 1 month
        "sl/2014_03_0.grib",
        *(f"pl/2014_01-02_{i}.grib" for i in range(3)),  # 3, 3, 1 variables x 2 months
        "sl/2014_01-02_0.grib",
    ]
    assert jobs["pl/2014_03_0.grib"].request["time"] == [f"{h:02d}:00" for h in range(24)]
    assert jobs["pl/2014_01-02_0.grib"].request["time"] == [f"{h:02d}:00" for h in range(0, 24, 3)]


def test_months_not_consecutive():
    jobs = era5_cds.make_jobs(months_3h("2010-07", "2010-09"), NL)
    assert jobs["sl/2010_07+09_0.grib"].request["month"] == ["07", "09"]
    assert len([n for n in jobs if n.startswith("pl/")]) == 3


def test_blocks_must_agree():
    month = pd.date_range("2014-03-01", "2014-03-31 23:00", freq="h")
    assert era5_cds.make_jobs([month, month], NL) == era5_cds.make_jobs(month, NL)
    with pytest.raises(ValueError, match="two blocks"):
        era5_cds.make_jobs([month, month[::3]], NL)


def test_download_retried(monkeypatch, tmp_path):
    """A failed download (e.g. truncated) is tried again; the request is not submitted again."""
    import ecmwf.datastores

    downloads, submitted = [], []

    class Remote:
        status = "successful"

        def download(self, target):
            downloads.append(target)
            if len(downloads) == 1:
                raise AssertionError("File size mismatch 0 bytes instead of 84854912")
            pathlib.Path(target).write_bytes(b"GRIB")

    class Client:
        def __init__(self, **kwargs):
            pass

        def submit(self, dataset, request):
            submitted.append(dataset)
            return type("R", (), {"request_id": "id"})()

        def get_remote(self, request_id):
            return Remote()

    monkeypatch.setattr(ecmwf.datastores, "Client", Client)
    jobs = era5_cds.make_jobs(pd.date_range("2017-01-01", periods=4, freq="6h"), NL, variables=["2m_temperature"])
    era5_cds._retrieve(jobs, tmp_path, poll=0)
    assert len(submitted) == 1 and len(downloads) == 2
    assert (tmp_path / "sl/2017_01_0.grib").read_bytes() == b"GRIB"

"""otprof.sources.era5_cds without the CDS: request planning, the plan in `root/jobs.json`, reading checks."""

import json

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
        era5_cds.fetch(["2017-01-01"], NL, tmp_path)
    era5_cds.download(pd.date_range("2017-01-01", periods=4, freq="6h"), NL, tmp_path)
    with pytest.raises(ValueError, match="missing"):  # planned, not downloaded
        era5_cds.fetch(["2017-01-01"], Points(lat=[52.0], lon=[5.0]), tmp_path)
    with pytest.raises(ValueError, match="not downloaded"):
        era5_cds.fetch(["2017-05-01"], NL, tmp_path)

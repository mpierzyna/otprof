"""The common interface of the download sources: `Area`, `Points`, `select`, `select_like`."""

import numpy as np
import pydantic
import pytest
import xarray as xr

from otprof.sources import Area, Points, select, select_like


def make_grid(lon: np.ndarray) -> xr.Dataset:
    """A global-style grid: latitudes north to south, longitudes as given; value = 1000 * row + column."""
    lat = np.arange(60.0, 39.0, -1.0)
    values = 1000 * np.arange(len(lat))[:, None] + np.arange(len(lon))[None, :]
    return xr.Dataset({"x": (("lat", "lon"), values)}, coords={"lat": lat, "lon": lon})


def test_area_and_points():
    assert Area(lon_min=340, lon_max=350, lat_min=25, lat_max=31).lon_min == -20.0
    with pytest.raises(pydantic.ValidationError, match="180"):
        Area(lon_min=170, lon_max=-170, lat_min=0, lat_max=1)
    with pytest.raises(pydantic.ValidationError, match="lat_min"):
        Area(lon_min=0, lon_max=1, lat_min=2, lat_max=1)
    p = Points.from_dict({"a": (52.0, 355.0), "b": (28.8, -17.9)})
    assert p.lon == (-5.0, -17.9) and p.names == ("a", "b")
    with pytest.raises(pydantic.ValidationError, match="same length"):
        Points(lat=[1.0], lon=[1.0, 2.0])


@pytest.mark.parametrize("lon", [np.arange(0.0, 360.0, 1.0), np.arange(-180.0, 180.0, 1.0)])
def test_select_area_across_0_deg(lon):
    """North to south, west to east, longitudes -180..180, whatever the convention of the grid."""
    out = select(make_grid(lon), Area(lon_min=-2, lon_max=2, lat_min=48, lat_max=50))
    assert out["lat"].values.tolist() == [50.0, 49.0, 48.0] and out["lon"].values.tolist() == [-2, -1, 0, 1, 2]
    row = int(np.flatnonzero(make_grid(lon)["lat"].values == 50.0)[0])
    col = int(np.flatnonzero(lon % 360 == 358)[0])
    assert int(out["x"].sel(lat=50.0, lon=-2)) == 1000 * row + col


def test_select_points():
    grid = make_grid(np.arange(0.0, 360.0, 1.0))
    out = select(grid, Points(lat=[49.6, 50.2], lon=[-0.4, 359.9], names=["a", "b"]))
    assert out["x"].dims == ("loc",) and out["name"].values.tolist() == ["a", "b"]
    assert out["lat"].values.tolist() == [49.6, 50.2] and out["lon"].values.tolist() == pytest.approx(
        [-0.4, -0.1]
    )  # points
    assert out["x"].values.tolist() == [10 * 1000 + 0, 10 * 1000 + 0]  # nearest: 50 N, 0 E (across 0 deg)


def test_select_like():
    """Another grid at the points / grid points of a selection (GFS: sflux on the 0.25 deg grid)."""
    coarse = make_grid(np.arange(0.0, 360.0, 1.0))
    fine = xr.Dataset(coords={"lat": [50.2, 49.8], "lon": [0.3, 0.6, 0.9]})
    out = select_like(coarse, fine)
    assert out["x"].dims == ("lat", "lon") and out["lon"].values.tolist() == [0.3, 0.6, 0.9]
    assert out["x"].values.tolist() == [[10_000, 10_001, 10_001]] * 2
    points = select(fine.assign(y=(("lat", "lon"), np.zeros((2, 3)))), Points(lat=[50.2], lon=[0.9], names=["a"]))
    assert select_like(coarse, points)["x"].values.tolist() == [10_001]
    assert select_like(coarse, points)["name"].values.tolist() == ["a"]

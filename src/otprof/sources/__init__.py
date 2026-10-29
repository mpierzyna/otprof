"""Source adapters: data of one source -> the canonical inputs (`otprof.schema`).

Download sources, all with the same functions:

| Source | Data | Call |
|---|---|---|
| `era5` | ERA5 from the ARCO-ERA5 store (Google Cloud), any time and area | `load(times, where)` |
| `era5_cds` | ERA5 from the Copernicus CDS, for long time series: bulk download first | `download(times, area, root)`, then `load(times, where, root)` |
| `gfs` | GFS forecasts (v16 and later, from 2021) from AWS, one forecast run | `load(cycle, times, where)` |
| `gfs_archive` | GFS of 2015-2020 from NCAR GDEX (hindcasts) | `load(times, where)` |

- `where`: an `Area` (the grid points inside it; dims `lat`, `lon`) or `Points` (the grid point nearest to each;
  dim `loc`, coordinates `lat`, `lon` and `name` of the points)
- `fetch(...)`: the data as the source delivers it (its names and units), selected to `where`
- `standardize(raw)`: -> the canonical inputs on the 31 pressure levels (dims time, lat, lon or loc, bottom_top)
- `load(...)` = `standardize(fetch(...))`

`wrf` converts WRF Cn2 simulations (training data) with `standardize` only.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pydantic
import xarray as xr


def to_180(lon):
    """Longitudes -> -180..180; values already in that range stay exactly as they are."""
    lon = np.asarray(lon, dtype=float)
    return np.where((lon < -180) | (lon >= 180), (lon + 180) % 360 - 180, lon)


class Area(pydantic.BaseModel):
    """A box in degrees north / east. Longitudes -180..180 or 0..360 (stored as -180..180); not across 180 deg."""

    model_config = pydantic.ConfigDict(frozen=True)

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    @pydantic.model_validator(mode="after")
    def _check(self) -> Self:
        lon_min, lon_max = to_180([self.lon_min, self.lon_max])
        if lon_min > lon_max:
            raise ValueError("areas across the 180 deg meridian are not supported")
        if self.lat_min > self.lat_max:
            raise ValueError("lat_min > lat_max")
        object.__setattr__(self, "lon_min", float(lon_min))
        object.__setattr__(self, "lon_max", float(lon_max))
        return self


class Points(pydantic.BaseModel):
    """Locations in degrees north / east (longitudes stored as -180..180), optionally with names."""

    model_config = pydantic.ConfigDict(frozen=True)

    lat: tuple[float, ...]
    lon: tuple[float, ...]
    names: tuple[str, ...] | None = None

    @pydantic.model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.lat) != len(self.lon) or (self.names is not None and len(self.names) != len(self.lat)):
            raise ValueError("lat, lon (and names) need the same length")
        object.__setattr__(self, "lon", tuple(float(x) for x in to_180(self.lon)))
        return self

    @classmethod
    def from_dict(cls, locations: dict[str, tuple[float, float]]) -> Points:
        """From {name: (lat, lon)}."""
        lat, lon = zip(*locations.values())
        return cls(lat=lat, lon=lon, names=tuple(locations))


Where = Area | Points


def get_nearest(grid: np.ndarray, values, periodic: bool = False) -> np.ndarray:
    """Index of the grid value nearest to each of `values` (`periodic`: longitudes, distance around the globe)."""
    d = np.asarray(grid, dtype=float)[:, None] - np.asarray(values, dtype=float)
    return np.abs((d + 180) % 360 - 180 if periodic else d).argmin(0)


def select(ds: xr.Dataset, where: Where, lat: str = "lat", lon: str = "lon") -> xr.Dataset:
    """The grid points of `ds` (a regular grid: 1-D coordinates `lat`, `lon` of any order and longitude convention) in
    an `Area` (north to south, west to east, longitudes -180..180) or nearest to `Points` (dim `loc`; coordinates
    `lat`, `lon` and `name` of the points)."""
    glat, glon = ds[lat].values, to_180(ds[lon].values)
    if isinstance(where, Area):
        i = np.flatnonzero((glat >= where.lat_min) & (glat <= where.lat_max))
        j = np.flatnonzero((glon >= where.lon_min) & (glon <= where.lon_max))
        if not len(i) or not len(j):
            raise ValueError(f"no grid points in {where}")
        i, j = i[np.argsort(-glat[i], kind="stable")], j[np.argsort(glon[j], kind="stable")]
        return ds.isel({lat: i, lon: j}).assign_coords({lon: glon[j]})
    i = xr.DataArray(get_nearest(glat, where.lat), dims="loc")
    j = xr.DataArray(get_nearest(glon, where.lon, periodic=True), dims="loc")
    out = ds.isel({lat: i, lon: j}).drop_vars([lat, lon])
    out = out.assign_coords(
        loc=np.arange(len(where.lat)), **{lat: ("loc", list(where.lat)), lon: ("loc", list(where.lon))}
    )
    return out if where.names is None else out.assign_coords(name=("loc", list(where.names)))


def select_like(ds: xr.Dataset, like: xr.Dataset, lat: str = "lat", lon: str = "lon") -> xr.Dataset:
    """`ds` at the grid points of `like` (another grid, e.g. GFS's Gaussian sflux grid on its 0.25 deg grid): the
    nearest point to each point (dim `loc`) or each grid point (dims `lat`, `lon`) of `like`."""
    if "loc" in like.dims:
        where = Points(lat=like[lat].values.tolist(), lon=like[lon].values.tolist())
        return select(ds, where, lat, lon).assign_coords({c: like[c] for c in like.coords if "loc" in like[c].dims})
    i = get_nearest(ds[lat].values, like[lat].values)
    j = get_nearest(to_180(ds[lon].values), like[lon].values, periodic=True)
    return ds.isel({lat: i, lon: j}).assign_coords({lat: like[lat].values, lon: like[lon].values})

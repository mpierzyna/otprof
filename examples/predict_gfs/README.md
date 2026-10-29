# Predict Cn2 profiles from GFS forecasts with the published model

Three small scripts, run one after the other. Each writes its result to `data/` as a NetCDF file you can open and
inspect (e.g. with `xarray` or `ncview`) before running the next one.

```
1 download GFS  →  data/gfs.nc      (GFS fields at the locations, as decoded from GRIB)
2 standardize   →  data/gfs_pl.nc   (otprof's names, units and levels)
3 predict       →  data/X.nc        (the model inputs)
                   data/y.nc        (Cn2 profiles, r0, scintillation)
```

## Choose what to predict

Edit `params.yaml`:

- `cycle`: the GFS run (UTC, 00, 06, 12 or 18). The archive on AWS starts in 2021.
- `leads`: the lead times [h], e.g. every 3 h of the first day.
- `locations`: name and [latitude, longitude]. Each location takes the nearest GFS grid point (0.25 deg for the
  profiles, about 13 km for the surface fluxes).
- `model`: leave at `null` for the published model, or give a local model directory.

GFS comes from the NOAA archive on AWS (no account needed). Only the needed GRIB messages are downloaded. Each is a
global field, so a lead time downloads about 120 MB, whatever the number of locations.

For an area instead of locations, use `Area(lon_min=2, lon_max=9, lat_min=49, lat_max=55)` instead of `Points`: the
0.25 deg grid points inside it, on dims (time, lat, lon) like ERA5 (same download size).

## Run

    uv run python step_1_download_gfs.py
    uv run python step_2_standardize.py
    uv run python step_3_predict.py

## Output (`data/y.nc`)

As in `examples/predict_era5`: for every time, location and height (`bottom_top`, 0 = lowest) `lcn2` (log10 Cn2),
`QKE`, `TSQ`, `EL_PBL` and `z_agl`, and per profile `r0` [cm] and `lsi2` (1550 nm). The coordinates `cycle` and `lead`
give the GFS run and lead time of every time step.

## Good to know

- The fields are those of the current GFS (v16 and later): all 31 pressure levels, instantaneous surface fluxes,
  friction velocity and PBL height. `otprof.sources.gfs_archive` reads the GFS of earlier years (2015-2020) from the
  NCAR GDEX archive, which lacks some of these fields; see its docstring.
- The surface CT2 input (`lct2_w71f`) clips the Obukhov length to the 2.5 / 97.5 percentiles of the data the inputs
  are computed from, as in training. For a few times and locations these percentiles are uncertain; fixed bounds are
  planned.
- The terrain height input is GFS's own orography. The training inputs used the WRF terrain of the target column.
- The model was trained on WRF in the Netherlands with ERA5 at inference; GFS inputs are not part of the paper's
  evaluation (arXiv:2604.09346).

## The same in Python

```python
import pandas as pd

from otprof.models import sqf_h
from otprof.predict import load_model, predict
from otprof.sources import Points, gfs

cycle = pd.Timestamp("2025-06-01 00:00")   # one forecast run; all times come from it
times = pd.date_range(cycle + pd.Timedelta(hours=3), periods=4, freq="3h")
ds = gfs.load(cycle, times, Points.from_dict({"cabauw": (51.97, 4.93), "la_palma": (28.76, -17.89)}))
predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)
X = predictor.make_inputs(ds)   # the model computes its own inputs
y = predict(predictor, X)
```

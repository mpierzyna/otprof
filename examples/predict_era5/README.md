# Predict Cn2 profiles from ERA5 with the published model

Three small scripts, run one after the other. Each writes its result to `data/` as a NetCDF file you can open and
inspect (e.g. with `xarray` or `ncview`) before running the next one.

```
1 download ERA5  →  data/era5.nc      (ERA5 pressure levels and surface fields, as downloaded)
2 standardize    →  data/era5_pl.nc   (otprof's names, units and levels)
3 predict        →  data/X.nc         (the model inputs)
                    data/y.nc         (Cn2 profiles, r0, scintillation)
```

## Choose what to predict

Edit `params.yaml`:

- `start`, `end`, `freq`: the ERA5 time steps (UTC), e.g. one day every 6 hours.
- `area`: the region on the ERA5 grid (0.25 deg). Longitudes run from 0 to 360, so 10 deg W is 350.
- `model`: leave at `null` for the published model, or give a local model directory.

ERA5 is downloaded from the public ARCO-ERA5 copy on Google Cloud (no account needed). It stores each time step for
the whole globe in one piece, so every time step downloads about 1 GB, however small the area. The default (4 time
steps) downloads 4.6 GB and keeps less than 1 MB.

## Run

    uv run python step_1_download_era5.py
    uv run python step_2_standardize.py
    uv run python step_3_predict.py

## Output (`data/y.nc`)

For every time, grid point and height (`bottom_top`, 0 = lowest):

| Variable | Meaning |
|---|---|
| `lcn2` | log10 of the refractive index structure parameter Cn2 [m^-2/3] |
| `QKE`, `TSQ`, `EL_PBL` | turbulence quantities of the WRF MYNN scheme the model was trained on |
| `z_agl` | height above ground [m] |

and per profile:

| Variable | Meaning |
|---|---|
| `r0` | Fried parameter [cm], 1550 nm |
| `lsi2` | log10 of the scintillation index, plane wave, 1550 nm |

The heights are the 99 levels the model was trained on (8 m to about 29 km). Other heights:
`predict(predictor, X, z_agl_query=[10, 50, 100, ...])`; `r0` and `lsi2` are then integrated over these heights
only. The output quantile mapping of the model is applied (`otprof_oqm = 1`); `predict(..., oqm=False)` gives the
uncorrected profiles.

## Good to know

- The surface CT2 input (`lct2_w71f`) clips the Obukhov length to the 2.5 / 97.5 percentiles of the data the inputs
  are computed from, as in training. For very few time steps or grid points these percentiles are uncertain; fixed bounds
  are planned.
- The terrain height input is ERA5's own orography. The training inputs used the WRF terrain of the target column.
- The model was trained on the Netherlands. How well it transfers to other regions is discussed in the paper
  (arXiv:2604.09346).

## The same in Python

```python
import pandas as pd

from otprof.models import sqf_h
from otprof.predict import load_model, predict
from otprof.sources import era5

times = pd.date_range("2023-07-01 00:00", "2023-07-01 18:00", freq="6h")
ds, sizes = era5.fetch(times, lon_min=3, lon_max=7, lat_min=51, lat_max=53)   # lazy; downloads on .load()
predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)
X = predictor.make_inputs(era5.standardize(ds.load()))   # the model computes its own inputs
y = predict(predictor, X)
```

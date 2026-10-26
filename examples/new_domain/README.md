# Building a training dataset for a new domain

These scripts turn WRF Cn2 simulations of a new domain into a training dataset for SQF-H. Each step is a small script
that calls a few `otprof` functions and writes NetCDF files you can open and check before running the next step. The
same steps run in the research repository as a DVC pipeline. `dvc.yaml` here does the same if you want caching.

```
WRF Cn2 output (otprof-wrf)  ─┐
                              ├─ 2 align ─┬─ 3 features + targets ─ 4 filter Cn2 ─→ y_wrf.nc       (targets)
ERA5 (1 download) ────────────┘           ├─────────────────────────────────────→ X_era5_pl.nc   (ERA5 inputs)
                                          └─ 5 WRF → pressure levels ───────────→ X_wrf_pl.nc    (training inputs)
```

## Setup

```bash
pip install "otprof[data]"     # or: uv sync --extra data, in a clone of otprof
```

1. Run the WRF simulations of your domain with [otprof-wrf](https://github.com/mpierzyna/otprof-wrf). The step
   scripts read its post-processed output, `wrfout_aux_*_cn2.nc`.
2. Add the domain to `params.yaml`: where the WRF output is, the ERA5 area to download, and two alignment settings
   (`crop`, `dist_th`, see step 2).
3. Run the steps one after another with the domain name as the only argument, e.g. `python step_2_align.py nl`. All
   output goes to `data/<domain>/` (or to `$OTPROF_DATA_ROOT/<domain>/`).

## Steps

| Step | Script | Writes | Check |
|---|---|---|---|
| 1 | `step_1_download_era5.py` | `1_raw/era5.nc` | the time steps cover the WRF period |
| 2 | `step_2_align.py` | `2_aligned/era5.nc`, `2_aligned/wrf.nc` | number of points, distances, common time steps |
| 3 | `step_3_features_targets.py` | `3_features/y_wrf_all.nc`, `3_features/X_era5_pl.nc` | profiles look plausible |
| 4 | `step_4_filter_cn2.py` | `3_features/y_wrf.nc` | share of profiles kept |
| 5 | `step_5_wrf_to_pl.py` | `4_features_interp/X_wrf_pl.nc` | WRF inputs resemble the ERA5 inputs |

**1. Download ERA5.** `otprof.dataset.wrf_times` reads the time steps of the WRF output and
`otprof.sources.era5.fetch` opens ERA5 (pressure levels and surface) for these times and the area from the cloud
copy of ERA5 (ARCO-ERA5). Every time step is downloaded for the whole globe and cut to the area, which takes a while.

**2. Align.** `otprof.dataset.align` pairs each ERA5 grid point with the nearest WRF column. `crop` drops that many
WRF cells along each edge (the WRF boundary zone); `dist_th` [deg] is the largest allowed distance between an ERA5
point and its WRF column, so only ERA5 points that fall on a WRF cell are kept. Only the time steps both have are kept.
Nothing is converted: ERA5 and WRF keep their own names, units and levels, so `2_aligned` is the raw comparison
dataset. WRF files are read in parallel (`N_WORKERS`); reading is limited by decompression.

**3. Features and targets.**
- `otprof.dataset.wrf_targets`: the WRF profiles on the model levels with canonical names (`cn2`, `QKE`, `TSQ`,
  `EL_PBL`, `z_agl`, ...) and the astroclimate parameters r0, tau0, th0 and the scintillation index for the surface
  layer, boundary layer, free atmosphere and full column.
- `otprof.sources.era5.standardize` converts ERA5 to the canonical inputs (`otprof.schema`: units, sign conventions,
  31 pressure levels 1000..20 hPa) and `otprof.features.make_inputs` builds the model inputs from them: the 25
  `otprof.features.FEATURES` on 30 levels (the 20 hPa level only enters the gradients of the top level). The terrain
  input is WRF's terrain, the true terrain of the column the model predicts.

**4. Filter Cn2.** `otprof.dataset.filter_cn2` masks (sets to NaN) Cn2 profiles that are zero at any level or whose
r0 or scintillation index lies above the 99.9 % quantile, and masks the astroclimate parameters accordingly. Masked
profiles are dropped when training.

**5. WRF on pressure levels.** `otprof.sources.wrf.standardize(grid="pressure")` interpolates WRF to the ERA5
pressure levels and `make_inputs` builds the same inputs as for ERA5. The model is trained on these WRF inputs and
applied to ERA5: training and inference inputs come from the same function.

## Adding a feature

1. Compute it in `otprof.features.make_inputs` from the canonical variables (`otprof.schema.VARIABLES`); if it needs a
   new variable, add it to the schema and to every source adapter in `otprof.sources`.
2. Bump `otprof.features.FEATURES_VERSION` and regenerate the golden test data (`tests/data/make_golden.py`).
3. Rerun steps 3 and 5, check the new variable in the output files, and retrain.

To try a feature without touching otprof, add it in your own step script after `make_inputs`, as the research
repository does for its extra inputs.

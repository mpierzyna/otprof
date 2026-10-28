# Building a training dataset for a new domain

These scripts turn WRF Cn2 simulations of a new domain into a training dataset. Each step is a small script
that calls a few `otprof` functions and writes NetCDF files you can open and check before running the next step. The
same steps run in the research repository as a DVC pipeline. `dvc.yaml` here does the same if you want caching.

```
WRF Cn2 output (otprof-wrf)  ─┐
                              ├─ 2 align ─┬─ 3 targets + inputs ─ 4 filter Cn2 ─→ y_wrf.nc     (targets)
ERA5 (1 download) ────────────┘           ├───────────────────────────────────→ era5_pl.nc   (ERA5 inputs)
                                          └─ 5 WRF → pressure levels ─────────→ wrf_pl.nc    (training inputs)
```

The dataset holds the inputs in the canonical form of `otprof.schema` (names, units, levels), not the features of a
particular model: each model computes its own inputs from them when it loads the data (SQF-H:
`otprof.models.sqf_h.inputs.make_inputs`), and the same function runs at inference.

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
| 3 | `step_3_standardize_era5_wrf.py` | `3_targets/y_wrf_all.nc`, `3_inputs/era5_pl.nc` | profiles look plausible |
| 4 | `step_4_filter_cn2.py` | `3_targets/y_wrf.nc` | share of profiles kept |
| 5 | `step_5_wrf_to_pl.py` | `4_inputs_interp/wrf_pl.nc` | WRF profiles resemble the ERA5 profiles |

**1. Download ERA5.** `otprof.dataset.get_wrf_times` reads the time steps of the WRF output and
`otprof.sources.era5.fetch` opens ERA5 (pressure levels and surface) for these times and the area from the cloud
copy of ERA5 (ARCO-ERA5). Every time step is downloaded for the whole globe and cut to the area, which takes a while.

**2. Align.** `otprof.dataset.align` pairs each ERA5 grid point with the nearest WRF column. `crop` drops that many
WRF cells along each edge (the WRF boundary zone); `dist_th` [deg] is the largest allowed distance between an ERA5
point and its WRF column, so only ERA5 points that fall on a WRF cell are kept. Only the time steps both have are kept.
Nothing is converted: ERA5 and WRF keep their own names, units and levels, so `2_aligned` is the raw comparison
dataset. WRF files are read in parallel (`N_WORKERS`); reading is limited by decompression.

**3. Standardize.**
- `otprof.dataset.make_wrf_targets`: the WRF profiles on the model levels with canonical names (`cn2`, `QKE`, `TSQ`,
  `EL_PBL`, `z_agl`, ...) and the astroclimate parameters r0, tau0, th0 and the scintillation index for the surface
  layer, boundary layer, free atmosphere and full column.
- `otprof.sources.era5.standardize` converts ERA5 to the canonical inputs (`otprof.schema`: units, sign conventions,
  31 pressure levels 1000..20 hPa). The terrain height of the paired WRF column is added as `z_sfc_target`: the true
  terrain of the column the model predicts (SQF-H uses it as its terrain input).

**4. Filter Cn2.** `otprof.dataset.filter_cn2` masks (sets to NaN) Cn2 profiles that are zero at any level or whose
r0 or scintillation index lies above the 99.9 % quantile, and masks the astroclimate parameters accordingly. Masked
profiles are dropped when training.

**5. WRF on pressure levels.** `otprof.sources.wrf.standardize(grid="pressure")` interpolates WRF to the ERA5
pressure levels, in the same canonical form as the ERA5 inputs. The model is trained on these WRF inputs and applied
to ERA5: training and inference inputs come from the same function.

## Adding a feature

Features are part of a model, not of the dataset: add it to the model's input function (SQF-H:
`otprof.models.sqf_h.inputs.make_inputs` and its table `INPUTS`, with a scaler), bump the model's `FEATURES_VERSION`
(MAJOR: the model's inputs change) and retrain. The dataset only needs rebuilding if the feature needs a variable it
does not hold yet: then add the variable to `otprof.schema` and to every source adapter in `otprof.sources`, and
rerun steps 3 and 5.

To try a feature without touching otprof, compute it in your own pipeline (`fe_fns`) or input function, as the
research repository does for its extra inputs.

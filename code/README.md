# OTProf: 

This is a standalone extract of the Squeezeformer model used to predict optical-turbulence (Cn²) profiles from 
coarse atmospheric state. It bundles the model/training/analysis code, the supporting `otprof` package.

## Layout

```
.
├── data/                     # NetCDF datasets (see "Data" below)
│   ├── wrf_highres.nc        # high-resolution WRF profiles (targets)
│   ├── wrf_lowres.nc         # WRF interpolated to ERA5 pressure levels (inputs)
│   ├── era5_pl.nc            # ERA5 pressure-level fields
│   └── era5_sfc.nc           # ERA5 surface fields
├── otprof/                   # core library: data pipelines, datasets, transforms, losses
├── research_tools/           # vendored subset of an internal utility library
│   ├── ot/                   # optical-turbulence integrals (r0, scint. index, …)
│   └── misc_tools/           # YAML config (de)serialization + HTML reporter
├── experiments.py            # experiment definitions (model + data config)
├── leap_sqf.py               # Squeezeformer model + building blocks
├── lit.py                    # LightningModule wrappers
├── train_sqf_decoupled.py    # training entry point
├── predict_decoupled.py      # prediction / reference-profile entry point
├── model_analysis.py / model_plotting.py   # evaluation + HTML report generation
└── pyproject.toml
```

## Setup

Requires Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e .
```

or with pip:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

## Running

Run the scripts from this directory so the local modules and packages
(`otprof`, `research_tools`, `experiments`, `lit`, …) are importable. The data
files are located relative to the `otprof` package, so the working directory does
not affect data loading.

```bash
# Train a model (edit __main__ to select the experiment)
python train_sqf_decoupled.py

# Predict with a trained checkpoint (edit the paths in __main__ first)
python predict_decoupled.py
```

`train_sqf_decoupled.py` writes checkpoints and the serialized experiment to
`lightning_logs/version_*/`. `predict_decoupled.py` reads a trained model from a
`lightning_logs/version_*` directory and writes predictions to `saved_models/`.

## Pretrained models

The trained checkpoints for the three experiments discussed in the paper are
bundled under `lightning_logs/`. Each directory holds the top-5 checkpoints (by
validation loss), the serialized data pipeline, and the experiment config:

| experiment (`get_exp` name) | directory                  | description                                  |
| --------------------------- | -------------------------- | -------------------------------------------- |
| `wrf_native`                | `lightning_logs/version_176` | native WRF grid → WRF                       |
| `wrf_pl`                    | `lightning_logs/version_175` | WRF on ERA5 pressure levels → WRF           |
| `era5_pl_direct`            | `lightning_logs/version_179` | ERA5 pressure levels → WRF (direct)         |

To predict with a bundled model, point `predict_decoupled.predict` at the matching
experiment and directory, e.g.:

```python
import pathlib, experiments, predict_decoupled
predict_decoupled.predict(
    exp=experiments.get_exp("wrf_native"),
    model_path=pathlib.Path("lightning_logs/version_176"),
    out_dir=pathlib.Path("saved_models/preds_wrf_native"),
    out_basename="wrfnative",
)
```

The best checkpoint is selected automatically from each directory by validation
score. The experiment (and its scalers) are rebuilt from the bundled data, so the
datasets above must be present.

## Data

The datasets are NetCDF files covering 2017–2020 (2019–2020 train, 2018
validation, 2017 test). They are renamed/flattened copies of the original
pipeline outputs:

| published file   | original source                              |
| ---------------- | -------------------------------------------- |
| `wrf_highres.nc` | `3_features/wrf_filtered.nc`                 |
| `wrf_lowres.nc`  | `4_features_interp/wrf_filtered_era5pl.nc`   |
| `era5_pl.nc`     | `3_features/era5_pl.nc`                       |
| `era5_sfc.nc`    | `3_features/era5_sfc.nc`                      |

Only the ERA5 pressure-level grid (`era5pl`) is shipped, so the `aifspl`/`ml`
vertical-grid variants and the `experimental` data pipelines are kept for
reference but cannot be run with this extract.

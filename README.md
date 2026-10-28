# OTProf

Vertical profiles of optical turbulence (the refractive index structure parameter Cn2) from reanalysis and forecast
data with deep learning. The model, SQF-H (height-aware SqueezeFormer), is trained on WRF simulations of Cn2 and
predicts the Cn2 profile, the WRF MYNN turbulence quantities, the Fried parameter r0 and the scintillation index from
ERA5 pressure-level data. Method and evaluation: [arXiv:2604.09346](https://arxiv.org/abs/2604.09346).

## Setup

We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs the right Python version
and all dependencies into a local environment. In a clone of this repository:

```bash
git clone https://github.com/mpierzyna/otprof.git
cd otprof
uv sync
```

`uv sync` creates `.venv/` with otprof, all its optional parts and the test tools. Everything in `examples/` and
`paper/` then runs with `uv run`, e.g. `uv run python examples/train.py ...`, and `uv run pytest` runs the tests.
Training needs a GPU; prediction runs on a CPU too.

To use otprof in your own uv project instead, add it with the optional parts you need:

```bash
uv add "otprof[torch,hub] @ git+https://github.com/mpierzyna/otprof"
```

| Optional part | For |
|---|---|
| `data` | building training data: ERA5 download, alignment with WRF, NetCDF / YAML files |
| `torch` | loading and running models |
| `hub` | downloading the published model from Hugging Face |
| `train` | training (Lightning, TensorBoard; includes `torch`) |

Without optional parts, otprof only needs numpy, xarray and pydantic (the physics, the input schema and the data
processing).

## Getting started

| Folder | What it shows |
|---|---|
| [`examples/predict_era5`](examples/predict_era5/README.md) | Cn2 profiles for any region from ERA5 with the published model |
| [`examples/new_domain`](examples/new_domain/README.md) | building a training dataset from WRF Cn2 simulations of a new domain |
| `examples/train.py`, `examples/configs` | training a model from an experiment config |
| [`paper`](paper/README.md) | reproducing the experiments of the paper |

In short, with the published model:

```python
from otprof.models import sqf_h
from otprof.predict import load_model, predict
from otprof.sources import era5

predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)
X = predictor.make_inputs(era5.standardize(ds))   # ds: ERA5 loaded with era5.fetch
y = predict(predictor, X)                         # lcn2, QKE, TSQ, EL_PBL, r0, lsi2
```

## Package

| Module | Contents |
|---|---|
| `otprof.schema` | the canonical input dataset: names, units, pressure levels every data source is converted to |
| `otprof.sources` | data sources: `era5` (download, `standardize`), `wrf` (WRF Cn2 output, `standardize`) |
| `otprof.dataset` | building a training dataset: alignment of ERA5 and WRF, targets, Cn2 filter |
| `otprof.features` | building blocks of model inputs (potential temperature, wind, surface CT2, time features) |
| `otprof.models` | one subpackage per model; `sqf_h`: its inputs, data processing, network and training module |
| `otprof.pipeline`, `otprof.transforms` | loading the dataset files, train / val / test split, scalers and transforms |
| `otprof.experiment`, `otprof.training` | experiment configs (YAML) and training |
| `otprof.predict`, `otprof.export` | prediction with a trained or published model, output quantile mapping, export |
| `otprof.physics` | atmospheric and optical turbulence physics (r0, scintillation, reference profiles, MOST) |

## License

GPL-3.0, see `COPYING`.

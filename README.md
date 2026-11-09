<!--
SPDX-FileCopyrightText: 2026 Maximilian Pierzyna

SPDX-License-Identifier: GPL-3.0-or-later
-->

# OTProf

Estimate high-resolution vertical profiles of optical turbulence, i.e., the refractive index structure parameter $C_n^2$, from reanalysis data.

This repository is part of a set of repositories:
- [`otprof`](https://github.com/mpierzyna/otprof) (**this repository**): Code to train a new OTProf model or run a pretrained model. A snapshot of this repository is available on Zenodo (DOI: 10.5281/zenodo.24084350). The weights of OTProf, pretrained on the Netherlands, are available on Hugging Face (https://huggingface.co/maxiimilian/otprof-sqfh).
- [`otprof-wrf`](https://github.com/mpierzyna/otprof-wrf): Code to generate the WRF-based $C_n^2$ training data. The generated training dataset and a snapshot of the `otprof-wrf` code are available on Zenodo (DOI: 10.5281/zenodo.23733024).
- [`otprof-ui`](https://github.com/mpierzyna/otprof-ui): Code of the web UI to show OTProf-based $C_n^2$ forecasts and climatology. Check it out at [**https://otprof.mpier.eu**](https://otprof.mpier.eu)!

📝 The **article** presenting the method in detail and evaluating its performance is available on arXiv: [arXiv:2604.09346](https://arxiv.org/abs/2604.09346).
Please refer to the [License](#license) and [How to cite](#how-to-cite) sections if you would like to use this software or its products for your own work.

This software has been developed by [**Maximilian Pierzyna**](https://orcid.org/0000-0002-0596-8374), Technische Universiteit Delft.

## Setup

We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs the right Python version
and all dependencies into a local environment. In a clone of this repository:

```bash
git clone https://github.com/mpierzyna/otprof.git
cd otprof
uv sync
```

`uv sync` creates `.venv/` with otprof, all its optional parts and the test tools, in the tested versions of
`uv.lock`. Everything in `examples/` and `paper/` then runs with `uv run`, e.g. `uv run python examples/train.py ...`,
and `uv run pytest` runs the tests. Training needs a GPU; prediction runs on a CPU too.

To use otprof in your own uv project instead, add it with the optional parts you need:

```bash
uv add "otprof[torch,hub] @ git+https://github.com/mpierzyna/otprof"
```

| Optional part | For |
|---|---|
| `data` | building training data: ERA5 download, alignment with WRF, NetCDF / YAML files |
| `torch` | loading and running models |
| `hub` | downloading the published model from Hugging Face |
| `gfs` | GFS forecasts (GRIB decoding with eccodes) |
| `train` | training (Lightning, TensorBoard; includes `torch`) |

Without optional parts, otprof only needs numpy, xarray and pydantic (the physics, the input schema and the data
processing).

## Getting started

| Folder | What it shows |
|---|---|
| [`examples/predict_era5`](examples/predict_era5/README.md) | $C_n^2$ profiles for any region from ERA5 with the published model |
| [`examples/predict_gfs`](examples/predict_gfs/README.md) | $C_n^2$ profile forecasts at given locations from GFS with the published model |
| [`examples/new_domain`](examples/new_domain/README.md) | building a training dataset from WRF $C_n^2$ simulations of a new domain |
| `examples/train.py`, `examples/configs` | training a model from an experiment config |
| [`paper`](paper/README.md) | reproducing the experiments of the paper |

In short, with the published model:

```python
from otprof.models import sqf_h
from otprof.predict import load_model, predict
from otprof.sources import Area, era5

ds = era5.load(["2023-07-01 12:00"], Area(lon_min=3, lon_max=7, lat_min=51, lat_max=53))
predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)
X = predictor.make_inputs(ds)
y = predict(predictor, X)                         # lcn2, QKE, TSQ, EL_PBL, r0, lsi2
```

## Package

| Module | Contents |
|---|---|
| `otprof.schema` | the canonical input dataset: names, units, pressure levels every data source is converted to |
| `otprof.sources` | data sources with one interface (`load(times, where)`, `where` an `Area` or `Points`): `era5`, `era5_cds` (bulk downloads), `gfs` (forecasts), `gfs_archive`; `wrf` (WRF $C_n^2$ output, training data) |
| `otprof.dataset` | building a training dataset: alignment of ERA5 and WRF, targets, $C_n^2$ filter |
| `otprof.features` | building blocks of model inputs (potential temperature, wind, surface CT2, time features) |
| `otprof.models` | one subpackage per model (`sqf_h`: its inputs, data processing, network and training module); saving and loading networks (`bundle`) |
| `otprof.pipeline`, `otprof.transforms` | loading the dataset files, train / val / test split, scalers and transforms |
| `otprof.experiment`, `otprof.training` | experiment configs (YAML) and training |
| `otprof.predict`, `otprof.export` | prediction with a trained or published model, output quantile mapping, export |
| `otprof.physics` | atmospheric and optical turbulence physics (r0, scintillation, reference profiles, MOST) |

## Adding a model

A model is a subpackage with a fixed interface (`otprof.models.ModelPackage`): its hyperparameters (`Config`, the
`model:` block of an experiment config), its inputs computed from the canonical input dataset (`make_inputs`,
`FEATURES_VERSION`), its data processing (`get_pipeline`), its network (called with the inputs and the forcings of the
query heights) and its Lightning module. `otprof.models.sqf_h` is the reference. Register it with one line,

```python
from otprof import models

models.MODELS["my_model"] = "my_package.my_model"
```

and experiments with `model: {name: my_model, ...}` train, export, load and predict with the same functions as SQF-H
(`tests/dummy_model` is a minimal example). A model developed elsewhere becomes part of otprof by moving its
subpackage to `otprof.models` and adding it to `MODELS` there.


## License
Unless otherwise noted, the content of this repository is licensed under **GNU General Public License v3.0 or later** (see [`COPYING`](./COPYING)).

### Third-party code
[`src/otprof/models/sqf_h/model.py`](src/otprof/models/sqf_h/model.py) is derived from the Kaggle notebook
["LEAP: Training 1"](https://www.kaggle.com/code/shlomoron/leap-training-1?scriptVersionId=188874411) by Ron Shlomo, licensed under the Apache License 2.0. The file remains under Apache 2.0 (see `LICENSES/Apache-2.0.txt`); modifications by Maximilian Pierzyna are noted in the file header.

## Copyright notice
Technische Universiteit Delft hereby disclaims all copyright interest in the program "OTProf" written by the Author(s). Henri Werij, Faculty of Aerospace Engineering, Technische Universiteit Delft.

© 2026, Maximilian Pierzyna (excluding third-party code listed above)

## How to cite
When using OTProf, please cite this **code repository** and the **related article**. You can do so either using the `./CITATION.cff` file or by using the following information:

### Code repository

> Pierzyna, M. (2026). *OTProf: Estimating High-Resolution Profiles of Optical Turbulence from Reanalysis Using Deep Learning* (Version 0.2.3) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24084350

### Article

> Pierzyna, M., Basu, S., & Saathof, R. (2026). OTProf: estimating high-resolution profiles of optical turbulence ($C_n^2$) from reanalysis using deep learning. *arXiv preprint* arXiv:2604.09346. https://doi.org/10.48550/arXiv.2604.09346

### Training dataset

Our training dataset is licensed under CC-BY-SA-4.0, so you may reuse it for your own work given that you attribute the original author as follows.

> Pierzyna, M. (2026). *OTProf: Estimating High-Resolution Profiles of Optical Turbulence (Cn2) from Reanalysis Using Deep Learning* (Version 0.2.1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23733024
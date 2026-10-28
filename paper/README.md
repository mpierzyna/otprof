# Paper: reproducing the comparisons

The settings and scripts behind the figures and tables of the paper (arXiv:2604.09346). They use otprof like any user
would and are not part of the installed package.

All commands are run from the top folder of the otprof repository. The example folder names below (`data/nl`,
`runs/...`, `preds/...`) can be anything; replace them with your own.

## What you need

- **The datasets** of the two domains, the Netherlands (`nl`) and the Canary Islands (`canaries`), downloaded from
  Zenodo (`<DOI>`) or built with `examples/new_domain`. Below they are in `data/nl` and `data/canaries`. Each holds
  - `4_inputs_interp/wrf_pl.nc`: WRF interpolated to the ERA5 pressure levels (inputs, "WRF-PL"),
  - `3_inputs/era5_pl.nc`: ERA5 at the same points (inputs),
  - `3_targets/y_wrf.nc`: the WRF Cn2 profiles (targets),
  - for the control experiment also `3_inputs/wrf_native.nc` (WRF on its own model levels; not built by
    `examples/new_domain`).

## The experiments

Each experiment is one settings file in `configs/`:

| Experiment in the paper | Settings file | Model inputs |
|---|---|---|
| WRF-PL and ERA5-PL (the decoupled model) | `configs/wrf_pl_h.yaml` | WRF on pressure levels |
| Ctl. (control) | `configs/wrf_native_h.yaml` | WRF on its model levels |
| ERA5-PL direct (appendix) | `configs/era5_pl_h_direct.yaml` | ERA5 |
| HV+W71 (baseline) | script `hv_w71.py` | no training |

## Step 1: train the three models

Each command trains one model on the nl years 2019-2020 (2018 for validation) and saves it in a run folder:

    uv run python examples/train.py paper/configs/wrf_pl_h.yaml --data-root data/nl --save-dir runs --version wrf_pl_h
    uv run python examples/train.py paper/configs/wrf_native_h.yaml --data-root data/nl --save-dir runs --version wrf_native_h
    uv run python examples/train.py paper/configs/era5_pl_h_direct.yaml --data-root data/nl --save-dir runs --version era5_pl_h_direct

The models end up in `runs/lightning_logs/wrf_pl_h` etc. Training needs a GPU.

## Step 2: predict the nl test year 2017

`predict_test.py` predicts the test samples of a settings file with a trained model and writes the predictions
(`<name>_y_pred.nc`) next to the WRF truth of the same samples (`<name>_y_test.nc`).

WRF-PL: the decoupled model with the WRF inputs it was trained on.

    uv run python paper/predict_test.py paper/configs/wrf_pl_h.yaml --model runs/lightning_logs/wrf_pl_h \
        --data-root data/nl --out preds/nl --basename wrfpl

ERA5-PL: the same model, but with ERA5 inputs (`--inputs` replaces the inputs file).

    uv run python paper/predict_test.py paper/configs/wrf_pl_h.yaml --model runs/lightning_logs/wrf_pl_h \
        --data-root data/nl --out preds/nl --basename era5pl --inputs 3_inputs/era5_pl.nc

Ctl. and ERA5-PL direct:

    uv run python paper/predict_test.py paper/configs/wrf_native_h.yaml --model runs/lightning_logs/wrf_native_h \
        --data-root data/nl --out preds/nl --basename wrfnative

    uv run python paper/predict_test.py paper/configs/era5_pl_h_direct.yaml --model runs/lightning_logs/era5_pl_h_direct \
        --data-root data/nl --out preds/nl --basename era5pl_direct

HV+W71 baseline:

    uv run python paper/hv_w71.py paper/configs/wrf_pl_h.yaml --data-root data/nl --out preds/nl

## Step 3: output quantile mapping on nl (NL-oQM)

The output quantile mapping corrects the distribution of the ERA5-driven Cn2 predictions towards WRF. It is fitted on
the training years, so first predict them with ERA5 inputs (`--sel-test` selects the years to predict):

    uv run python paper/predict_test.py paper/configs/wrf_pl_h.yaml --model runs/lightning_logs/wrf_pl_h \
        --data-root data/nl --out preds/nl --basename era5pl_trainyears --inputs 3_inputs/era5_pl.nc \
        --sel-test '{"year": [2019, 2020]}'

Then fit the mapping and apply it to the 2017 ERA5 predictions:

    uv run python paper/fit_oqm.py nl --pred-dir preds/nl

This writes the mapping (`preds/nl/oqm_nl.json`; the published model carries the same mapping) and the corrected
predictions `preds/nl/era5pl_oqmsine_y_pred.nc`.

## Step 4: the Canary Islands (zero-shot)

The nl model is applied to the Canary Islands without any retraining. The canaries data cover 12 months of 5 days,
split into three groups of four months: A (months 1, 4, 7, 10), B (2, 5, 8, 11) and C (3, 6, 9, 12). Each group is
predicted separately, with WRF and with ERA5 inputs (`--no-oqm`: without any output quantile mapping). Group A:

    uv run python paper/predict_test.py paper/configs/wrf_pl_h.yaml --model runs/lightning_logs/wrf_pl_h \
        --data-root data/canaries --out preds/canaries/A --basename wrfpl \
        --sel-test '{"month": [1, 4, 7, 10]}' --no-oqm

    uv run python paper/predict_test.py paper/configs/wrf_pl_h.yaml --model runs/lightning_logs/wrf_pl_h \
        --data-root data/canaries --out preds/canaries/A --basename era5pl --inputs 3_inputs/era5_pl.nc \
        --sel-test '{"month": [1, 4, 7, 10]}' --no-oqm

and the same for group B (`--out preds/canaries/B --sel-test '{"month": [2, 5, 8, 11]}'`) and group C
(`--out preds/canaries/C --sel-test '{"month": [3, 6, 9, 12]}'`). The HV+W71 baseline of a group:

    uv run python paper/hv_w71.py paper/configs/wrf_pl_h.yaml --data-root data/canaries --out preds/canaries/A \
        --sel-test '{"month": [1, 4, 7, 10]}'

Finally the output quantile mappings:

    uv run python paper/fit_oqm.py canaries --nl-oqm preds/nl/oqm_nl.json --data-root data/canaries \
        --group A=preds/canaries/A --group B=preds/canaries/B --group C=preds/canaries/C

- NL-oQM: the mapping fitted on nl, applied to every group (`era5pl_oqmcn2nl_y_pred.nc`).
- CI-oQM: a mapping fitted on one canaries group, separately for WRF land and sea points, applied to the two other
  groups (`era5pl_oqmcn2fitA_y_pred.nc` in the folders of B and C, etc.).

## Using the published model instead of training

The published model is the WRF-PL model with the NL-oQM included. Use `--model maxiimilian/otprof-sqfh` instead of
`--model runs/lightning_logs/wrf_pl_h`. It applies the NL-oQM by default, so add `--no-oqm` wherever uncorrected
predictions are needed (the WRF-PL and ERA5-PL cases and the training-year predictions of step 3).

"""Predict the test samples of an experiment with a trained model, for the evaluation of the paper.

    uv run python paper/predict_test.py paper/configs/wrf_pl_h.yaml --model <bundle or run dir> \\
        --data-root <domain data dir> --out <dir> --basename wrfpl

The config defines the dataset and the split (the domain's data are loaded, nothing is fitted); the model brings its
own processing chain (scalers fitted on its training domain, oQM). `--inputs` replaces the inputs file (e.g.
`3_inputs/era5_pl.nc`: the decoupled WRF-PL model run on ERA5), `--sel-test` the test selection (e.g.
'{"month": [3, 6, 9, 12]}' for the canaries test months, '{"year": [2019, 2020]}' for the nl training years used to fit
the oQM). The query heights are the target heights of each sample.

Writes `<out>/<basename>_y_pred.nc` (targets, z_agl, r0, lsi2) and `<out>/<basename>_y_test.nc` (WRF targets of the
same samples); `--save-X` also the unscaled inputs `<basename>_X_test.nc`.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from otprof.experiment import Experiment, ExperimentConfig
from otprof.predict import load_model, predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--model", required=True, help="model bundle, run directory or Hugging Face repo id")
    parser.add_argument("--data-root", type=pathlib.Path, help="domain data dir (default: the config's data.root)")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--basename", required=True)
    parser.add_argument("--inputs", help="inputs file relative to the data root (default: the config's)")
    parser.add_argument("--sel-test", type=json.loads, help="test selection as JSON (default: the config's)")
    parser.add_argument("--no-oqm", action="store_true", help="without the model's output quantile mapping")
    parser.add_argument("--save-X", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    data = {k: v for k, v in [("inputs", args.inputs), ("sel_test", args.sel_test)] if v is not None}
    config = config.model_copy(update={"data": config.data.model_copy(update=data)})
    exp = Experiment.from_config(config, data_root=args.data_root)
    exp.pipeline.prepare_data()
    X_test, y_test = exp.pipeline.split("test", transform=False)

    predictor = load_model(args.model)
    y_pred = predict(
        predictor,
        X_test,
        z_agl_query=y_test["z_agl"],
        oqm=not args.no_oqm,
        batch_size=args.batch_size,
        device=args.device,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    y_pred.reset_index("sample").to_netcdf(args.out / f"{args.basename}_y_pred.nc")
    y_test.reset_index("sample").to_netcdf(args.out / f"{args.basename}_y_test.nc")
    if args.save_X:
        X_test.reset_index("sample").to_netcdf(args.out / f"{args.basename}_X_test.nc")
    print(f"Predictions saved to {args.out / args.basename}_y_{{pred,test}}.nc ({X_test.sizes['sample']} samples)")

"""Output quantile mapping (oQM) of the paper: one monotone map of lcn2 for the whole column (sine-spaced quantiles),
fitted on ERA5-driven predictions against WRF; r0 and lsi2 then follow from the mapped profile.

NL-oQM: fitted on the predictions of the nl training years (2019-2020), applied to the 2017 test predictions. Its
fitted state is what the published model carries in `post_y`.

    uv run python paper/fit_oqm.py nl --pred-dir <dir>
        reads  <dir>/era5pl_trainyears_y_{pred,test}.nc, <dir>/era5pl_y_pred.nc   (paper/predict_test.py, no oQM)
        writes <dir>/oqm_nl.json, <dir>/era5pl_oqmsine_y_pred.nc

Canaries (zero-shot, month groups A = 1, 4, 7, 10; B = 2, 5, 8, 11; C = 3, 6, 9, 12; each predicted with
`--basename era5pl` into its own directory): NL-oQM applied to every group, and CI-oQM fitted on one group with one map
for WRF land and one for WRF sea locations, applied to the two other groups.

    uv run python paper/fit_oqm.py canaries --nl-oqm <dir>/oqm_nl.json --data-root <canaries data dir> \\
        --group A=<dir A> --group B=<dir B> --group C=<dir C>
        writes era5pl_oqmcn2nl_y_pred.nc in every group dir, era5pl_oqmcn2fit<X>_y_pred.nc in the dirs of the two
        groups not fitted on

The `_y_test.nc` of every output is a link to `era5pl_y_test.nc` (same samples).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import xarray as xr

from otprof.predict import fit_oqm, get_integrated_params
from otprof.transforms import QuantileMapper


def apply_oqm(pred: xr.Dataset, qm: QuantileMapper, mask: np.ndarray | None = None) -> xr.Dataset:
    """lcn2 of `pred` mapped with `qm` (only the samples in `mask`, if given), r0 and lsi2 recomputed."""
    out = pred.copy()
    if mask is None:
        out["lcn2"] = qm.transform(pred[["lcn2"]])["lcn2"]
    else:
        lcn2 = out["lcn2"].values.copy()
        lcn2[mask] = qm.transform(pred[["lcn2"]].isel(sample=mask))["lcn2"].values
        out["lcn2"] = (pred["lcn2"].dims, lcn2)
    return out.assign(get_integrated_params(out["lcn2"], out["z_agl"])).assign_attrs(otprof_oqm=1)


def write(ds: xr.Dataset, d: pathlib.Path, base: str) -> None:
    ds.to_netcdf(d / f"{base}_y_pred.nc")
    link = d / f"{base}_y_test.nc"
    if not link.exists():
        link.symlink_to("era5pl_y_test.nc")
    print(f"Wrote {d / base}_y_pred.nc")


def load(path: pathlib.Path) -> xr.Dataset:
    return xr.load_dataset(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="case", required=True)
    p_nl = sub.add_parser("nl")
    p_nl.add_argument("--pred-dir", type=pathlib.Path, required=True)
    p_cn = sub.add_parser("canaries")
    p_cn.add_argument("--nl-oqm", type=pathlib.Path, required=True)
    p_cn.add_argument("--data-root", type=pathlib.Path, required=True, help="canaries data dir (WRF land-sea mask)")
    p_cn.add_argument("--group", action="append", required=True, help="NAME=DIR, e.g. A=preds/trainmonths")
    args = parser.parse_args()

    if args.case == "nl":
        d = args.pred_dir
        qm = fit_oqm(pred=load(d / "era5pl_trainyears_y_pred.nc"), truth=load(d / "era5pl_trainyears_y_test.nc"))
        (d / "oqm_nl.json").write_text(json.dumps(qm.to_dict()))
        write(apply_oqm(load(d / "era5pl_y_pred.nc"), qm), d, "era5pl_oqmsine")

    else:
        groups = {name: pathlib.Path(path) for name, path in (g.split("=", 1) for g in args.group)}
        qm_nl = QuantileMapper.from_dict(json.loads(args.nl_oqm.read_text()))
        # WRF land-sea mask per location (ERA5's own mask marks only half of the WRF land points as land)
        lsm = xr.open_dataset(args.data_root / "4_features_interp" / "X_wrf_pl.nc")["lsm"]
        land_locs = lsm["loc"].values[(lsm > 0.5).any("time").values]

        for name, d in groups.items():
            write(apply_oqm(load(d / "era5pl_y_pred.nc"), qm_nl), d, "era5pl_oqmcn2nl")

        for fit_name, fit_dir in groups.items():
            pred_fit, truth_fit = load(fit_dir / "era5pl_y_pred.nc"), load(fit_dir / "era5pl_y_test.nc")
            land_fit = np.isin(truth_fit["loc"].values, land_locs)
            qms = {is_land: fit_oqm(pred_fit.isel(sample=land_fit == is_land), truth_fit.isel(sample=land_fit == is_land))
                   for is_land in [False, True]}  # fmt: skip
            for name, d in groups.items():
                if name == fit_name:
                    continue
                pred = load(d / "era5pl_y_pred.nc")
                land = np.isin(pred["loc"].values, land_locs)
                out = apply_oqm(apply_oqm(pred, qms[False], ~land), qms[True], land)
                write(out, d, f"era5pl_oqmcn2fit{fit_name}")

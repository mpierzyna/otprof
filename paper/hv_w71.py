"""HV+W71 baseline of the paper: Hufnagel-Valley Cn2 profile scaled to the Walters (1971) surface value.

The surface Cn2 comes from the W71 surface CT2 of the inputs (`lct2_w71f`) with the pressure of the lowest input level
and the 2 m temperature; the HV profile uses the wind speed of the input profile and is evaluated at the target
heights. No training. Writes `<out>/<basename>_ref.nc` (`lcn2_hv` on the test samples of the experiment's split;
`--sel-test` replaces the test selection as in `predict_test.py`).

    uv run python paper/hv_w71.py examples/configs/wrf_pl_h.yaml --data-root <domain data dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import xarray as xr

from otprof.experiment import Experiment, ExperimentConfig
from otprof.features import get_wind_mag_dir
from otprof.physics import profiles


def compute_hv_w71(X: xr.Dataset, y: xr.Dataset) -> xr.Dataset:
    """log10 HV+W71 Cn2 at the target heights `y["z_agl"]` from the unscaled inputs `X` (samples, `bottom_top`)."""
    p_0 = X["p"].isel(bottom_top=0) / 100  # hPa
    ct2_0 = 10 ** X["lct2_w71f"]
    cn2_0 = (79e-6 * p_0 / X["tk2"] ** 2) ** 2 * ct2_0  # CT2 -> Cn2 (optical, dry air)
    cn2_hv = profiles.hufnagel_valley_xr(
        cn2_0=cn2_0, m=get_wind_mag_dir(X["u"], X["v"])["m"], z=X["z_agl"], z_eval=y["z_agl"], dim="bottom_top"
    )
    return xr.Dataset({"lcn2_hv": np.log10(cn2_hv)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--data-root", type=pathlib.Path, help="domain data dir (default: the config's data.root)")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--basename", default="wrfpl")
    parser.add_argument("--sel-test", type=json.loads, help="test selection as JSON (default: the config's)")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.sel_test is not None:
        config = config.model_copy(update={"data": config.data.model_copy(update={"sel_test": args.sel_test})})
    exp = Experiment.from_config(config, data_root=args.data_root)
    exp.pipeline.prepare_data()  # no fitting needed for unscaled inputs
    X_test, y_test = exp.pipeline.split("test", transform=False)
    args.out.mkdir(parents=True, exist_ok=True)
    compute_hv_w71(X_test, y_test).reset_index("sample").to_netcdf(args.out / f"{args.basename}_ref.nc")
    print(f"Reference profiles saved to {args.out / args.basename}_ref.nc")

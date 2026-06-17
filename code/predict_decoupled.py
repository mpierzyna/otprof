from __future__ import annotations

import logging
import pathlib

import lightning as L
import numpy as np
import torch
import xarray as xr

logging.basicConfig(level=logging.INFO)

import experiments
from otprof import pipelines
from research_tools.ot import profiles


def get_best_ckpt(model_path: pathlib.Path) -> pathlib.Path:
    """Get best checkpoint from `model_path` based on validation score."""

    def _get_score(f: pathlib.Path) -> float:
        data = torch.load(f, map_location="cpu")
        (ckpt_state,) = [v for k, v in data["callbacks"].items() if k.startswith("ModelCheckpoint")]
        return ckpt_state["current_score"].item()

    # Pick best model checkpoint
    ckpt_dir = model_path / "checkpoints"
    ckpt_files = sorted(list(ckpt_dir.glob("*.ckpt")))
    ckpt_scores = [_get_score(f) for f in ckpt_files]
    best_ckpt = ckpt_files[torch.tensor(ckpt_scores).argmin()]
    print(f"Using best checkpoint: {best_ckpt} with score {_get_score(best_ckpt):.4f}")

    return best_ckpt


def predict(
    exp: experiments.SqueezeFormerExp,
    model_path: pathlib.Path,
    out_dir: pathlib.Path,
    out_basename: str,
    z_hr_pred: np.ndarray | None = None,
):
    """Make predictions for `exp` using model stored in `model_path` and save to `out_dir / out_name`."""
    assert model_path.exists(), f"Model path {model_path} does not exist."
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare test data loader and model
    best_ckpt = get_best_ckpt(model_path)
    loader_test = exp.get_torch_loader(stage="test")
    model = exp.get_lit_model(data_train=loader_test.dataset)  # noqa: note, not train but should be fine

    # For simplicity, save also the untransformed test data
    X_test, y_test = exp.dp.split("test", transform=False)
    X_test.reset_index("sample").to_netcdf(out_dir / f"{out_basename}_X_test.nc")
    y_test.reset_index("sample").to_netcdf(out_dir / f"{out_basename}_y_test.nc")

    # Optionally, set fixed z_hr levels for prediction
    if z_hr_pred is not None:
        model.z_hr_pred = torch.tensor(z_hr_pred, dtype=torch.float32)
        out_basename = f"{out_basename}_nzhr={len(z_hr_pred)}"
        # todo: save z_hr_pred to disk

    # Make predictions
    trainer = L.Trainer(
        accelerator="auto",  # GPU if available, else CPU
        devices="auto",
        num_nodes=1,
        enable_checkpointing=False,
        logger=False,
    )
    preds = trainer.predict(model, loader_test, ckpt_path=best_ckpt)

    # deterministic training with confidence output
    y_pred, y_conf = zip(*preds)
    y_pred = torch.concat(y_pred, dim=0)  # (samples, levels, vars)
    y_conf = torch.concat(y_conf, dim=0)  # (samples, levels, vars)

    # Inverse trafo
    ds_y_pred = loader_test.dataset.unstack_y(y_pred, sample_coord="self")  # noqa
    _, ds_y_pred = exp.dp.inverse_transform(ds_y=ds_y_pred)
    ds_y_pred.reset_index("sample").to_netcdf(out_dir / f"{out_basename}_y_pred.nc")
    print(f"Predictions saved to {out_dir / out_basename}*.nc")


def wrf_to_era5(
    exp: experiments.SqueezeFormerExp,
    quantile_mapping: bool,
) -> experiments.SqueezeFormerExp:
    """Create new experiment using ERA5 data matched to WRF levels."""
    features = exp.features + ["m"]  # m is needed for HV profile
    if quantile_mapping:
        # Exclude features which are the same in both datasets (temp features)
        # LSM is excluded because interpolation on binary mask needs extra treatment
        qm_exclude = ["lsm", "hr_sin", "hr_cos", "doy_sin", "doy_cos"]
        features_qm = [f for f in features if f not in qm_exclude]
    else:
        features_qm = None

    # Prepare ERA5 data matched to WRF levels
    dp_era5 = pipelines.era5_wrf.p_era5pl_wrf.setup()
    dp_era5 = pipelines.era5_wrf.era5_match_wrf(
        dp_era5=dp_era5,
        dp_wrf=exp.dp,
        features=features,
        quantile_mapping=quantile_mapping,
        features_qm=features_qm,
    )

    # Create updated experiment with ERA5 data pipeline
    exp_era5 = exp.model_copy(update={"name": f"{exp.name}_era5", "dp": dp_era5}, deep=True)
    return exp_era5


def predict_ref(exp: experiments.SqueezeFormerExp, out_dir: pathlib.Path, out_basename: str):
    # Get data for experiment
    X_test, y_test = exp.dp.split("test", transform=False)
    if "bottom_top_h" in X_test.dims:
        X_test = X_test.rename(bottom_top_h="bottom_top")  # todo: inconsistency in data pipelines
    if "bottom_top_h" in y_test.dims:
        y_test = y_test.rename(bottom_top_h="bottom_top")  # todo: inconsistency in data pipelines

    ## Hufnagel-Valley profile
    # Convert surface CT2 from W71 to Cn2
    p_0 = X_test["p"].isel(bottom_top=0) / 100  # hPa
    tk = X_test["tk2"]
    ct2_0 = 10 ** X_test["lct2_w71f"]
    cn2_0 = (79e-6 * p_0 / tk**2) ** 2 * ct2_0

    # Evaluate HV
    print("Computing Hufnagel-Valley Cn2 profile")
    cn2_hv = profiles.hufnagel_valley_xr(
        cn2_0=cn2_0,
        m=X_test["m"],  # use X for W computation
        z=X_test["z_agl"],
        z_eval=y_test["z_agl"],  # evaluate using y levels
        dim="bottom_top",
    )  # will have dim of outputs

    ## Osborn-Sarazin profile
    print("Computing Osborn-Sarazin Cn2 profile")
    cn2_os = profiles.osborn_sarazin_xr(
        th=X_test["th"],
        Gamma=X_test["dth_dz"],
        S=X_test["S"],
        p_hPa=X_test["p"] / 100,
        dim="bottom_top",
    )
    cn2_os = cn2_os.rename(bottom_top="bottom_top_")  # will have dim of inputs

    ds = xr.Dataset(
        {
            "lcn2_hv": np.log10(cn2_hv),
            "lcn2_os": np.log10(cn2_os),
        }
    )
    ds.reset_index("sample").to_netcdf(out_dir / f"{out_basename}_ref.nc")
    print(f"Reference profiles saved to {out_dir / out_basename}_ref.nc")


if __name__ == "__main__":
    # Bundled pretrained checkpoints (see README)
    model_root = pathlib.Path("lightning_logs")
    out_dir = pathlib.Path("saved_models/preds")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plain squeezeformer experiments
    wrf_native = experiments.get_exp("wrf_native")
    wrf_pl = experiments.get_exp("wrf_pl")
    era5_pl_direct = experiments.get_exp("era5_pl_direct")

    # The WRF-PL model applied to ERA5 inputs matched to WRF pressure levels,
    # without and with quantile mapping
    era5pl = wrf_to_era5(wrf_pl, quantile_mapping=False)
    era5pl_qm = wrf_to_era5(wrf_pl, quantile_mapping=True)

    predict(wrf_native, model_root / "version_176", out_dir=out_dir, out_basename="wrfnative")
    predict(wrf_pl, model_root / "version_175", out_dir=out_dir, out_basename="wrfpl")
    predict(era5pl, model_root / "version_175", out_dir=out_dir, out_basename="era5pl")
    predict(era5pl_qm, model_root / "version_175", out_dir=out_dir, out_basename="era5pl_qm")
    predict(era5_pl_direct, model_root / "version_179", out_dir=out_dir, out_basename="era5pl_direct")

    # Literature reference profiles (Hufnagel-Valley) based on WRF-PL inputs
    predict_ref(wrf_pl, out_dir=out_dir, out_basename="wrfpl")

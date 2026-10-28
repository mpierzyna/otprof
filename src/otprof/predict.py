"""Prediction: a trained model with its processing chain (`Predictor`), loaded from a bundle, the hub or a training run,
and `predict` on the model inputs.

    predictor = load_model(sqf_h.HF_REPO_ID, revision=sqf_h.HF_REVISION)   # or a bundle / run directory
    X = predictor.make_inputs(era5.standardize(ds))   # the model's inputs from canonical data (time, loc, bottom_top)
    y = predict(predictor, X)

The chain is the one of training and evaluation: `pipeline.forward_X` -> network -> `pipeline.backward_y` (inverse
target scaling and transforms, then the `post_y` steps, e.g. the output quantile mapping, if `oqm`). The pipeline is
loaded without data. Evaluation uses the same `predict` on `pipeline.split("test", transform=False)` with the target
heights as query heights.

Needs `otprof[torch]` (and `otprof[hub]` to download from Hugging Face).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import warnings

import numpy as np
import torch
import xarray as xr

from otprof.experiment import ExperimentConfig
from otprof.models import get_model
from otprof.models.bundle import check_versions, load_network, load_network_from_checkpoint
from otprof.physics import ot
from otprof.pipeline import BaseDataPipeline
from otprof.transforms import QuantileMapper

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Predictor:
    """A network with its processing chain and variable order.

    `vars_X`: input variables in tensor order (surface first), `forcings`: variables of the query heights the network
    is called with (made by the pipeline's `make_targets` from `z_agl`, as in training), `targets`: output variables in
    tensor order, `z_agl_default`: query heights if none are given (mean heights of the
    training targets), `config`: the bundle's `config.json` (versions, hparams).
    """

    model: torch.nn.Module
    pipeline: BaseDataPipeline
    vars_X: list[str]
    forcings: list[str]
    targets: list[str]
    z_agl_default: np.ndarray | None = None
    config: dict = dataclasses.field(default_factory=dict)

    def make_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        """The model's inputs (unscaled) from canonical inputs (`otprof.schema`, e.g. `era5.standardize(...)`)."""
        return self.pipeline.make_inputs(ds)


def get_best_ckpt(run_dir: pathlib.Path) -> pathlib.Path:
    """The checkpoint of `run_dir/checkpoints` with the lowest validation loss (as monitored in training)."""

    def _get_score(f: pathlib.Path) -> float:
        data = torch.load(f, map_location="cpu")
        (ckpt_state,) = [v for k, v in data["callbacks"].items() if k.startswith("ModelCheckpoint")]
        return ckpt_state["current_score"].item()

    ckpt_files = sorted((pathlib.Path(run_dir) / "checkpoints").glob("*.ckpt"))
    ckpt_scores = [_get_score(f) for f in ckpt_files]
    best_ckpt = ckpt_files[torch.tensor(ckpt_scores).argmin()]
    logger.info(f"Using best checkpoint: {best_ckpt} with score {min(ckpt_scores):.4f}")
    return best_ckpt


def load_model(
    path_or_repo_id: str | pathlib.Path,
    revision: str | None = None,
    ckpt: str | pathlib.Path | None = None,
    device: str = "cpu",
) -> Predictor:
    """Load a model for prediction from
    - a bundle directory (`config.json`, `model.safetensors`, `pipeline.json`, `steps.json`; see `otprof.export`),
    - a Hugging Face repo id (downloaded at `revision`, needs `otprof[hub]`; the published models and their pinned
      revisions are in their model packages, e.g. `otprof.models.sqf_h.HF_REPO_ID` / `HF_REVISION`), or
    - a training run directory (`config.yaml`, `data_pipeline/`, `checkpoints/`): the best checkpoint or `ckpt`.
    """
    path = pathlib.Path(path_or_repo_id)
    if not path.exists():
        from huggingface_hub import snapshot_download  # otprof[hub]

        path = pathlib.Path(snapshot_download(str(path_or_repo_id), revision=revision))

    if (path / "model.safetensors").exists():
        package = get_model(json.loads((path / "config.json").read_text())["model"])
        model, config = load_network(path, package, map_location=device)
        pipeline = BaseDataPipeline.from_disk(path, pipelines=package.PIPELINE_CLASSES)
        z_default = None if config.get("z_agl_default") is None else np.array(config["z_agl_default"])
        return Predictor(
            model=model.to(device).eval(),
            pipeline=pipeline,
            vars_X=config["vars_X"],
            forcings=config["forcings"],
            targets=config["targets"],
            z_agl_default=z_default,
            config=config,
        )

    if (path / "config.yaml").exists():
        exp_config = ExperimentConfig.from_yaml(path / "config.yaml")
        package = get_model(exp_config.model.name)
        ckpt = get_best_ckpt(path) if ckpt is None else pathlib.Path(ckpt)
        model = load_network_from_checkpoint(ckpt, package.get_network_class(), map_location=device)
        pipeline = BaseDataPipeline.from_disk(path / "data_pipeline", pipelines=package.PIPELINE_CLASSES)
        features = pipeline.vars_X if exp_config.features is None else exp_config.features
        return Predictor(
            model=model.to(device).eval(),
            pipeline=pipeline,
            vars_X=pipeline.get_input_order(features),
            forcings=exp_config.forcings,
            targets=exp_config.targets,
        )

    raise FileNotFoundError(f"{path}: neither a model bundle (model.safetensors) nor a training run (config.yaml)")


def _sample_dims(X: xr.Dataset) -> list[str]:
    """Dims of the inputs other than `bottom_top`, in order of appearance."""
    dims = []
    for da in X.data_vars.values():
        dims += [d for d in da.dims if d != "bottom_top" and d not in dims]
    return dims


def _stack(ds: xr.Dataset | xr.DataArray, dims: list[str]):
    return ds if dims == ["sample"] else ds.stack(sample=dims)


def predict(
    predictor: Predictor,
    X: xr.Dataset,
    z_agl_query: xr.DataArray | np.ndarray | None = None,
    oqm: bool = True,
    batch_size: int = 4096,
    device: str | None = None,
) -> xr.Dataset:
    """Predict the targets at the query heights from inputs `X` (`make_inputs` output with dims (..., bottom_top), e.g.
    (time, loc, bottom_top) or the stacked (sample, bottom_top) of a pipeline split; not scaled).

    `z_agl_query`: heights above ground [m], increasing; one profile (`bottom_top` or 1-D array) for all samples, or
    per sample (dims of `X` + `bottom_top`). Default: `predictor.z_agl_default`. `oqm`: apply the `post_y` steps
    (output quantile mapping of lcn2), recorded in the attrs.

    Returns the targets (`lcn2`, `QKE`, `TSQ`, `EL_PBL`), `z_agl` and, integrated over the query heights from the
    returned `lcn2` (1550 nm), the Fried parameter `r0` [cm] and the log10 plane-wave scintillation index `lsi2`;
    dims (..., bottom_top) with `bottom_top` the query levels.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    for key in ["features_version", "schema_version"]:
        if f"otprof_{key}" in X.attrs and key in predictor.config:
            check_versions(predictor.config, {key: X.attrs[f"otprof_{key}"]}, source="model")

    # Inputs -> model space, stacked to (sample, bottom_top)
    dims = _sample_dims(X[predictor.vars_X])
    X_s = predictor.pipeline.forward_X(_stack(X[predictor.vars_X], dims))
    # Per-column variables repeated over the levels, constant profiles (e.g. `p` on pressure levels) over the samples
    shape = (X_s.sizes["sample"], X_s.sizes["bottom_top"])
    template = xr.DataArray(np.empty(shape, dtype=bool), dims=("sample", "bottom_top"))
    x = np.stack(
        [X_s[v].broadcast_like(template).transpose("sample", "bottom_top").values for v in predictor.vars_X],
        axis=-1,
    ).astype(np.float32)

    # Query heights per sample
    if z_agl_query is None:
        if predictor.z_agl_default is None:
            raise ValueError("No query heights: pass z_agl_query (the model has no default heights)")
        z_agl_query = predictor.z_agl_default
    if not isinstance(z_agl_query, xr.DataArray):
        z_agl_query = xr.DataArray(np.asarray(z_agl_query), dims="bottom_top")
    if z_agl_query.dims == ("bottom_top",):
        z_q = np.broadcast_to(z_agl_query.values, (X_s.sizes["sample"], z_agl_query.size))
    else:
        z_q = _stack(z_agl_query, dims).transpose("sample", "bottom_top").values
    _warn_coverage(z_q, predictor.z_agl_default)

    # Forcings of the query heights in model space, as in training
    ds_q = predictor.pipeline.make_targets(xr.Dataset({"z_agl": (("sample", "bottom_top"), z_q)}))
    ds_q = predictor.pipeline.forward_y(ds_q)
    frc = np.stack([ds_q[v].values for v in predictor.forcings], axis=-1).astype(np.float32)

    # Network, in batches
    model = predictor.model.to(device).eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            out = model(
                torch.from_numpy(x[i : i + batch_size]).to(device),
                torch.from_numpy(frc[i : i + batch_size]).to(device),
            )
            preds.append((out[0] if isinstance(out, tuple) else out).cpu())
    y = torch.cat(preds).numpy()

    # Model space -> physical targets (and oQM)
    ds_y = xr.Dataset(
        {v: (("sample", "bottom_top"), y[..., i]) for i, v in enumerate(predictor.targets)},
        coords={"sample": X_s["sample"]},
    )
    ds_y = predictor.pipeline.backward_y(ds_y, post=oqm)
    ds_y["z_agl"] = (("sample", "bottom_top"), z_q)
    if "lcn2" in ds_y:
        ds_y = ds_y.assign(get_integrated_params(ds_y["lcn2"], ds_y["z_agl"]))
    ds_y = ds_y.assign_attrs(otprof_oqm=int(oqm and bool(predictor.pipeline.post_y)))
    return ds_y if dims == ["sample"] else ds_y.unstack("sample").transpose(*dims, "bottom_top")


def get_integrated_params(lcn2: xr.DataArray, z_agl: xr.DataArray) -> dict[str, xr.DataArray]:
    """Fried parameter `r0` [cm] and log10 plane-wave scintillation index `lsi2` of log10 Cn2 profiles (1550 nm),
    integrated over `bottom_top`."""
    cn2 = 10**lcn2
    r0 = (ot.fried_r0_xr(cn2=cn2, z=z_agl, dim="bottom_top") * 100).assign_attrs(
        long_name="Fried parameter", units="cm"
    )
    si = ot.scint_index_xr(cn2=cn2, z=z_agl, dim="bottom_top", mode="plane")
    lsi2 = np.log10(si).assign_attrs(long_name="log10 scintillation index (plane wave)", units="1")
    return {"r0": r0, "lsi2": lsi2}


def _warn_coverage(z_q: np.ndarray, z_default: np.ndarray | None) -> None:
    """r0 and lsi2 integrate over the query heights: warn if they span less than the training column."""
    if z_default is None:
        return
    if z_q.min() > z_default[0] * 1.01 or z_q.max() < z_default[-1] * 0.99:
        warnings.warn(
            f"Query heights {z_q.min():.0f}-{z_q.max():.0f} m do not cover the training column "
            f"{z_default[0]:.0f}-{z_default[-1]:.0f} m: r0 and lsi2 integrate over the query heights only",
            stacklevel=3,
        )


def fit_oqm(pred: xr.Dataset, truth: xr.Dataset, n_quantiles: int = 100, grid: str = "sine") -> QuantileMapper:
    """Output quantile mapping of lcn2: one transfer function for the whole column (all levels pooled), fitted on
    predictions `pred` against the reference `truth` (same samples, e.g. WRF). Append it to `pipeline.post_y` to apply
    it in `predict`."""
    qm = QuantileMapper(n_quantiles=n_quantiles, vars=["lcn2"], per_level=False, grid=grid)
    return qm.fit(X_obs=truth[["lcn2"]], X_model=pred[["lcn2"]])

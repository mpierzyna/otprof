"""Processing steps between the dataset files (or `make_inputs` at inference) and the model.

A pipeline (`otprof.pipeline`) holds three ordered lists of steps: `pre_X` (inputs -> model), `pre_y` (targets <->
model space, inverted for the outputs) and `post_y` (outputs only, e.g. output quantile mapping). Each step works on
xarray Datasets and serializes to plain JSON (`to_dict` / `from_dict`); `STEPS` maps the name stored in the JSON to
the class. Steps:

- `Transforms`: named, fixed functions per variable (e.g. log10 of QKE), from the plain dict `TRANSFORMS`
- `DatasetScaler`: location / scale per variable, fitted on the training data
- `QuantileMapper`: quantile mapping, fitted on model output vs reference

A new kind of step is a class with the methods of `Step` (or `InvertibleStep`) plus an entry in `STEPS`. Fitting is
not part of the interface: steps fit on different data (`pre_*` on the training split, `post_y` on predictions).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Protocol, Self, TypeVar, runtime_checkable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Scaler configuration per variable: (dims to aggregate over (None: all), scaler kind, extra arguments).
# "robust" and "quantile" take (qmin, qmax) as extra arguments.
TScalerConfig = tuple[list[str] | None, Literal["standard", "robust", "minmax", "quantile"], tuple | None]


@runtime_checkable
class Step(Protocol):
    """One processing step on xr.Datasets that serializes to plain JSON."""

    def transform(self, ds: xr.Dataset) -> xr.Dataset: ...

    def to_dict(self) -> dict:
        """Config and fitted state; the key "step" holds the name in `STEPS`."""
        ...

    @classmethod
    def from_dict(cls, d: dict) -> Self: ...


@runtime_checkable
class InvertibleStep(Step, Protocol):
    """A step whose transform can be undone (needed for the targets: model outputs are mapped back)."""

    def inverse_transform(self, ds: xr.Dataset) -> xr.Dataset: ...


## Named transforms


def log10_clip(da: T, min: float) -> T:
    return np.log10(da.clip(min=min))


def log10_clip_inv(x: T, min: float) -> T:
    return 10**x


# Name -> (forward, inverse). Both get the variable and the parameters stored with the name, e.g.
# {"QKE": ["log10_clip", {"min": 1e-8}]}.
TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "log10_clip": (log10_clip, log10_clip_inv),
}


class Transforms:
    """Fixed functions per variable, looked up by name in `transforms` (default: `TRANSFORMS`).

    `config` maps a variable to [name, parameters], e.g. {"QKE": ["log10_clip", {"min": 1e-8}]}. Variables not in the
    dataset are skipped, variables not in `config` pass through.
    """

    def __init__(self, config: dict[str, tuple[str, dict[str, Any]]], transforms: dict | None = None):
        self.transforms = TRANSFORMS if transforms is None else transforms
        unknown = {name for name, _ in config.values()} - set(self.transforms)
        if unknown:
            raise ValueError(f"Unknown transforms {sorted(unknown)}. Known: {sorted(self.transforms)}")
        self.config = {v: (name, dict(params)) for v, (name, params) in config.items()}

    def _apply(self, ds: xr.Dataset, inverse: bool) -> xr.Dataset:
        ds = ds.copy()
        for v, (name, params) in self.config.items():
            if v in ds:
                logger.info(f"{'Inverting' if inverse else 'Applying'} transformation for variable '{v}'")
                fn = self.transforms[name][1 if inverse else 0]
                ds[v] = fn(ds[v], **params)
        return ds

    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        return self._apply(ds, inverse=False)

    def inverse_transform(self, ds: xr.Dataset) -> xr.Dataset:
        return self._apply(ds, inverse=True)

    def to_dict(self) -> dict:
        return {"step": "transforms", "config": {v: [name, params] for v, (name, params) in self.config.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(d["config"])


## Serialization of fitted arrays


def _array_to_dict(a: np.ndarray) -> dict:
    """Exact JSON form of an array (Python floats round-trip float32 and float64 exactly)."""
    a = np.asarray(a)
    return {"dtype": str(a.dtype), "data": a.tolist()}


def _array_from_dict(d: dict) -> np.ndarray:
    return np.array(d["data"], dtype=d["dtype"])


def _da_to_dict(da: xr.DataArray) -> dict:
    return {
        "name": da.name,
        "dims": list(da.dims),
        **_array_to_dict(da.values),
        "coords": {str(k): {"dims": list(c.dims), **_array_to_dict(c.values)} for k, c in da.coords.items()},
        "attrs": dict(da.attrs),
    }


def _da_from_dict(d: dict) -> xr.DataArray:
    coords = {k: (c["dims"], _array_from_dict(c)) for k, c in d["coords"].items()}
    return xr.DataArray(_array_from_dict(d), dims=d["dims"], coords=coords, name=d["name"], attrs=d["attrs"])


## Scaler


class DatasetScaler:
    def __init__(
        self,
        config: dict[str, TScalerConfig],
        remainder: Literal["passthrough", "drop"] = "passthrough",
    ):
        """Scaler for xarray.Dataset
        allowing different scalers and different aggregations dimensions per variable.

        Parameters
        ----------
        config : Dict
            Configuration for each variable to scale.
            The key is the variable name and the value is a tuple with:
            - Optional[List[str]]: Dimensions to aggregate over. If None, all dimensions are used.
            - Literal["standard", "robust", "minmax", "quantile"]: Scaler to use.
            - Optional[Tuple]: Additional arguments for the scaler. Provide qmin and qmax for "quantile".
        """
        self.config = config
        self.remainder = remainder

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "loc_")

    def fit(self, ds: xr.Dataset) -> Self:
        # Compute location
        self.loc_ = {}
        self.scale_ = {}

        for v in ds:
            # Cast variable name to str (in case it's not)
            v = str(v)

            # Check if config provided for variable
            try:
                dims, scaler, args = self.config[v]
            except KeyError:
                logger.warning(f"No scaler configured for {v}.")
                continue

            if scaler == "standard":
                # Standard scaling: zero mean, unit variance
                self.loc_[v] = ds[v].mean(dims)
                self.scale_[v] = ds[v].std(dims)

            elif scaler == "robust":
                # Robust scaling: zero median, quantile range
                # Assume args are qmin and qmax
                qmin, qmax = args
                v_qmin = ds[v].quantile(qmin, dims).drop_vars("quantile")
                v_qmax = ds[v].quantile(qmax, dims).drop_vars("quantile")
                v_range = v_qmax - v_qmin
                self.loc_[v] = ds[v].median(dims)
                self.scale_[v] = v_range

            elif scaler == "minmax":
                # Min-Max scaling
                v_min = ds[v].min(dims)
                v_max = ds[v].max(dims)
                v_range = v_max - v_min
                self.loc_[v] = v_min
                self.scale_[v] = v_range

            elif scaler == "quantile":
                # Min-Max scaling based on quantiles
                # Assume args are qmin and qmax
                qmin, qmax = args
                v_qmin = ds[v].quantile(qmin, dims).drop_vars("quantile")
                v_qmax = ds[v].quantile(qmax, dims).drop_vars("quantile")
                v_range = v_qmax - v_qmin
                self.loc_[v] = v_qmin
                self.scale_[v] = v_range

            else:
                raise ValueError(f"Unknown scaler {scaler}")

        return self

    def transform(self, ds: xr.Dataset) -> xr.Dataset:
        ds_scaled = xr.Dataset()
        for v in ds:
            if v in self.config:
                tf_fn = self.get_tf_fn(v)
                ds_scaled[v] = tf_fn(ds[v])
            else:
                if self.remainder == "passthrough":
                    ds_scaled[v] = ds[v]

        return ds_scaled

    def inverse_transform(self, ds: xr.Dataset) -> xr.Dataset:
        ds_unscaled = xr.Dataset()
        for v in ds:
            if v in self.config:
                da = ds[v]
                inv_tf_fn = self.get_inv_tf_fn(v)

                # Unscale and cast back to original dtype.
                # Otherwise, memory might blow up going from float32 to float64.
                da_dtype = da.dtype
                da = inv_tf_fn(da)
                da = da.astype(da_dtype)

                if "quantile" in da.coords:
                    da = da.drop_vars("quantile")
                ds_unscaled[v] = da
            else:
                if self.remainder == "passthrough":
                    ds_unscaled[v] = ds[v]

        return ds_unscaled

    def get_tf_fn(self, v: str, mode: Literal["xr", "torch"] = "xr") -> Callable[[T], T]:
        """Get transform function for variable v."""
        if v not in self.config:
            raise ValueError(f"No scaler configured for {v}")

        loc = self.loc_[v]
        scale = self.scale_[v]

        if mode == "torch":
            import torch

            scale = torch.from_numpy(scale.values).to(torch.float32)
            loc = torch.from_numpy(loc.values).to(torch.float32)

        def fn(x: T) -> T:
            return (x - loc) / scale

        return fn

    def get_inv_tf_fn(self, v: str, mode: Literal["xr", "torch"] = "xr") -> Callable[[T], T]:
        """Get inverse transform function for variable v."""
        if v not in self.config:
            raise ValueError(f"No scaler configured for {v}")

        loc = self.loc_[v]
        scale = self.scale_[v]

        if mode == "torch":
            import torch

            scale = torch.from_numpy(scale.values).to(torch.float32)
            loc = torch.from_numpy(loc.values).to(torch.float32)

        def fn(x: T) -> T:
            return x * scale + loc

        return fn

    def to_dict(self) -> dict:
        d = {
            "step": "scaler",
            "config": {v: [dims, scaler, args] for v, (dims, scaler, args) in self.config.items()},
            "remainder": self.remainder,
        }
        if self.is_fitted:
            d["loc"] = {v: _da_to_dict(da) for v, da in self.loc_.items()}
            d["scale"] = {v: _da_to_dict(da) for v, da in self.scale_.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        config = {
            v: (dims, scaler, None if args is None else tuple(args)) for v, (dims, scaler, args) in d["config"].items()
        }
        scaler = cls(config, remainder=d["remainder"])
        if "loc" in d:
            scaler.loc_ = {v: _da_from_dict(da) for v, da in d["loc"].items()}
            scaler.scale_ = {v: _da_from_dict(da) for v, da in d["scale"].items()}
        return scaler


## Quantile mapping


class QuantileMapper:
    """Bias correction using quantile mapping for xarray Datasets.

    With `per_level=False` (default) quantiles are pooled over all dimensions per variable.
    With `per_level=True`, profile variables (those with both a sample and a bottom_top*
    dimension in obs AND model data) get one transfer function per vertical level; all other
    variables fall back to the pooled mapping. Obs and model may use different vertical dim
    names (e.g. bottom_top_h vs bottom_top) but must have the same level count.

    `grid`: probabilities of the quantiles. "linear" spaces them evenly over [0, 1] and extrapolates
    the outer segments linearly. "sine" uses p = (1 - cos(pi u)) / 2 for even u without the end
    points u = 0, 1: dense at both tails and coarse in the center, but without the sample minimum
    and maximum. Beyond the outermost quantiles values get the constant correction of those
    quantiles (slope 1): the extreme tails, where the model does not resolve the reference's
    variability, are shifted instead of stretched onto the reference extremes.
    """

    @staticmethod
    def _z_dim(da: xr.DataArray) -> str | None:
        """Vertical dim of a profile variable, or None if not a (sample, level) profile."""
        z_dims = [d for d in da.dims if str(d).startswith("bottom_top")]
        if z_dims and "sample" in da.dims:
            return str(z_dims[0])
        return None

    def __init__(
        self, n_quantiles: int, vars: list, per_level: bool = False, grid: Literal["linear", "sine"] = "linear"
    ):
        self.n_quantiles = n_quantiles
        self.grid = grid
        if grid == "linear":
            self.prob_grid = np.linspace(0, 1, n_quantiles)
        else:
            u = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # without the end points (min / max)
            self.prob_grid = (1 - np.cos(np.pi * u)) / 2
        self.q_obs_ = {}
        self.q_mod_fit_ = {}
        self.vars = vars
        self.per_level = per_level
        self.per_level_vars_ = set()

    @property
    def is_fitted(self) -> bool:
        return bool(self.q_obs_)

    def fit(self, X_obs: xr.Dataset, X_model: xr.Dataset) -> Self:
        """
        X_obs: Reference/Observed dataset (F_O)
        X_model: Training/Historical period of the model (F_M)
        """
        for var in self.vars:
            zd_obs = self._z_dim(X_obs[var])
            zd_mod = self._z_dim(X_model[var])
            if self.per_level and zd_obs and zd_mod:
                da_obs = X_obs[var].transpose("sample", zd_obs)
                da_mod = X_model[var].transpose("sample", zd_mod)
                if da_obs.sizes[zd_obs] != da_mod.sizes[zd_mod]:
                    raise ValueError(
                        f"Level count mismatch for '{var}': obs {da_obs.sizes[zd_obs]} vs model {da_mod.sizes[zd_mod]}"
                    )
                # Quantiles per level: (n_quantiles, levels)
                self.q_obs_[var] = np.quantile(da_obs.values, self.prob_grid, axis=0)
                self.q_mod_fit_[var] = np.quantile(da_mod.values, self.prob_grid, axis=0)
                self.per_level_vars_.add(var)
            else:
                # Quantiles of observations / model pooled over all dimensions
                self.q_obs_[var] = X_obs[var].quantile(self.prob_grid).values
                self.q_mod_fit_[var] = X_model[var].quantile(self.prob_grid).values

        return self

    def _transfer(self, x: np.ndarray, q_mod: np.ndarray, q_obs: np.ndarray) -> np.ndarray:
        """Transfer function model quantiles -> observed quantiles applied to x (see `grid`)."""
        if self.grid == "linear":
            from scipy.interpolate import interp1d

            return interp1d(q_mod, q_obs, kind="linear", bounds_error=False, fill_value="extrapolate")(x)
        y = np.interp(x, q_mod, q_obs)
        y = np.where(x > q_mod[-1], x + (q_obs[-1] - q_mod[-1]), y)  # constant correction beyond the ends
        return np.where(x < q_mod[0], x + (q_obs[0] - q_mod[0]), y)

    def transform(self, X_model: xr.Dataset) -> xr.Dataset:
        """Map future/test model data to the observed distribution."""
        X_out = X_model.copy()

        for var in self.vars:
            # 1. Map model values to their cumulative probabilities (F_M)
            # 2. Map those probabilities to observed values (F_O_inv)
            # This effectively does: x_out = interp_obs(interp_model_inv(x_mod))
            if var in self.per_level_vars_:
                zd = self._z_dim(X_model[var])
                da = X_model[var].transpose("sample", zd)
                vals = da.values
                out = np.empty_like(vals, dtype=float)
                for lvl in range(vals.shape[1]):
                    out[:, lvl] = self._transfer(vals[:, lvl], self.q_mod_fit_[var][:, lvl], self.q_obs_[var][:, lvl])
                X_out[var] = (da.dims, out.astype(vals.dtype, copy=False))
            else:
                vals = X_model[var].values
                out = self._transfer(vals.flatten(), self.q_mod_fit_[var], self.q_obs_[var]).reshape(vals.shape)
                X_out[var].values = out.astype(vals.dtype, copy=False)  # keep the dtype, as per level

        return X_out

    def to_dict(self) -> dict:
        return {
            "step": "quantile_map",
            "n_quantiles": self.n_quantiles,
            "vars": list(self.vars),
            "per_level": self.per_level,
            "grid": self.grid,
            "q_obs": {v: _array_to_dict(q) for v, q in self.q_obs_.items()},
            "q_mod": {v: _array_to_dict(q) for v, q in self.q_mod_fit_.items()},
            "per_level_vars": sorted(self.per_level_vars_),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        qm = cls(n_quantiles=d["n_quantiles"], vars=d["vars"], per_level=d["per_level"], grid=d["grid"])
        qm.q_obs_ = {v: _array_from_dict(q) for v, q in d["q_obs"].items()}
        qm.q_mod_fit_ = {v: _array_from_dict(q) for v, q in d["q_mod"].items()}
        qm.per_level_vars_ = set(d["per_level_vars"])
        return qm


# Name in the JSON ("step") -> class
STEPS: dict[str, type] = {
    "transforms": Transforms,
    "scaler": DatasetScaler,
    "quantile_map": QuantileMapper,
}


def step_from_dict(d: dict, steps: dict[str, type] | None = None) -> Step:
    """Rebuild a step from its `to_dict` output; `steps` (default: `STEPS`) maps the stored name to the class."""
    steps = STEPS if steps is None else steps
    if d["step"] not in steps:
        raise ValueError(f"Unknown step '{d['step']}'. Known: {sorted(steps)}")
    return steps[d["step"]].from_dict(d)

from typing import Callable, Dict, Literal, TypeVar, Self

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin

from otprof.logging import get_logger
from otprof.types import TScalerConfig

logger = get_logger()

T = TypeVar("T")


class DatasetScaler(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        config: Dict[str, TScalerConfig],
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

    def fit(self, ds: xr.Dataset):
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


class QuantileMapper(BaseEstimator, TransformerMixin):
    """Bias correction using quantile mapping for xarray Datasets."""

    def __init__(self, n_quantiles: int, vars: list):
        self.n_quantiles = n_quantiles
        self.prob_grid = np.linspace(0, 1, n_quantiles)
        self.q_obs_ = {}
        self.q_mod_fit_ = {}
        self.vars = vars

    def fit(self, X_obs: xr.Dataset, X_model: xr.Dataset) -> Self:
        """
        X_obs: Reference/Observed dataset (F_O)
        X_model: Training/Historical period of the model (F_M)
        """
        for var in self.vars:
            # Quantiles of observations
            self.q_obs_[var] = X_obs[var].quantile(self.prob_grid).values
            # Quantiles of the model during the calibration period
            self.q_mod_fit_[var] = X_model[var].quantile(self.prob_grid).values

        return self

    def transform(self, X_model: xr.Dataset) -> xr.Dataset:
        """Map future/test model data to the observed distribution."""
        X_out = X_model.copy()

        for var in self.vars:
            original_shape = X_model[var].shape
            flat_values = X_model[var].values.flatten()

            # 1. Map model values to their cumulative probabilities (F_M)
            # 2. Map those probabilities to observed values (F_O_inv)
            # This effectively does: x_out = interp_obs(interp_model_inv(x_mod))

            # Create a transfer function: Model Quantiles -> Observed Quantiles
            transfer_func = interp1d(
                self.q_mod_fit_[var],
                self.q_obs_[var],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            X_out[var].values = transfer_func(flat_values).reshape(original_shape)

        return X_out

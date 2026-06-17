from __future__ import annotations

from typing import Callable, Tuple, Optional, List, Literal, Union

import torch
import xarray as xr
from matplotlib import pyplot as plt

# Callable for plotting results.
# Expects ds_X, ds_y_true, ds_y_pred and has to return figure.
# Alternatively, it can accept additional arguments.
TPlotFn = Union[
    Callable[["BaseExperiment", xr.Dataset, xr.Dataset, xr.Dataset], plt.Figure],
    Callable[["BaseExperiment", xr.Dataset, xr.Dataset, xr.Dataset, ...], plt.Figure],
]

# Tuple for dataset scaler configuration
# Expects (dims, scaler type, additional arguments)
TScalerConfig = Tuple[
    Optional[List[str]],
    Literal["standard", "robust", "minmax", "quantile"],
    Optional[Tuple],
]

# Callable computing loss
TLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # unweighted
TWeightLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # weighted
TStage = Literal["train", "val", "test"]

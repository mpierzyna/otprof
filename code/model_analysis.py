from __future__ import annotations
from typing import Tuple

import dataclasses
import logging
import pathlib
from typing import Dict, List, Union, Callable
from enum import StrEnum

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
import jax
import jax.numpy as jnp

from research_tools.misc_tools.reporter import BaseReport
from research_tools.ot import ot

# Disable jit for debugging
# jax.config.update("jax_disable_jit", True)

# Stay on CPU
jax.config.update("jax_platforms", "cpu")

# En/disable certain plots
PLOT_SCORES = True
FORCE_KDE = False

# Type for dimension argument
TDim = Union[None, str, List[str]]

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger.setLevel("INFO")


class DatasetKey(StrEnum):
    TRUE = "true"
    WRF_NATIVE = "pred_wrf_native"
    WRF_PL = "pred_wrfpl"
    ERA5_PL = "pred_era5pl"
    ERA5_PL_QM = "pred_era5pl_qm"
    ERA5_PL_DIRECT = "pred_era5pl_direct"
    HV_WRFPL = "pred_hv_wrfpl"
    # HV_ERA5PL = "pred_hv_era5pl"


@dataclasses.dataclass
class Pair:
    """Pair of one variable (DataArrays)"""

    name: str
    a: xr.DataArray
    b: xr.DataArray

    @property
    def is_profile(self) -> bool:
        return "bottom_top" in self.a.dims


@dataclasses.dataclass
class Array:
    """Array of one variable from different datasets"""

    name: str
    z: xr.DataArray
    data: Dict[DatasetKey, xr.DataArray]

    def __post_init__(self):
        # Ensure all required keys are present
        for key in DatasetKey:
            if key not in self.data:
                raise ValueError(f"Missing key {key} in Array data")

    def map(self, fn: Callable, *args, **kwargs) -> Array:
        """Apply function `fn` to each DataArray in the triplet."""
        return Array(
            name=self.name,
            z=self.z,
            data={k: fn(v, *args, **kwargs) for k, v in self.data.items()},
        )

    def map_(self, fname: str, *args, **kwargs) -> Array:
        """Apply DataArray method `fname` to each DataArray in the triplet."""
        return Array(
            name=self.name,
            z=self.z,
            data={k: getattr(v, fname)(*args, **kwargs) for k, v in self.data.items()},
        )

    def __getitem__(self, item: str | DatasetKey) -> xr.DataArray:
        """Get DataArray for dataset key `item`."""
        return self.data[item]

    def as_dict(self) -> Dict[DatasetKey, xr.DataArray]:
        return self.data

    def as_pairs(self) -> Dict[str, Pair]:
        pairs = {}
        true_da = self.data[DatasetKey.TRUE]
        for k, v in self.data.items():
            if k != DatasetKey.TRUE:
                pairs[f"{DatasetKey.TRUE}/{k}"] = Pair(self.name, true_da, v)
        return pairs

    def sel(self, sel) -> Array:
        """Select data from each DataArray in the triplet."""
        return self.map_("sel", sel)

    def isel(self, indexers) -> Array:
        """Index data from each DataArray in the triplet."""
        return self.map_("isel", indexers)

    def rename(self, name: str) -> Array:
        """Rename the triplet variable."""
        return Array(name=name, z=self.z, data=self.data)

    @property
    def is_profile(self) -> bool:
        return "bottom_top" in self.data[DatasetKey.TRUE].dims


@dataclasses.dataclass
class Dataset:
    """Triplet of datasets"""

    z: xr.DataArray  # vertical coordinate for all datasets
    z_lr: xr.DataArray  # average vertical coordinate for ERA5 dataset
    data: Dict[DatasetKey, xr.Dataset]

    def __post_init__(self):
        # Ensure all required keys are present
        for key in DatasetKey:
            if key not in self.data:
                raise ValueError(f"Missing key {key} in Dataset data")

    def __getitem__(self, v) -> Array:
        """Get triplet for variable `v`."""
        return Array(name=v, z=self.z, data={k: ds[v] for k, ds in self.data.items()})

    def sel(self, sel) -> Dataset:
        """Select data from each Dataset in the triplet."""
        return Dataset(
            z=self.z.sel(sel),
            z_lr=self.z_lr,
            data={k: ds.sel(sel) for k, ds in self.data.items()},
        )

    def isel(self, indexers) -> Dataset:
        """Index data from each Dataset in the triplet."""
        return Dataset(
            z=self.z.isel(indexers),
            z_lr=self.z_lr,
            data={k: ds.isel(indexers) for k, ds in self.data.items()},
        )


def open_datasets(pred_dir: pathlib.Path | str, use_dask: bool) -> Dataset:
    pred_dir = pathlib.Path(pred_dir)
    chunks = {"sample": "auto", "bottom_top": -1} if use_dask else None

    # WRF PL -> WRF NATIVE
    y_pred_wrfpl = (
        xr.open_dataset(pred_dir / "wrfpl_y_pred.nc", chunks=chunks)
        .set_index(sample=["time", "loc"])
        .rename(level="bottom_top")
    )
    # ERA5 PL -> WRF NATIVE
    y_pred_era5 = (
        xr.open_dataset(pred_dir / "era5pl_y_pred.nc", chunks=chunks)
        .set_index(sample=["time", "loc"])
        .rename(level="bottom_top")
    )
    # ERA5 PL -> QM -> WRF NATIVE
    y_pred_era5pl_qm = (
        xr.open_dataset(pred_dir / "era5pl_qm_y_pred.nc", chunks=chunks)
        .set_index(sample=["time", "loc"])
        .rename(level="bottom_top")
    )
    # WRF NATIVE -> WRF NATIVE
    y_pred_wrf_native = (
        # this comes from minmax experiment to assess if normalization results in qq-plot kink
        # -> unchanged plots, so normalization is not issue
        # xr.open_dataset(pred_dir / "wrfnative_minmax_y_pred.nc", chunks=chunks)
        xr.open_dataset(pred_dir / "wrfnative_y_pred.nc", chunks=chunks)
        .set_index(sample=["time", "loc"])
        .rename(level="bottom_top")
    )
    # ERA5 direct training
    y_pred_era5pl_direct = (
        xr.open_dataset(pred_dir / "era5pl_direct_y_pred.nc", chunks=chunks)
        .set_index(sample=["time", "loc"])
        .rename(level="bottom_top")
    )

    # Get ERA5 average vertical height
    X_wrfpl = xr.open_dataset(pred_dir / "wrfpl_X_test.nc", chunks=chunks).set_index(sample=["time", "loc"])
    z_era5 = X_wrfpl["z_agl"].mean("sample")  # average vertical coordinate over all samples

    # True WRF NATIVE
    y_true = xr.open_dataset(pred_dir / "wrfpl_y_test.nc", chunks=chunks).set_index(sample=["time", "loc"])

    # Literature references based on WRF PL inputs
    y_wrfpl_ref = xr.open_dataset(pred_dir / "wrfpl_ref.nc", chunks=chunks).set_index(sample=["time", "loc"])

    data = {
        DatasetKey.TRUE: y_true,
        DatasetKey.WRF_NATIVE: y_pred_wrf_native,
        DatasetKey.WRF_PL: y_pred_wrfpl,
        DatasetKey.ERA5_PL: y_pred_era5,
        DatasetKey.ERA5_PL_QM: y_pred_era5pl_qm,
        DatasetKey.ERA5_PL_DIRECT: y_pred_era5pl_direct,
        DatasetKey.HV_WRFPL: y_wrfpl_ref[["lcn2_hv"]].rename({"lcn2_hv": "lcn2"}).assign_coords(y_true.coords),
        # DatasetKey.HV_ERA5PL: y_era5pl_ref[["lcn2_hv"]].rename({"lcn2_hv": "lcn2"}).assign_coords(y_true.coords),
    }
    return Dataset(z=y_true["z_agl"], z_lr=z_era5, data=data)


def compute_metrics(pair: Pair, dim: TDim, per_profile: bool) -> xr.Dataset:
    """Compute metrics between a pair of DataArrays.
    If `per_profile` is True and the DataArrays are profiles, compute metrics per profile first,
    then average over other dimensions.
    """

    logger.debug(f"Computing metrics for {pair.name} over dimensions: {dim}")

    def _pearson_r(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute Pearson correlation coefficient between two DataArrays."""
        a_mean = a.mean(dim=dim)
        b_mean = b.mean(dim=dim)
        cov = ((a - a_mean) * (b - b_mean)).mean(dim=dim)
        a_std = a.std(dim=dim)
        b_std = b.std(dim=dim)
        return cov / (a_std * b_std)

    def _wasserstein(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute 1D Wasserstein distance between two DataArrays."""
        from scipy.stats import wasserstein_distance

        logger.debug("Computing Wasserstein distance for profiles")

        return xr.apply_ufunc(
            wasserstein_distance,
            a,
            b,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    def _bias(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute bias between two DataArrays."""
        return b.mean(dim) - a.mean(dim)

    def _rmse(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute RMSE between two DataArrays."""
        return np.sqrt(((a - b) ** 2).mean(dim=dim))

    def _crmse(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute centered RMSE between two DataArrays."""
        a_ = a - a.mean(dim)
        b_ = b - b.mean(dim)
        return _rmse(a_, b_, dim=dim)

    def _r2(a: xr.DataArray, b: xr.DataArray, dim: TDim) -> xr.DataArray:
        """Compute coefficient of determination (R^2) between two DataArrays.
        Attention! Asymmetric metric, assuming `a` is the true value and `b` the prediction.
        """
        ss_res = ((a - b) ** 2).sum(dim=dim)
        ss_tot = ((a - a.mean(dim)) ** 2).sum(dim=dim)
        return 1 - ss_res / ss_tot

    metrics = {
        "lat": pair.a["lat"][0],
        "lon": pair.a["lon"][0],
    }

    if pair.is_profile and per_profile:
        # For profiles, we compute metrics first between profiles, then average over other dims.
        dim_z = "bottom_top"
        dim_ = dim if dim is None else [d for d in dim if d != dim_z]

        metrics["bias"] = _bias(pair.a, pair.b, dim=dim_z).mean(dim=dim_)
        metrics["rmse"] = _rmse(pair.a, pair.b, dim=dim_z).mean(dim=dim_)
        metrics["crmse"] = _crmse(pair.a, pair.b, dim=dim_z).mean(dim=dim_)
        metrics["r"] = _pearson_r(pair.a, pair.b, dim=dim_z).mean(dim=dim_)
        metrics["r2"] = _r2(pair.a, pair.b, dim=dim_z).mean(dim=dim_)

        # Compute Wasserstein distance for all profiles
        wd = _wasserstein(pair.a, pair.b, dim=dim_z)
        metrics["wasserstein"] = wd.mean(dim=dim_)
    else:
        metrics["bias"] = _bias(pair.a, pair.b, dim=dim)
        metrics["rmse"] = _rmse(pair.a, pair.b, dim=dim)
        metrics["crmse"] = _crmse(pair.a, pair.b, dim=dim)
        metrics["r"] = _pearson_r(pair.a, pair.b, dim=dim)
        metrics["r2"] = _r2(pair.a, pair.b, dim=dim)

    return xr.Dataset(metrics)


def compute_metrics_t(t: Array, dim: TDim, per_profile: bool = True) -> xr.Dataset:
    """Compute metrics DataFrame for all pairs in a Triplet."""
    logger.debug(f"Computing metrics for triplet {t.name}")
    pairs = t.as_pairs()
    metrics_list = []
    for n, pair in pairs.items():
        logger.debug(f"Computing metrics for pair: {n}")
        metrics = compute_metrics(pair, dim=dim, per_profile=per_profile)
        metrics = metrics.expand_dims(pair=[n])
        metrics_list.append(metrics)
    return xr.concat(metrics_list, dim="pair")


def plot_hist_qq(t: Array, kde: bool, **kwargs) -> plt.Figure:
    fig, (ax_hist, ax_qq) = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)
    if kde:
        logger.warning("KDE plotting enabled. This can be slow for large datasets.")

    # Plot histograms
    for i, (k, v) in enumerate(t.as_dict().items()):
        v.plot.hist(**kwargs, label=k, alpha=0.25, ax=ax_hist, color=f"C{i}", density=True)
        if kde:
            sns.kdeplot(x=v.values.flatten(), ax=ax_hist, color=f"C{i}", fill=False)
    ax_hist.legend()

    # Plot qq
    t_flat_sorted = t.map(lambda x: np.sort(x.values.flatten()))
    vmin = np.nanmin(t_flat_sorted.data[DatasetKey.TRUE])
    vmax = np.nanmax(t_flat_sorted.data[DatasetKey.TRUE])
    for i, (n, p) in enumerate(t_flat_sorted.as_pairs().items()):
        ax_qq.scatter(p.a, p.b, label=n, s=1, color=f"C{i+1}")
    ax_qq.plot([vmin, vmax], [vmin, vmax], ls="--", color="grey")
    ax_qq.text(0.01, 0.99, "overestimation", transform=ax_qq.transAxes, va="top", ha="left", color="grey")
    ax_qq.text(0.99, 0.01, "underestimation", transform=ax_qq.transAxes, va="bottom", ha="right", color="grey")
    ax_qq.set_xlabel(f"{t.name} true")
    ax_qq.set_ylabel(f"{t.name} pred")
    ax_qq.legend(loc="lower right")
    ax_qq.set_aspect("equal")

    return fig


def plot_metrics(metrics: xr.Dataset, **kwargs) -> plt.Figure:
    df = metrics.to_dataframe()
    n, m = df.shape

    fig, ax = plt.subplots(figsize=(m + 1, n), constrained_layout=True)
    sns.heatmap(df, annot=True, fmt=".3f", ax=ax, **kwargs)
    return fig


def plot_id_map(t: Array) -> plt.Figure:
    """Simple map of the locations and their numerical ID"""
    da = t.data[DatasetKey.TRUE].unstack("sample").isel(time=0)
    lat = da["lat"]
    lon = da["lon"]
    loc = da["loc"]

    fig: plt.Figure
    fig, ax = plt.subplots(
        figsize=(6, 4),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for la, lo, l in zip(lat.values, lon.values, loc.values):
        ax.plot(lo, la, marker="o", color="red", markersize=5, transform=ccrs.PlateCarree())
        ax.text(lo, la, str(l), transform=ccrs.PlateCarree(), fontsize=6)

    ax.coastlines()
    ax.set_title("Location IDs")
    return fig


def plot_metrics_map(t: Array, metrics: xr.Dataset, rel: bool) -> plt.Figure:
    # Make values relative to global mean
    if rel:
        metrics_mean = metrics.mean("loc")
        metrics = metrics - metrics_mean
        cmap = "bwr"
        vmax_ds = np.abs(metrics).max("loc")
        vmin_ds = -vmax_ds
    else:
        metrics_mean = None
        cmap = "viridis"
        vmin_ds, vmax_ds = None, None

    lat = metrics.lat
    lon = metrics.lon

    n = len(metrics)
    p = metrics.sizes["pair"]

    fig: plt.Figure
    fig, axarr = plt.subplots(
        nrows=n,
        ncols=p,
        figsize=(4 * p, 3 * n),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if n == 1:
        axarr = np.expand_dims(axarr, axis=0)  # type: ignore
    for ax_row, m in zip(axarr, metrics):
        for ax, i in zip(ax_row, range(p)):
            data = metrics[m].isel(pair=i)
            vmin = vmin_ds[m].isel(pair=i) if vmin_ds is not None else None
            vmax = vmax_ds[m].isel(pair=i) if vmax_ds is not None else None

            s = ax.scatter(lon, lat, c=data, cmap=cmap, s=20, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
            fig.colorbar(s, ax=ax)

            ax.coastlines()
            if rel:
                ax.set_title(f"{m} -- {data.pair.item()} ({metrics_mean[m].isel(pair=i).item():.3f} mean)")
            else:
                ax.set_title(f"{m} -- {data.pair.item()}")

    return fig


def plot_metrics_temp(t: Array) -> plt.Figure:
    # Unstack and compute metrics
    # Note, need to recompute because temporal compared to spatial metrics computed in `make_report`
    t = t.map_("unstack", "sample")
    if t.is_profile:
        # for profiles
        metrics = compute_metrics_t(t, dim=["loc", "bottom_top"])
    else:
        # for surface level variables
        metrics = compute_metrics_t(t, dim=["loc"])

    # Aggregate metrics per month
    metrics = metrics.groupby("time.month").mean()

    n = len(metrics)
    p = metrics.sizes["pair"]

    fig: plt.Figure
    fig, axarr = plt.subplots(
        nrows=n,
        ncols=p,
        figsize=(4 * p, 3 * n),
        constrained_layout=True,
        subplot_kw={"projection": "polar", "theta_direction": -1},
    )
    if n == 1:
        axarr = np.expand_dims(axarr, axis=0)  # type: ignore
    for ax_row, m in zip(axarr, metrics):
        for ax, i in zip(ax_row, range(p)):
            data = metrics[m].isel(pair=i)
            mon_rad = data.month * 2 * np.pi / 12  # convert month to radians
            ax.scatter(mon_rad, data.values)
            ax.set_title(f"{m} -- {data.pair.item()}")

            # set months as theta ticks
            ax.set_xticks(mon_rad)
            ax.set_xticklabels(data.month.values)
            ax.set_theta_zero_location("N")

    return fig


def plot_metrics_profile(t: Array) -> plt.Figure:
    # Unstack and compute metrics
    # Need to recompute as we don't want aggregated values
    t = t.map_("unstack", "sample")
    metrics = compute_metrics_t(t, dim=["time"], per_profile=False)  # per level plotting, so don't aggregate profiles
    metrics_mean = metrics.mean("loc")
    metrics -= metrics_mean  # substract location mean

    ncols = metrics.sizes["pair"] + 1  # one extra for loc-averaged profiles
    nrows = len(metrics)
    fig, axarr = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(4 * ncols, 4 * nrows),
        constrained_layout=True,
        sharey="all",
        width_ratios=[1] + [3] * (ncols - 1),
    )
    if nrows == 1:
        axarr = np.expand_dims(axarr, axis=0)  # type: ignore
    for ax_row, m in zip(axarr, metrics):
        ax_avg = ax_row[0]  # plot average over all locations here
        for ax, i in zip(ax_row[1:], range(metrics.sizes["pair"])):
            ax_avg.plot(
                metrics_mean[m].isel(pair=i),
                np.arange(metrics_mean.sizes["bottom_top"]),
            )
            metrics[m].isel(pair=i).plot(ax=ax)

    return fig


def argquant(x: xr.DataArray, q: List[float], dim: TDim) -> xr.DataArray:
    """Return indices of quantiles along a given dimension."""

    def _argquant(x):
        v = np.quantile(x, q)
        inds = np.searchsorted(x, v=v)
        inds = np.clip(inds, 0, len(x) - 1).astype(int)
        inds = np.argsort(x)[inds]  # translate sorted indices to original indices
        return xr.DataArray(inds, dims=["quantile"])

    i_quant = xr.apply_ufunc(
        _argquant,
        x,
        input_core_dims=[[dim]],
        output_core_dims=[["quantile"]],
        vectorize=True,
    )
    i_quant = i_quant.assign_coords(quantile=q)
    return i_quant


def plot_ts(t: Array, q: List[float]) -> List[go.Figure]:
    # Unstack and compute metrics
    t = t.map_("unstack", "sample")
    try:
        # for profiles
        metrics = compute_metrics_t(t, dim=["time", "bottom_top"])
    except Exception:
        # for surface level variables
        metrics = compute_metrics_t(t, dim=["time"])

    # Select locations for each pair based on RMSE quantiles
    m = "rmse"
    inds = argquant(metrics[m], q=q, dim="loc")
    t = t.sel({"loc": inds})
    t_ds = xr.Dataset(t.as_dict())
    vars = list(t_ds.data_vars)
    colors = [f"C{i}" for i in range(len(vars))]

    figs = []

    for p in t_ds.pair.values:
        # Convert to dataframe for plotly
        t_df = t_ds.sel(pair=p).to_dataframe().reset_index()

        # Plot
        fig = go.Figure()
        for qi in q:
            for var, c in zip(vars, colors):
                fig.add_trace(
                    go.Scatter(
                        x=t_df.loc[t_df["quantile"] == qi, "time"],
                        y=t_df.loc[t_df["quantile"] == qi, var],
                        mode="lines",
                        # name=f"{var} (loc={loc})",
                        legendgroup=f"q={qi}",
                        legendgrouptitle={"text": f"q={qi}"},
                        line={"color": c},
                    )
                )
        figs.append(fig)

    return figs


def plot_hexbin(t: Array, **kwargs) -> plt.Figure:
    t_pairs = t.as_pairs()
    m = len(t_pairs)
    vmin = t.data[DatasetKey.TRUE].min()
    vmax = t.data[DatasetKey.TRUE].max()

    fig, axarr = plt.subplots(nrows=1, ncols=m, figsize=(m * 3.5, 3), constrained_layout=True)
    for ax, (n, pair) in zip(axarr, t_pairs.items()):
        h = ax.hexbin(pair.a.values.flatten(), pair.b.values.flatten(), **kwargs)
        ax.plot([vmin, vmax], [vmin, vmax], ls="--", color="grey")
        ax.text(0.01, 0.99, "overestimation", transform=ax.transAxes, va="top", ha="left", color="grey")
        ax.text(0.99, 0.01, "underestimation", transform=ax.transAxes, va="bottom", ha="right", color="grey")
        a_label, b_label = n.split("/")
        ax.set_xlabel(f"{pair.name} {a_label}")
        ax.set_ylabel(f"{pair.name} {b_label}")
        ax.set_aspect("equal")
        fig.colorbar(h, ax=ax)

    return fig


def plot_random_profiles(t: Array, n_rand: int, seed: int) -> plt.Figure:
    """Plot random profiles from the triplet."""
    rng = np.random.default_rng(seed)
    inds = rng.choice(t.data[DatasetKey.TRUE].sizes["sample"], n_rand, replace=False)

    n_cols = 10
    n_rows = np.ceil(n_rand / n_cols).astype(int)

    fig, axarr = plt.subplots(
        ncols=n_cols,
        nrows=n_rows,
        figsize=(2 * n_cols, 3 * n_rows),
        tight_layout=True,
        sharex="all",
        sharey="all",
    )
    for ax, i in zip(axarr.flatten(), inds):
        # Plot profiles
        for j, (k, v) in enumerate(t.as_dict().items()):
            ax.plot(v.isel(sample=i), t.z.isel(sample=i), label=k, zorder=100, color=f"C{j}")

        # Add meta data
        p = t.data[DatasetKey.TRUE].isel(sample=i)
        time_str = pd.to_datetime(p["time"].item()).strftime("%Y-%m-%d %H:%M")
        ax.text(
            0.05,
            0.95,
            f"time: {time_str}\nloc: {p["loc"].item()}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )

        # Set sqrt scaling for y axis to stretch lower levels
        ax.set_yscale("function", functions=(lambda x: np.sqrt(x), lambda x: x**2))
        ax.margins(0)

    axarr[0, 0].legend(fontsize=8)  # first axis gets legend

    return fig


def plot_boxplot_hr(t: Array) -> plt.Figure:
    """Boxplot comparison of hourly data at all locations"""
    t_df = xr.Dataset(t.as_dict()).unstack("sample").isel(loc=0).to_dataframe()
    value_vars = list(t.as_dict().keys())
    t_df = t_df[value_vars].reset_index()
    # t_df["hour"] = t_df["time"].dt.hour
    t_df["month"] = t_df["time"].dt.month
    t_df = t_df.drop(columns=["time"]).melt(id_vars=["month"], value_vars=value_vars)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.boxplot(data=t_df, x="month", y="value", hue="variable", ax=ax)
    return fig


def make_report(t: Array) -> None:
    # Precompute spatial metrics
    if PLOT_SCORES:
        logger.info("Computing error metrics...")
        metrics = compute_metrics_t(
            t.map_("unstack", "sample"),  # unstack
            dim=["time", "bottom_top"] if t.is_profile else ["time"],  # for profiles, agg vertically
        ).drop_vars("time")

    logger.info(f"Generating report for {t.name}...")
    with BaseReport("Model Analysis", f"analysis_{t.name}.html") as r:
        if PLOT_SCORES:
            # Compute overall scores
            logger.info("Computing overall scores...")
            r.add_heading("Metrics", level=2)

            logger.info(f"Plotting global metrics")
            r.add_heading("Global", level=3)
            r.add_mpl_fig(
                plot_metrics(metrics.mean("loc")),
                caption=f"Metrics for {t.name} at all levels/locations/times.",
            )

            logger.info(f"Plotting spatial metrics map")
            r.add_heading("Spatial", level=3)
            r.add_mpl_fig(plot_id_map(t), caption="Location IDs on map.")
            r.add_mpl_fig(
                plot_metrics_map(t, metrics=metrics, rel=True),
                caption=f"Spatial error distribution of {t.name}, relative to global metric mean.",
            )

            logger.info(f"Plotting temporal/seasonal metrics")
            r.add_heading("Temporal/seasonal", level=3)
            r.add_mpl_fig(plot_metrics_temp(t), caption=f"Seasonal error distribution of {t.name}.")

        if t.is_profile:
            if PLOT_SCORES:
                logger.info(f"Plotting vertical profile metrics for {t.name}")
                r.add_heading("Vertical", level=3)
                r.add_mpl_fig(plot_metrics_profile(t))

            ## Smoothness
            r.add_heading("Smoothness", level=2)
            logger.info("Computing and plotting smoothness metrics...")
            smoothness = compute_smoothness_t(t.map_("unstack", "sample"), dim=["time"])  # smoothness per location
            r.add_mpl_fig(
                plot_metrics_map(t, metrics=smoothness, rel=False),
                caption="Smoothness metrics per location.",
            )

            # Structure function
            logger.info("Computing and plotting structure function")
            r.add_heading("Structure Function", level=3)
            r.add_mpl_fig(
                plot_sf(t=lcn2_t, show_bin_freq=False),
                caption=f"Structure function of {t.name}.",
            )
            r.add_mpl_fig(
                plot_sf(t=lcn2_t, show_bin_freq=True),
                caption=f"Structure function of {t.name} where error bars indicate "
                f"inverse fraction of number of samples per bins",
            )

        # Plot basic histogram
        logger.info("Plotting histograms...")
        r.add_heading("Histograms", level=2)
        r.add_mpl_fig(
            plot_hist_qq(t, kde=(not t.is_profile) or FORCE_KDE, bins=50),
            caption=f"Histogram and qq plot of {t.name}.",
        )

        # Plot hexbin
        logger.info("Plotting hexbins...")
        r.add_heading("Hexbin Plots", level=2)
        r.add_mpl_fig(plot_hexbin(t, bins="log"), caption=f"Hexbin plot of {t.name}.")

        # Boxplots
        logger.info("Plotting boxplots...")
        r.add_heading("Boxplots", level=2)
        r.add_mpl_fig(plot_boxplot_hr(t), caption=f"Hourly distribution of {t.name} at all locations.")

        # Plot time series at selected locations
        # if not t.is_profile:
        #     logger.info("Plotting time series...")
        #     r.add_heading("Time Series", level=2)
        #     figs = plot_ts(t, q=[0.01, 0.25, 0.5, 0.75, 0.99])
        #     for i, fig in enumerate(figs):
        #         logger.info(f"Adding time series plot {i+1}/{len(figs)}")
        #         r.add_plotly_fig(fig)

        # Plot randomly selected profiles
        if t.is_profile:
            n = 100
            seed = 1337

            logger.info(f"Plotting {n} random profiles...")
            r.add_heading(f"Randomly drawn profiles (seed = {seed})", level=2)
            fig = plot_random_profiles(t, n_rand=n, seed=seed)
            r.add_mpl_fig(fig, caption=f"Randomly drawn profiles of {t.name}.")

    logger.info("Done!")


def resample_day(da, mode: str, agg: str) -> xr.DataArray:
    """Resample DataArray `da` to daily statistics based on `mode` and `agg`."""

    def _max(da):
        return da.max("time", skipna=True)

    def _min(da):
        return da.min("time", skipna=True)

    agg_dict = {"max": _max, "min": _min}

    da_ = da.unstack("sample")

    if mode == "night":
        # Select hours around midnight
        hrs_sel = (24 + np.arange(-3, 3)) % 24  # 21:00 UTC previous day until 03:00 UTC same day
        da_ = da_.sel(time=da_["time"].dt.hour.isin(hrs_sel))
        # Set resample window, so it contains full night (incl selected midnight hours)
        da_res = da_.resample(time="1D", offset="12h")  # 12:00 UTC until 11:00 UTC next day
        da_res = da_res.apply(agg_dict[agg])
    elif mode == "day":
        # Select hours around noon
        hrs_sel = 12 + np.arange(-3, 3)  # 09:00 UTC until 15:00 UTC same day
        da_ = da_.sel(time=da_["time"].dt.hour.isin(hrs_sel))
        # Set resample window, so it contains full day (incl selected noon hours)
        da_res = da_.resample(time="1D", offset="00h")  # 00:00 UTC until 23:00 UTC same day
        da_res = da_res.apply(agg_dict[agg])
    else:
        raise ValueError(f"Unknown resample mode: {mode}")

    # Reset coordinates, which get lost during resampling
    da_res = da_res.assign_coords(
        lat=da_["lat"].isel(time=0).drop_vars("time"),
        lon=da_["lon"].isel(time=0).drop_vars("time"),
    )
    da_res = da_res.stack(sample=("time", "loc"))
    return da_res


@jax.jit
def compute_ensemble_sf(
    x: jnp.ndarray,
    y: jnp.ndarray,
    bin_edges: jnp.ndarray,
    p: int = 2,
    use_abs: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ensemble structure function S^p(l) = <[y(x+l) - y(x)]^p> for non-uniformly sampled data using JAX.

    Steps are
    1. For each realization, compute all pairwise distances and increments
    2. Bin the displacements into lag bins and accumulate sums and counts
    3. Average over all realizations

    Parameters
    ----------
    x: jnp.ndarray
        Shape (N, n) - Non-uniform sampling positions, unit: length
    y: jnp.ndarray
        Shape (N, n) - Signal values, unit: signal
    num_lags: int
        Number of bins for lag distance l
    p: int
        Order of the structure function (default is 2 for S2)
    """
    # Determine global scale for binning and setup bins
    # max_l = jnp.max(x) - jnp.min(x)  # unit: length
    # bin_edges = jnp.linspace(100, max_l, num_lags + 1)  # unit: length
    # bin_edges = jnp.logspace(2, jnp.log10(max_l), num_lags + 1)  # unit: length
    num_lags = len(bin_edges) - 1

    # Precompute upper triangular indices for pairwise differences
    n = x.shape[1]
    ii, jj = jnp.triu_indices(n, k=1)

    def scan_body(carry, inputs):
        sum_acc, count_acc = carry
        xk, yk = inputs

        # Compute pairwise distances
        dx_pairs = jnp.abs(xk[ii] - xk[jj])

        # Compute pairwise increments (abs if specified, order p)
        if use_abs:
            dy_p_pairs = jnp.abs(yk[ii] - yk[jj]) ** p  # unit: signal^p
        else:
            dy_p_pairs = (yk[ii] - yk[jj]) ** p  # unit: signal^p

        # Bin the pairwise distances and accumulate sums and counts
        bin_sums, _ = jnp.histogram(dx_pairs, bins=bin_edges, weights=dy_p_pairs)  # weights act like sum of y per bin
        bin_counts, _ = jnp.histogram(dx_pairs, bins=bin_edges)

        return (sum_acc + bin_sums, count_acc + bin_counts), None

    # Initialize accumulators
    init_carry = (jnp.zeros(num_lags), jnp.zeros(num_lags))

    # Efficiently loops over N realizations with fixed memory overhead
    (total_sums, total_counts), _ = jax.lax.scan(scan_body, init_carry, (x, y))

    Sp = jnp.where(total_counts > 0, total_sums / total_counts, 0.0)  # ensemble averaging, unit: signal^p
    dx = (bin_edges[:-1] + bin_edges[1:]) / 2  # unit: length
    dx_freq = total_counts / jnp.sum(total_counts)  # unit: fraction

    return dx, Sp, dx_freq


def get_tf_edges(x_min: float, x_max: float, n_bins: int, tf: Callable | None, tf_inv: Callable | None) -> np.ndarray:
    """Get bin edges in transformed space for uniform binning in original space."""
    edges_tf = np.linspace(tf(x_min), tf(x_max), n_bins)
    dx = edges_tf[1] - edges_tf[0]  # in tf'd space, uniform sampling
    edges_tf = edges_tf - 0.5 * dx  # shift to get left edges
    edges_tf = np.append(edges_tf, edges_tf[-1] + dx)  # add rightmost edge
    edges = tf_inv(edges_tf)
    return edges


def plot_sf(t: Array, show_bin_freq: bool) -> plt.Figure:
    # Precompute bins
    z_ = t.z.median("sample").values
    ii, jj = np.triu_indices(len(z_), k=1)
    dz = np.abs(z_[ii] - z_[jj])  # all aggregated pairwise distances
    dz_edges = get_tf_edges(x_min=50, x_max=20_000, n_bins=150, tf=np.sqrt, tf_inv=lambda x: x**2)
    # dz_bins = get_tf_edges(x_min=10, x_max=30_000, n_bins=50, tf=lambda x: x, tf_inv=lambda x: x)
    # dz_edges = get_tf_edges(x_min=10, x_max=30_000, n_bins=50, tf=np.log10, tf_inv=lambda x: 10**x)

    # Debugging figure for uniform plotting
    fig, ax = plt.subplots()
    ax.hist(dz, bins=dz_edges)
    ax.set_xscale("log")
    fig.show()

    # Plot structure functions
    fig_sf, ax_sf = plt.subplots()
    for name, da in t.as_dict().items():
        print(f"Computing structure function for {name}")
        l, s2, l_freq = compute_ensemble_sf(
            x=jnp.array(t.z.isel(sample=slice(None, None, 10)).values),
            y=jnp.array(da.isel(sample=slice(None, None, 10)).values),
            bin_edges=jnp.array(dz_edges),
        )
        is_valid = l_freq >= 1e-3
        ax_sf.errorbar(
            np.log10(l[is_valid]),
            np.log10(s2[is_valid]),
            yerr=l_freq.max() / l_freq[is_valid] * 0.05 if show_bin_freq else None,
            label=name,
        )

    dz = np.linspace(0, 0.5)
    ax_sf.plot(dz + 3.5, -0.75 + 2 * dz, c="grey")
    ax_sf.plot(dz + 3.5, -0.75 + 1 * dz, c="grey")
    # ax_sf.plot(dz + 3.5, -0.75 + (2 / 3) * dz, c="grey")
    ax_sf.plot(dz + 3.5, -0.75 + 0 * dz, c="grey")

    ax_sf.set_xlabel("dz, m")
    ax_sf.set_ylabel(f"S^2(dz) of {t.name}")
    ax_sf.legend()

    return fig_sf


def compute_smoothness(da: xr.DataArray, dim: TDim) -> xr.Dataset:
    dim_z = "bottom_top"

    # Compute mean total variation
    tv_dy = np.abs(da.diff(dim_z)).sum(dim_z).mean(dim)

    # Mean RMS of differences
    rms_dy = np.sqrt((da.diff(dim_z) ** 2).mean(dim_z).mean(dim))

    # Weighted 2nd order difference
    # don't allow dim_z as coordinates because differencing otherwise messed up
    # assert dim_z not in z.coords
    # assert dim_z not in da.coords
    # y1 = da.isel({dim_z: slice(1, -1)})
    # y0 = da.isel({dim_z: slice(0, -2)})
    # y2 = da.isel({dim_z: slice(2, None)})
    # z1 = z.isel({dim_z: slice(1, -1)})
    # z0 = z.isel({dim_z: slice(0, -2)})
    # z2 = z.isel({dim_z: slice(2, None)})
    # roughness = ((y2 - y1) / (z2 - z1) - (y1 - y0) / (z1 - z0)) ** 2
    # roughness = roughness.sum(dim_z).mean(dim)

    return xr.Dataset(
        {
            "tv_dy": tv_dy,
            "rms_dy": rms_dy,
            # "roughness": roughness,
            "lat": da["lat"].isel(time=0),
            "lon": da["lon"].isel(time=0),
        }
    )


def compute_smoothness_t(t: Array, dim: TDim) -> xr.Dataset:
    """Compute smoothness metrics for all members in a Triplet."""
    logger.debug(f"Computing smoothness metrics for triplet {t.name}")
    smoothness_list = []
    for n, da in t.as_dict().items():
        logger.debug(f"Computing smoothness for member: {n}")
        smoothness = compute_smoothness(da, dim=dim)
        smoothness = smoothness.expand_dims(pair=[n])  # todo: pair for legacy compatability
        smoothness_list.append(smoothness)
    return xr.concat(smoothness_list, dim="pair")  # todo: pair for legacy compatability


if __name__ == "__main__":
    USE_DASK = False
    if USE_DASK:
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster()
        client = Client(cluster)

    logger.info("Opening datasets...")
    ds = open_datasets("saved_models/preds_175_176_178", use_dask=USE_DASK)

    lcn2_t = ds["lcn2"]
    r0_t = lcn2_t.map(lambda lcn2: ot.fried_r0_xr(cn2=10**lcn2, z=ds.z, dim="bottom_top")).rename("r0")
    lsi2_t = lcn2_t.map(
        lambda lcn2: np.log10(ot.scint_index_xr(cn2=10**lcn2, z=ds.z, dim="bottom_top", mode="plane"))
    ).rename("lsi2")
    # th_0_t = lcn2_t.map(
    #     lambda lcn2: ot.isoplanatic_angle_xr(cn2=10**lcn2, z=ds.z, dim="bottom_top") * 180 / np.pi * 60 * 60  # arcsec
    # ).rename("th0")

    logger.debug("Computing r0 resampled statistics...")

    r0_t_min = r0_t.map(resample_day, mode="day", agg="min").rename("r0_day_min")  # r0 min during day
    r0_t_max = r0_t.map(resample_day, mode="night", agg="max").rename("r0_night_max")  # r0 max during night

    si2_t_min = lsi2_t.map(resample_day, mode="day", agg="max").rename("lsi2_day_max")  # max si2 during day
    si2_t_max = lsi2_t.map(resample_day, mode="night", agg="min").rename("lsi2_night_min")  # min si2 during night

    make_report(lcn2_t)
    make_report(r0_t)
    make_report(r0_t_min)
    make_report(r0_t_max)

    make_report(lsi2_t)
    make_report(si2_t_min)
    make_report(si2_t_max)
    # make_report(th_0_t)
    # make_report(eps)

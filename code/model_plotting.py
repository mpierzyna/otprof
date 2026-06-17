import logging
import pathlib
from typing import Tuple, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import model_analysis as ma
import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr
from research_tools.ot import ot

logger = logging.getLogger(__name__)

sns.set_palette("colorblind")
plt.rcParams.update(
    {
        # "font.family": "serif",
        "font.size": 8,
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{amsmath}",
        "figure.dpi": 300,
        # "figure.labelsize": 8,
        "lines.linewidth": 0.75,
        "hatch.linewidth": 0.5,
    }
)

# BASELINE_UPPER = "true/pred_wrf_native"
# BASELINE_LOWER = "true/pred_hv_wrfpl"

# Scale figures because otherwise very small
FIG_WIDTH = 7.25  # in
FIG_WIDTH *= 0.9

VARS_PRETTY = {
    "lcn2": r"$\log_{10}C_n^2$",
    "r0": r"$r_0$",
    "lsi2": r"$\log_{10}\sigma_I^2$",
}
VARS_UNITS = {
    "r0": "cm",
}

EXP_PRETTY = {
    ma.DatasetKey.TRUE: "Ref.",
    ma.DatasetKey.WRF_NATIVE: "Wn",
    ma.DatasetKey.WRF_PL: "Wpl",
    ma.DatasetKey.ERA5_PL: "E5pl no-QM",
    ma.DatasetKey.ERA5_PL_QM: "E5pl QM",
    ma.DatasetKey.ERA5_PL_DIRECT: "E5pl train",
    ma.DatasetKey.HV_WRFPL: "HV+W71",
}
EXP_COLORS = {
    ma.DatasetKey.TRUE: "k",
    ma.DatasetKey.WRF_NATIVE: "C1",
    ma.DatasetKey.WRF_PL: "C6",
    ma.DatasetKey.ERA5_PL: "C0",
    ma.DatasetKey.ERA5_PL_QM: "C3",  # todo: check color
    ma.DatasetKey.ERA5_PL_DIRECT: "C9",
    ma.DatasetKey.HV_WRFPL: "C7",
}

METRICS_PLOT = [
    "bias",
    "crmse",
    "r",
    "r2",
    # "wasserstein",
]
METRICS_DIM = ["bias", "crmse"]  # dimensional metrics
METRICS_PRETTY = {
    "bias": r"Bias ($\downarrow$)",
    "crmse": r"cRMSE ($\downarrow$)",
    "r": r"$r$ ($\uparrow$)",
    "r2": r"$R^2$ ($\uparrow$)",
    "wasserstein": r"W ($\downarrow$)",
}

PAIR_COLORS = {f"true/{e}": c for (e, c) in EXP_COLORS.items() if not e == ma.DatasetKey.TRUE}
PAIR_MARKERS = {
    f"true/{ma.DatasetKey.WRF_NATIVE}": "^",
    f"true/{ma.DatasetKey.WRF_PL}": "o",
    f"true/{ma.DatasetKey.ERA5_PL}": "o",
    f"true/{ma.DatasetKey.ERA5_PL_QM}": "o",
    f"true/{ma.DatasetKey.ERA5_PL_DIRECT}": "o",
    f"true/{ma.DatasetKey.HV_WRFPL}": "v",
}
PAIR_PRETTY = {f"true/{k}": v for k, v in EXP_PRETTY.items()}


def plot_scores(a: ma.Array, pairs_plot) -> Tuple[plt.Figure, str]:
    metrics = ma.compute_metrics_t(a, dim=["sample", "bottom_top"] if a.is_profile else ["sample"])
    metrics = metrics.to_dataframe()[METRICS_PLOT]
    metrics = metrics.loc[pairs_plot]

    fig, axarr = plt.subplots(
        figsize=(FIG_WIDTH * 2 / 3, FIG_WIDTH / 4.5),
        constrained_layout=True,
        nrows=2,
    )

    for ax, is_dim in zip(axarr, [True, False]):
        metrics_subset = [m for m in METRICS_PLOT if (m in METRICS_DIM) == is_dim]  # Only select dim/non-dim metrics
        m_space = np.arange(len(metrics_subset))[::-1]
        for i, m in zip(m_space, metrics_subset):
            for p in pairs_plot:
                label = PAIR_PRETTY[p] if i == m_space[0] else None  # only label once
                ax.scatter(metrics[m].loc[p], i, c=PAIR_COLORS[p], marker=PAIR_MARKERS[p], label=label)

        ax.set_yticks(m_space)
        ax.set_ylim(-0.5, len(metrics_subset) - 0.5)
        ax.set_yticklabels([f"{METRICS_PRETTY[m]}" for m in metrics_subset])

        if not is_dim:
            m_min = metrics[metrics_subset].min().min()
            ax.set_xlim(min(0.5, m_min), 1)  # at least [0.5 , 1]

    axarr[0].legend(ncols=len(pairs_plot), bbox_to_anchor=(0.5, 1.1), loc="lower center")

    # Convert dataframe to latex table
    metrics_latex = metrics.T
    metrics_latex.columns = [PAIR_PRETTY[c] for c in metrics_latex.columns]
    metrics_latex = metrics_latex.reset_index()
    metrics_latex["index"] = metrics_latex["index"].map(METRICS_PRETTY)
    metrics_latex = metrics_latex.set_index("index")
    metrics_latex = metrics_latex.to_latex(
        escape=False,
        float_format="%.3f",
        index=False,
        column_format="c" * len(metrics_latex.columns),
    )

    return fig, metrics_latex


def plot_hist_qq(
    a: ma.Array,
    exps_plot,
    pairs_plot,
    xmin: float | None = None,
    xmax: float | None = None,
) -> plt.Figure:
    a_dict = a.as_dict()

    # Determine limits
    a_flt_srtd = a.map(lambda x: np.sort(x.values.flatten()))
    vmin = np.nanmin(a_flt_srtd.data[ma.DatasetKey.TRUE]) if xmin is None else xmin
    vmax = np.nanmax(a_flt_srtd.data[ma.DatasetKey.TRUE]) if xmax is None else xmax

    fig, (ax_hist, ax_qq) = plt.subplots(
        ncols=2,
        figsize=(FIG_WIDTH, FIG_WIDTH / 3),
        constrained_layout=True,
        width_ratios=[2, 1],
    )

    unit = "" if not a.name in VARS_UNITS else f", {VARS_UNITS[a.name]}"

    # Plot histograms
    for e in exps_plot:
        v = a_dict[e]
        l = EXP_PRETTY[e]
        c = EXP_COLORS[e]
        # v.plot.hist(label=l, alpha=0.25, ax=ax_hist, color=c, density=True)
        sns.kdeplot(x=v.values.flatten(), ax=ax_hist, color=c, fill=False, label=l, lw=1.5)
    ax_hist.set_xlabel(VARS_PRETTY[a.name] + unit)
    ax_hist.legend()
    ax_hist.set_xlim(vmin, vmax)

    # Plot qq
    a_flt_srtd_pair = a_flt_srtd.as_pairs()
    for p in pairs_plot:
        pair = a_flt_srtd_pair[p]
        l = PAIR_PRETTY[p]
        c = PAIR_COLORS[p]
        ax_qq.scatter(pair.a, pair.b, label=l, s=1, color=c, rasterized=True)

    ax_qq.plot([vmin, vmax], [vmin, vmax], ls="--", color="k")
    ax_qq.text(0.01, 0.99, "overestimation", transform=ax_qq.transAxes, va="top", ha="left", color="grey")
    ax_qq.text(0.99, 0.01, "underestimation", transform=ax_qq.transAxes, va="bottom", ha="right", color="grey")
    ax_qq.set_xlabel(f"Ref. {VARS_PRETTY[a.name]}" + unit)
    ax_qq.set_ylabel(f"Estimated {VARS_PRETTY[a.name]}" + unit)
    ax_qq.margins(0)
    ax_qq.set_aspect("equal")

    return fig


def plot_sf(a: ma.Array, exps_plot) -> plt.Figure:
    """Plot structure function of variable a."""
    # Precompute bins
    z_ = a.z.median("sample").values
    ii, jj = np.triu_indices(len(z_), k=1)
    dz = np.abs(z_[ii] - z_[jj])  # all aggregated pairwise distances
    dz_edges = ma.get_tf_edges(x_min=50, x_max=20_000, n_bins=100, tf=np.sqrt, tf_inv=lambda x: x**2)
    # dz_bins = get_tf_edges(x_min=10, x_max=30_000, n_bins=50, tf=lambda x: x, tf_inv=lambda x: x)
    # dz_edges = get_tf_edges(x_min=10, x_max=30_000, n_bins=50, tf=np.log10, tf_inv=lambda x: 10**x)

    # # Debugging figure for uniform plotting
    # fig, ax = plt.subplots()
    # ax.hist(dz, bins=dz_edges)
    # ax.set_xscale("log")
    # fig.show()

    # Plot structure functions
    fig, (ax_full, ax_zoom) = plt.subplots(ncols=2, figsize=(FIG_WIDTH, FIG_WIDTH / 3), constrained_layout=True)
    for e in exps_plot:
        print(f"Computing structure function for {e}")
        l, s2, l_freq = ma.compute_ensemble_sf(
            x=jnp.array(a.z.isel(sample=slice(None, None, 10)).values),
            y=jnp.array(a[e].isel(sample=slice(None, None, 10)).values),
            bin_edges=jnp.array(dz_edges),
        )
        is_valid = l_freq >= 1e-3
        for ax in [ax_full, ax_zoom]:
            ax.plot(
                l[is_valid],
                s2[is_valid],
                label=EXP_PRETTY[e],
                color=EXP_COLORS[e],
                lw=1,
            )

    for ax in [ax_full, ax_zoom]:
        ax.set_xscale("log")
        ax.set_yscale("log")

        # dz = np.linspace(0, 0.5)
        # ax_sf.plot(dz + 3.5, -0.75 + 2 * dz, c="grey")
        # ax_sf.plot(dz + 3.5, -0.75 + 1 * dz, c="grey")
        # # ax_sf.plot(dz + 3.5, -0.75 + (2 / 3) * dz, c="grey")
        # ax_sf.plot(dz + 3.5, -0.75 + 0 * dz, c="grey")

        ax.set_xlabel(r"$\Delta z$, m")
        ax.margins(x=0)

    # only full axis
    ax_full.set_ylabel(rf"$S^2(\Delta z)$ of {VARS_PRETTY[a.name]}")
    ax_full.legend(ncols=2, loc="lower right")

    # Zoomed axis limits
    ax_zoom.set_xlim(2e3, None)
    ax_zoom.set_ylim(8e-1, 10)

    # Draw matching box on full axis
    ax_full.indicate_inset_zoom(ax_zoom, edgecolor="black")

    return fig


def plot_random_profiles(a: ma.Array, exps_plot, z_lr: xr.DataArray, ncols: int, nrows: int, seed: int) -> plt.Figure:
    """Plot random profiles from the triplet."""
    rng = np.random.default_rng(seed)
    inds = rng.choice(a.data[ma.DatasetKey.TRUE].sizes["sample"], ncols * nrows, replace=False)
    a_dict = a.as_dict()

    fig, axarr = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_WIDTH * nrows / 3),
        constrained_layout=True,
        sharex="all",
        sharey="all",
    )
    if axarr.ndim == 1:
        axarr = axarr[:, None]  # make 2d for consistent indexing

    legend_handles = []
    for ax, i in zip(axarr.flatten(), inds):
        # Plot profiles
        for e in exps_plot:
            v = a_dict[e]
            p = ax.plot(
                v.isel(sample=i),
                a.z.isel(sample=i),
                label=EXP_PRETTY[e],
                color=EXP_COLORS[e],
                zorder=100,
                lw=1.5,
            )
            if i == inds[0]:
                legend_handles.append(p[0])

        # Add meta data (use last v because all same loc and time)
        lat = v.isel(sample=i).lat.item()
        lat = f"{lat:.2f} °N" if lat >= 0 else f"{-lat:.2f} °S"
        lon = v.isel(sample=i).lon.item()
        lon = f"{lon:.2f} °E" if lon >= 0 else f"{-lon:.2f} °W"
        time = v.isel(sample=i).time.dt.strftime("%Y-%m-%d %H:%M").item()

        ax.text(
            0.99,
            0.99,
            f"{lat}, {lon}\n{time}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=6.5,
        )

        # Add ERA5 levels in the background
        # This is not so helpful for the story
        # for z in z_lr:
        #     ax.axhline(z, color="lightgrey", lw=0.5, ls="--", zorder=-100)

        # Set sqrt scaling for y axis to stretch lower levels
        ax.set_yscale("function", functions=(lambda x: np.sqrt(x), lambda x: x**2))
        ax.margins(y=0)
        ax.set_ylim(0, None)

        # add major and minor ticks in y direction
        ax.yaxis.set_major_locator(plt.MultipleLocator(5000))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))

    for ax in axarr[:, 0]:
        ax.set_ylabel("Height, m")
    for ax in axarr[-1, :]:
        ax.set_xlabel(f"{VARS_PRETTY[a.name]}")

    # Legend above figure
    fig.legend(
        handles=legend_handles,
        fontsize=8,
        ncol=len(exps_plot),
        loc="outside upper center",
    )

    return fig


def plot_markers() -> plt.Figure:
    """Simple figure to show marker styles used in other plots."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH / 10 * len(PAIR_COLORS), FIG_WIDTH / 10), constrained_layout=True)
    for i, p in enumerate(PAIR_COLORS):
        ax.scatter(i, 0, c=PAIR_COLORS[p], marker=PAIR_MARKERS[p])
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_levels(data_root: pathlib.Path) -> plt.Figure:
    # Open native high-res X
    ds_hr = xr.open_dataset(data_root / "wrfnative_X_test.nc")
    ds_lr = xr.open_dataset(data_root / "wrfpl_X_test.nc")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH / 3, FIG_WIDTH / 1.5), constrained_layout=True)

    z = ds_hr["z_agl"].mean("sample")
    p = ds_hr["p"].mean("sample") / 100
    ax.scatter(p + 50, z, s=2, label="WRF +50hPa")

    z = ds_lr["z_agl"].mean("sample")
    p = ds_lr["p"].mean("sample") / 100
    ax.scatter(p, z, s=2, marker="D", label="ERA5-PL")

    ax.set_yscale("function", functions=(lambda x: np.sqrt(x), lambda x: x**2))
    ax.set_xlabel("Pressure, hPa")
    ax.set_ylabel("Height, m")
    ax.legend()

    # add major and minor ticks in y direction
    ax.yaxis.set_major_locator(plt.MultipleLocator(5000))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))
    ax.set_xlim(0, None)
    # ax.set_yticklabels([0, 1000, 5000, 10000, 15000, 20000, 25000])

    ax.margins(0)

    return fig


def print_percentiles(a: ma.Array, a_vals: List[float], exp_plots) -> None:
    """Compute percentiles for values in `a_vals` and print them."""
    res = {}
    for e in exp_plots:
        q = np.linspace(0, 1, 200)  # quantiles to compute
        q_v = a[e].quantile(q).values  # compute quantiles
        res[e] = np.interp(a_vals, q_v, q)  # interpolate
    res = pd.DataFrame(res, index=a_vals, columns=exp_plots)
    print(res)


def save_and_show(fig: plt.Figure, fpath: pathlib.Path, **kwargs) -> None:
    fig.savefig(fpath, transparent=True, pad_inches=0, **kwargs)
    fig.show()


if __name__ == "__main__":
    # Base dir
    plot_dir = pathlib.Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Open datasets
    logger.info("Opening datasets...")
    data_root = pathlib.Path("saved_models/preds_175_176_178")
    ds = ma.open_datasets(data_root, use_dask=False)
    ds = ds.isel({"sample": slice(None, None, 10)})

    # Load/compute variables for plotting
    lcn2 = ds["lcn2"]
    r0 = lcn2.map(lambda lcn2: ot.fried_r0_xr(cn2=10**lcn2, z=ds.z, dim="bottom_top") * 100).rename("r0")
    lsi2 = lcn2.map(
        lambda lcn2: np.log10(ot.scint_index_xr(cn2=10**lcn2, z=ds.z, dim="bottom_top", mode="plane"))
    ).rename("lsi2")

    # Experiments for main text figures
    exp_main = [
        ma.DatasetKey.TRUE,
        ma.DatasetKey.WRF_NATIVE,
        ma.DatasetKey.WRF_PL,
        ma.DatasetKey.ERA5_PL_QM,
        # ma.DatasetKey.ERA5_PL,
        # ma.DatasetKey.ERA5_PL_DIRECT,
        ma.DatasetKey.HV_WRFPL,
    ]
    # Experiments for appendix figures
    exp_app = [
        ma.DatasetKey.TRUE,
        # ma.DatasetKey.WRF_NATIVE,
        # ma.DatasetKey.WRF_PL,
        ma.DatasetKey.ERA5_PL_QM,
        ma.DatasetKey.ERA5_PL,
        ma.DatasetKey.ERA5_PL_DIRECT,
        # ma.DatasetKey.HV_WRFPL,
    ]
    exp_pcaop = [
        ma.DatasetKey.TRUE,
        ma.DatasetKey.WRF_PL,
        ma.DatasetKey.ERA5_PL_QM,
        ma.DatasetKey.HV_WRFPL,
    ]

    # pcAOP plots
    out_dir = "pcaop"
    (plot_dir / out_dir).mkdir(parents=True, exist_ok=True)
    pairs_plot = [f"true/{e}" for e in exp_pcaop if not e == ma.DatasetKey.TRUE]
    save_and_show(
        plot_random_profiles(lcn2, exps_plot=exp_pcaop, z_lr=ds.z_lr, ncols=4, nrows=1, seed=42),
        plot_dir / out_dir / "lcn2_profiles.pdf",
    )
    save_and_show(
        plot_hist_qq(r0, exps_plot=exp_pcaop, pairs_plot=pairs_plot, xmin=0, xmax=150),
        plot_dir / out_dir / "r0_hist_qq.pdf",
        dpi=650,
    )
    # # lsi2 hist/qq
    save_and_show(
        plot_hist_qq(lsi2, exps_plot=exp_pcaop, pairs_plot=pairs_plot, xmin=-2.6, xmax=-0.5),
        plot_dir / out_dir / "lsi2_hist_qq.pdf",
        dpi=650,
    )

    # for exps_plot, out_dir in [(exp_main, "main"), (exp_app, "appendix")]:
    #     # Create output dir
    #     (plot_dir / out_dir).mkdir(parents=True, exist_ok=True)

    #     # print_percentiles(r0, [50, 100], exps_plot)

    #     # # Pairs to plot
    #     pairs_plot = [f"true/{e}" for e in exps_plot if not e == ma.DatasetKey.TRUE]

    #     # # logCn2 profiles
    #     save_and_show(
    #         plot_random_profiles(lcn2, exps_plot=exps_plot, z_lr=ds.z_lr, ncols=4, nrows=3, seed=42),
    #         plot_dir / out_dir / "lcn2_profiles.pdf",
    #     )

    #     # # logCn2 scores
    #     fig, metrics_latex = plot_scores(lcn2, pairs_plot=pairs_plot)
    #     save_and_show(fig, plot_dir / out_dir / "lcn2_scores.pdf")
    #     (plot_dir / out_dir / "lcn2_scores.tex").write_text(metrics_latex)

    #     # # logCn2 hist/qq
    #     save_and_show(
    #         plot_hist_qq(lcn2, exps_plot=exps_plot, pairs_plot=pairs_plot),
    #         plot_dir / out_dir / "lcn2_hist_qq.pdf",
    #         dpi=650,
    #     )

    #     # # logCn2 structure function
    #     save_and_show(plot_sf(lcn2, exps_plot=exps_plot), plot_dir / out_dir / "lcn2_sf.pdf")

    #     # # r0 scores
    #     fig, metrics_latex = plot_scores(r0, pairs_plot=pairs_plot)
    #     save_and_show(fig, plot_dir / out_dir / "r0_scores.pdf")
    #     (plot_dir / out_dir / "r0_scores.tex").write_text(metrics_latex)

    #     # # r0 hist/qq
    #     save_and_show(
    #         plot_hist_qq(r0, exps_plot=exps_plot, pairs_plot=pairs_plot, xmin=0, xmax=150),
    #         plot_dir / out_dir / "r0_hist_qq.pdf",
    #         dpi=650,
    #     )

    #     # # lsi2 scores
    #     fig, metrics_latex = plot_scores(lsi2, pairs_plot=pairs_plot)
    #     save_and_show(fig, plot_dir / out_dir / "lsi2_scores.pdf")
    #     (plot_dir / out_dir / "lsi2_scores.tex").write_text(metrics_latex)

    #     # # lsi2 hist/qq
    #     save_and_show(
    #         plot_hist_qq(lsi2, exps_plot=exps_plot, pairs_plot=pairs_plot, xmin=-2.6, xmax=-0.5),
    #         plot_dir / out_dir / "lsi2_hist_qq.pdf",
    #         dpi=650,
    #     )

    # This is general
    # save_and_show(plot_levels(data_root), plot_dir / "levels.pdf")
    # save_and_show(plot_markers(), plot_dir / "marker_legend.pdf")

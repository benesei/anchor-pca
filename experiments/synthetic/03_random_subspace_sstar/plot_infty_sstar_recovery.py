"""Plot publication figures for random-subspace S_star recovery simulations.

The standard figures read outputs from ``run_infty_sstar_recovery.py``. If
matching FindS_star outputs from ``run_main_seqtest_dimension.py`` exist, the
corresponding recovery figures overlay the FindS_star dimension and subspace
recovery curves. If the small-E stress output from
``run_g2_low_environment_stress.py`` exists, the script also writes the combined
Gaussian extended-N recovery/tolerance figure for the hard ``E=2, p=8, k=5,
m=2`` configuration. The hard recovery panels use the same FindS_star overlay
style.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from anchorpca.reproducibility import repo_relative_path, software_versions


DEFAULT_PROFILES = ("balanced", "threshold_stress")
DEFAULT_DISTRIBUTIONS = ("gaussian", "gaussian_mixture")
DEFAULT_MAIN_CONFIG = "g5_p10_k5_m2"
DEFAULT_STRESS_CONFIG = "g2_p8_k5_m2"

PROFILE_LABELS = {
    "balanced": "easy",
    "threshold_stress": "hard",
}
PROFILE_COLORS = {
    "balanced": "tab:blue",
    "threshold_stress": "#D62728",
}
DISTRIBUTION_LABELS = {
    "gaussian": "Gaussian",
    "gaussian_mixture": "Gaussian mixture",
    "scale_mixture": "scale mixture",
}
DISTRIBUTION_LINESTYLES = {
    "gaussian": "-",
    "gaussian_mixture": "--",
    "scale_mixture": ":",
}
METHOD_MARKERS = {
    "anchorpca": "o",
    "seqtest": "s",
}
METHOD_LABELS = {
    "anchorpca": r"$\mathrm{AnchorPCA}_{\infty}$",
    "seqtest": r"$\mathtt{FindS}_{\star}$",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot AnchorPCA_infty S_star recovery simulation summaries."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir / "results" / "infty_sstar_recovery",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=script_dir / "figures",
    )
    parser.add_argument(
        "--stress-results-dir",
        type=Path,
        default=script_dir / "results" / "infty_sstar_recovery_g2_low_environment_stress",
        help=(
            "Results directory from run_g2_low_environment_stress.py. If present, "
            "the combined extended Gaussian stress figure is written."
        ),
    )
    parser.add_argument(
        "--main-seqtest-results-dir",
        type=Path,
        default=script_dir / "results" / "infty_sstar_recovery_main_seqtest",
        help=(
            "Results directory from run_main_seqtest_dimension.py. If present, "
            "the selected paper figures overlay FindS_star dimension recovery."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
        choices=list(PROFILE_LABELS),
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=list(DEFAULT_DISTRIBUTIONS),
        choices=list(DISTRIBUTION_LABELS),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help=(
            "Config ids or base config labels to plot. If omitted, all available "
            "nonzero-m configs in the summary are plotted."
        ),
    )
    parser.add_argument(
        "--no-bands",
        action="store_true",
        help="Disable 10%-90% bands. Bands are shown by default.",
    )
    parser.add_argument(
        "--skip-appendix",
        action="store_true",
        help="Only write the main two-panel figure.",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Do not write the small-E extended-N stress figure.",
    )
    return parser.parse_args()


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_figure(fig, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")


def read_summaries(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_path = results_dir / "summary_by_curve.csv"
    design_path = results_dir / "summary_by_design.csv"
    if not curve_path.exists():
        raise FileNotFoundError(f"Missing curve summary: {curve_path}")
    if not design_path.exists():
        raise FileNotFoundError(f"Missing design summary: {design_path}")
    return pd.read_csv(curve_path), pd.read_csv(design_path)


def base_config_label(row: pd.Series) -> str:
    return f"g{int(row['g'])}_p{int(row['p'])}_k{int(row['k'])}_m{int(row['m'])}"


def select_plot_rows(
    summary: pd.DataFrame,
    *,
    configs: list[str] | None,
    profiles: list[str],
    distributions: list[str],
) -> pd.DataFrame:
    rows = summary[
        summary["profile"].isin(profiles)
        & summary["distribution"].isin(distributions)
    ].copy()
    rows["base_config"] = rows.apply(base_config_label, axis=1)
    if configs:
        rows = rows[rows["config_id"].isin(configs) | rows["base_config"].isin(configs)]
    rows = rows[rows["m"].astype(int) > 0]
    if rows.empty:
        raise ValueError("No rows matched the requested plot selection.")
    return rows.sort_values(
        ["base_config", "distribution", "profile", "N"]
    ).reset_index(drop=True)


def plot_recovery_two_panel(
    summary: pd.DataFrame,
    *,
    seqtest_summary: pd.DataFrame | None = None,
    title: str | None = None,
    show_bands: bool = True,
    figsize: tuple[float, float] | None = None,
    overlay_panel_gap: float | None = None,
    overlay_legend_gap: float | None = None,
    overlay_legend_width: float | None = None,
    axis_labelsize: float | None = None,
    tick_labelsize: float | None = None,
    legend_fontsize: float | None = None,
) -> plt.Figure:
    distributions = sorted(summary["distribution"].unique().tolist())
    if len(distributions) != 1:
        raise ValueError(
            "plot_recovery_two_panel expects rows for exactly one distribution; "
            f"got {distributions}."
        )
    distribution = distributions[0]
    has_seqtest_overlay = seqtest_summary is not None and not seqtest_summary.empty
    if has_seqtest_overlay:
        required_seqtest_columns = {
            "profile",
            "N",
            "seqtest_dim_correct_mean",
            "seqtest_top_m_projector_error_op_median",
        }
        missing_columns = sorted(required_seqtest_columns.difference(seqtest_summary.columns))
        if missing_columns:
            raise ValueError(
                "FindS_star overlay requires summary columns "
                f"{missing_columns}. Rerun the corresponding FindS_star "
                "simulation script so the top-m_hat subspace-error summaries "
                "are available."
            )

    if has_seqtest_overlay:
        fig = plt.figure(figsize=figsize or (7.6, 2.65))
        if overlay_panel_gap is None and overlay_legend_gap is None:
            grid = fig.add_gridspec(
                1,
                3,
                width_ratios=[1.12, 1.12, 0.58],
                wspace=0.27,
            )
            ax_dim = fig.add_subplot(grid[0, 0])
            ax_error = fig.add_subplot(grid[0, 1], sharex=ax_dim)
            legend_ax = fig.add_subplot(grid[0, 2])
        else:
            grid = fig.add_gridspec(
                1,
                5,
                width_ratios=[
                    1.12,
                    overlay_panel_gap if overlay_panel_gap is not None else 0.14,
                    1.12,
                    overlay_legend_gap if overlay_legend_gap is not None else 0.04,
                    overlay_legend_width if overlay_legend_width is not None else 0.58,
                ],
                wspace=0.0,
            )
            ax_dim = fig.add_subplot(grid[0, 0])
            ax_error = fig.add_subplot(grid[0, 2], sharex=ax_dim)
            legend_ax = fig.add_subplot(grid[0, 4])
        legend_ax.axis("off")
        axes = (ax_dim, ax_error)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize or (6.8, 2.65), sharex=True)
        ax_dim, ax_error = axes
        legend_ax = None

    for profile, group in summary.groupby(
        "profile",
        sort=False,
    ):
        group = group.sort_values("N")
        color = PROFILE_COLORS.get(profile, "0.25")
        label = PROFILE_LABELS.get(profile, profile)

        ax_dim.plot(
            group["N"],
            group["dim_correct_mean"],
            color=color,
            linestyle="-",
            marker="o",
            linewidth=1.9,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=("_nolegend_" if has_seqtest_overlay else label),
        )
        if show_bands:
            ax_dim.fill_between(
                group["N"].to_numpy(dtype=float),
                group["dim_correct_q10"].to_numpy(dtype=float),
                group["dim_correct_q90"].to_numpy(dtype=float),
                color=color,
                alpha=0.13,
                linewidth=0,
            )
        if has_seqtest_overlay:
            seq_group = seqtest_summary[seqtest_summary["profile"] == profile].sort_values("N")
            if not seq_group.empty:
                ax_dim.plot(
                    seq_group["N"],
                    seq_group["seqtest_dim_correct_mean"],
                    color=color,
                    linestyle="--",
                    marker=METHOD_MARKERS["seqtest"],
                    linewidth=1.8,
                    markersize=4.8,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    label="_nolegend_",
                )
        ax_error.plot(
            group["N"],
            group["first_block_projector_error_op_median"],
            color=color,
            linestyle="-",
            marker="o",
            linewidth=1.9,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=label,
        )
        if show_bands:
            ax_error.fill_between(
                group["N"].to_numpy(dtype=float),
                group["first_block_projector_error_op_q10"].to_numpy(dtype=float),
                group["first_block_projector_error_op_q90"].to_numpy(dtype=float),
                color=color,
                alpha=0.13,
                linewidth=0,
            )
        if has_seqtest_overlay:
            seq_group = seqtest_summary[seqtest_summary["profile"] == profile].sort_values("N")
            if not seq_group.empty:
                ax_error.plot(
                    seq_group["N"],
                    seq_group["seqtest_top_m_projector_error_op_median"],
                    color=color,
                    linestyle="--",
                    marker=METHOD_MARKERS["seqtest"],
                    linewidth=1.8,
                    markersize=4.8,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                )

    ax_dim.set_xscale("log")
    ax_error.set_xscale("log")
    ax_dim.set_ylim(-0.03, 1.03)
    ax_error.set_ylim(bottom=0.0)
    ax_dim.set_xlabel(r"$N$ per environment")
    ax_error.set_xlabel(r"$N$ per environment")
    ax_dim.set_ylabel(r"$\Pr(\widehat m=m)$" if has_seqtest_overlay else r"$\Pr(\widehat m_\infty=m)$")
    ax_error.set_ylabel(
        r"$\|\Pi_{\widehat{\mathcal{S}}_\star}-\Pi_{\mathcal{S}_\star}\|_{\mathrm{op}}$"
    )
    ax_dim.set_title("Dimension recovery" if has_seqtest_overlay else "First-block dimension")
    ax_error.set_title("Subspace error" if has_seqtest_overlay else "First-block subspace error")

    for ax in axes:
        ax.grid(False)
        if axis_labelsize is not None:
            ax.xaxis.label.set_size(axis_labelsize)
            ax.yaxis.label.set_size(axis_labelsize)
        if tick_labelsize is not None:
            ax.tick_params(axis="both", labelsize=tick_labelsize)

    distribution_label = DISTRIBUTION_LABELS.get(distribution, distribution)
    handles, labels = ax_dim.get_legend_handles_labels()
    if has_seqtest_overlay:
        method_handles = [
            plt.Line2D(
                [0],
                [0],
                color="0.20",
                linestyle="-",
                marker=METHOD_MARKERS["anchorpca"],
                linewidth=1.9,
                markersize=5.0,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=METHOD_LABELS["anchorpca"],
            ),
            plt.Line2D(
                [0],
                [0],
                color="0.20",
                linestyle="--",
                marker=METHOD_MARKERS["seqtest"],
                linewidth=1.8,
                markersize=4.8,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=METHOD_LABELS["seqtest"],
            ),
        ]
        dgp_handles = [
            plt.Line2D(
                [0],
                [0],
                color=PROFILE_COLORS[profile],
                linestyle="-",
                linewidth=2.0,
                label=PROFILE_LABELS[profile],
            )
            for profile in DEFAULT_PROFILES
            if profile in set(summary["profile"])
        ]
        method_legend = legend_ax.legend(
            handles=method_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            title="texture = method",
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=2.0,
            labelspacing=0.75,
            borderaxespad=0.0,
            title_fontsize=legend_fontsize or 7.4,
        )
        legend_ax.add_artist(method_legend)
        legend_ax.legend(
            handles=dgp_handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 0.02),
            title="color = DGP",
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=2.0,
            labelspacing=0.75,
            borderaxespad=0.0,
            title_fontsize=legend_fontsize or 7.4,
        )
    else:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=max(1, len(labels)),
            frameon=False,
            handlelength=2.3,
            columnspacing=1.2,
            title=distribution_label,
            title_fontsize=7.5,
        )
    if title:
        fig.suptitle(title, y=1.20, fontsize=9)
    if has_seqtest_overlay:
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.88)
    else:
        fig.tight_layout(w_pad=1.5)
    return fig


def _plot_tolerance_mechanism_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    short_labels: bool = False,
) -> None:
    gaussian = summary[summary["distribution"] == "gaussian"].copy()
    if gaussian.empty:
        raise ValueError("Agreement-distance plot requires Gaussian rows.")

    tolerance = (
        gaussian.groupby("N", sort=True)["block_tol_median"]
        .median()
        .reset_index()
    )
    ax.plot(
        tolerance["N"],
        tolerance["block_tol_median"],
        color="#08306B",
        linestyle="-",
        marker="D",
        linewidth=1.8,
        markersize=4.3,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="block tolerance",
    )

    gap_markers = {"balanced": "o", "threshold_stress": "^"}
    for profile, group in gaussian.groupby("profile", sort=False):
        group = group.sort_values("N")
        color = PROFILE_COLORS.get(profile, "0.25")
        ax.plot(
            group["N"],
            group["gap_m_median"],
            color=color,
            linestyle="--",
            marker=gap_markers.get(profile, "o"),
            linewidth=1.8,
            markersize=4.7,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=(
                PROFILE_LABELS.get(profile, profile)
                if short_labels
                else (
                    f"{PROFILE_LABELS.get(profile, profile)}: "
                    r"$\widehat\gamma_\Pi$"
                )
            ),
        )
        ax.fill_between(
            group["N"].to_numpy(dtype=float),
            group["gap_m_q10"].to_numpy(dtype=float),
            group["gap_m_q90"].to_numpy(dtype=float),
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$N$ per environment")
    ax.set_ylabel("agreement-separation gap")
    ax.grid(False)


def _positive_tolerance_axis_values(summary: pd.DataFrame) -> list[float]:
    gaussian = summary[summary["distribution"] == "gaussian"].copy()
    if gaussian.empty:
        return []

    values: list[float] = []
    tolerance = gaussian.groupby("N", sort=True)["block_tol_median"].median()
    for value in tolerance:
        if pd.notna(value) and float(value) > 0:
            values.append(float(value))

    for column in ("gap_m_median", "gap_m_q10", "gap_m_q90"):
        if column not in gaussian.columns:
            continue
        for value in gaussian[column]:
            if pd.notna(value) and float(value) > 0:
                values.append(float(value))
    return values


def _align_tolerance_axes_logscale(
    axes: tuple[plt.Axes, plt.Axes],
    *summaries: pd.DataFrame,
) -> None:
    values: list[float] = []
    for summary in summaries:
        values.extend(_positive_tolerance_axis_values(summary))
    if not values:
        return

    lower = min(values) * 0.8
    upper = max(values) * 1.25
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)


def _plot_recovery_axes(
    ax_dim: plt.Axes,
    ax_error: plt.Axes,
    summary: pd.DataFrame,
    *,
    seqtest_summary: pd.DataFrame,
    show_bands: bool = True,
    axis_labelsize: float = 9.5,
    tick_labelsize: float = 8.8,
) -> None:
    required_seqtest_columns = {
        "profile",
        "N",
        "seqtest_dim_correct_mean",
        "seqtest_top_m_projector_error_op_median",
    }
    missing_columns = sorted(required_seqtest_columns.difference(seqtest_summary.columns))
    if missing_columns:
        raise ValueError(f"FindS_star overlay requires summary columns {missing_columns}.")

    for profile, group in summary.groupby("profile", sort=False):
        group = group.sort_values("N")
        color = PROFILE_COLORS.get(profile, "0.25")

        ax_dim.plot(
            group["N"],
            group["dim_correct_mean"],
            color=color,
            linestyle="-",
            marker=METHOD_MARKERS["anchorpca"],
            linewidth=1.9,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        if show_bands:
            ax_dim.fill_between(
                group["N"].to_numpy(dtype=float),
                group["dim_correct_q10"].to_numpy(dtype=float),
                group["dim_correct_q90"].to_numpy(dtype=float),
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        seq_group = seqtest_summary[seqtest_summary["profile"] == profile].sort_values("N")
        if not seq_group.empty:
            ax_dim.plot(
                seq_group["N"],
                seq_group["seqtest_dim_correct_mean"],
                color=color,
                linestyle="--",
                marker=METHOD_MARKERS["seqtest"],
                linewidth=1.8,
                markersize=4.8,
                markeredgecolor="white",
                markeredgewidth=0.9,
            )

        ax_error.plot(
            group["N"],
            group["first_block_projector_error_op_median"],
            color=color,
            linestyle="-",
            marker=METHOD_MARKERS["anchorpca"],
            linewidth=1.9,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        if show_bands:
            ax_error.fill_between(
                group["N"].to_numpy(dtype=float),
                group["first_block_projector_error_op_q10"].to_numpy(dtype=float),
                group["first_block_projector_error_op_q90"].to_numpy(dtype=float),
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        if not seq_group.empty:
            ax_error.plot(
                seq_group["N"],
                seq_group["seqtest_top_m_projector_error_op_median"],
                color=color,
                linestyle="--",
                marker=METHOD_MARKERS["seqtest"],
                linewidth=1.8,
                markersize=4.8,
                markeredgecolor="white",
                markeredgewidth=0.9,
            )

    ax_dim.set_xscale("log")
    ax_error.set_xscale("log")
    ax_dim.set_ylim(-0.03, 1.03)
    ax_error.set_ylim(bottom=0.0)
    ax_dim.set_xlabel(r"$N$ per environment")
    ax_error.set_xlabel(r"$N$ per environment")
    ax_dim.set_ylabel(r"$\Pr(\widehat m=m)$")
    ax_error.set_ylabel(
        r"$\|\Pi_{\widehat{\mathcal{S}}_\star}-\Pi_{\mathcal{S}_\star}\|_{\mathrm{op}}$"
    )
    ax_dim.set_title("Dimension recovery")
    ax_error.set_title("Subspace error")
    for ax in (ax_dim, ax_error):
        ax.grid(False)
        ax.xaxis.label.set_size(axis_labelsize)
        ax.yaxis.label.set_size(axis_labelsize)
        ax.tick_params(axis="both", labelsize=tick_labelsize)


def _add_recovery_side_legend(
    legend_ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    legend_fontsize: float = 9.5,
) -> None:
    legend_ax.axis("off")
    method_handles = [
        plt.Line2D(
            [0], [0], color="0.20", linestyle="-", marker=METHOD_MARKERS["anchorpca"],
            linewidth=1.9, markersize=5.0, markeredgecolor="white", markeredgewidth=0.9,
            label=METHOD_LABELS["anchorpca"],
        ),
        plt.Line2D(
            [0], [0], color="0.20", linestyle="--", marker=METHOD_MARKERS["seqtest"],
            linewidth=1.8, markersize=4.8, markeredgecolor="white", markeredgewidth=0.9,
            label=METHOD_LABELS["seqtest"],
        ),
    ]
    dgp_handles = [
        plt.Line2D([0], [0], color=PROFILE_COLORS[profile], linestyle="-", linewidth=2.0, label=PROFILE_LABELS[profile])
        for profile in DEFAULT_PROFILES
        if profile in set(summary["profile"])
    ]
    method_legend = legend_ax.legend(
        handles=method_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
        title="texture = method", frameon=False, fontsize=legend_fontsize,
        handlelength=2.0, labelspacing=0.75, borderaxespad=0.0,
        title_fontsize=legend_fontsize,
    )
    legend_ax.add_artist(method_legend)
    legend_ax.legend(
        handles=dgp_handles, loc="lower left", bbox_to_anchor=(0.0, 0.02),
        title="color = DGP", frameon=False, fontsize=legend_fontsize,
        handlelength=2.0, labelspacing=0.75, borderaxespad=0.0,
        title_fontsize=legend_fontsize,
    )


def _add_tolerance_side_legend(
    legend_ax: plt.Axes,
    ax_source: plt.Axes,
    *,
    legend_fontsize: float = 9.0,
) -> None:
    legend_ax.axis("off")
    handles, labels = ax_source.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if "block tolerance" in by_label:
        block_legend = legend_ax.legend(
            [by_label["block tolerance"]], ["block tolerance\n" + r"$\mathrm{tol}_N$"], loc="upper left",
            bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=legend_fontsize,
            handlelength=2.0, borderaxespad=0.0,
        )
        legend_ax.add_artist(block_legend)

    gap_labels = [label for label in ["easy", "hard"] if label in by_label]
    if gap_labels:
        display_labels = [rf"$\widehat{{\gamma}}_\Pi$ ({label} regime)" for label in gap_labels]
        legend_ax.legend(
            [by_label[label] for label in gap_labels], display_labels, loc="lower left",
            bbox_to_anchor=(0.0, 0.02), frameon=False, fontsize=legend_fontsize,
            handlelength=2.0, labelspacing=0.75, borderaxespad=0.0,
        )


def plot_small_e_recovery_tolerance_four_panel(
    *,
    stress_summary: pd.DataFrame,
    stress_seqtest_summary: pd.DataFrame,
    main_tolerance_summary: pd.DataFrame,
) -> plt.Figure:
    fig = plt.figure(figsize=(7.05, 4.95))
    grid = fig.add_gridspec(
        2, 5,
        width_ratios=[1.12, 0.34, 1.12, 0.03, 0.72],
        height_ratios=[1.0, 1.0],
        wspace=0.0,
        hspace=0.64,
    )
    ax_dim = fig.add_subplot(grid[0, 0])
    ax_error = fig.add_subplot(grid[0, 2], sharex=ax_dim)
    ax_top_legend = fig.add_subplot(grid[0, 4])
    ax_main_tol = fig.add_subplot(grid[1, 0])
    ax_stress_tol = fig.add_subplot(grid[1, 2])
    ax_bottom_legend = fig.add_subplot(grid[1, 4])

    _plot_recovery_axes(
        ax_dim, ax_error, stress_summary,
        seqtest_summary=stress_seqtest_summary,
        show_bands=True,
    )
    _add_recovery_side_legend(ax_top_legend, stress_summary)

    _plot_tolerance_mechanism_axis(ax_main_tol, main_tolerance_summary, short_labels=True)
    _plot_tolerance_mechanism_axis(ax_stress_tol, stress_summary, short_labels=True)
    _align_tolerance_axes_logscale(
        (ax_main_tol, ax_stress_tol),
        main_tolerance_summary,
        stress_summary,
    )
    ax_main_tol.set_title("Main random-subspace configuration")
    ax_stress_tol.set_title(r"Small-$E$ configuration")
    ax_stress_tol.set_ylabel("")
    _add_tolerance_side_legend(ax_bottom_legend, ax_main_tol)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.95)
    return fig

def plot_tolerance_mechanism(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.9, 2.65))
    _plot_tolerance_mechanism_axis(ax, summary, short_labels=False)
    ax.legend(
        frameon=False,
        ncol=1,
        fontsize=7.2,
        handlelength=2.4,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    return fig


def plot_tolerance_mechanism_comparison(
    main_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), sharey=False)
    panel_specs = [
        (axes[0], main_summary, "Main random-subspace design"),
        (axes[1], stress_summary, r"Small-$E$ stress design"),
    ]
    for ax, rows, title in panel_specs:
        _plot_tolerance_mechanism_axis(ax, rows, short_labels=True)
        ax.set_title(title)
    axes[1].set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = ["block tolerance", "easy", "hard"]
    fig.legend(
        [by_label[label] for label in ordered_labels if label in by_label],
        [label for label in ordered_labels if label in by_label],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.tight_layout(w_pad=1.6)
    fig.subplots_adjust(top=0.78)
    return fig


def plot_agreement_gap_scatter(design_summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65))
    final_n = int(design_summary["N"].max())
    rows = design_summary[design_summary["N"] == final_n].copy()
    if "agreement_gap" not in rows.columns:
        axes[0].text(0.5, 0.5, "agreement_gap unavailable", ha="center", va="center")
        axes[1].text(0.5, 0.5, "agreement_gap unavailable", ha="center", va="center")
        return fig

    for (profile, distribution), group in rows.groupby(["profile", "distribution"], sort=False):
        color = PROFILE_COLORS.get(profile, "0.25")
        marker = "o" if distribution == "gaussian" else "s"
        label = f"{PROFILE_LABELS.get(profile, profile)}, {DISTRIBUTION_LABELS.get(distribution, distribution)}"
        axes[0].scatter(
            group["agreement_gap"],
            group["dim_correct_mean"],
            s=18,
            color=color,
            marker=marker,
            alpha=0.70,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
        axes[1].scatter(
            group["agreement_gap"],
            group["first_block_projector_error_op_median"],
            s=18,
            color=color,
            marker=marker,
            alpha=0.70,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )

    axes[0].set_ylabel(r"$\Pr(\widehat m_\infty=m)$")
    axes[1].set_ylabel(r"median subspace error")
    for ax in axes:
        ax.set_xlabel(r"$1-\lambda_{m+1}(\bar\Pi)$")
        ax.grid(False)
    axes[0].set_title(f"Dimension recovery at N={final_n}")
    axes[1].set_title(f"Subspace error at N={final_n}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.tight_layout(w_pad=1.4)
    return fig


def config_stem(rows: pd.DataFrame) -> str:
    first = rows.iloc[0]
    return f"g{int(first['g'])}_p{int(first['p'])}_k{int(first['k'])}_m{int(first['m'])}"


def figure_stem_for(rows: pd.DataFrame, distribution: str) -> str:
    base = config_stem(rows)
    suffix = "_main" if base == DEFAULT_MAIN_CONFIG else ""
    return f"infty_sstar_recovery_{distribution}_{base}{suffix}"


def matching_seqtest_rows(
    seqtest_summary: pd.DataFrame | None,
    *,
    base_config: str,
    distribution: str,
    profiles: list[str],
) -> pd.DataFrame | None:
    if seqtest_summary is None or seqtest_summary.empty:
        return None
    rows = seqtest_summary[
        seqtest_summary["profile"].isin(profiles)
        & (seqtest_summary["distribution"] == distribution)
    ].copy()
    if rows.empty:
        return None
    rows["base_config"] = rows.apply(base_config_label, axis=1)
    rows = rows[rows["base_config"] == base_config]
    if rows.empty:
        return None
    return rows.sort_values(["profile", "N"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    configure_publication_style()

    results_dir = args.results_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    summary, design_summary = read_summaries(results_dir)
    main_seqtest_summary = None
    main_seqtest_results_dir = args.main_seqtest_results_dir.resolve()
    main_seqtest_path = main_seqtest_results_dir / "summary_by_curve_seqtest.csv"
    if main_seqtest_path.exists():
        main_seqtest_summary = pd.read_csv(main_seqtest_path)

    plot_rows = select_plot_rows(
        summary,
        configs=args.configs,
        profiles=list(args.profiles),
        distributions=list(args.distributions),
    )
    plot_rows = plot_rows[
        ~(
            (plot_rows["base_config"] == "g5_p8_k5_m1")
            & (plot_rows["distribution"] == "gaussian_mixture")
        )
    ].reset_index(drop=True)
    written = []
    for (base_config, distribution), rows in plot_rows.groupby(
        ["base_config", "distribution"],
        sort=False,
    ):
        seqtest_rows = matching_seqtest_rows(
            main_seqtest_summary,
            base_config=base_config,
            distribution=distribution,
            profiles=list(args.profiles),
        )
        has_seqtest_overlay = seqtest_rows is not None and not seqtest_rows.empty
        fig = plot_recovery_two_panel(
            rows,
            seqtest_summary=seqtest_rows,
            show_bands=not bool(args.no_bands),
            figsize=(7.05, 2.28) if has_seqtest_overlay else None,
            overlay_panel_gap=0.34 if has_seqtest_overlay else None,
            overlay_legend_gap=0.03 if has_seqtest_overlay else None,
            overlay_legend_width=0.72 if has_seqtest_overlay else None,
            axis_labelsize=9.5 if has_seqtest_overlay else None,
            tick_labelsize=8.8 if has_seqtest_overlay else None,
            legend_fontsize=9.5 if has_seqtest_overlay else None,
        )
        stem = figure_stem_for(rows, distribution)
        save_figure(fig, figures_dir, stem)
        plt.close(fig)
        written.append(stem)

    main_rows = plot_rows[plot_rows["base_config"] == DEFAULT_MAIN_CONFIG]
    gaussian_main_tolerance_rows = pd.DataFrame()
    if not args.skip_appendix and not main_rows.empty:
        gaussian_main = main_rows[main_rows["distribution"] == "gaussian"]
        if not gaussian_main.empty:
            gaussian_main_tolerance_rows = gaussian_main.copy()

    stress_results_dir = args.stress_results_dir.resolve()
    stress_written = []
    if not args.skip_stress and stress_results_dir.exists():
        stress_summary, _ = read_summaries(stress_results_dir)
        stress_rows = select_plot_rows(
            stress_summary,
            configs=[DEFAULT_STRESS_CONFIG],
            profiles=list(args.profiles),
            distributions=["gaussian"],
        )

        seq_path = stress_results_dir / "summary_by_curve_seqtest.csv"
        if not seq_path.exists():
            raise FileNotFoundError(
                f"Missing hard-config FindS_star summary: {seq_path}. "
                "Run run_g2_low_environment_stress.py without --skip-seqtest."
            )
        seq_summary = pd.read_csv(seq_path)
        seq_rows = select_plot_rows(
            seq_summary,
            configs=[DEFAULT_STRESS_CONFIG],
            profiles=list(args.profiles),
            distributions=["gaussian"],
        )

        if not gaussian_main_tolerance_rows.empty:
            fig = plot_small_e_recovery_tolerance_four_panel(
                stress_summary=stress_rows,
                stress_seqtest_summary=seq_rows,
                main_tolerance_summary=gaussian_main_tolerance_rows,
            )
            save_figure(
                fig,
                figures_dir,
                "infty_sstar_recovery_g2_p8_k5_m2_combined_extended_n100000",
            )
            plt.close(fig)
            stress_written.append(
                "infty_sstar_recovery_g2_p8_k5_m2_combined_extended_n100000"
            )

    metadata = {
        "software_versions": software_versions(),
        "results_dir": repo_relative_path(results_dir),
        "main_seqtest_results_dir": repo_relative_path(main_seqtest_results_dir),
        "stress_results_dir": repo_relative_path(stress_results_dir),
        "figures_dir": repo_relative_path(figures_dir),
        "profiles": list(args.profiles),
        "distributions": list(args.distributions),
        "configs": args.configs,
        "bands": not bool(args.no_bands),
        "main_seqtest_overlay": bool(main_seqtest_summary is not None),
        "written_stems": written,
        "stress_written_stems": stress_written,
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "infty_sstar_recovery_plot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()

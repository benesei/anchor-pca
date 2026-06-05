"""Create the cleaned publication rolling-split source/target summary plot.

The plot reads the full per-batch explained-variance CSV produced by
``compute_rolling_split_explained_variance.py`` and averages the same methods
over source batches and target batches. It shows k in {10, 20, 30} and
s in {3, ..., 8}; k=5, k=40, and s=9 are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from compute_rolling_split_explained_variance import to_jsonable  # noqa: E402
from anchorpca.reproducibility import repo_relative_path, software_versions  # noqa: E402


EXPLAINED_VARIANCE_CSV = "rolling_publication_explained_variance_all.csv"
ROLLING_PLOT_K_VALUES = (10, 20, 30)
ROLLING_PLOT_LAST_SOURCE_BATCHES = tuple(range(3, 9))
ROLLING_PLOT_METHODS = ("poolPCA", "AnchorPCA_lambda=1", "AnchorPCA_infty")
ROLLING_PLOT_SPLITS = ("source", "target")
ROLLING_PLOT_BAND_METHODS = ("poolPCA", "AnchorPCA_infty")
ROLLING_PLOT_ROW_YLIMS = {
    "source": (60.0, 100.0),
    "target": (20.0, 100.0),
}

METHOD_LABELS = {
    "poolPCA": "poolPCA",
    "AnchorPCA_lambda=1": r"AnchorPCA$_{\lambda=1}$",
    "AnchorPCA_infty": r"AnchorPCA$_\infty$",
}
METHOD_STYLES = {
    "poolPCA": {"color": "tab:blue", "linestyle": "-", "marker": "o"},
    "AnchorPCA_lambda=1": {"color": "#6A3D9A", "linestyle": "-", "marker": "o"},
    "AnchorPCA_infty": {"color": "#D62728", "linestyle": "-", "marker": "o"},
}


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
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


def adaptive_ylim(*frames: pd.DataFrame) -> tuple[float, float]:
    values = []
    for frame in frames:
        for column in ["min_ev", "mean_ev", "max_ev"]:
            if column in frame.columns:
                values.extend(frame[column].dropna().astype(float).tolist())
    if not values:
        return 0.0, 100.0
    ymin = min(values)
    ymax = max(values)
    span = max(ymax - ymin, 1.0)
    pad = max(2.0, 0.12 * span)
    return max(0.0, ymin - pad), min(100.0, ymax + pad)


def select_rolling_plot_data(
    all_results: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = ROLLING_PLOT_K_VALUES,
    last_source_batches: tuple[int, ...] = ROLLING_PLOT_LAST_SOURCE_BATCHES,
    method_order: tuple[str, ...] = ROLLING_PLOT_METHODS,
) -> pd.DataFrame:
    """Aggregate exactly the rows shown in the rolling publication plot."""
    required = {
        "last_source_batch",
        "k",
        "split",
        "batch",
        "method_id",
        "percent_explained_variance",
    }
    missing = sorted(required.difference(all_results.columns))
    if missing:
        raise ValueError(f"Explained-variance CSV is missing required columns: {missing}")

    filtered = all_results[
        all_results["k"].isin(k_values)
        & all_results["last_source_batch"].isin(last_source_batches)
        & all_results["method_id"].isin(method_order)
        & all_results["split"].isin(ROLLING_PLOT_SPLITS)
    ].copy()

    expected_raw_n = 0
    for s in last_source_batches:
        expected_raw_n += 10 * len(k_values) * len(method_order)
    if len(filtered) != expected_raw_n:
        raise ValueError(
            f"Expected {expected_raw_n} selected per-batch rows but found {len(filtered)}."
        )

    excluded_k = {5, 40}
    if filtered["k"].isin(excluded_k).any():
        raise RuntimeError("k=5 and k=40 must be excluded from the rolling publication plot.")
    if (filtered["last_source_batch"] == 9).any():
        raise RuntimeError("s=9 must be excluded from the rolling publication plot.")
    if (filtered["method_id"] == "AnchorPCA_lambda=10").any():
        raise RuntimeError("AnchorPCA_lambda=10 must be excluded from this plot.")

    split_rows = (
        filtered.groupby(["split", "last_source_batch", "k", "method_id"], as_index=False)
        .agg(
            mean_ev=("percent_explained_variance", "mean"),
            min_ev=("percent_explained_variance", "min"),
            max_ev=("percent_explained_variance", "max"),
            n_batches=("batch", "nunique"),
        )
        .copy()
    )

    expected_summary_n = (
        len(ROLLING_PLOT_SPLITS) * len(k_values) * len(last_source_batches) * len(method_order)
    )
    if len(split_rows) != expected_summary_n:
        raise ValueError(
            f"Expected {expected_summary_n} source/target summary rows but found {len(split_rows)}."
        )

    expected_counts = {
        ("source", s): s for s in last_source_batches
    } | {
        ("target", s): 10 - s for s in last_source_batches
    }
    for row in split_rows.itertuples(index=False):
        expected_n = expected_counts[(row.split, int(row.last_source_batch))]
        if int(row.n_batches) != expected_n:
            raise ValueError(
                "Unexpected number of batches while aggregating rolling plot data: "
                f"split={row.split}, s={row.last_source_batch}, k={row.k}, "
                f"method={row.method_id}, expected {expected_n}, got {row.n_batches}."
            )

    method_rank = {method: rank for rank, method in enumerate(method_order)}
    split_rank = {split: rank for rank, split in enumerate(ROLLING_PLOT_SPLITS)}
    split_rows["method_rank"] = split_rows["method_id"].map(method_rank)
    split_rows["split_rank"] = split_rows["split"].map(split_rank)
    split_rows = split_rows.sort_values(
        ["split_rank", "k", "last_source_batch", "method_rank"]
    ).reset_index(drop=True)
    return split_rows.drop(columns=["method_rank", "split_rank"])


def plot_rolling_summary(
    plot_rows: pd.DataFrame,
    figures_dir: Path,
    *,
    stem: str = "rolling_publication_target_ev_combined",
) -> None:
    configure_publication_style()
    k_values = tuple(sorted(plot_rows["k"].unique()))
    fig, axes = plt.subplots(
        len(ROLLING_PLOT_SPLITS),
        len(k_values),
        figsize=(7.0, 4.25),
        sharex=True,
        sharey="row",
    )
    axes = np.atleast_2d(axes)

    legend_handles = None
    legend_labels = None
    row_labels = {
        "source": "% explained variance\n(source batches)",
        "target": "% explained variance\n(target batches)",
    }

    for row_idx, split in enumerate(ROLLING_PLOT_SPLITS):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            sub = plot_rows[(plot_rows["split"] == split) & (plot_rows["k"] == k)].copy()
            for method_id in ROLLING_PLOT_METHODS:
                mdf = sub[sub["method_id"] == method_id].sort_values("last_source_batch")
                style = METHOD_STYLES[method_id]
                x = mdf["last_source_batch"].to_numpy()
                y = mdf["mean_ev"].to_numpy()
                ymin = mdf["min_ev"].to_numpy()
                ymax = mdf["max_ev"].to_numpy()
                if method_id in ROLLING_PLOT_BAND_METHODS:
                    ax.fill_between(
                        x,
                        ymin,
                        ymax,
                        color=style["color"],
                        alpha=0.08,
                        linewidth=0,
                    )
                ax.plot(
                    x,
                    y,
                    label=METHOD_LABELS[method_id],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=1.9,
                    markersize=5.8,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                )

            if row_idx == 0:
                ax.set_title(f"k={int(k)}")
            if row_idx == len(ROLLING_PLOT_SPLITS) - 1:
                ax.set_xlabel("Last source batch s")
            ax.set_xticks(sorted(sub["last_source_batch"].unique()))
            ax.set_ylim(*ROLLING_PLOT_ROW_YLIMS[split])
            ax.grid(False)
            if col_idx == 0:
                ax.set_ylabel(row_labels[split])
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(3, len(legend_labels) if legend_labels else 3),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), w_pad=0.8, h_pad=1.0)
    save_figure(fig, figures_dir, stem)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else experiment_dir / "results"
    figures_dir = args.figures_dir.resolve() if args.figures_dir else experiment_dir / "figures"
    ev_csv = args.explained_variance_csv.resolve() if args.explained_variance_csv else results_dir / EXPLAINED_VARIANCE_CSV

    all_results = pd.read_csv(ev_csv)
    plot_rows = select_rolling_plot_data(all_results)
    plot_rolling_summary(plot_rows, figures_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_rows.to_csv(
        results_dir / "rolling_publication_target_ev_combined_plot_data.csv",
        index=False,
    )
    with (results_dir / "rolling_publication_target_ev_combined_metadata.json").open("w") as handle:
        json.dump(
            {
                "software_versions": software_versions(),
                "explained_variance_csv": repo_relative_path(ev_csv),
                "splits": list(ROLLING_PLOT_SPLITS),
                "k_values": list(ROLLING_PLOT_K_VALUES),
                "last_source_batches": list(ROLLING_PLOT_LAST_SOURCE_BATCHES),
                "methods": list(ROLLING_PLOT_METHODS),
                "band_methods": list(ROLLING_PLOT_BAND_METHODS),
                "row_y_limits": ROLLING_PLOT_ROW_YLIMS,
                "excluded": {
                    "k": [5, 40],
                    "last_source_batch": [9],
                    "method_id": ["AnchorPCA_lambda=10"],
                },
                "note": (
                    "This plot reads rolling_publication_explained_variance_all.csv. "
                    "The top row shows mean source-batch explained variance and the "
                    "bottom row shows mean target-batch explained variance. Shaded "
                    "bands are min-max over batches in the corresponding split and "
                    "are shown only for poolPCA and AnchorPCA_infty. Source-row "
                    "y-limits are fixed at 60-100%; target-row y-limits are fixed "
                    "at 20-100%."
                ),
            },
            handle,
            indent=2,
            default=to_jsonable,
        )

    print(f"Wrote figure: {figures_dir / 'rolling_publication_target_ev_combined.pdf'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Gas-sensor experiment directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing the publication summary CSVs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where the figure is written.",
    )
    parser.add_argument(
        "--explained-variance-csv",
        type=Path,
        default=None,
        help="Optional explicit full explained-variance CSV path.",
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

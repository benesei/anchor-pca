"""Create the publication source/target gas-sensor explained-variance plot.

The plot is intentionally CSV-only: it reads
``rolling_publication_explained_variance_all.csv`` produced by
``compute_rolling_split_explained_variance.py`` and does not refit or recompute
any representation.
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

from compute_rolling_split_explained_variance import (  # noqa: E402
    format_batch_label,
    make_source_target_batches,
    to_jsonable,
)
from anchorpca.reproducibility import repo_relative_path, software_versions  # noqa: E402


DEFAULT_K = 20
DEFAULT_LAST_SOURCE_BATCH = 6
EXPLAINED_VARIANCE_CSV = "rolling_publication_explained_variance_all.csv"

PLOT_METHOD_ORDER = (
    "poolPCA",
    "AnchorPCA_lambda=1",
    "AnchorPCA_infty",
    "norm-maxRegret",
)
LEGEND_METHOD_ORDER = (
    "AnchorPCA_infty",
    "AnchorPCA_lambda=1",
    "norm-maxRegret",
    "poolPCA",
)
PLOT_LABELS = {
    "poolPCA": "poolPCA",
    "AnchorPCA_lambda=1": r"AnchorPCA$_{\lambda=1}$",
    "AnchorPCA_infty": r"AnchorPCA$_\infty$",
    "norm-maxRegret": "norm-maxRegret",
}
PLOT_STYLES = {
    "poolPCA": {"color": "tab:blue", "linestyle": "-", "zorder": 4},
    "AnchorPCA_lambda=1": {"color": "#6A3D9A", "linestyle": "-", "zorder": 1},
    "AnchorPCA_infty": {"color": "#D62728", "linestyle": "-", "zorder": 3},
    "norm-maxRegret": {"color": "#E69F00", "linestyle": "-", "zorder": 2},
}


def configure_publication_style() -> None:
    """Use the wcPCA-style source/target plot appearance."""
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_figure(fig, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")


def make_plot_stem(*, k: int, last_source_batch: int) -> str:
    return f"gas_sensor_publication_b1_b{last_source_batch}_source_target_k{k}"


def adaptive_ylim(values: pd.Series) -> tuple[float, float]:
    ymin = float(values.min())
    ymax = float(values.max())
    span = max(ymax - ymin, 1.0)
    pad = max(2.0, 0.15 * span)
    return max(0.0, ymin - pad), min(100.0, ymax + pad)


def select_source_target_rows(
    all_results: pd.DataFrame,
    *,
    last_source_batch: int,
    k: int,
    method_order: tuple[str, ...] = PLOT_METHOD_ORDER,
) -> pd.DataFrame:
    """Extract exactly the rows shown in the source/target plot."""
    source_batches, target_batches = make_source_target_batches(last_source_batch)
    expected_batches = set(source_batches).union(target_batches)
    required_columns = {
        "last_source_batch",
        "k",
        "split",
        "batch",
        "method_id",
        "percent_explained_variance",
    }
    missing_columns = sorted(required_columns.difference(all_results.columns))
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

    rows = all_results[
        (all_results["last_source_batch"] == int(last_source_batch))
        & (all_results["k"] == int(k))
        & (all_results["method_id"].isin(method_order))
        & (all_results["batch"].isin(expected_batches))
    ].copy()

    expected_n = len(expected_batches) * len(method_order)
    if len(rows) != expected_n:
        raise ValueError(
            f"Expected {expected_n} rows for s={last_source_batch}, k={k}, "
            f"methods={list(method_order)}, but found {len(rows)}."
        )

    for method_id in method_order:
        method_rows = rows[rows["method_id"] == method_id]
        if sorted(method_rows["batch"].astype(int).tolist()) != sorted(expected_batches):
            raise ValueError(
                f"Method {method_id} does not contain exactly the selected batches."
            )

    split_by_batch = {
        **{batch: "source" for batch in source_batches},
        **{batch: "target" for batch in target_batches},
    }
    expected_split = rows["batch"].map(split_by_batch)
    if not (rows["split"].to_numpy() == expected_split.to_numpy()).all():
        raise ValueError("Source/target split labels in the CSV do not match s.")

    method_rank = {method: rank for rank, method in enumerate(method_order)}
    rows["method_rank"] = rows["method_id"].map(method_rank)
    rows = rows.sort_values(["split", "batch", "method_rank"]).reset_index(drop=True)
    rows["method_label"] = rows["method_id"].map(PLOT_LABELS)
    return rows.drop(columns=["method_rank"])


def compute_anchor_infty_target_improvement(rows: pd.DataFrame) -> dict[str, object]:
    """Return max target-batch 100*(AnchorPCA_infty - poolPCA)/poolPCA."""
    target = rows[rows["split"] == "target"]
    pivot = target.pivot(
        index="batch",
        columns="method_id",
        values="percent_explained_variance",
    )
    missing = {"poolPCA", "AnchorPCA_infty"}.difference(pivot.columns)
    if missing:
        raise ValueError(f"Rows are missing methods needed for the arrow: {sorted(missing)}")

    baseline = pivot["poolPCA"]
    anchor = pivot["AnchorPCA_infty"]
    if (baseline <= 0).any():
        bad_batches = baseline[baseline <= 0].index.astype(int).tolist()
        raise ValueError(f"poolPCA target EV must be positive for arrow computation: {bad_batches}")

    relative_gain = 100.0 * (anchor - baseline) / baseline
    batch = int(relative_gain.idxmax())
    return {
        "batch": batch,
        "baseline_method_id": "poolPCA",
        "method_id": "AnchorPCA_infty",
        "poolPCA_percent_explained_variance": float(baseline.loc[batch]),
        "method_percent_explained_variance": float(anchor.loc[batch]),
        "relative_improvement_percent": float(relative_gain.loc[batch]),
    }


def draw_improvement_arrow(
    ax,
    *,
    improvement: dict[str, object],
    target_batches: tuple[int, ...],
    label_position: str,
) -> None:
    if improvement["relative_improvement_percent"] <= 0:
        return

    batch = int(improvement["batch"])
    x_pos = list(target_batches).index(batch)
    y_start = float(improvement["poolPCA_percent_explained_variance"])
    y_end = float(improvement["method_percent_explained_variance"])

    ax.annotate(
        "",
        xy=(x_pos, y_end),
        xytext=(x_pos, y_start),
        arrowprops=dict(
            arrowstyle="-|>",
            color="tab:green",
            lw=3.2,
            mutation_scale=18,
        ),
        zorder=20,
    )

    if label_position == "left-peak":
        x_text = x_pos - 0.18
        y_text = y_end
        ha = "right"
        va = "center"
    elif label_position == "left-bottom":
        x_text = x_pos - 0.18
        y_text = y_start
        ha = "right"
        va = "center"
    elif label_position == "below":
        x_offset = -0.22 if x_pos >= len(target_batches) - 1 else 0.16
        ha = "right" if x_offset < 0 else "left"
        x_text = x_pos + x_offset
        y_text = y_start
        va = "center"
    else:
        raise ValueError(f"Unknown arrow label position: {label_position!r}.")

    ax.text(
        x_text,
        y_text,
        f"{improvement['relative_improvement_percent']:.1f}%",
        color="tab:green",
        fontsize=11.5,
        fontweight="bold",
        ha=ha,
        va=va,
        zorder=21,
    )


def plot_source_target(
    rows: pd.DataFrame,
    figures_dir: Path,
    *,
    source_batches: tuple[int, ...],
    target_batches: tuple[int, ...],
    plot_stem: str,
    arrow_label_position: str,
    target_spacing_scale: float,
    plot_arrow: bool = True,
) -> dict[str, object]:
    """Create the two-panel source/target line plot."""
    configure_publication_style()

    source_intervals = max(len(source_batches) - 1, 1)
    target_intervals = max(len(target_batches) - 1, 1)
    width_ratios = [
        float(source_intervals),
        float(target_intervals) * float(target_spacing_scale),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.0, 2.25),
        sharey=True,
        width_ratios=width_ratios,
    )
    shared_ylim = adaptive_ylim(rows["percent_explained_variance"])
    panel_specs = [
        (axes[0], "source", source_batches, "Source batches"),
        (axes[1], "target", target_batches, "Target batches"),
    ]

    for ax, split, batches, xlabel in panel_specs:
        sub = rows[rows["split"] == split]
        x = np.arange(len(batches))
        for method_id in PLOT_METHOD_ORDER:
            method_rows = (
                sub[sub["method_id"] == method_id]
                .set_index("batch")
                .loc[list(batches)]
                .reset_index()
            )
            style = PLOT_STYLES[method_id]
            ax.plot(
                x,
                method_rows["percent_explained_variance"],
                label=PLOT_LABELS[method_id],
                color=style["color"],
                marker="o",
                linestyle=style["linestyle"],
                linewidth=2.4,
                markersize=7.5,
                markeredgecolor="white",
                markeredgewidth=1.3,
                zorder=style["zorder"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"B{batch}" for batch in batches])
        ax.tick_params(axis="x", rotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.set_xlabel(xlabel)
        ax.set_ylim(*shared_ylim)
        ax.grid(False)

    axes[0].set_ylabel("% explained variance")
    improvement = compute_anchor_infty_target_improvement(rows)
    if plot_arrow:
        draw_improvement_arrow(
            axes[1],
            improvement=improvement,
            target_batches=target_batches,
            label_position=arrow_label_position,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    fig.legend(
        [handles_by_label[PLOT_LABELS[method_id]] for method_id in LEGEND_METHOD_ORDER],
        [PLOT_LABELS[method_id] for method_id in LEGEND_METHOD_ORDER],
        title="Method",
        loc="center left",
        bbox_to_anchor=(0.82, 0.62),
        frameon=False,
        borderaxespad=0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0), w_pad=1.2)
    save_figure(fig, figures_dir, plot_stem)
    plt.close(fig)
    return improvement


def run(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else experiment_dir / "results"
    figures_dir = args.figures_dir.resolve() if args.figures_dir else experiment_dir / "figures"
    input_csv = args.input_csv.resolve() if args.input_csv else results_dir / EXPLAINED_VARIANCE_CSV

    all_results = pd.read_csv(input_csv)
    rows = select_source_target_rows(
        all_results,
        last_source_batch=args.last_source_batch,
        k=args.k,
    )
    source_batches, target_batches = make_source_target_batches(args.last_source_batch)
    plot_stem = make_plot_stem(k=args.k, last_source_batch=args.last_source_batch)
    improvement = plot_source_target(
        rows,
        figures_dir,
        source_batches=source_batches,
        target_batches=target_batches,
        plot_stem=plot_stem,
        arrow_label_position=args.arrow_label_position,
        target_spacing_scale=args.target_spacing_scale,
        plot_arrow=args.plot_arrow,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(results_dir / f"{plot_stem}_plot_data.csv", index=False)
    with (results_dir / f"{plot_stem}_metadata.json").open("w") as handle:
        json.dump(
            {
                "software_versions": software_versions(),
                "input_csv": repo_relative_path(input_csv),
                "k": int(args.k),
                "last_source_batch": int(args.last_source_batch),
                "source_batches": list(source_batches),
                "target_batches": list(target_batches),
                "source_batches_label": format_batch_label(source_batches),
                "target_batches_label": format_batch_label(target_batches),
                "methods": list(PLOT_METHOD_ORDER),
                "best_relative_target_improvement": improvement,
                "plot_arrow": bool(args.plot_arrow),
                "arrow_label_position": args.arrow_label_position,
                "target_spacing_scale": float(args.target_spacing_scale),
                "note": (
                    "The plot reads only rows from the publication CSV. "
                    "No method is refit and no EV value is recomputed here."
                ),
            },
            handle,
            indent=2,
            default=to_jsonable,
        )

    print(f"Wrote figure: {figures_dir / (plot_stem + '.pdf')}")
    print(f"Wrote plot data: {results_dir / (plot_stem + '_plot_data.csv')}")
    print("\nBest AnchorPCA_infty relative target improvement over poolPCA")
    print(pd.DataFrame([improvement]).to_string(index=False))


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
        help="Directory containing rolling_publication_explained_variance_all.csv.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where the figure is written.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional explicit explained-variance CSV path.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--last-source-batch", type=int, default=DEFAULT_LAST_SOURCE_BATCH)
    parser.add_argument(
        "--arrow-label-position",
        choices=["below", "left-peak", "left-bottom"],
        default="below",
    )
    parser.add_argument(
        "--plot-arrow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to draw the green best target-improvement arrow. "
            "Use --no-plot-arrow to omit it."
        ),
    )
    parser.add_argument(
        "--target-spacing-scale",
        type=float,
        default=1.0,
        help="Horizontal spacing multiplier for target-batch intervals.",
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

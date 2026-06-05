"""Plot estimated S_star against same-dimensional poolPCA on B9 and B10.

For each rolling split and k, this diagnostic compares two subspaces with the
same estimated dimension m_hat:

* AnchorPCA_infty's estimated first-block S_star subspace;
* the top m_hat source-only poolPCA directions.

The plot shows the percentage of explained variance on target batches B9 and
B10 separately. It reuses the same source-only preprocessing, covariance
construction, poolPCA fitting, and EV checks as the main gas-sensor pipeline.
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

import compute_rolling_split_explained_variance as compute  # noqa: E402
import plot_sstar_poolpca_tradeoff as same_dim  # noqa: E402
from anchorpca.reproducibility import repo_relative_path, software_versions  # noqa: E402


DEFAULT_TARGET_BATCHES = (9, 10)
PLOT_STEM = "sstar_poolpca_same_dim_b9_b10_ev"
TARGET_BATCH_YLIMS = {
    9: (0.0, 40.0),
    10: (60.0, 100.0),
}

METHOD_ORDER = ("poolPCA_top_same_dim", "AnchorPCA_infty_Sstar_first_block")
METHOD_LABELS = {
    "poolPCA_top_same_dim": r"poolPCA top $\hat m$",
    "AnchorPCA_infty_Sstar_first_block": r"estimated $S_\star$",
}
METHOD_STYLES = {
    "poolPCA_top_same_dim": {
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
        "zorder": 3,
    },
    "AnchorPCA_infty_Sstar_first_block": {
        "color": "#D62728",
        "marker": "o",
        "linestyle": "-",
        "zorder": 4,
    },
}


def select_b9_b10_plot_data(
    batch_table: pd.DataFrame,
    *,
    target_batches: tuple[int, ...] = DEFAULT_TARGET_BATCHES,
    k_values: tuple[int, ...] = same_dim.DEFAULT_K_VALUES,
    last_source_batches: tuple[int, ...] = same_dim.DEFAULT_LAST_SOURCE_BATCHES,
) -> pd.DataFrame:
    """Return the long-form rows plotted for B9/B10 same-dimension EV."""
    required = {
        "last_source_batch",
        "k",
        "split",
        "batch",
        "sstar_percent_ev",
        "poolpca_top_d_percent_ev",
        "invariant_n_selected",
        "invariant_dim_estimate",
        "block_tol",
        "block_tol_mode",
    }
    missing = sorted(required.difference(batch_table.columns))
    if missing:
        raise ValueError(f"Same-dimension batch table is missing columns: {missing}")

    selected = batch_table[
        (batch_table["split"] == "target")
        & batch_table["batch"].isin(target_batches)
        & batch_table["k"].isin(k_values)
        & batch_table["last_source_batch"].isin(last_source_batches)
    ].copy()

    expected_n = len(target_batches) * len(k_values) * len(last_source_batches)
    if len(selected) != expected_n:
        raise ValueError(
            f"Expected {expected_n} selected target-batch rows but found {len(selected)}."
        )

    for s in last_source_batches:
        invalid_targets = [batch for batch in target_batches if batch <= s]
        if invalid_targets:
            raise ValueError(
                f"Target batches {invalid_targets} are not held out when last_source_batch={s}."
            )

    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        common = {
            "last_source_batch": int(row.last_source_batch),
            "k": int(row.k),
            "target_batch": int(row.batch),
            "invariant_dim_estimate": int(row.invariant_dim_estimate),
            "invariant_n_selected": int(row.invariant_n_selected),
            "block_tol": float(row.block_tol),
            "block_tol_mode": str(row.block_tol_mode),
            "dimension_label": rf"$\hat{{m}}={int(row.invariant_n_selected)}$",
        }
        records.append(
            {
                **common,
                "method_id": "poolPCA_top_same_dim",
                "method_label": METHOD_LABELS["poolPCA_top_same_dim"],
                "percent_explained_variance": float(row.poolpca_top_d_percent_ev),
            }
        )
        records.append(
            {
                **common,
                "method_id": "AnchorPCA_infty_Sstar_first_block",
                "method_label": METHOD_LABELS["AnchorPCA_infty_Sstar_first_block"],
                "percent_explained_variance": float(row.sstar_percent_ev),
            }
        )

    plot_rows = pd.DataFrame.from_records(records)
    expected_long_n = 2 * expected_n
    if len(plot_rows) != expected_long_n:
        raise RuntimeError(f"Expected {expected_long_n} plot rows but got {len(plot_rows)}.")

    method_rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    plot_rows["method_rank"] = plot_rows["method_id"].map(method_rank)
    return (
        plot_rows.sort_values(["target_batch", "k", "last_source_batch", "method_rank"])
        .drop(columns=["method_rank"])
        .reset_index(drop=True)
    )


def adaptive_ylim(values: pd.Series) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    lower = float(vals.min())
    upper = float(vals.max())
    span = max(upper - lower, 1.0)
    pad = max(2.0, 0.08 * span)
    return max(0.0, lower - pad), min(100.0, upper + pad)


def plot_b9_b10_same_dim_ev(
    plot_rows: pd.DataFrame,
    figures_dir: Path,
    *,
    stem: str = PLOT_STEM,
) -> None:
    same_dim.configure_publication_style()
    target_batches = tuple(sorted(plot_rows["target_batch"].unique()))
    k_values = tuple(sorted(plot_rows["k"].unique()))
    fig, axes = plt.subplots(
        len(target_batches),
        len(k_values),
        figsize=(7.0, 4.1),
        sharex=True,
        sharey="row",
    )
    axes = np.atleast_2d(axes)
    legend_handles = None
    legend_labels = None

    for row_idx, batch in enumerate(target_batches):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            sub = plot_rows[(plot_rows["target_batch"] == batch) & (plot_rows["k"] == k)]
            for method_id in METHOD_ORDER:
                mdf = sub[sub["method_id"] == method_id].sort_values("last_source_batch")
                style = METHOD_STYLES[method_id]
                ax.plot(
                    mdf["last_source_batch"],
                    mdf["percent_explained_variance"],
                    label=METHOD_LABELS[method_id],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.9,
                    markersize=5.8,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=style["zorder"],
                )

            if row_idx == 0:
                ax.set_title(f"k={int(k)}")
            if row_idx == len(target_batches) - 1:
                ax.set_xlabel("Last source batch s")
                dimension_rows = (
                    sub[sub["method_id"] == "AnchorPCA_infty_Sstar_first_block"]
                    .sort_values("last_source_batch")
                    .drop_duplicates(["last_source_batch", "k"])
                )
                y0, y1 = TARGET_BATCH_YLIMS.get(
                    int(batch),
                    adaptive_ylim(sub["percent_explained_variance"]),
                )
                label_y = y0 + 0.04 * (y1 - y0)
                for point in dimension_rows.itertuples(index=False):
                    ax.text(
                        point.last_source_batch,
                        label_y,
                        point.dimension_label,
                        ha="center",
                        va="bottom",
                        fontsize=5.8,
                        color="black",
                    )
            if col_idx == 0:
                ax.set_ylabel(f"B{int(batch)}\n% explained variance")
            split_values = sorted(sub["last_source_batch"].unique())
            ax.set_xticks(split_values)
            ax.set_xlim(min(split_values) - 0.35, max(split_values) + 0.35)
            ax.set_ylim(*TARGET_BATCH_YLIMS.get(int(batch), adaptive_ylim(sub["percent_explained_variance"])))
            ax.grid(False)
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0), w_pad=0.9, h_pad=1.0)
    same_dim.save_figure(fig, figures_dir, stem)
    plt.close(fig)


def build_same_dim_batch_table_from_inputs(
    *,
    data_dir: Path,
    sstar_csv: Path,
    k_values: tuple[int, ...],
    last_source_batches: tuple[int, ...],
    scale_mode: str,
    force_download: bool,
    skip_sha256_check: bool,
) -> pd.DataFrame:
    sstar_results = pd.read_csv(sstar_csv)
    sstar_rows = same_dim.select_sstar_rows(
        sstar_results,
        k_values=k_values,
        last_source_batches=last_source_batches,
    )
    dataset = compute.load_dataset(
        data_dir,
        force_download=force_download,
        skip_sha256_check=skip_sha256_check,
    )
    pool_rows = same_dim.evaluate_poolpca_top_same_dim(
        dataset,
        sstar_rows,
        k_values=k_values,
        last_source_batches=last_source_batches,
        scale_mode=scale_mode,
    )
    return same_dim.build_same_dim_batch_table(sstar_rows, pool_rows)


def run(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else experiment_dir / "results"
    figures_dir = args.figures_dir.resolve() if args.figures_dir else experiment_dir / "figures"
    data_dir = args.data_dir.resolve() if args.data_dir else experiment_dir / "data"
    sstar_csv = args.sstar_csv.resolve() if args.sstar_csv else results_dir / same_dim.SSTAR_CSV
    k_values = tuple(int(k) for k in args.k_values)
    last_source_batches = tuple(int(s) for s in args.last_source_batches)
    target_batches = tuple(int(batch) for batch in args.target_batches)

    print("Same-dimension Sstar versus poolPCA on target batches")
    print("=" * 72)
    print(f"k values: {list(k_values)}")
    print(f"Last source batches s: {list(last_source_batches)}")
    print(f"Target batches shown separately: {[f'B{batch}' for batch in target_batches]}")
    print("Leakage check: poolPCA_top_m is fitted only on source covariances.")
    print("Targets are used only for explained-variance evaluation.")

    batch_table = build_same_dim_batch_table_from_inputs(
        data_dir=data_dir,
        sstar_csv=sstar_csv,
        k_values=k_values,
        last_source_batches=last_source_batches,
        scale_mode=args.scale_mode,
        force_download=bool(args.force_download),
        skip_sha256_check=bool(args.skip_sha256_check),
    )
    plot_rows = select_b9_b10_plot_data(
        batch_table,
        target_batches=target_batches,
        k_values=k_values,
        last_source_batches=last_source_batches,
    )
    plot_b9_b10_same_dim_ev(plot_rows, figures_dir, stem=args.plot_stem)

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_data_path = results_dir / f"{args.plot_stem}_plot_data.csv"
    metadata_path = results_dir / f"{args.plot_stem}_metadata.json"
    plot_rows.to_csv(plot_data_path, index=False)
    with metadata_path.open("w") as handle:
        json.dump(
            {
                "software_versions": software_versions(),
                "sstar_csv": repo_relative_path(sstar_csv),
                "k_values": list(k_values),
                "last_source_batches": list(last_source_batches),
                "target_batches": list(target_batches),
                "target_batch_y_limits": {
                    str(batch): TARGET_BATCH_YLIMS.get(batch)
                    for batch in target_batches
                    if batch in TARGET_BATCH_YLIMS
                },
                "scale_mode": args.scale_mode,
                "comparison": (
                    "For each (s, k), compare AnchorPCA_infty estimated S_star "
                    "to source-only poolPCA with n_components equal to the "
                    "estimated first-block dimension m_hat."
                ),
                "plotted_value": "percentage explained variance on each target batch",
                "leakage_safeguard": (
                    "Feature scaling and poolPCA fitting use only source batches "
                    "B1-Bs. Target batches are used only for EV evaluation."
                ),
            },
            handle,
            indent=2,
            default=compute.to_jsonable,
        )

    print(f"Wrote figure: {figures_dir / (args.plot_stem + '.pdf')}")
    print(f"Wrote plot data: {plot_data_path}")


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
        help="Directory containing S_star CSVs and receiving diagnostic CSVs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where the diagnostic figure is written.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing or receiving the UCI gas-sensor data.",
    )
    parser.add_argument(
        "--sstar-csv",
        type=Path,
        default=None,
        help="Optional explicit rolling_publication_anchor_infty_sstar_all.csv path.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(same_dim.DEFAULT_K_VALUES),
        help="k values to include.",
    )
    parser.add_argument(
        "--last-source-batches",
        type=int,
        nargs="+",
        default=list(same_dim.DEFAULT_LAST_SOURCE_BATCHES),
        help="Last source batch values s to include.",
    )
    parser.add_argument(
        "--target-batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_BATCHES),
        help="Held-out target batches to show separately.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["source-standard", "none"],
        default="source-standard",
        help="Preprocessing mode. Use source-standard for publication figures.",
    )
    parser.add_argument(
        "--plot-stem",
        default=PLOT_STEM,
        help="Filename stem for the figure and CSV outputs.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the UCI archive even if a cached archive exists.",
    )
    parser.add_argument(
        "--skip-sha256-check",
        action="store_true",
        help="Skip the UCI archive SHA256 check.",
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

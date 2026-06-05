"""Plot the same-dimension S_star versus poolPCA source/target tradeoff.

For each rolling split and k, this diagnostic compares the estimated
AnchorPCA_infty first-block subspace to the top poolPCA subspace of exactly the
same estimated dimension. The x-axis is the mean source-batch explained
variance difference and the y-axis is the mean target-batch explained variance
difference, both in percentage points:

    S_star_hat - poolPCA_top_d_hat.

The script reuses the main gas-sensor pipeline's source-only preprocessing,
source covariance construction, poolPCA fitting, and explained-variance
trace/projection checks. Target batches are used only for evaluation.
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
from anchorpca.reproducibility import repo_relative_path, software_versions  # noqa: E402


DEFAULT_K_VALUES = (10, 20, 30)
DEFAULT_LAST_SOURCE_BATCHES = tuple(range(3, 9))
SSTAR_CSV = "rolling_publication_anchor_infty_sstar_all.csv"
PLOT_STEM = "sstar_poolpca_same_dim_source_target_tradeoff"


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


def select_sstar_rows(
    sstar_results: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    last_source_batches: tuple[int, ...] = DEFAULT_LAST_SOURCE_BATCHES,
) -> pd.DataFrame:
    """Extract the S_star rows used by this diagnostic."""
    required = {
        "last_source_batch",
        "n_source_batches",
        "k",
        "split",
        "batch",
        "percent_explained_variance",
        "invariant_dim_estimate",
        "invariant_n_selected",
        "block_tol",
        "block_tol_mode",
        "preprocessing_mode",
    }
    missing = sorted(required.difference(sstar_results.columns))
    if missing:
        raise ValueError(f"S_star CSV is missing required columns: {missing}")

    rows = sstar_results[
        sstar_results["k"].isin(k_values)
        & sstar_results["last_source_batch"].isin(last_source_batches)
    ].copy()

    expected_n = 10 * len(k_values) * len(last_source_batches)
    if len(rows) != expected_n:
        raise ValueError(
            f"Expected {expected_n} S_star per-batch rows but found {len(rows)}."
        )

    for (s, k), group in rows.groupby(["last_source_batch", "k"], sort=False):
        batches = set(group["batch"].astype(int))
        expected_batches = set(range(1, 11))
        if batches != expected_batches:
            raise ValueError(
                f"S_star rows for s={s}, k={k} do not contain exactly batches B1-B10."
            )
        source_batches, target_batches = compute.make_source_target_batches(int(s))
        source_seen = set(group.loc[group["split"] == "source", "batch"].astype(int))
        target_seen = set(group.loc[group["split"] == "target", "batch"].astype(int))
        if source_seen != set(source_batches) or target_seen != set(target_batches):
            raise ValueError(f"S_star source/target labels are inconsistent for s={s}, k={k}.")
        for column in ["invariant_dim_estimate", "invariant_n_selected", "block_tol"]:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"S_star column {column} is not constant for s={s}, k={k}.")

    return rows.reset_index(drop=True)


def fit_poolpca_top_same_dim(
    split_data: compute.SplitData,
    *,
    comparison_dim: int,
    anchor_k: int,
) -> compute.FittedRepresentation:
    """Fit source-only poolPCA with the same dimension as estimated S_star."""
    d = int(comparison_dim)
    if d <= 0:
        raise ValueError(f"comparison_dim must be positive; got {comparison_dim}.")

    min_source_n = min(split_data.batch_stats[batch].n_obs for batch in split_data.source_batches)
    max_rank = min(compute.N_FEATURES, min_source_n - 1)
    if d > max_rank:
        raise ValueError(
            f"comparison_dim={d} exceeds rank budget {max_rank} for "
            f"source batches {list(split_data.source_batches)}."
        )

    pool = compute.pool_pca_from_covariances(
        split_data.source_covariances,
        n_components=d,
    )
    return compute.FittedRepresentation(
        method_id="poolPCA",
        directions=np.asarray(pool["directions"], dtype=float),
        fit_source="anchorpca",
        details={
            "anchor_k": int(anchor_k),
            "comparison_dim": d,
            "note": "poolPCA fitted on source covariances only with n_components=d_hat.",
        },
    )


def evaluate_poolpca_top_same_dim(
    dataset: compute.Dataset,
    sstar_rows: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    last_source_batches: tuple[int, ...] = DEFAULT_LAST_SOURCE_BATCHES,
    scale_mode: str = "source-standard",
) -> pd.DataFrame:
    """Compute poolPCA-top-d EV using the same source-only splits as S_star."""
    records: list[dict[str, object]] = []
    dimension_table = (
        sstar_rows.groupby(["last_source_batch", "k"], sort=False)
        .agg(
            invariant_dim_estimate=("invariant_dim_estimate", "first"),
            invariant_n_selected=("invariant_n_selected", "first"),
            block_tol=("block_tol", "first"),
            block_tol_mode=("block_tol_mode", "first"),
        )
        .reset_index()
    )

    for s in last_source_batches:
        split_data = compute.prepare_split_data(
            dataset.X_raw,
            dataset.metadata,
            last_source_batch=int(s),
            scale_mode=scale_mode,
        )
        for k in k_values:
            match = dimension_table[
                (dimension_table["last_source_batch"] == int(s))
                & (dimension_table["k"] == int(k))
            ]
            if len(match) != 1:
                raise ValueError(f"Could not find exactly one S_star dimension for s={s}, k={k}.")

            comparison_dim = int(match["invariant_n_selected"].iloc[0])
            representation = fit_poolpca_top_same_dim(
                split_data,
                comparison_dim=comparison_dim,
                anchor_k=int(k),
            )
            for split, batches in [
                ("source", split_data.source_batches),
                ("target", split_data.target_batches),
            ]:
                for batch in batches:
                    row = compute.explained_variance_row(
                        representation,
                        split_data.batch_stats[batch],
                        k=comparison_dim,
                        split=split,
                    )
                    row["k"] = int(k)
                    row["comparison_dim"] = comparison_dim
                    row["method_id"] = "poolPCA_top_same_dim"
                    row["method_label"] = r"poolPCA top $\hat d$"
                    row["last_source_batch"] = int(s)
                    row["n_source_batches"] = len(split_data.source_batches)
                    row["source_batches_label"] = compute.format_batch_label(
                        split_data.source_batches
                    )
                    row["target_batches_label"] = compute.format_batch_label(
                        split_data.target_batches
                    )
                    row["preprocessing_mode"] = scale_mode
                    row["invariant_dim_estimate"] = int(match["invariant_dim_estimate"].iloc[0])
                    row["invariant_n_selected"] = comparison_dim
                    row["block_tol"] = float(match["block_tol"].iloc[0])
                    row["block_tol_mode"] = str(match["block_tol_mode"].iloc[0])
                    records.append(row)

    pool_rows = pd.DataFrame.from_records(records)
    expected_n = 10 * len(k_values) * len(last_source_batches)
    if len(pool_rows) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} poolPCA same-dimension rows but got {len(pool_rows)}."
        )
    return pool_rows


def build_same_dim_batch_table(
    sstar_rows: pd.DataFrame,
    pool_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-batch S_star and poolPCA-top-d EV values."""
    join_cols = ["last_source_batch", "k", "split", "batch"]
    sstar_keep = join_cols + [
        "percent_explained_variance",
        "invariant_dim_estimate",
        "invariant_n_selected",
        "block_tol",
        "block_tol_mode",
        "preprocessing_mode",
    ]
    pool_keep = join_cols + ["percent_explained_variance", "comparison_dim"]

    left = sstar_rows[sstar_keep].rename(
        columns={"percent_explained_variance": "sstar_percent_ev"}
    )
    right = pool_rows[pool_keep].rename(
        columns={"percent_explained_variance": "poolpca_top_d_percent_ev"}
    )
    merged = left.merge(right, on=join_cols, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            "Per-batch S_star and poolPCA same-dimension rows do not align one-to-one."
        )
    if not (merged["comparison_dim"].astype(int) == merged["invariant_n_selected"].astype(int)).all():
        raise ValueError("poolPCA comparison dimensions do not match S_star selected dimensions.")

    merged["delta_sstar_minus_poolpca_pp"] = (
        merged["sstar_percent_ev"] - merged["poolpca_top_d_percent_ev"]
    )
    return merged.sort_values(join_cols).reset_index(drop=True)


def build_tradeoff_summary(batch_table: pd.DataFrame) -> pd.DataFrame:
    """Build one source-cost/target-gain point for each (s, k)."""
    required = {
        "last_source_batch",
        "k",
        "split",
        "batch",
        "sstar_percent_ev",
        "poolpca_top_d_percent_ev",
        "delta_sstar_minus_poolpca_pp",
        "invariant_dim_estimate",
        "invariant_n_selected",
        "block_tol",
        "block_tol_mode",
    }
    missing = sorted(required.difference(batch_table.columns))
    if missing:
        raise ValueError(f"Batch table is missing required columns: {missing}")

    grouped = (
        batch_table.groupby(["last_source_batch", "k", "split"], sort=False)
        .agg(
            mean_sstar_ev=("sstar_percent_ev", "mean"),
            mean_poolpca_top_d_ev=("poolpca_top_d_percent_ev", "mean"),
            mean_delta_pp=("delta_sstar_minus_poolpca_pp", "mean"),
            min_delta_pp=("delta_sstar_minus_poolpca_pp", "min"),
            max_delta_pp=("delta_sstar_minus_poolpca_pp", "max"),
            n_batches=("batch", "nunique"),
            invariant_dim_estimate=("invariant_dim_estimate", "first"),
            invariant_n_selected=("invariant_n_selected", "first"),
            block_tol=("block_tol", "first"),
            block_tol_mode=("block_tol_mode", "first"),
        )
        .reset_index()
    )

    source = grouped[grouped["split"] == "source"].drop(columns=["split"]).rename(
        columns={
            "mean_sstar_ev": "mean_source_sstar_ev",
            "mean_poolpca_top_d_ev": "mean_source_poolpca_top_d_ev",
            "mean_delta_pp": "source_delta_pp",
            "min_delta_pp": "min_source_delta_pp",
            "max_delta_pp": "max_source_delta_pp",
            "n_batches": "n_source_batches",
        }
    )
    target = grouped[grouped["split"] == "target"].drop(columns=["split"]).rename(
        columns={
            "mean_sstar_ev": "mean_target_sstar_ev",
            "mean_poolpca_top_d_ev": "mean_target_poolpca_top_d_ev",
            "mean_delta_pp": "target_delta_pp",
            "min_delta_pp": "worst_target_delta_pp",
            "max_delta_pp": "best_target_delta_pp",
            "n_batches": "n_target_batches",
        }
    )
    duplicate_cols = [
        "invariant_dim_estimate",
        "invariant_n_selected",
        "block_tol",
        "block_tol_mode",
    ]
    target = target.drop(columns=duplicate_cols)
    summary = source.merge(
        target,
        on=["last_source_batch", "k"],
        how="inner",
        validate="one_to_one",
    )

    wins = (
        batch_table[batch_table["split"] == "target"]
        .assign(target_win=lambda frame: frame["delta_sstar_minus_poolpca_pp"] > 0)
        .groupby(["last_source_batch", "k"], sort=False)
        .agg(n_target_wins=("target_win", "sum"))
        .reset_index()
    )
    summary = summary.merge(wins, on=["last_source_batch", "k"], how="left", validate="one_to_one")
    summary["n_target_wins"] = summary["n_target_wins"].astype(int)
    summary["source_difference_pp"] = summary["source_delta_pp"]
    summary["target_difference_pp"] = summary["target_delta_pp"]
    return summary.sort_values(["k", "last_source_batch"]).reset_index(drop=True)


def axis_limits(values: pd.Series) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    lower = float(min(vals.min(), 0.0))
    upper = float(max(vals.max(), 0.0))
    span = max(upper - lower, 1.0)
    pad = max(1.0, 0.15 * span)
    return lower - pad, upper + pad


def plot_tradeoff(
    summary: pd.DataFrame,
    figures_dir: Path,
    *,
    stem: str = PLOT_STEM,
) -> None:
    configure_publication_style()
    k_values = tuple(sorted(summary["k"].unique()))
    fig, axes = plt.subplots(
        1,
        len(k_values),
        figsize=(7.0, 2.45),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    xlim = axis_limits(summary["source_difference_pp"])
    ylim = axis_limits(summary["target_difference_pp"])

    for ax, k in zip(axes, k_values):
        sub = summary[summary["k"] == k].sort_values("last_source_batch")
        ax.axvline(0.0, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
        ax.scatter(
            sub["source_difference_pp"],
            sub["target_difference_pp"],
            s=45,
            color="#D62728",
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for row in sub.itertuples(index=False):
            ax.annotate(
                f"s={int(row.last_source_batch)}",
                (row.source_difference_pp, row.target_difference_pp),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.8,
                color="0.15",
            )
        ax.set_title(f"k={int(k)}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(False)

    axes[0].set_ylabel("Target %EV difference\nSstar - poolPCA top d (pp)")
    for ax in axes:
        ax.set_xlabel("Source %EV difference\nSstar - poolPCA top d (pp)")
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, figures_dir, stem)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else experiment_dir / "results"
    figures_dir = args.figures_dir.resolve() if args.figures_dir else experiment_dir / "figures"
    data_dir = args.data_dir.resolve() if args.data_dir else experiment_dir / "data"
    sstar_csv = args.sstar_csv.resolve() if args.sstar_csv else results_dir / SSTAR_CSV
    k_values = tuple(int(k) for k in args.k_values)
    last_source_batches = tuple(int(s) for s in args.last_source_batches)

    print("Same-dimension Sstar versus poolPCA tradeoff")
    print("=" * 72)
    print(f"k values: {list(k_values)}")
    print(f"Last source batches s: {list(last_source_batches)}")
    print("Leakage check: poolPCA_top_d is fitted only on source covariances.")
    print("Targets are used only for explained-variance evaluation.")

    sstar_results = pd.read_csv(sstar_csv)
    sstar_rows = select_sstar_rows(
        sstar_results,
        k_values=k_values,
        last_source_batches=last_source_batches,
    )
    dataset = compute.load_dataset(
        data_dir,
        force_download=bool(args.force_download),
        skip_sha256_check=bool(args.skip_sha256_check),
    )
    pool_rows = evaluate_poolpca_top_same_dim(
        dataset,
        sstar_rows,
        k_values=k_values,
        last_source_batches=last_source_batches,
        scale_mode=args.scale_mode,
    )
    batch_table = build_same_dim_batch_table(sstar_rows, pool_rows)
    summary = build_tradeoff_summary(batch_table)
    plot_tradeoff(summary, figures_dir, stem=args.plot_stem)

    results_dir.mkdir(parents=True, exist_ok=True)
    batch_path = results_dir / f"{args.plot_stem}_batch_values.csv"
    summary_path = results_dir / f"{args.plot_stem}_summary.csv"
    metadata_path = results_dir / f"{args.plot_stem}_metadata.json"
    batch_table.to_csv(batch_path, index=False)
    summary.to_csv(summary_path, index=False)
    with metadata_path.open("w") as handle:
        json.dump(
            {
                "software_versions": software_versions(),
                "sstar_csv": repo_relative_path(sstar_csv),
                "k_values": list(k_values),
                "last_source_batches": list(last_source_batches),
                "scale_mode": args.scale_mode,
                "comparison": (
                    "For each (s, k), compare AnchorPCA_infty estimated S_star "
                    "to source-only poolPCA with n_components equal to the "
                    "estimated first-block dimension d_hat."
                ),
                "x_axis": "mean source %EV(Sstar_hat) - mean source %EV(poolPCA_top_d_hat)",
                "y_axis": "mean target %EV(Sstar_hat) - mean target %EV(poolPCA_top_d_hat)",
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
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote per-batch values: {batch_path}")


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
        default=list(DEFAULT_K_VALUES),
        help="k values to include.",
    )
    parser.add_argument(
        "--last-source-batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAST_SOURCE_BATCHES),
        help="Last source batch values s to include.",
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

"""Compute the same-dimensional comparison used in the target-batch figure."""

from __future__ import annotations

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


DEFAULT_K_VALUES = (10, 20, 30)
DEFAULT_LAST_SOURCE_BATCHES = tuple(range(3, 9))
SSTAR_CSV = "rolling_publication_anchor_infty_sstar_all.csv"


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
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


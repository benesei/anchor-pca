"""Visualize poolPCA and AnchorPCA_infty gas-sensor loading structure.

Default diagnostic: source batches B1--B6, k=20. The script fits only poolPCA
and AnchorPCA_infty on source covariances after source-only standardization.
Target batches are not used.

Outputs:
* feature-type loading heatmaps for the top 10 directions;
* sensor-level and feature-type leverage summaries using the full 128 x k
  loading matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import compute_rolling_split_explained_variance as compute  # noqa: E402
from anchorpca.reproducibility import repo_relative_path, software_versions  # noqa: E402


DEFAULT_K = 20
DEFAULT_LAST_SOURCE_BATCH = 6
DEFAULT_TOP_DIRECTIONS = 10
PLOT_STEM = "gas_sensor_b1_b6_k20_loading_diagnostics"

FEATURE_TYPES = (
    "DR",
    "|DR|",
    "EMAi0.001",
    "EMAi0.01",
    "EMAi0.1",
    "EMAd0.001",
    "EMAd0.01",
    "EMAd0.1",
)
N_SENSORS = 16

METHOD_ORDER = ("poolPCA", "AnchorPCA_infty")
METHOD_LABELS = {
    "poolPCA": "poolPCA",
    "AnchorPCA_infty": r"AnchorPCA$_\infty$",
}
METHOD_COLORS = {
    "poolPCA": "tab:blue",
    "AnchorPCA_infty": "#D62728",
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
            "legend.fontsize": 7,
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


def parse_block_tol(value: str) -> str | float:
    if value == "auto":
        return "auto"
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block_tol must be 'auto' or a nonnegative number.") from exc
    if not np.isfinite(numeric) or numeric < 0:
        raise argparse.ArgumentTypeError("block_tol must be 'auto' or a nonnegative number.")
    return numeric


def feature_index_table() -> pd.DataFrame:
    """Return the deterministic 128-feature layout."""
    records = []
    for sensor in range(1, N_SENSORS + 1):
        for feature_type_index, feature_type in enumerate(FEATURE_TYPES):
            feature_index = (sensor - 1) * len(FEATURE_TYPES) + feature_type_index
            records.append(
                {
                    "feature_index_zero_based": feature_index,
                    "feature_index_one_based": feature_index + 1,
                    "sensor": sensor,
                    "feature_type_index": feature_type_index,
                    "feature_type": feature_type,
                }
            )
    table = pd.DataFrame.from_records(records)
    if len(table) != compute.N_FEATURES:
        raise RuntimeError(f"Expected {compute.N_FEATURES} feature rows, got {len(table)}.")
    return table


def fit_pool_and_anchor(
    dataset: compute.Dataset,
    *,
    last_source_batch: int,
    k: int,
    scale_mode: str,
    block_tol: str | float,
) -> tuple[compute.SplitData, dict[str, object]]:
    """Fit poolPCA and AnchorPCA_infty using source batches only."""
    split_data = compute.prepare_split_data(
        dataset.X_raw,
        dataset.metadata,
        last_source_batch=last_source_batch,
        scale_mode=scale_mode,
    )
    min_source_n = min(split_data.batch_stats[batch].n_obs for batch in split_data.source_batches)
    max_rank = min(compute.N_FEATURES, min_source_n - 1)
    if int(k) > max_rank:
        raise ValueError(f"k={k} exceeds source rank budget {max_rank}.")

    pool = compute.pool_pca_from_covariances(split_data.source_covariances, n_components=int(k))
    anchor = compute.AnchorPCAInfty(n_components=int(k), block_tol=block_tol).fit_covariances(
        split_data.source_covariances,
        n_obs=split_data.source_n_obs,
    )
    fitted = {
        "poolPCA": {
            "directions": np.asarray(pool["directions"], dtype=float),
            "details": {},
        },
        "AnchorPCA_infty": {
            "directions": np.asarray(anchor.directions_, dtype=float),
            "details": {
                "block_tol": float(anchor.block_tol_),
                "block_tol_mode": anchor.block_tol_mode_,
                "block_tol_alpha": float(anchor.block_tol_alpha_),
                "block_tol_c": float(anchor.block_tol_c_),
                "block_tol_max": float(anchor.block_tol_max_),
                "invariant_dim_estimate": int(anchor.invariant_dim_estimate_),
                "invariant_n_selected": int(anchor.agreement_blocks_[0]["n_selected"]),
                "agreement_blocks": anchor.agreement_blocks_,
                "barPi_eigenvalues": np.asarray(anchor.barPi_eigenvalues_, dtype=float),
            },
        },
    }
    return split_data, fitted


def validate_directions(directions: np.ndarray, *, k: int, method_id: str) -> None:
    directions = np.asarray(directions, dtype=float)
    if directions.shape != (compute.N_FEATURES, int(k)):
        raise ValueError(
            f"{method_id} directions have shape {directions.shape}; "
            f"expected {(compute.N_FEATURES, int(k))}."
        )
    gram = directions.T @ directions
    if not np.allclose(gram, np.eye(int(k)), atol=1e-5):
        raise ValueError(f"{method_id} directions are not orthonormal.")


def direction_feature_type_leverage(
    directions: np.ndarray,
    *,
    method_id: str,
    top_directions: int,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compute feature-type loading energy for each plotted direction.

    For method matrix V and direction ell, this returns
    100 * sum_{sensor i} V[(i, feature_type), ell]^2. Each direction sums to
    100 over the eight feature types because each loading vector has unit norm.
    """
    directions = np.asarray(directions, dtype=float)
    top = min(int(top_directions), directions.shape[1])
    records = []
    for ell in range(top):
        squared = directions[:, ell] ** 2
        total = float(np.sum(squared))
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"{method_id} direction {ell + 1} has squared norm {total}.")
        frame = feature_table.copy()
        frame["squared_loading"] = squared
        grouped = (
            frame.groupby(["feature_type_index", "feature_type"], sort=True)["squared_loading"]
            .sum()
            .reset_index()
        )
        for row in grouped.itertuples(index=False):
            records.append(
                {
                    "method_id": method_id,
                    "direction": ell + 1,
                    "feature_type_index": int(row.feature_type_index),
                    "feature_type": row.feature_type,
                    "leverage_percent": 100.0 * float(row.squared_loading),
                }
            )
    result = pd.DataFrame.from_records(records)
    expected_n = top * len(FEATURE_TYPES)
    if len(result) != expected_n:
        raise RuntimeError(f"Expected {expected_n} heatmap rows, got {len(result)}.")
    return result


def subspace_sensor_importance(
    directions: np.ndarray,
    *,
    method_id: str,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate full-subspace leverage by sensor.

    For an orthonormal p x k loading matrix V, row leverage is
    h_j = sum_{ell=1}^k V[j, ell]^2. Sensor importance is the sum of h_j over
    the eight features belonging to that sensor, normalized by k.
    """
    directions = np.asarray(directions, dtype=float)
    k = directions.shape[1]
    frame = feature_table.copy()
    frame["row_leverage"] = np.sum(directions**2, axis=1)
    grouped = frame.groupby("sensor", sort=True)["row_leverage"].sum().reset_index()
    grouped["method_id"] = method_id
    grouped["importance_percent"] = 100.0 * grouped["row_leverage"] / float(k)
    if not np.isclose(grouped["importance_percent"].sum(), 100.0, atol=1e-6):
        raise RuntimeError(f"Sensor importance for {method_id} does not sum to 100%.")
    return grouped[["method_id", "sensor", "row_leverage", "importance_percent"]]


def subspace_feature_type_importance(
    directions: np.ndarray,
    *,
    method_id: str,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate full-subspace leverage by feature type."""
    directions = np.asarray(directions, dtype=float)
    k = directions.shape[1]
    frame = feature_table.copy()
    frame["row_leverage"] = np.sum(directions**2, axis=1)
    grouped = (
        frame.groupby(["feature_type_index", "feature_type"], sort=True)["row_leverage"]
        .sum()
        .reset_index()
    )
    grouped["method_id"] = method_id
    grouped["importance_percent"] = 100.0 * grouped["row_leverage"] / float(k)
    if not np.isclose(grouped["importance_percent"].sum(), 100.0, atol=1e-6):
        raise RuntimeError(f"Feature-type importance for {method_id} does not sum to 100%.")
    return grouped[
        ["method_id", "feature_type_index", "feature_type", "row_leverage", "importance_percent"]
    ]


def plot_top_direction_feature_heatmaps(
    heatmap_rows: pd.DataFrame,
    figures_dir: Path,
    *,
    stem: str,
    m_hat: int,
) -> None:
    configure_publication_style()
    methods = list(METHOD_ORDER)
    directions = sorted(heatmap_rows["direction"].unique())
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(7.0, 2.8),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    vmax = float(heatmap_rows["leverage_percent"].max())
    for ax, method_id in zip(axes, methods):
        sub = heatmap_rows[heatmap_rows["method_id"] == method_id]
        matrix = (
            sub.pivot(index="feature_type_index", columns="direction", values="leverage_percent")
            .loc[list(range(len(FEATURE_TYPES))), directions]
            .to_numpy()
        )
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(METHOD_LABELS[method_id])
        ax.set_xticks(range(len(directions)))
        ax.set_xticklabels([str(direction) for direction in directions])
        ax.set_xlabel("Direction")
        ax.set_yticks(range(len(FEATURE_TYPES)))
        ax.set_yticklabels(FEATURE_TYPES)
        if method_id == "AnchorPCA_infty":
            ax.add_patch(
                Rectangle(
                    (-0.5, -0.5),
                    float(m_hat),
                    float(len(FEATURE_TYPES)),
                    fill=False,
                    edgecolor="white",
                    linewidth=3.0,
                    zorder=4,
                )
            )
            ax.add_patch(
                Rectangle(
                    (-0.5, -0.5),
                    float(m_hat),
                    float(len(FEATURE_TYPES)),
                    fill=False,
                    edgecolor="black",
                    linewidth=1.4,
                    zorder=5,
                )
            )

    axes[0].set_ylabel("Feature type")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
    colorbar.set_label("% of one direction's loading energy")
    block_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.4)
    fig.legend(
        [block_handle],
        [rf"AnchorPCA$_\infty$: outlined first $\hat{{m}}={int(m_hat)}$ directions are estimated $S_\star$"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        frameon=False,
        fontsize=7,
    )
    save_figure(fig, figures_dir, stem)
    plt.close(fig)


def add_difference_column(
    long_rows: pd.DataFrame,
    *,
    index_column: str,
) -> pd.DataFrame:
    pivot = long_rows.pivot(index=index_column, columns="method_id", values="importance_percent")
    required = set(METHOD_ORDER)
    missing = required.difference(pivot.columns)
    if missing:
        raise ValueError(f"Importance table is missing methods: {sorted(missing)}")
    diff = pivot["AnchorPCA_infty"] - pivot["poolPCA"]
    return diff.reset_index(name="anchor_minus_pool_pp")


def plot_importance_summaries(
    sensor_rows: pd.DataFrame,
    feature_type_rows: pd.DataFrame,
    figures_dir: Path,
    *,
    stem: str,
) -> None:
    configure_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1))
    sensor_diff = add_difference_column(sensor_rows, index_column="sensor")
    feature_diff = add_difference_column(feature_type_rows, index_column="feature_type")

    sensors = sorted(sensor_rows["sensor"].unique())
    x_sensor = np.arange(len(sensors))
    width = 0.38
    for offset, method_id in [(-width / 2, "poolPCA"), (width / 2, "AnchorPCA_infty")]:
        sub = sensor_rows[sensor_rows["method_id"] == method_id].sort_values("sensor")
        axes[0, 0].bar(
            x_sensor + offset,
            sub["importance_percent"],
            width=width,
            label=METHOD_LABELS[method_id],
            color=METHOD_COLORS[method_id],
            alpha=0.9,
        )
    axes[0, 0].set_xticks(x_sensor)
    axes[0, 0].set_xticklabels([str(sensor) for sensor in sensors])
    axes[0, 0].set_xlabel("Sensor")
    axes[0, 0].set_ylabel("% of subspace leverage")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(
        x_sensor,
        sensor_diff.sort_values("sensor")["anchor_minus_pool_pp"],
        color="0.25",
    )
    axes[0, 1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0, 1].set_xticks(x_sensor)
    axes[0, 1].set_xticklabels([str(sensor) for sensor in sensors])
    axes[0, 1].set_xlabel("Sensor")
    axes[0, 1].set_ylabel("Anchor - pool (pp)")

    feature_order = list(FEATURE_TYPES)
    x_feature = np.arange(len(feature_order))
    for offset, method_id in [(-width / 2, "poolPCA"), (width / 2, "AnchorPCA_infty")]:
        sub = (
            feature_type_rows[feature_type_rows["method_id"] == method_id]
            .set_index("feature_type")
            .loc[feature_order]
            .reset_index()
        )
        axes[1, 0].bar(
            x_feature + offset,
            sub["importance_percent"],
            width=width,
            label=METHOD_LABELS[method_id],
            color=METHOD_COLORS[method_id],
            alpha=0.9,
        )
    axes[1, 0].set_xticks(x_feature)
    axes[1, 0].set_xticklabels(feature_order, rotation=30, ha="right")
    axes[1, 0].set_xlabel("Feature type")
    axes[1, 0].set_ylabel("% of subspace leverage")

    feature_diff = feature_diff.set_index("feature_type").loc[feature_order].reset_index()
    axes[1, 1].bar(x_feature, feature_diff["anchor_minus_pool_pp"], color="0.25")
    axes[1, 1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 1].set_xticks(x_feature)
    axes[1, 1].set_xticklabels(feature_order, rotation=30, ha="right")
    axes[1, 1].set_xlabel("Feature type")
    axes[1, 1].set_ylabel("Anchor - pool (pp)")

    for ax in axes.ravel():
        ax.grid(False)
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    save_figure(fig, figures_dir, stem)
    plt.close(fig)


def build_diagnostic_tables(
    fitted: dict[str, object],
    *,
    k: int,
    top_directions: int,
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    heatmap_frames = []
    sensor_frames = []
    feature_type_frames = []
    for method_id in METHOD_ORDER:
        directions = fitted[method_id]["directions"]
        validate_directions(directions, k=k, method_id=method_id)
        heatmap_frames.append(
            direction_feature_type_leverage(
                directions,
                method_id=method_id,
                top_directions=top_directions,
                feature_table=feature_table,
            )
        )
        sensor_frames.append(
            subspace_sensor_importance(
                directions,
                method_id=method_id,
                feature_table=feature_table,
            )
        )
        feature_type_frames.append(
            subspace_feature_type_importance(
                directions,
                method_id=method_id,
                feature_table=feature_table,
            )
        )

    return (
        pd.concat(heatmap_frames, ignore_index=True),
        pd.concat(sensor_frames, ignore_index=True),
        pd.concat(feature_type_frames, ignore_index=True),
    )


def run(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else experiment_dir / "data"
    results_dir = args.results_dir.resolve() if args.results_dir else experiment_dir / "results"
    figures_dir = args.figures_dir.resolve() if args.figures_dir else experiment_dir / "figures"
    k = int(args.k)
    last_source_batch = int(args.last_source_batch)
    top_directions = int(args.top_directions)

    print("Gas-sensor loading diagnostics")
    print("=" * 72)
    print(f"Split: source B1-B{last_source_batch}; targets not used")
    print(f"k={k}, top directions in heatmap={top_directions}")

    dataset = compute.load_dataset(
        data_dir,
        force_download=bool(args.force_download),
        skip_sha256_check=bool(args.skip_sha256_check),
    )
    split_data, fitted = fit_pool_and_anchor(
        dataset,
        last_source_batch=last_source_batch,
        k=k,
        scale_mode=args.scale_mode,
        block_tol=args.block_tol,
    )
    feature_table = feature_index_table()
    heatmap_rows, sensor_rows, feature_type_rows = build_diagnostic_tables(
        fitted,
        k=k,
        top_directions=top_directions,
        feature_table=feature_table,
    )

    anchor_details = fitted["AnchorPCA_infty"]["details"]
    m_hat = int(anchor_details["invariant_n_selected"])
    if m_hat > top_directions:
        raise ValueError(
            f"Estimated Sstar dimension m_hat={m_hat} exceeds top_directions={top_directions}."
        )

    split_label = f"b1_b{last_source_batch}_k{k}"
    heatmap_stem = f"{args.plot_stem}_{split_label}_top{top_directions}_feature_heatmap"
    importance_stem = f"{args.plot_stem}_{split_label}_importance_summary"
    plot_top_direction_feature_heatmaps(
        heatmap_rows,
        figures_dir,
        stem=heatmap_stem,
        m_hat=m_hat,
    )
    plot_importance_summaries(
        sensor_rows,
        feature_type_rows,
        figures_dir,
        stem=importance_stem,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = results_dir / f"{heatmap_stem}.csv"
    sensor_path = results_dir / f"{importance_stem}_sensor.csv"
    feature_type_path = results_dir / f"{importance_stem}_feature_type.csv"
    metadata_path = results_dir / f"{args.plot_stem}_{split_label}_metadata.json"
    heatmap_rows.to_csv(heatmap_path, index=False)
    sensor_rows.to_csv(sensor_path, index=False)
    feature_type_rows.to_csv(feature_type_path, index=False)
    with metadata_path.open("w") as handle:
        json.dump(
            {
                "software_versions": software_versions(),
                "last_source_batch": last_source_batch,
                "source_batches": list(split_data.source_batches),
                "target_batches_used": [],
                "k": k,
                "top_directions": top_directions,
                "scale_mode": args.scale_mode,
                "block_tol_requested": args.block_tol,
                "anchor_infty": anchor_details,
                "feature_types": list(FEATURE_TYPES),
                "notes": [
                    "All loadings are fit using source batches only.",
                    "Row leverage h_j = sum_l V[j,l]^2 is invariant to rotations inside the fitted subspace.",
                    "Sensor and feature-type importance are normalized by k and sum to 100 percent for each method.",
                ],
            },
            handle,
            indent=2,
            default=compute.to_jsonable,
        )

    print(f"AnchorPCA_infty estimated m_hat={m_hat}")
    print(f"Wrote heatmap figure: {figures_dir / (heatmap_stem + '.pdf')}")
    print(f"Wrote importance figure: {figures_dir / (importance_stem + '.pdf')}")
    print(f"Wrote CSVs under: {results_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Gas-sensor experiment directory.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--last-source-batch", type=int, default=DEFAULT_LAST_SOURCE_BATCH)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--top-directions", type=int, default=DEFAULT_TOP_DIRECTIONS)
    parser.add_argument(
        "--scale-mode",
        choices=["source-standard", "none"],
        default="source-standard",
        help="Preprocessing mode. Use source-standard for publication figures.",
    )
    parser.add_argument(
        "--block-tol",
        type=parse_block_tol,
        default="auto",
        help="AnchorPCA_infty block tolerance. Default uses the package auto rule.",
    )
    parser.add_argument(
        "--plot-stem",
        default=PLOT_STEM,
        help="Filename prefix for generated figures and CSVs.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-sha256-check", action="store_true")
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

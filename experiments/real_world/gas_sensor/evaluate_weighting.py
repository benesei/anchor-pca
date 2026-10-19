"""Compare uniform-domain and observation-weighted gas-sensor fits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "src" / "anchorpca").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anchorpca import fit_two_e_lambda_from_covariances  # noqa: E402
from anchorpca.reproducibility import software_versions  # noqa: E402
from analysis_utils import (  # noqa: E402
    load_dataset,
    total_ev_fraction,
    write_frame,
    write_protocol,
)
import compute_rolling_split_explained_variance as gas  # noqa: E402


DEFAULT_S_VALUES = tuple(range(3, 9))
DEFAULT_K_VALUES = (10, 20, 30)


def run(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.data_dir, force_download=args.force_download)
    rows = []
    for last_source in args.s_values:
        split = gas.prepare_split_data(
            dataset.X_raw,
            dataset.metadata,
            last_source_batch=last_source,
            scale_mode="source-standard",
        )
        for rank in args.k_values:
            for weighting in ("uniform", "observations"):
                for method_id, coefficient in (("poolPCA", 0.0), ("Anchor_PCA_infinity", np.inf)):
                    model = fit_two_e_lambda_from_covariances(
                        split.source_covariances,
                        rank,
                        coefficient,
                        weighting=weighting,
                        n_obs=split.source_n_obs,
                    )
                    for batch in split.target_batches:
                        rows.append(
                            {
                                "last_source_batch": last_source,
                                "k": rank,
                                "weighting": weighting,
                                "method_id": method_id,
                                "target_batch": batch,
                                "target_ev_percent_total": 100.0
                                * total_ev_fraction(
                                    model.projection_, split.batch_stats[batch].covariance
                                ),
                            }
                        )
    frame = pd.DataFrame(rows)
    by_split = (
        frame.groupby(["last_source_batch", "k", "weighting", "method_id"], sort=False)
        .agg(
            mean_target_ev_percent_total=("target_ev_percent_total", "mean"),
            minimum_target_ev_percent_total=("target_ev_percent_total", "min"),
            n_target_batches=("target_batch", "nunique"),
        )
        .reset_index()
    )
    wide = by_split.pivot(
        index=["last_source_batch", "k", "weighting"],
        columns="method_id",
        values="mean_target_ev_percent_total",
    ).reset_index()
    wide["infinity_minus_poolPCA_pp"] = (
        wide["Anchor_PCA_infinity"] - wide["poolPCA"]
    )
    summary = (
        wide.groupby(["k", "weighting"], sort=False)
        .agg(
            mean_infinity_minus_poolPCA_pp=("infinity_minus_poolPCA_pp", "mean"),
            minimum_infinity_minus_poolPCA_pp=("infinity_minus_poolPCA_pp", "min"),
            splits_infinity_above_poolPCA=(
                "infinity_minus_poolPCA_pp",
                lambda values: int((values > 0).sum()),
            ),
            n_splits=("last_source_batch", "nunique"),
        )
        .reset_index()
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_frame(
        frame,
        output / "weighting_target_ev_by_batch.csv",
        ("last_source_batch", "k", "weighting", "method_id", "target_batch"),
    )
    write_frame(
        wide,
        output / "weighting_by_split.csv",
        ("last_source_batch", "k", "weighting"),
    )
    write_frame(summary, output / "weighting_summary.csv", ("k", "weighting"))
    write_protocol(
        output / "protocol.json",
        {
            "dataset": "UCI Gas Sensor Array Drift Dataset",
            "archive_sha256": gas.EXPECTED_SHA256,
            "last_source_batches": list(args.s_values),
            "ranks": list(args.k_values),
            "preprocessing": "standardized using all source observations in each split",
            "weighting_schemes": ["uniform", "observations"],
            "target_aggregation": "equal target-batch mean within each split",
            "target_used_for_fitting": False,
            "software_versions": software_versions(),
        },
    )
    print(f"Wrote gas weighting tables to {output}")


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=directory / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=directory / "results" / "weighting",
    )
    parser.add_argument("--s-values", type=int, nargs="+", default=DEFAULT_S_VALUES)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""Select the gas-sensor Anchor PCA penalty using source batches only."""

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

from anchorpca import (  # noqa: E402
    DEFAULT_TWO_E_LAMBDA_GRID,
    fit_two_e_lambda_from_covariances,
)
from anchorpca.reproducibility import software_versions  # noqa: E402
from analysis_utils import (  # noqa: E402
    load_dataset,
    source_validation_folds,
    standard_error,
    total_ev_fraction,
    write_frame,
    write_protocol,
)
import compute_rolling_split_explained_variance as gas  # noqa: E402


DEFAULT_S_VALUES = tuple(range(3, 9))
DEFAULT_K_VALUES = (10, 20, 30)


def run(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.data_dir, force_download=args.force_download)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    candidates = DEFAULT_TWO_E_LAMBDA_GRID
    fold_rows = []
    candidate_rows = []
    target_rows = []
    all_ev_rows = []
    setting_rows = []

    for last_source in args.s_values:
        source_batches = tuple(range(1, last_source + 1))
        folds = source_validation_folds(dataset, source_batches)
        split = gas.prepare_split_data(
            dataset.X_raw,
            dataset.metadata,
            last_source_batch=last_source,
            scale_mode="source-standard",
        )
        for rank in args.k_values:
            scores = np.empty((len(candidates), len(folds)), dtype=float)
            for fold_index, fold in enumerate(folds):
                for candidate_index, coefficient in enumerate(candidates):
                    model = fit_two_e_lambda_from_covariances(
                        fold.training_covariances,
                        rank,
                        coefficient,
                        n_obs=fold.training_n_obs,
                        weighting="uniform",
                    )
                    score = total_ev_fraction(model.projection_, fold.heldout_covariance)
                    scores[candidate_index, fold_index] = score
                    fold_rows.append(
                        {
                            "last_source_batch": last_source,
                            "k": rank,
                            "candidate_order": candidate_index,
                            "two_e_lambda": coefficient,
                            "heldout_source_batch": fold.heldout_batch,
                            "training_source_batches": ",".join(
                                str(value) for value in fold.training_batches
                            ),
                            "raw_lambda_fold": (
                                np.inf if np.isinf(coefficient) else model.lambda_
                            ),
                            "heldout_total_ev_fraction": score,
                        }
                    )
            means = scores.mean(axis=1)
            selected_index = int(np.argmax(means))
            selected = candidates[selected_index]
            for candidate_index, coefficient in enumerate(candidates):
                candidate_rows.append(
                    {
                        "last_source_batch": last_source,
                        "k": rank,
                        "candidate_order": candidate_index,
                        "two_e_lambda": coefficient,
                        "mean_total_ev_fraction": means[candidate_index],
                        "standard_error_total_ev_fraction": standard_error(
                            scores[candidate_index]
                        ),
                        "selected": int(candidate_index == selected_index),
                    }
                )

            methods = {
                "poolPCA": 0.0,
                "source_selected_Anchor_PCA": selected,
                "Anchor_PCA_infinity": np.inf,
            }
            target_means = {}
            for method_id, coefficient in methods.items():
                model = fit_two_e_lambda_from_covariances(
                    split.source_covariances,
                    rank,
                    coefficient,
                    n_obs=split.source_n_obs,
                    weighting="uniform",
                )
                values = []
                for batch in (*split.source_batches, *split.target_batches):
                    value = 100.0 * total_ev_fraction(
                        model.projection_, split.batch_stats[batch].covariance
                    )
                    all_ev_rows.append(
                        {
                            "last_source_batch": last_source,
                            "k": rank,
                            "method_id": method_id,
                            "selected_two_e_lambda": selected,
                            "method_two_e_lambda": coefficient,
                            "split": "source" if batch in split.source_batches else "target",
                            "batch": batch,
                            "percent_explained_variance": value,
                        }
                    )
                    if batch in split.source_batches:
                        continue
                    values.append(value)
                    target_rows.append(
                        {
                            "last_source_batch": last_source,
                            "k": rank,
                            "method_id": method_id,
                            "selected_two_e_lambda": selected,
                            "method_two_e_lambda": coefficient,
                            "target_batch": batch,
                            "target_ev_percent_total": value,
                        }
                    )
                target_means[method_id] = float(np.mean(values))
            setting_rows.append(
                {
                    "last_source_batch": last_source,
                    "k": rank,
                    "selected_two_e_lambda": selected,
                    "selected_type": (
                        "infinity"
                        if np.isinf(selected)
                        else ("poolPCA" if selected == 0 else "positive_finite")
                    ),
                    "mean_target_ev_poolPCA": target_means["poolPCA"],
                    "mean_target_ev_source_selected": target_means[
                        "source_selected_Anchor_PCA"
                    ],
                    "mean_target_ev_infinity": target_means["Anchor_PCA_infinity"],
                    "selected_minus_poolPCA_pp": (
                        target_means["source_selected_Anchor_PCA"]
                        - target_means["poolPCA"]
                    ),
                    "absolute_selected_minus_infinity_pp": abs(
                        target_means["source_selected_Anchor_PCA"]
                        - target_means["Anchor_PCA_infinity"]
                    ),
                }
            )

    write_frame(
        pd.DataFrame(fold_rows),
        output / "penalty_validation_folds.csv",
        ("last_source_batch", "k", "candidate_order", "heldout_source_batch"),
    )
    write_frame(
        pd.DataFrame(candidate_rows),
        output / "penalty_validation_summary.csv",
        ("last_source_batch", "k", "candidate_order"),
    )
    write_frame(
        pd.DataFrame(target_rows),
        output / "selected_target_ev_by_batch.csv",
        ("last_source_batch", "k", "method_id", "target_batch"),
    )
    write_frame(
        pd.DataFrame(all_ev_rows),
        output / "selected_ev_by_batch.csv",
        ("last_source_batch", "k", "method_id", "batch"),
    )
    write_frame(
        pd.DataFrame(setting_rows),
        output / "selection_summary.csv",
        ("last_source_batch", "k"),
    )
    write_protocol(
        output / "protocol.json",
        {
            "dataset": "UCI Gas Sensor Array Drift Dataset",
            "archive_sha256": gas.EXPECTED_SHA256,
            "source_standardization": "refitted in every validation fold",
            "source_weighting": "uniform domain",
            "last_source_batches": list(args.s_values),
            "ranks": list(args.k_values),
            "two_e_lambda_grid": [
                "infinity" if np.isinf(value) else value for value in candidates
            ],
            "validation_score": "held-out fraction of total explained variance",
            "target_used_for_selection": False,
            "software_versions": software_versions(),
        },
    )
    print(f"Wrote gas penalty-selection tables to {output}")


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=directory / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=directory / "results" / "penalty_selection",
    )
    parser.add_argument("--s-values", type=int, nargs="+", default=DEFAULT_S_VALUES)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

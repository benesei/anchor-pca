"""Run the Fama--French temporal explained-variance experiment."""

from __future__ import annotations

import argparse
import json
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
    select_two_e_lambda_from_covariances,
)
from anchorpca.reproducibility import software_versions  # noqa: E402
from data import (  # noqa: E402
    DATA_URL,
    archive_provenance,
    load_periods,
    period_data_hash,
    resolve_archive,
)


DEFAULT_RANKS = tuple(range(2, 11))
PRIMARY_RANK = 2
MINPCA_SEED = 4159490659


def covariance(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - np.mean(values, axis=0)
    return (centered.T @ centered) / float(centered.shape[0] - 1)


def total_ev_percent(projection: np.ndarray, covariance_matrix: np.ndarray) -> float:
    return 100.0 * float(np.trace(projection @ covariance_matrix)) / float(
        np.trace(covariance_matrix)
    )


def fit_norm_maxregret(
    covariances: list[np.ndarray],
    rank: int,
    *,
    restarts: int,
    iterations: int,
) -> np.ndarray:
    try:
        import torch
        from minPCA.minpca import minPCA
    except ImportError as exc:
        raise RuntimeError("The norm-maxRegret baseline requires minPCA==0.1.0.") from exc
    np.random.seed(MINPCA_SEED)
    torch.manual_seed(MINPCA_SEED)
    model = minPCA(n_components=rank, function="maxregret", norm=True)
    model.fit(
        [np.asarray(item, dtype=np.float32) for item in covariances],
        n_restarts=restarts,
        n_iters=iterations,
        lr=0.01,
        verbose=False,
    )
    directions, _ = np.linalg.qr(model.v_.detach().cpu().numpy())
    directions = np.asarray(directions[:, :rank], dtype=float)
    return directions @ directions.T


def run(args: argparse.Namespace) -> None:
    archive = resolve_archive(args.archive, args.data_dir)
    periods = load_periods(archive)
    sources = [period for period in periods if period.role == "source"]
    targets = [period for period in periods if period.role == "target"]
    source_covariances = [covariance(period.values) for period in sources]
    source_counts = np.asarray([period.values.shape[0] for period in sources], dtype=float)
    target_covariances = {period.period_id: covariance(period.values) for period in targets}
    ranks = tuple(int(rank) for rank in args.ranks)
    if not ranks or any(rank < 1 or rank > 49 for rank in ranks):
        raise ValueError("ranks must contain integers between 1 and 49.")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "period_id": period.period_id,
                "role": period.role,
                "start_year": period.start_year,
                "end_year": period.end_year,
                "n_observations": period.values.shape[0],
                "raw_rows": period.raw_rows,
                "incomplete_rows_removed": period.incomplete_rows_removed,
            }
            for period in periods
        ]
    ).to_csv(output / "period_counts.csv", index=False)

    scree_rows = []
    for period, sigma in zip(sources, source_covariances):
        eigenvalues = np.linalg.eigvalsh(sigma)[::-1]
        for component, eigenvalue in enumerate(eigenvalues, start=1):
            scree_rows.append(
                {
                    "source_period": period.period_id,
                    "start_year": period.start_year,
                    "end_year": period.end_year,
                    "component": component,
                    "eigenvalue": float(eigenvalue),
                    "explained_variance_fraction": float(eigenvalue / eigenvalues.sum()),
                }
            )
    pd.DataFrame(scree_rows).to_csv(output / "source_scree.csv", index=False)

    fold_rows = []
    validation_rows = []
    target_rows = []
    for rank in ranks:
        selection = select_two_e_lambda_from_covariances(
            source_covariances,
            rank,
            n_obs=source_counts,
            weighting="uniform",
        )
        for candidate_index, candidate in enumerate(selection.candidates):
            for fold_index, source in enumerate(sources):
                fold_rows.append(
                    {
                        "k": rank,
                        "candidate_order": candidate_index,
                        "two_e_lambda": candidate,
                        "heldout_source_period": source.period_id,
                        "heldout_start_year": source.start_year,
                        "heldout_end_year": source.end_year,
                        "heldout_total_ev_fraction": selection.fold_scores[
                            candidate_index, fold_index
                        ],
                    }
                )
            validation_rows.append(
                {
                    "k": rank,
                    "candidate_order": candidate_index,
                    "two_e_lambda": candidate,
                    "mean_total_ev_fraction": selection.mean_scores[candidate_index],
                    "standard_error_total_ev_fraction": selection.standard_errors[
                        candidate_index
                    ],
                    "selected": int(candidate_index == selection.selected_index),
                }
            )

        methods = {
            "poolPCA": fit_two_e_lambda_from_covariances(
                source_covariances, rank, 0.0, n_obs=source_counts
            ).projection_,
            "source_selected_Anchor_PCA": selection.estimator.projection_,
        }
        if rank == PRIMARY_RANK:
            methods["norm-maxRegret"] = fit_norm_maxregret(
                source_covariances,
                rank,
                restarts=args.minpca_restarts,
                iterations=args.minpca_iterations,
            )
        for method_id, projection in methods.items():
            coefficient = (
                selection.selected_two_e_lambda
                if method_id == "source_selected_Anchor_PCA"
                else (0.0 if method_id == "poolPCA" else np.nan)
            )
            for target in targets:
                target_rows.append(
                    {
                        "k": rank,
                        "method_id": method_id,
                        "two_e_lambda": coefficient,
                        "target_period": target.period_id,
                        "start_year": target.start_year,
                        "end_year": target.end_year,
                        "target_ev_percent_total": total_ev_percent(
                            projection, target_covariances[target.period_id]
                        ),
                    }
                )

    folds = pd.DataFrame(fold_rows)
    validation = pd.DataFrame(validation_rows)
    targets_frame = pd.DataFrame(target_rows)
    folds.to_csv(output / "penalty_validation_folds.csv", index=False)
    validation.to_csv(output / "penalty_validation_summary.csv", index=False)
    targets_frame.to_csv(output / "target_ev_by_period.csv", index=False)

    pool = targets_frame[targets_frame.method_id == "poolPCA"][
        ["k", "target_period", "target_ev_percent_total"]
    ].rename(columns={"target_ev_percent_total": "poolPCA_target_ev_percent_total"})
    comparison = targets_frame.merge(pool, on=["k", "target_period"], validate="many_to_one")
    comparison["difference_vs_poolPCA_pp"] = (
        comparison.target_ev_percent_total - comparison.poolPCA_target_ev_percent_total
    )
    summary = (
        comparison.groupby(["k", "method_id"], sort=False, dropna=False)
        .agg(
            two_e_lambda=("two_e_lambda", "first"),
            mean_target_ev_percent_total=("target_ev_percent_total", "mean"),
            minimum_target_ev_percent_total=("target_ev_percent_total", "min"),
            maximum_target_ev_percent_total=("target_ev_percent_total", "max"),
            mean_difference_vs_poolPCA_pp=("difference_vs_poolPCA_pp", "mean"),
            minimum_difference_vs_poolPCA_pp=("difference_vs_poolPCA_pp", "min"),
            maximum_difference_vs_poolPCA_pp=("difference_vs_poolPCA_pp", "max"),
            wins_vs_poolPCA=(
                "difference_vs_poolPCA_pp",
                lambda values: int((values > 1e-10).sum()),
            ),
            n_target_periods=("target_period", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(output / "target_ev_summary.csv", index=False)

    protocol = {
        "dataset": "Fama--French 49 Industry Portfolios, value-weighted daily returns",
        "data_url": DATA_URL,
        **archive_provenance(archive),
        "used_periods_sha256": period_data_hash(periods),
        "source_periods": [[period.start_year, period.end_year] for period in sources],
        "target_periods": [[period.start_year, period.end_year] for period in targets],
        "centering": "within period",
        "source_weighting": "uniform domain",
        "ranks": list(ranks),
        "primary_rank": PRIMARY_RANK,
        "two_e_lambda_grid": [
            "infinity" if np.isinf(value) else value
            for value in DEFAULT_TWO_E_LAMBDA_GRID
        ],
        "target_used_for_selection": False,
        "norm_maxRegret": {
            "rank": PRIMARY_RANK,
            "restarts": args.minpca_restarts,
            "iterations": args.minpca_iterations,
            "learning_rate": 0.01,
            "seed": MINPCA_SEED,
        },
        "software_versions": software_versions(
            ("anchorpca", "numpy", "scipy", "pandas", "torch", "minPCA")
        ),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote Fama--French result tables to {output}")


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--data-dir", type=Path, default=directory / "data")
    parser.add_argument("--output-dir", type=Path, default=directory / "results")
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--minpca-restarts", type=int, default=10)
    parser.add_argument("--minpca-iterations", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

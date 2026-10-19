"""Evaluate learned subspaces on an independently generated target domain."""

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
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from anchorpca import AnchorPCAInfty, pool_pca_from_covariances  # noqa: E402
from anchorpca.reproducibility import software_versions  # noqa: E402
import generate_design_bank as design_bank  # noqa: E402
import recovery_utils as recovery  # noqa: E402


MASTER_SEED = 20260723
PROFILE_LABELS = {"balanced": "easy", "threshold_stress": "hard"}
P = 10
K = 5
M = 2
N_SOURCE_DOMAINS = 5
MIXTURE_MEAN_SCALE = 0.75


def empirical_covariance(sample: np.ndarray) -> np.ndarray:
    centered = np.asarray(sample, dtype=float) - np.mean(sample, axis=0)
    return (centered.T @ centered) / float(centered.shape[0] - 1)


def projector_check(projection: np.ndarray, rank: int) -> None:
    if not np.allclose(projection, projection.T, atol=1e-8):
        raise RuntimeError("Fitted projection is not symmetric.")
    if not np.allclose(projection @ projection, projection, atol=1e-8):
        raise RuntimeError("Fitted projection is not idempotent.")
    if np.linalg.matrix_rank(projection, tol=1e-7) != rank:
        raise RuntimeError("Fitted projection has the wrong rank.")


def run(args: argparse.Namespace) -> None:
    rows = []
    design_rows = []
    for profile in args.profiles:
        for design_replicate in range(args.n_designs):
            design_seed = design_bank.stable_seed(
                MASTER_SEED, "population", profile, design_replicate
            )
            design = design_bank.build_single_design(
                design_seed=design_seed,
                g=N_SOURCE_DOMAINS + 1,
                p=P,
                k=K,
                m=M,
                profile_name=profile,
                rank_tol=design_bank.DEFAULT_RANK_TOL,
                min_span_singular_value=design_bank.DEFAULT_MIN_SPAN_SINGULAR_VALUE,
                max_attempts=design_bank.DEFAULT_MAX_ATTEMPTS,
            )
            population = np.asarray(design["covariances"], dtype=float)
            source_population = population[:N_SOURCE_DOMAINS]
            target_population = population[N_SOURCE_DOMAINS]
            target_total = float(np.trace(target_population))
            true_projector = np.asarray(design["sstar_projector"], dtype=float)
            source_bar_pi = np.mean(
                np.asarray(design["top_projectors"], dtype=float)[:N_SOURCE_DOMAINS],
                axis=0,
            )
            agreement_values = np.linalg.eigvalsh(source_bar_pi)[::-1]
            pairwise_distances = [
                np.linalg.norm(population[left] - population[right], ord="fro")
                for left in range(N_SOURCE_DOMAINS + 1)
                for right in range(left + 1, N_SOURCE_DOMAINS + 1)
            ]
            design_rows.append(
                {
                    "profile": profile,
                    "regime": PROFILE_LABELS[profile],
                    "design_replicate": design_replicate,
                    "source_intersection_gap": float(1.0 - agreement_values[M]),
                    "minimum_pairwise_covariance_distance_fro": float(
                        np.min(pairwise_distances)
                    ),
                    "shared_subspace_rank": int(
                        np.linalg.matrix_rank(true_projector, tol=1e-7)
                    ),
                }
            )

            for sample_replicate in range(args.sample_repetitions):
                sample_seed = design_bank.stable_seed(
                    MASTER_SEED,
                    "source-sample",
                    profile,
                    design_replicate,
                    sample_replicate,
                )
                rng = np.random.default_rng(sample_seed)
                source_covariances = [
                    empirical_covariance(
                        recovery.sample_gaussian_mixture_same_covariance(
                            rng,
                            covariance,
                            args.n_source,
                            mean_scale=MIXTURE_MEAN_SCALE,
                        )
                    )
                    for covariance in source_population
                ]
                counts = np.full(N_SOURCE_DOMAINS, args.n_source, dtype=float)
                pool = pool_pca_from_covariances(
                    source_covariances,
                    n_components=K,
                    n_obs=counts,
                    weighting="uniform",
                )
                infinity = AnchorPCAInfty(
                    n_components=K,
                    weighting="uniform",
                    block_tol="auto",
                ).fit_covariances(source_covariances, n_obs=counts)
                methods = {
                    "poolPCA": (pool["projection"], pool["directions"]),
                    "Anchor_PCA_infinity": (
                        infinity.projection_,
                        infinity.directions_,
                    ),
                }
                for method_id, (projection, directions) in methods.items():
                    projection = np.asarray(projection, dtype=float)
                    directions = np.asarray(directions, dtype=float)
                    projector_check(projection, K)
                    leading_m = directions[:, :M] @ directions[:, :M].T
                    rows.append(
                        {
                            "profile": profile,
                            "regime": PROFILE_LABELS[profile],
                            "design_replicate": design_replicate,
                            "sample_replicate": sample_replicate,
                            "design_seed": design_seed,
                            "sample_seed": sample_seed,
                            "method_id": method_id,
                            "target_population_ev_percent_total": 100.0
                            * float(np.trace(projection @ target_population))
                            / target_total,
                            "leading_m_projector_error_op": float(
                                np.clip(
                                    np.linalg.norm(leading_m - true_projector, ord=2),
                                    0.0,
                                    1.0,
                                )
                            ),
                        }
                    )

    runs = pd.DataFrame(rows).sort_values(
        ["profile", "design_replicate", "sample_replicate", "method_id"],
        kind="mergesort",
    )
    by_design = (
        runs.groupby(
            ["profile", "regime", "method_id", "design_replicate"], sort=False
        )[["target_population_ev_percent_total", "leading_m_projector_error_op"]]
        .mean()
        .reset_index()
    )
    summary = (
        by_design.groupby(["profile", "regime", "method_id"], sort=False)
        .agg(
            n_population_designs=("design_replicate", "nunique"),
            mean_target_population_ev_percent_total=(
                "target_population_ev_percent_total",
                "mean",
            ),
            mean_leading_m_projector_error_op=("leading_m_projector_error_op", "mean"),
        )
        .reset_index()
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "runs.csv", index=False)
    by_design.to_csv(output / "by_design.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(design_rows).sort_values(
        ["profile", "design_replicate"], kind="mergesort"
    ).to_csv(output / "design_checks.csv", index=False)
    protocol = {
        "profiles": list(args.profiles),
        "regime_labels": PROFILE_LABELS,
        "n_population_designs_per_profile": args.n_designs,
        "sample_repetitions_per_design": args.sample_repetitions,
        "n_source_domains": N_SOURCE_DOMAINS,
        "n_independently_generated_target_domains": 1,
        "n_source_observations_per_domain": args.n_source,
        "p": P,
        "k": K,
        "shared_subspace_dimension_m": M,
        "distribution": "symmetric Gaussian mixture with the generated covariance",
        "mixture_mean_scale": MIXTURE_MEAN_SCALE,
        "methods": ["poolPCA", "Anchor_PCA_infinity"],
        "target_data_used_for_fitting": False,
        "target_evaluation": "population covariance",
        "aggregation": (
            "average source-sample repetitions within each population design, "
            "then average population designs equally"
        ),
        "software_versions": software_versions(),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote unseen-target simulation tables to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "unseen_target_evaluation",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_LABELS),
        default=tuple(PROFILE_LABELS),
    )
    parser.add_argument("--n-designs", type=int, default=50)
    parser.add_argument("--sample-repetitions", type=int, default=5)
    parser.add_argument("--n-source", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

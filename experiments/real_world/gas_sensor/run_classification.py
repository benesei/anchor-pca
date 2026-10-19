"""Run the seven reported gas-type classification pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


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

from anchorpca import (  # noqa: E402
    DEFAULT_TWO_E_LAMBDA_GRID,
    fit_two_e_lambda_from_covariances,
)
from anchorpca.reproducibility import software_versions  # noqa: E402
from analysis_utils import (  # noqa: E402
    load_dataset,
    source_validation_folds,
    standard_error,
    write_frame,
)
from classification_models import (  # noqa: E402
    fit_exact_group_dro,
    fit_logistic,
    fit_vrex,
)
import compute_rolling_split_explained_variance as gas  # noqa: E402


SOURCE_BATCHES = tuple(range(1, 7))
TARGET_BATCHES = tuple(range(7, 11))
RANK = 20
SEED = 20260724
DEFAULT_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3)
DEFAULT_L2_GRID = (1e-4, 1e-3, 1e-2)
DEFAULT_BETA_GRID = (0.0, 0.1, 1.0, 10.0, 100.0, 1e3)
METHOD_ORDER = (
    "poolPCA_l2",
    "source_selected_Anchor_PCA_l2",
    "Anchor_PCA_infinity_l2",
    "unreduced_l2",
    "unreduced_l1",
    "linear_GroupDRO",
    "linear_VREx",
)


def prediction_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    classes = np.unique(labels)
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(
            np.mean([np.mean(predictions[labels == value] == value) for value in classes])
        ),
    }


def mean_candidate_rows(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, sort=False, dropna=False)
        .agg(
            candidate_order=("candidate_order", "first"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            standard_error_balanced_accuracy=("balanced_accuracy", standard_error),
            all_converged=("converged", "min"),
            n_source_folds=("heldout_source_batch", "nunique"),
        )
        .reset_index()
    )


def select_row(summary: pd.DataFrame) -> pd.Series:
    eligible = summary[summary.all_converged == 1].sort_values(
        ["mean_balanced_accuracy", "candidate_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    if eligible.empty:
        raise RuntimeError("No converged source-validation candidate is available.")
    return eligible.iloc[0]


def fit_vrex_candidate(features, labels, domains, beta, l2_penalty, max_iter):
    if beta >= 1000.0:
        continuation = fit_vrex(
            features,
            labels,
            domains,
            beta=100.0,
            l2_penalty=l2_penalty,
            max_iter=max_iter,
        )
        if not continuation.converged:
            raise RuntimeError("The VREx beta=100 continuation fit did not converge.")
        return fit_vrex(
            features,
            labels,
            domains,
            beta=beta,
            l2_penalty=l2_penalty,
            max_iter=max(3000, max_iter),
            initial_parameters=continuation.parameters,
        )
    return fit_vrex(
        features,
        labels,
        domains,
        beta=beta,
        l2_penalty=l2_penalty,
        max_iter=max_iter,
    )


def run(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.data_dir, force_download=args.force_download)
    labels = dataset.metadata["gas_class"].to_numpy(dtype=int)
    batches = dataset.metadata["batch"].to_numpy(dtype=int)
    folds = source_validation_folds(dataset, SOURCE_BATCHES)
    c_grid = tuple(float(value) for value in args.c_values)
    rho_grid = tuple(float(value) for value in args.finite_two_e_lambda) + (np.inf,)
    l2_grid = tuple(float(value) for value in args.dg_l2_values)
    beta_grid = tuple(float(value) for value in args.beta_values)
    logistic_tasks = []
    for fold in folds:
        for rho_order, coefficient in enumerate(rho_grid):
            representation = fit_two_e_lambda_from_covariances(
                fold.training_covariances,
                RANK,
                coefficient,
                n_obs=fold.training_n_obs,
                weighting="uniform",
            )
            train_features = fold.X_standardized[fold.training_mask] @ representation.directions_
            validation_features = (
                fold.X_standardized[fold.validation_mask] @ representation.directions_
            )
            for c_order, c_value in enumerate(c_grid):
                logistic_tasks.append(
                    (
                        "Anchor_PCA_l2",
                        rho_order * len(c_grid) + c_order,
                        fold,
                        coefficient,
                        c_value,
                        "l2",
                        train_features,
                        validation_features,
                    )
                )
        for penalty, family in (("l2", "unreduced_l2"), ("l1", "unreduced_l1")):
            for c_order, c_value in enumerate(c_grid):
                logistic_tasks.append(
                    (
                        family,
                        c_order,
                        fold,
                        np.nan,
                        c_value,
                        penalty,
                        fold.X_standardized[fold.training_mask],
                        fold.X_standardized[fold.validation_mask],
                    )
                )

    def evaluate_logistic(task):
        (
            family,
            order,
            fold,
            coefficient,
            c_value,
            penalty,
            train_features,
            validation_features,
        ) = task
        fit = fit_logistic(
            train_features,
            labels[fold.training_mask],
            c_value=c_value,
            penalty=penalty,
            seed=SEED,
        )
        return {
            "candidate_family": family,
            "candidate_order": order,
            "heldout_source_batch": fold.heldout_batch,
            "two_e_lambda": coefficient,
            "C": c_value,
            "beta": np.nan,
            "l2_penalty": np.nan,
            "converged": int(fit.converged),
            "n_iter": fit.n_iter,
            **prediction_metrics(
                labels[fold.validation_mask], fit.predict(validation_features)
            ),
        }

    with threadpool_limits(limits=1):
        source_rows = Parallel(n_jobs=args.n_jobs, backend="threading", batch_size=1)(
            delayed(evaluate_logistic)(task) for task in logistic_tasks
        )
    print("Completed source validation for the L2 and L1 logistic models.", flush=True)

    dg_tasks = []
    for fold in folds:
        for order, l2_penalty in enumerate(l2_grid):
            dg_tasks.append((fold, "linear_GroupDRO", order, np.nan, l2_penalty))
        order = 0
        for beta in beta_grid:
            for l2_penalty in l2_grid:
                dg_tasks.append((fold, "linear_VREx", order, beta, l2_penalty))
                order += 1

    def evaluate_dg(task):
        fold, family, order, beta, l2_penalty = task
        train_labels = labels[fold.training_mask]
        train_domains = batches[fold.training_mask]
        if family == "linear_GroupDRO":
            fit = fit_exact_group_dro(
                fold.X_standardized[fold.training_mask],
                train_labels,
                train_domains,
                l2_penalty=l2_penalty,
                max_iter=args.dg_max_iter,
            )
        else:
            fit = fit_vrex_candidate(
                fold.X_standardized[fold.training_mask],
                train_labels,
                train_domains,
                beta,
                l2_penalty,
                args.dg_max_iter,
            )
        return {
            "candidate_family": family,
            "candidate_order": order,
            "heldout_source_batch": fold.heldout_batch,
            "two_e_lambda": np.nan,
            "C": np.nan,
            "beta": beta,
            "l2_penalty": l2_penalty,
            "converged": int(fit.converged),
            "n_iter": fit.n_iter,
            **prediction_metrics(
                labels[fold.validation_mask],
                fit.predict(fold.X_standardized[fold.validation_mask]),
            ),
        }

    with threadpool_limits(limits=1):
        dg_rows = Parallel(n_jobs=args.n_jobs, backend="threading", batch_size=1)(
            delayed(evaluate_dg)(task) for task in dg_tasks
        )
    print("Completed source validation for GroupDRO and VREx.", flush=True)
    source = pd.DataFrame([*source_rows, *dg_rows])

    anchor_summary = mean_candidate_rows(
        source[source.candidate_family == "Anchor_PCA_l2"],
        ["two_e_lambda", "C"],
    )
    pool_selection = select_row(anchor_summary[anchor_summary.two_e_lambda == 0.0])
    selected_anchor = select_row(anchor_summary)
    infinity_selection = select_row(anchor_summary[np.isinf(anchor_summary.two_e_lambda)])
    unreduced_l2 = select_row(
        mean_candidate_rows(source[source.candidate_family == "unreduced_l2"], ["C"])
    )
    unreduced_l1 = select_row(
        mean_candidate_rows(source[source.candidate_family == "unreduced_l1"], ["C"])
    )
    group_dro = select_row(
        mean_candidate_rows(source[source.candidate_family == "linear_GroupDRO"], ["l2_penalty"])
    )
    vrex = select_row(
        mean_candidate_rows(
            source[source.candidate_family == "linear_VREx"],
            ["beta", "l2_penalty"],
        )
    )
    chosen = {
        "poolPCA_l2": pool_selection,
        "source_selected_Anchor_PCA_l2": selected_anchor,
        "Anchor_PCA_infinity_l2": infinity_selection,
        "unreduced_l2": unreduced_l2,
        "unreduced_l1": unreduced_l1,
        "linear_GroupDRO": group_dro,
        "linear_VREx": vrex,
    }
    selection_rows = []
    for method_id in METHOD_ORDER:
        row = chosen[method_id]
        selection_rows.append(
            {
                "method_id": method_id,
                "two_e_lambda": row.get("two_e_lambda", np.nan),
                "C": row.get("C", np.nan),
                "beta": row.get("beta", np.nan),
                "l2_penalty": row.get("l2_penalty", np.nan),
                "source_mean_balanced_accuracy": row.mean_balanced_accuracy,
                "source_standard_error_balanced_accuracy": row.standard_error_balanced_accuracy,
                "all_source_fits_converged": int(row.all_converged),
            }
        )
    selections = pd.DataFrame(selection_rows)

    X_standardized, _ = gas.preprocess_features(
        dataset.X_raw, dataset.metadata, SOURCE_BATCHES, scale_mode="source-standard"
    )
    source_stats = gas.build_batch_stats(X_standardized, dataset.metadata, SOURCE_BATCHES)
    source_covariances = [source_stats[batch].covariance for batch in SOURCE_BATCHES]
    source_counts = np.asarray(
        [source_stats[batch].n_obs for batch in SOURCE_BATCHES],
        dtype=float,
    )
    source_mask = np.isin(batches, SOURCE_BATCHES)
    final_fits = {}
    final_features = {}
    final_checks = []
    for method_id in ("poolPCA_l2", "source_selected_Anchor_PCA_l2", "Anchor_PCA_infinity_l2"):
        row = chosen[method_id]
        model = fit_two_e_lambda_from_covariances(
            source_covariances,
            RANK,
            float(row.two_e_lambda),
            n_obs=source_counts,
            weighting="uniform",
        )
        features = X_standardized @ model.directions_
        fit = fit_logistic(
            features[source_mask],
            labels[source_mask],
            c_value=float(row.C),
            penalty="l2",
            seed=SEED,
        )
        final_fits[method_id] = fit
        final_features[method_id] = features
        final_checks.append(
            {
                "method_id": method_id,
                "representation_dimension": RANK,
                "directions_orthonormal": int(
                    np.allclose(model.directions_.T @ model.directions_, np.eye(RANK), atol=1e-7)
                ),
                "converged": int(fit.converged),
                "n_iter": fit.n_iter,
            }
        )
    for method_id, penalty in (("unreduced_l2", "l2"), ("unreduced_l1", "l1")):
        row = chosen[method_id]
        fit = fit_logistic(
            X_standardized[source_mask],
            labels[source_mask],
            c_value=float(row.C),
            penalty=penalty,
            seed=SEED,
        )
        final_fits[method_id] = fit
        final_features[method_id] = X_standardized
        final_checks.append(
            {
                "method_id": method_id,
                "representation_dimension": X_standardized.shape[1],
                "directions_orthonormal": np.nan,
                "converged": int(fit.converged),
                "n_iter": fit.n_iter,
            }
        )
    group_row = chosen["linear_GroupDRO"]
    final_fits["linear_GroupDRO"] = fit_exact_group_dro(
        X_standardized[source_mask],
        labels[source_mask],
        batches[source_mask],
        l2_penalty=float(group_row.l2_penalty),
        max_iter=args.dg_max_iter,
    )
    vrex_row = chosen["linear_VREx"]
    final_fits["linear_VREx"] = fit_vrex_candidate(
        X_standardized[source_mask],
        labels[source_mask],
        batches[source_mask],
        float(vrex_row.beta),
        float(vrex_row.l2_penalty),
        args.dg_max_iter,
    )
    for method_id in ("linear_GroupDRO", "linear_VREx"):
        final_features[method_id] = X_standardized
        fit = final_fits[method_id]
        final_checks.append(
            {
                "method_id": method_id,
                "representation_dimension": X_standardized.shape[1],
                "directions_orthonormal": np.nan,
                "converged": int(fit.converged),
                "n_iter": fit.n_iter,
                "optimizer_success": int(fit.optimizer_success),
                "gradient_norm": fit.gradient_norm,
                "constraint_violation": fit.constraint_violation,
            }
        )

    target_rows = []
    for method_id in METHOD_ORDER:
        fit = final_fits[method_id]
        features = final_features[method_id]
        for batch in TARGET_BATCHES:
            mask = batches == batch
            target_rows.append(
                {
                    "method_id": method_id,
                    "target_batch": batch,
                    "n_target_observations": int(mask.sum()),
                    **prediction_metrics(labels[mask], fit.predict(features[mask])),
                }
            )
    target = pd.DataFrame(target_rows)
    target_summary = (
        target.groupby("method_id", sort=False)
        .agg(
            n_target_batches=("target_batch", "nunique"),
            n_target_observations=("n_target_observations", "sum"),
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    if not all(row["converged"] == 1 for row in final_checks):
        raise RuntimeError("At least one selected final classification fit did not converge.")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_frame(
        source,
        output / "source_validation.csv",
        ("candidate_family", "candidate_order", "heldout_source_batch"),
    )
    write_frame(selections, output / "selections.csv", ("method_id",))
    write_frame(target, output / "target_metrics_by_batch.csv", ("method_id", "target_batch"))
    write_frame(target_summary, output / "target_metrics_summary.csv", ("method_id",))
    write_frame(pd.DataFrame(final_checks), output / "fit_checks.csv", ("method_id",))
    protocol = {
        "dataset": "UCI Gas Sensor Array Drift Dataset",
        "archive_sha256": gas.EXPECTED_SHA256,
        "source_batches": list(SOURCE_BATCHES),
        "target_batches": list(TARGET_BATCHES),
        "representation_rank": RANK,
        "seed": SEED,
        "source_standardization": "refitted in every validation fold",
        "anchor_weighting": "uniform domain",
        "class_loss_weighting": "class balanced",
        "selection_metric": "equal source-batch mean balanced accuracy",
        "source_validation_blas_threads_per_worker": 1,
        "blas_threads": 10,
        "C_grid": list(c_grid),
        "two_e_lambda_grid": ["infinity" if np.isinf(value) else value for value in rho_grid],
        "group_dro_l2_grid": list(l2_grid),
        "vrex_beta_grid": list(beta_grid),
        "vrex_l2_grid": list(l2_grid),
        "target_used_for_selection": False,
        "methods": list(METHOD_ORDER),
        "software_versions": software_versions(
            ("anchorpca", "numpy", "scipy", "pandas", "scikit-learn", "joblib")
        ),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote gas-classification tables to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "classification",
    )
    parser.add_argument("--c-values", type=float, nargs="+", default=DEFAULT_C_GRID)
    parser.add_argument(
        "--finite-two-e-lambda",
        type=float,
        nargs="+",
        default=[value for value in DEFAULT_TWO_E_LAMBDA_GRID if np.isfinite(value)],
    )
    parser.add_argument("--dg-l2-values", type=float, nargs="+", default=DEFAULT_L2_GRID)
    parser.add_argument("--beta-values", type=float, nargs="+", default=DEFAULT_BETA_GRID)
    parser.add_argument("--dg-max-iter", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Match the paper run; validation workers use one BLAS thread.
    with threadpool_limits(limits=10):
        run(parse_args())

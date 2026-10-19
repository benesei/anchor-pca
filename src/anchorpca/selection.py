"""Source-domain selection of the effective Anchor PCA penalty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .estimators import AnchorPCAInfty, AnchorPCALambda


DEFAULT_TWO_E_LAMBDA_GRID = (
    0.0,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    1e2,
    1e3,
    np.inf,
)


@dataclass(frozen=True)
class TwoELambdaSelectionResult:
    """Result of leave-one-source-domain-out penalty selection."""

    candidates: Tuple[float, ...]
    fold_scores: np.ndarray
    mean_scores: np.ndarray
    standard_errors: np.ndarray
    selected_index: int
    selected_two_e_lambda: float
    estimator: Union[AnchorPCALambda, AnchorPCAInfty]


def fit_two_e_lambda_from_covariances(
    covariances,
    n_components,
    two_e_lambda,
    *,
    weighting="uniform",
    n_obs=None,
    block_tol="auto",
):
    """Fit Anchor PCA for an effective coefficient ``two_e_lambda = 2 E lambda``.

    ``np.inf`` selects :class:`AnchorPCAInfty`. A finite coefficient is converted
    using the number of supplied source domains, so that it is the coefficient of
    the averaged local-projector matrix in the fitted population matrix.
    """
    covariances, counts = _validate_covariance_inputs(covariances, n_obs=n_obs)
    coefficient = _validate_coefficient(two_e_lambda)
    n_environments = len(covariances)

    if np.isinf(coefficient):
        estimator = AnchorPCAInfty(
            n_components=n_components,
            weighting=weighting,
            block_tol=block_tol,
        )
    else:
        estimator = AnchorPCALambda(
            n_components=n_components,
            lambda_=coefficient / (2.0 * n_environments),
            weighting=weighting,
        )

    estimator.fit_covariances(covariances, n_obs=counts)
    estimator.two_e_lambda_ = coefficient
    return estimator


def select_two_e_lambda_from_covariances(
    covariances,
    n_components,
    *,
    two_e_lambda_grid=DEFAULT_TWO_E_LAMBDA_GRID,
    weighting="uniform",
    n_obs=None,
    block_tol="auto",
):
    """Select ``2 E lambda`` by leave-one-source-domain-out validation.

    Each candidate is fitted on all but one source covariance and scored by the
    fraction of total variance it explains in the held-out source. Scores are
    averaged equally across source domains. Ties are resolved by candidate order,
    and the selected candidate is refitted on all supplied source domains.
    """
    covariances, counts = _validate_covariance_inputs(covariances, n_obs=n_obs)
    if len(covariances) < 3:
        raise ValueError("Penalty selection requires at least three source domains.")
    candidates = _validate_grid(two_e_lambda_grid)

    traces = np.asarray([np.trace(covariance) for covariance in covariances], dtype=float)
    if np.any(traces <= 0):
        raise ValueError("Each held-out covariance must have positive total variance.")

    n_candidates = len(candidates)
    n_environments = len(covariances)
    fold_scores = np.empty((n_candidates, n_environments), dtype=float)

    for fold_index in range(n_environments):
        train_covariances = [
            covariance
            for index, covariance in enumerate(covariances)
            if index != fold_index
        ]
        train_counts = None if counts is None else np.delete(counts, fold_index)
        held_out = covariances[fold_index]

        for candidate_index, coefficient in enumerate(candidates):
            estimator = fit_two_e_lambda_from_covariances(
                train_covariances,
                n_components,
                coefficient,
                weighting=weighting,
                n_obs=train_counts,
                block_tol=block_tol,
            )
            explained = np.trace(estimator.projection_ @ held_out)
            fold_scores[candidate_index, fold_index] = explained / traces[fold_index]

    mean_scores = fold_scores.mean(axis=1)
    standard_errors = fold_scores.std(axis=1, ddof=1) / np.sqrt(n_environments)
    selected_index = int(np.argmax(mean_scores))
    selected_coefficient = candidates[selected_index]
    estimator = fit_two_e_lambda_from_covariances(
        covariances,
        n_components,
        selected_coefficient,
        weighting=weighting,
        n_obs=counts,
        block_tol=block_tol,
    )

    fold_scores.setflags(write=False)
    mean_scores.setflags(write=False)
    standard_errors.setflags(write=False)
    return TwoELambdaSelectionResult(
        candidates=candidates,
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        standard_errors=standard_errors,
        selected_index=selected_index,
        selected_two_e_lambda=selected_coefficient,
        estimator=estimator,
    )


def _validate_covariance_inputs(covariances, *, n_obs):
    if len(covariances) == 0:
        raise ValueError("covariances must be non-empty.")

    validated = []
    expected_shape = None
    for index, covariance in enumerate(covariances):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"covariances[{index}] must be a square matrix.")
        if expected_shape is None:
            expected_shape = covariance.shape
        elif covariance.shape != expected_shape:
            raise ValueError(
                f"covariances[{index}] has shape {covariance.shape}, expected {expected_shape}."
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"covariances[{index}] must contain only finite values.")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"covariances[{index}] must be symmetric.")
        covariance = 0.5 * (covariance + covariance.T)
        scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
        if float(np.linalg.eigvalsh(covariance)[0]) < -1e-10 * scale:
            raise ValueError(f"covariances[{index}] must be positive semidefinite.")
        validated.append(covariance)

    if n_obs is None:
        counts = None
    else:
        counts = np.asarray(n_obs, dtype=float)
        if counts.shape != (len(validated),):
            raise ValueError(f"n_obs must have shape {(len(validated),)}.")
        if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
            raise ValueError("n_obs must contain positive finite values.")
    return validated, counts


def _validate_grid(two_e_lambda_grid):
    try:
        candidates = tuple(_validate_coefficient(value) for value in two_e_lambda_grid)
    except TypeError as exc:
        raise ValueError("two_e_lambda_grid must be a non-empty iterable.") from exc
    if len(candidates) == 0:
        raise ValueError("two_e_lambda_grid must be non-empty.")
    if sum(np.isinf(value) for value in candidates) != 1:
        raise ValueError("two_e_lambda_grid must contain exactly one infinity candidate.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("two_e_lambda_grid must not contain duplicate candidates.")
    return candidates


def _validate_coefficient(value):
    try:
        coefficient = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Each 2 E lambda candidate must be nonnegative.") from exc
    if np.isnan(coefficient) or coefficient < 0:
        raise ValueError("Each 2 E lambda candidate must be nonnegative.")
    return coefficient

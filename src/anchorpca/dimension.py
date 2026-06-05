from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.stats import chi2

from .covariance import empirical_covariances_from_envs
from .linalg import symmetrize


@dataclass(frozen=True)
class SchottTestResult:
    """Result of one Schott partial-CPCA span-dimension test."""

    statistic: float
    df: int
    p_value: float
    s: int
    p: int
    k: int
    selected_rank: int
    n_obs: np.ndarray


@dataclass(frozen=True)
class SStarDimensionTestResult:
    """Sequential estimate of m = dim(S_star)."""

    m_hat: int
    r_hat: int
    alpha: float
    tested_s: np.ndarray
    p_values: np.ndarray
    statistics: np.ndarray
    stopped_early: bool
    schott_results: List[SchottTestResult]


class SStarDimensionWarning(UserWarning):
    """Warning raised when S_star dimension diagnostics disagree."""


def estimate_sstar_dimension(
    X_envs,
    n_components,
    *,
    alpha=0.05,
    center=True,
    assume_gaussian=False,
    check_gaps=True,
    eig_rtol=1e-10,
    eig_atol=1e-12,
):
    """Estimate m = dim(S_star) by sequential bottom-space Schott tests.

    This test uses the Gaussian fixed-p large-sample calibration. Set
    ``assume_gaussian=True`` to acknowledge this assumption.
    """
    _require_gaussian_assumption(assume_gaussian)
    covariances, _, n_obs = empirical_covariances_from_envs(X_envs, center=center)
    return estimate_sstar_dimension_from_covariances(
        covariances,
        n_components,
        n_obs=n_obs,
        alpha=alpha,
        assume_gaussian=True,
        check_gaps=check_gaps,
        eig_rtol=eig_rtol,
        eig_atol=eig_atol,
    )


def estimate_sstar_dimension_from_covariances(
    covariances,
    n_components,
    *,
    n_obs,
    alpha=0.05,
    assume_gaussian=False,
    check_gaps=True,
    eig_rtol=1e-10,
    eig_atol=1e-12,
):
    """Estimate m = dim(S_star) from empirical covariances and sample sizes."""
    _require_gaussian_assumption(assume_gaussian)
    alpha = _validate_alpha(alpha)
    covariances, n_obs, p, k = _validate_covariance_inputs(
        covariances,
        n_components,
        n_obs,
    )

    g = len(covariances)
    q = p - k
    t = min(g * q, p)

    tested_s = []
    p_values = []
    statistics = []
    schott_results = []

    for s in range(q, t):
        result = _schott_partial_cpca_test_from_covariances(
            covariances,
            n_obs,
            k,
            s,
            check_gaps=check_gaps,
            eig_rtol=eig_rtol,
            eig_atol=eig_atol,
        )
        tested_s.append(s)
        p_values.append(result.p_value)
        statistics.append(result.statistic)
        schott_results.append(result)

        if result.p_value >= alpha:
            r_hat = int(s)
            return SStarDimensionTestResult(
                m_hat=int(p - r_hat),
                r_hat=r_hat,
                alpha=alpha,
                tested_s=np.asarray(tested_s, dtype=int),
                p_values=np.asarray(p_values, dtype=float),
                statistics=np.asarray(statistics, dtype=float),
                stopped_early=True,
                schott_results=schott_results,
            )

    r_hat = int(t)
    return SStarDimensionTestResult(
        m_hat=int(p - r_hat),
        r_hat=r_hat,
        alpha=alpha,
        tested_s=np.asarray(tested_s, dtype=int),
        p_values=np.asarray(p_values, dtype=float),
        statistics=np.asarray(statistics, dtype=float),
        stopped_early=False,
        schott_results=schott_results,
    )


def _schott_partial_cpca_test_from_covariances(
    covariances,
    n_obs,
    k,
    s,
    *,
    check_gaps,
    eig_rtol,
    eig_atol,
):
    g = len(covariances)
    p = covariances[0].shape[0]
    selected_idx = np.arange(k, p)
    complement_idx = np.arange(0, k)
    q = p - k
    t = min(g * q, p)
    if not (q <= s < t):
        raise ValueError(
            f"Need q <= s < min(g*q, p), got q={q}, s={s}, min(g*q, p)={t}."
        )

    n_i = np.asarray(n_obs, dtype=float) - 1.0
    n_total = float(np.sum(n_i))
    df = int((g * q - s) * (p - s))

    evals = []
    evecs = []
    selected_bases = []
    selected_projectors = []

    for index, covariance in enumerate(covariances):
        values, vectors = _eigh_desc(covariance)
        if check_gaps:
            gap = float(values[k - 1] - values[k])
            threshold = max(eig_atol, eig_rtol * max(abs(float(values[k - 1])), 1.0))
            if gap <= threshold:
                raise ValueError(
                    "Selected eigenspace is numerically ill-defined for "
                    f"environment {index}: the top-k/bottom-(p-k) eigengap is too small."
                )

        selected_basis = vectors[:, selected_idx]
        evals.append(values)
        evecs.append(vectors)
        selected_bases.append(selected_basis)
        selected_projectors.append(_projector_from_basis(selected_basis))

    projector_sum = np.sum(selected_projectors, axis=0)
    projector_sum_values, projector_sum_vectors = _eigh_desc(projector_sum)
    if check_gaps:
        gap_s = float(projector_sum_values[s - 1] - projector_sum_values[s])
        threshold = max(
            eig_atol,
            eig_rtol * max(abs(float(projector_sum_values[s - 1])), 1.0),
        )
        if gap_s <= threshold:
            raise ValueError(
                "The estimated common bottom-space span of dimension s is "
                "numerically ill-defined because its spectral gap is too small."
            )

    common_span_basis = projector_sum_vectors[:, :s]
    common_span_complement = projector_sum_vectors[:, s:]
    inv_positive = _safe_inverse_positive(
        projector_sum_values[:s],
        rtol=eig_rtol,
        atol=eig_atol,
    )
    projector_sum_pinv = common_span_basis @ np.diag(inv_positive) @ common_span_basis.T
    projector_sum_pinv = symmetrize(projector_sum_pinv)

    block_size = q * (p - s)
    total_dim = g * block_size
    Wstar = np.zeros((total_dim, total_dim), dtype=float)

    Ystar_blocks = []
    for environment_index in range(g):
        values = evals[environment_index]
        vectors = evecs[environment_index]
        sample_df = float(n_i[environment_index])
        Ystar = np.zeros((block_size, block_size), dtype=float)

        for local_j, j in enumerate(selected_idx):
            lambda_j = float(values[j])
            ej = np.zeros((q, 1), dtype=float)
            ej[local_j, 0] = 1.0
            Ej = ej @ ej.T

            for ell in complement_idx:
                lambda_l = float(values[ell])
                denominator = (lambda_j - lambda_l) ** 2
                scale = max(abs(lambda_j), abs(lambda_l), 1.0)
                if denominator <= max(eig_atol, eig_rtol * scale) ** 2:
                    raise ValueError(
                        "Encountered a near-zero Schott covariance denominator "
                        f"in environment {environment_index}."
                    )
                coefficient = (n_total / sample_df) * (lambda_j * lambda_l) / denominator
                ql = vectors[:, [ell]]
                restricted_complement = common_span_complement.T @ (ql @ ql.T) @ common_span_complement
                Ystar += coefficient * np.kron(Ej, symmetrize(restricted_complement))

        Ystar = symmetrize(Ystar)
        Ystar_blocks.append(Ystar)
        start = environment_index * block_size
        Wstar[start : start + block_size, start : start + block_size] += Ystar

    middle_cache = {}
    for left in range(g):
        for right in range(g):
            middle = selected_bases[left].T @ projector_sum_pinv @ selected_bases[right]
            middle_cache[(left, right)] = np.kron(middle, np.eye(p - s))

    for left in range(g):
        for right in range(g):
            V = np.zeros((block_size, block_size), dtype=float)
            middle = middle_cache[(left, right)]
            for environment_index in range(g):
                V += (
                    middle_cache[(left, environment_index)]
                    @ Ystar_blocks[environment_index]
                    @ middle_cache[(environment_index, right)]
                )
            V -= middle @ Ystar_blocks[right]
            V -= Ystar_blocks[left] @ middle
            V = symmetrize(V)

            row = left * block_size
            col = right * block_size
            Wstar[row : row + block_size, col : col + block_size] += V

    Wstar = symmetrize(Wstar)
    vstar = np.concatenate(
        [_vecF(common_span_complement.T @ basis) for basis in selected_bases]
    )
    _, Wstar_pinv = _restricted_pseudoinverse(Wstar, df, rtol=eig_rtol, atol=eig_atol)

    statistic = float(n_total * (vstar.T @ Wstar_pinv @ vstar))
    p_value = float(chi2.sf(statistic, df))
    return SchottTestResult(
        statistic=statistic,
        df=df,
        p_value=p_value,
        s=int(s),
        p=int(p),
        k=int(k),
        selected_rank=int(q),
        n_obs=np.asarray(n_obs, dtype=float),
    )


def _validate_covariance_inputs(covariances, n_components, n_obs):
    covariances = [symmetrize(np.asarray(covariance, dtype=float)) for covariance in covariances]
    if len(covariances) < 2:
        raise ValueError("At least two empirical covariances are required.")

    p = covariances[0].shape[0]
    k = int(n_components)
    for index, covariance in enumerate(covariances):
        if covariance.shape != (p, p):
            raise ValueError(
                f"covariances[{index}] has shape {covariance.shape}, expected {(p, p)}."
            )
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"covariances[{index}] contains non-finite values.")
    if not (1 <= k < p):
        raise ValueError(f"Need 1 <= n_components < p for dimension testing, got k={k}, p={p}.")

    if n_obs is None:
        raise ValueError("n_obs is required for covariance-based dimension testing.")
    n_obs = np.asarray(n_obs, dtype=float)
    if n_obs.ndim != 1 or n_obs.size != len(covariances):
        raise ValueError(f"n_obs must be one-dimensional with length {len(covariances)}.")
    if not np.all(np.isfinite(n_obs)) or np.any(n_obs <= 1):
        raise ValueError("n_obs must contain finite values greater than one.")
    return covariances, n_obs, p, k


def _validate_alpha(alpha):
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a scalar in (0, 1).") from exc
    if not np.isscalar(alpha) or not np.isfinite(alpha_value) or not (0.0 < alpha_value < 1.0):
        raise ValueError("alpha must be a scalar in (0, 1).")
    return alpha_value


def _require_gaussian_assumption(assume_gaussian):
    if not isinstance(assume_gaussian, (bool, np.bool_)) or not bool(assume_gaussian):
        raise ValueError(
            "Sequential S_star dimension testing uses a Gaussian asymptotic "
            "calibration. Set assume_gaussian=True to run it."
        )


def _eigh_desc(matrix):
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=float), _canonicalize_eigenvectors(vectors[:, order])


def _canonicalize_eigenvectors(vectors):
    vectors = np.asarray(vectors, dtype=float).copy()
    for column_index in range(vectors.shape[1]):
        column = vectors[:, column_index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0:
            vectors[:, column_index] = -column
    return vectors


def _projector_from_basis(basis):
    return symmetrize(np.asarray(basis, dtype=float) @ np.asarray(basis, dtype=float).T)


def _vecF(matrix):
    return np.asarray(matrix, dtype=float).reshape(-1, order="F")


def _safe_inverse_positive(values, *, rtol, atol):
    values = np.asarray(values, dtype=float)
    scale = max(float(np.max(np.abs(values))), 1.0)
    threshold = max(atol, rtol * scale)
    output = np.zeros_like(values)
    mask = values > threshold
    output[mask] = 1.0 / values[mask]
    return output


def _restricted_pseudoinverse(matrix, rank, *, rtol, atol):
    values, vectors = _eigh_desc(matrix)
    if rank < 0 or rank > values.size:
        raise ValueError("Invalid restricted pseudoinverse rank.")
    selected_values = values[:rank]
    inverse_values = _safe_inverse_positive(selected_values, rtol=rtol, atol=atol)
    if rank == 0:
        return values, np.zeros_like(matrix)
    pseudoinverse = vectors[:, :rank] @ np.diag(inverse_values) @ vectors[:, :rank].T
    return values, symmetrize(pseudoinverse)

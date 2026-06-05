from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eigh

from .covariance import build_covariance_context, empirical_covariances_from_envs
from .dimension import SStarDimensionWarning, estimate_sstar_dimension_from_covariances
from .linalg import projector_from_basis, symmetrize, top_eigpairs_symmetric
from .metrics import explained_variance


class AnchorPCALambda:
    """Finite-invariance AnchorPCA estimator indexed by lambda."""

    def __init__(self, n_components, lambda_, *, center=True, weighting="uniform"):
        self.n_components = n_components
        self.lambda_ = lambda_
        self.center = center
        self.weighting = weighting

    def fit_covariances(self, covariances, *, n_obs=None):
        lambda_value = float(self.lambda_)
        if lambda_value < 0:
            raise ValueError("lambda_ must be nonnegative.")

        context = build_covariance_context(
            covariances,
            n_components=self.n_components,
            n_obs=n_obs,
            weighting=self.weighting,
        )
        matrix = symmetrize(
            context.barSigma
            + 2.0 * float(context.n_environments) * lambda_value * context.barPi
        )
        eigenvalues, directions = top_eigpairs_symmetric(matrix, context.n_components)

        _store_common_fit_attributes(self, context, directions)
        self.lambda_ = lambda_value
        self.M_lambda_ = matrix
        self.eigenvalues_ = eigenvalues
        return self

    def fit(self, X_envs):
        covariances, means, n_obs = empirical_covariances_from_envs(
            X_envs,
            center=self.center,
        )
        self.fit_covariances(covariances, n_obs=n_obs)
        self.means_ = means
        return self

    def transform(self, X, mean=None):
        _check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        if mean is not None:
            X = X - mean
        return X @ self.directions_

    def fit_transform(self, X_envs):
        self.fit(X_envs)
        transformed = []
        for X, mean in zip(X_envs, self.means_):
            transformed.append(self.transform(X, mean=mean if self.center else None))
        return transformed

    def explained_variance(self, covariances=None, weights=None):
        _check_is_fitted(self)
        if covariances is None:
            covariances = self.covariances_
        if weights is None:
            weights = self.weights_
        return explained_variance(self.directions_, covariances, weights)


class AnchorPCAInfty:
    """Blockwise hard-agreement AnchorPCA estimator."""

    def __init__(
        self,
        n_components,
        *,
        center=True,
        weighting="uniform",
        block_tol="auto",
        block_tol_alpha=0.4,
        block_tol_c=0.5,
        block_tol_max=0.05,
        sstar_test_mode="off",
        sstar_test_alpha=0.05,
        assume_gaussian=False,
    ):
        self.n_components = n_components
        self.center = center
        self.weighting = weighting
        self.block_tol = block_tol
        self.block_tol_alpha = block_tol_alpha
        self.block_tol_c = block_tol_c
        self.block_tol_max = block_tol_max
        self.sstar_test_mode = sstar_test_mode
        self.sstar_test_alpha = sstar_test_alpha
        self.assume_gaussian = assume_gaussian

    def fit_covariances(self, covariances, *, n_obs=None):
        sstar_test_mode = _validate_sstar_test_mode(self.sstar_test_mode)
        if sstar_test_mode != "off":
            _require_sstar_test_gaussian_assumption(self.assume_gaussian)
            if n_obs is None:
                raise ValueError("n_obs is required when sstar_test_mode is enabled.")

        block_tol, block_tol_mode = _resolve_block_tol(
            self.block_tol,
            n_obs,
            block_tol_alpha=self.block_tol_alpha,
            block_tol_c=self.block_tol_c,
            block_tol_max=self.block_tol_max,
        )
        context = build_covariance_context(
            covariances,
            n_components=self.n_components,
            n_obs=n_obs,
            weighting=self.weighting,
        )

        sstar_test_result = None
        invariant_dim_uncalibrated = None
        if sstar_test_mode != "off":
            sstar_test_result = estimate_sstar_dimension_from_covariances(
                context.covariances,
                context.n_components,
                n_obs=context.n_obs,
                alpha=self.sstar_test_alpha,
                assume_gaussian=self.assume_gaussian,
            )

        if sstar_test_mode == "calibrate":
            uncalibrated_solution = _blockwise_hard_agreement_solution(
                context.barSigma,
                context.barPi,
                context.n_components,
                block_tol=block_tol,
            )
            invariant_dim_uncalibrated = int(
                uncalibrated_solution["agreement_blocks"][0]["dimension"]
            )
            block_tol = _calibrated_block_tol_for_first_block(
                context.barPi,
                sstar_test_result.m_hat,
            )
            block_tol_mode = "test_calibrated"

        solution = _blockwise_hard_agreement_solution(
            context.barSigma,
            context.barPi,
            context.n_components,
            block_tol=block_tol,
        )

        _store_common_fit_attributes(self, context, solution["directions"])
        self.block_tol_ = block_tol
        self.block_tol_mode_ = block_tol_mode
        self.block_tol_alpha_ = float(self.block_tol_alpha)
        self.block_tol_c_ = float(self.block_tol_c)
        self.block_tol_max_ = float(self.block_tol_max)
        self.agreement_blocks_ = solution["agreement_blocks"]
        self.invariant_dim_estimate_ = int(self.agreement_blocks_[0]["dimension"])
        self.rho_by_component_ = solution["rho_by_component"]
        self.variance_by_component_ = solution["variance_by_component"]
        self.barPi_eigenvalues_ = solution["barPi_eigenvalues"]
        if sstar_test_mode != "off":
            self.sstar_dimension_test_result_ = sstar_test_result
            self.sstar_test_mode_ = sstar_test_mode
            self.sstar_test_alpha_ = float(sstar_test_result.alpha)
            if sstar_test_mode == "calibrate":
                self.invariant_dim_estimate_uncalibrated_ = invariant_dim_uncalibrated
                mismatch = invariant_dim_uncalibrated != int(sstar_test_result.m_hat)
            else:
                mismatch = self.invariant_dim_estimate_ != int(sstar_test_result.m_hat)
            self.sstar_dimension_test_mismatch_ = bool(mismatch)
            if sstar_test_mode == "warn" and mismatch:
                warnings.warn(
                    "Tolerance-stabilized AnchorPCAInfty estimated "
                    f"dim(S_star)={self.invariant_dim_estimate_}, while the "
                    "sequential Gaussian Schott test estimated "
                    f"dim(S_star)={sstar_test_result.m_hat}.",
                    SStarDimensionWarning,
                    stacklevel=2,
                )
        return self

    def fit(self, X_envs):
        sstar_test_mode = _validate_sstar_test_mode(self.sstar_test_mode)
        if sstar_test_mode != "off":
            _require_sstar_test_gaussian_assumption(self.assume_gaussian)
            if not self.center:
                raise ValueError(
                    "AnchorPCAInfty S_star dimension testing requires center=True when fitting raw data."
                )
        covariances, means, n_obs = empirical_covariances_from_envs(
            X_envs,
            center=self.center,
        )
        self.fit_covariances(covariances, n_obs=n_obs)
        self.means_ = means
        return self

    def transform(self, X, mean=None):
        _check_is_fitted(self)
        X = np.asarray(X, dtype=float)
        if mean is not None:
            X = X - mean
        return X @ self.directions_

    def fit_transform(self, X_envs):
        self.fit(X_envs)
        transformed = []
        for X, mean in zip(X_envs, self.means_):
            transformed.append(self.transform(X, mean=mean if self.center else None))
        return transformed

    def explained_variance(self, covariances=None, weights=None):
        _check_is_fitted(self)
        if covariances is None:
            covariances = self.covariances_
        if weights is None:
            weights = self.weights_
        return explained_variance(self.directions_, covariances, weights)


def _store_common_fit_attributes(estimator, context, directions):
    estimator.n_components_ = context.n_components
    estimator.n_features_ = context.n_features
    estimator.n_environments_ = context.n_environments
    estimator.n_obs_ = np.asarray(context.n_obs, dtype=float)
    estimator.weights_ = np.asarray(context.weights, dtype=float)
    estimator.weighting_ = context.weighting
    estimator.covariances_ = list(context.covariances)
    estimator.barSigma_ = context.barSigma
    estimator.barPi_ = context.barPi
    estimator.local_directions_ = list(context.local_directions)
    estimator.local_projectors_ = list(context.local_projectors)
    estimator.directions_ = np.asarray(directions, dtype=float)
    estimator.components_ = estimator.directions_.T
    estimator.projection_ = projector_from_basis(estimator.directions_)


def _resolve_block_tol(
    block_tol,
    n_obs,
    *,
    block_tol_alpha,
    block_tol_c,
    block_tol_max,
):
    block_tol_alpha = _positive_finite_scalar(block_tol_alpha, "block_tol_alpha")
    block_tol_c = _positive_finite_scalar(block_tol_c, "block_tol_c")
    block_tol_max = _positive_finite_scalar(block_tol_max, "block_tol_max")

    if isinstance(block_tol, str):
        if block_tol != "auto":
            raise ValueError("block_tol must be 'auto' or a nonnegative numeric scalar.")
        if n_obs is None:
            raise ValueError(
                "block_tol='auto' requires observation counts. Pass n_obs to "
                "fit_covariances, or set a nonnegative numeric block_tol when "
                "fitting population covariances."
            )
        n_obs = np.asarray(n_obs, dtype=float)
        if n_obs.ndim != 1 or n_obs.size == 0:
            raise ValueError("n_obs must be a non-empty one-dimensional array in auto mode.")
        if not np.all(np.isfinite(n_obs)) or np.any(n_obs <= 0):
            raise ValueError("n_obs must contain positive finite values in auto mode.")
        n_min = float(np.min(n_obs))
        return min(block_tol_max, block_tol_c * n_min ** (-block_tol_alpha)), "auto"

    try:
        numeric_tol = float(block_tol)
    except (TypeError, ValueError) as exc:
        raise ValueError("block_tol must be 'auto' or a nonnegative numeric scalar.") from exc

    if not np.isscalar(block_tol) or not np.isfinite(numeric_tol) or numeric_tol < 0:
        raise ValueError("block_tol must be a nonnegative finite numeric scalar.")
    return numeric_tol, "manual"


def _positive_finite_scalar(value, name):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive.") from exc
    if not np.isscalar(value) or not np.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"{name} must be positive.")
    return numeric_value


def _validate_sstar_test_mode(mode):
    if mode not in {"off", "warn", "calibrate"}:
        raise ValueError("sstar_test_mode must be one of 'off', 'warn', or 'calibrate'.")
    return mode


def _require_sstar_test_gaussian_assumption(assume_gaussian):
    if not isinstance(assume_gaussian, (bool, np.bool_)) or not bool(assume_gaussian):
        raise ValueError(
            "AnchorPCAInfty S_star dimension testing uses a Gaussian asymptotic "
            "calibration. Set assume_gaussian=True to run it."
        )


def _calibrated_block_tol_for_first_block(barPi, m_hat):
    m_hat = int(m_hat)
    if m_hat == 0:
        raise ValueError(
            "Cannot calibrate AnchorPCAInfty block_tol to m_hat=0 because the first "
            "agreement block cannot have dimension zero."
        )

    rho_all, _ = eigh(symmetrize(barPi))
    rho_all = np.asarray(rho_all[::-1], dtype=float)
    p = rho_all.size
    if not (1 <= m_hat < p):
        raise ValueError(f"Need 1 <= m_hat < p for block tolerance calibration, got {m_hat}.")

    lower_gap = float(rho_all[0] - rho_all[m_hat - 1])
    upper_gap = float(rho_all[0] - rho_all[m_hat])
    if not upper_gap > lower_gap:
        raise ValueError(
            "Cannot calibrate AnchorPCAInfty block_tol: the empirical barPi spectrum "
            f"does not separate the first {m_hat} dimensions from the next dimension."
        )
    return 0.5 * (lower_gap + upper_gap)


def _blockwise_hard_agreement_solution(barSigma, barPi, n_components, block_tol):
    p = barPi.shape[0]
    rho_all, V_all = eigh(symmetrize(barPi))
    rho_all = rho_all[::-1]
    V_all = V_all[:, ::-1]

    blocks = []
    current = [0]
    for j in range(1, p):
        if abs(float(rho_all[j]) - float(rho_all[current[0]])) <= block_tol:
            current.append(j)
        else:
            blocks.append(current)
            current = [j]
    blocks.append(current)

    remaining = int(n_components)
    direction_blocks = []
    agreement_blocks = []
    rho_by_component = []
    variance_by_component = []

    for block_index, block_indices in enumerate(blocks):
        if remaining <= 0:
            break

        rho = float(rho_all[block_indices[0]])
        block_basis = V_all[:, block_indices]
        restricted = symmetrize(block_basis.T @ barSigma @ block_basis)
        restricted_values, restricted_vectors = eigh(restricted)
        order = np.argsort(restricted_values)[::-1]
        restricted_values = np.asarray(restricted_values[order], dtype=float)
        restricted_vectors = restricted_vectors[:, order]

        n_selected = min(remaining, len(block_indices))
        directions = block_basis @ restricted_vectors[:, :n_selected]

        direction_blocks.append(directions)
        rho_by_component.append(np.full(n_selected, rho, dtype=float))
        variance_by_component.append(restricted_values[:n_selected])
        agreement_blocks.append(
            {
                "block_index": int(block_index),
                "rho": rho,
                "dimension": int(len(block_indices)),
                "n_selected": int(n_selected),
                "restricted_eigenvalues": restricted_values,
            }
        )
        remaining -= n_selected

    if not direction_blocks:
        raise RuntimeError("No agreement block was selected.")

    directions = np.hstack(direction_blocks)
    return {
        "directions": directions,
        "agreement_blocks": agreement_blocks,
        "rho_by_component": np.concatenate(rho_by_component),
        "variance_by_component": np.concatenate(variance_by_component),
        "barPi_eigenvalues": np.asarray(rho_all, dtype=float),
    }


def _check_is_fitted(estimator):
    if not hasattr(estimator, "directions_"):
        raise RuntimeError("Estimator is not fitted yet.")

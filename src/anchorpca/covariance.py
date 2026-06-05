from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .linalg import projector_from_basis, symmetrize, top_eigpairs_symmetric


@dataclass
class CovarianceContext:
    n_environments: int
    n_features: int
    n_components: int
    covariances: List[np.ndarray]
    n_obs: np.ndarray
    weights: np.ndarray
    weighting: str
    barSigma: np.ndarray
    barPi: np.ndarray
    local_directions: List[np.ndarray]
    local_projectors: List[np.ndarray]


def empirical_covariance(X, center=True):
    """Compute the empirical covariance of a two-dimensional data matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n, p = X.shape
    if n < 2:
        raise ValueError("Each environment must contain at least 2 observations.")

    if center:
        mean = X.mean(axis=0)
        X_centered = X - mean
    else:
        mean = np.zeros(p, dtype=float)
        X_centered = X

    covariance = symmetrize((X_centered.T @ X_centered) / (n - 1))
    return covariance, mean, int(n)


def empirical_covariances_from_envs(X_envs, center=True):
    """Compute empirical covariances for a non-empty list of environments."""
    if len(X_envs) == 0:
        raise ValueError("X_envs must be non-empty.")

    arrays = [np.asarray(X, dtype=float) for X in X_envs]
    p = arrays[0].shape[1]
    covariances = []
    means = []
    n_obs = []

    for i, X in enumerate(arrays):
        if X.ndim != 2:
            raise ValueError(f"X_envs[{i}] must be 2D.")
        if X.shape[1] != p:
            raise ValueError(f"X_envs[{i}] has {X.shape[1]} features, expected {p}.")
        covariance, mean, n = empirical_covariance(X, center=center)
        covariances.append(covariance)
        means.append(mean)
        n_obs.append(n)

    return covariances, means, np.asarray(n_obs, dtype=float)


def environment_weights(n_environments, weighting="uniform", n_obs=None):
    """Build environment weights."""
    E = int(n_environments)
    if E <= 0:
        raise ValueError("n_environments must be positive.")

    if weighting == "uniform":
        weights = np.ones(E, dtype=float) / float(E)
        if n_obs is None:
            n_obs = np.ones(E, dtype=float)
        else:
            n_obs = _validate_n_obs(n_obs, E)
        return weights, n_obs

    if weighting == "observations":
        if n_obs is None:
            raise ValueError("n_obs is required when weighting='observations'.")
        n_obs = _validate_n_obs(n_obs, E)
        return n_obs / n_obs.sum(), n_obs

    raise ValueError("weighting must be either 'uniform' or 'observations'.")


def build_covariance_context(
    covariances,
    *,
    n_components,
    n_obs=None,
    weighting="uniform",
):
    """Construct weighted covariance and projector averages."""
    if len(covariances) == 0:
        raise ValueError("covariances must be non-empty.")

    covs = [symmetrize(np.asarray(covariance, dtype=float)) for covariance in covariances]
    E = len(covs)
    p = covs[0].shape[0]
    k = int(n_components)

    for i, covariance in enumerate(covs):
        if covariance.shape != (p, p):
            raise ValueError(f"covariances[{i}] has shape {covariance.shape}, expected {(p, p)}.")
    if not (1 <= k <= p):
        raise ValueError(f"Need 1 <= n_components <= p, got {k} and p={p}.")

    weights, n_obs = environment_weights(E, weighting=weighting, n_obs=n_obs)
    barSigma = np.zeros((p, p), dtype=float)
    barPi = np.zeros((p, p), dtype=float)
    local_directions = []
    local_projectors = []

    for weight, covariance in zip(weights, covs):
        barSigma += float(weight) * covariance
        _, directions = top_eigpairs_symmetric(covariance, k)
        local_directions.append(directions)
        projector = projector_from_basis(directions)
        local_projectors.append(projector)
        barPi += float(weight) * projector

    return CovarianceContext(
        n_environments=E,
        n_features=p,
        n_components=k,
        covariances=covs,
        n_obs=np.asarray(n_obs, dtype=float),
        weights=np.asarray(weights, dtype=float),
        weighting=str(weighting),
        barSigma=symmetrize(barSigma),
        barPi=symmetrize(barPi),
        local_directions=local_directions,
        local_projectors=local_projectors,
    )


def _validate_n_obs(n_obs, n_environments):
    n_obs = np.asarray(n_obs, dtype=float)
    if n_obs.ndim != 1:
        raise ValueError("n_obs must be one-dimensional.")
    if n_obs.size != int(n_environments):
        raise ValueError(f"n_obs must have length {n_environments}.")
    if np.any(n_obs <= 0):
        raise ValueError("All observation counts must be positive.")
    return n_obs


from __future__ import annotations

import numpy as np


def explained_variance(directions, covariances, weights=None):
    """Compute per-environment and weighted explained variance."""
    directions = np.asarray(directions, dtype=float)
    covariances = [np.asarray(covariance, dtype=float) for covariance in covariances]
    E = len(covariances)
    if E == 0:
        raise ValueError("covariances must be non-empty.")

    per_env = np.asarray(
        [np.trace(directions.T @ covariance @ directions) for covariance in covariances],
        dtype=float,
    )
    if weights is None:
        weights = np.ones(E, dtype=float) / float(E)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (E,):
        raise ValueError(f"weights must have shape {(E,)}.")

    return {
        "per_env": per_env,
        "average": float(weights @ per_env),
        "worst_case": float(per_env.min()),
        "worst_env": int(per_env.argmin()),
    }


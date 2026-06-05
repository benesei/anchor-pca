from __future__ import annotations

from .covariance import build_covariance_context, empirical_covariances_from_envs
from .linalg import projector_from_basis, top_eigpairs_symmetric


def pool_pca_from_covariances(
    covariances,
    n_components,
    *,
    n_obs=None,
    weighting="uniform",
):
    """PCA on the weighted average covariance used as the paper poolPCA baseline."""
    context = build_covariance_context(
        covariances,
        n_components=n_components,
        n_obs=n_obs,
        weighting=weighting,
    )
    eigenvalues, directions = top_eigpairs_symmetric(context.barSigma, context.n_components)
    return {
        "directions": directions,
        "components": directions.T,
        "projection": projector_from_basis(directions),
        "eigenvalues": eigenvalues,
        "barSigma": context.barSigma,
        "weights": context.weights,
        "weighting": context.weighting,
        "n_obs": context.n_obs,
    }


def pool_pca_from_envs(
    X_envs,
    n_components,
    *,
    center=True,
    weighting="uniform",
):
    """PCA on the weighted average empirical covariance of the environments."""
    covariances, _, n_obs = empirical_covariances_from_envs(X_envs, center=center)
    return pool_pca_from_covariances(
        covariances,
        n_components,
        n_obs=n_obs,
        weighting=weighting,
    )


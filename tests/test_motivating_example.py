from __future__ import annotations

import numpy as np

from anchorpca import (
    AnchorPCAInfty,
    AnchorPCALambda,
    explained_variance,
    pool_pca_from_covariances,
)


def rotated_pair(phi_deg):
    phi = np.deg2rad(phi_deg)
    direction = np.array([0.0, 0.0, np.cos(phi), np.sin(phi)])
    complement = np.array([0.0, 0.0, -np.sin(phi), np.cos(phi)])
    return direction, complement


def covariance_from_eigensystem(directions, eigenvalues):
    covariance = np.zeros((4, 4))
    for direction, value in zip(directions, eigenvalues):
        covariance += float(value) * np.outer(direction, direction)
    return covariance


def subspace_capture(directions, vector):
    return float(np.linalg.norm(directions.T @ vector) ** 2)


def motivating_covariances():
    c1 = np.array([1.0, 0.0, 0.0, 0.0])
    c2 = np.array([0.0, 1.0, 0.0, 0.0])
    u, u_perp = rotated_pair(0.0)
    v, v_perp = rotated_pair(50.0)
    w, w_perp = rotated_pair(100.0)

    sigma1 = covariance_from_eigensystem([u, c1, c2, u_perp], [220.0, 140.0, 90.0, 25.0])
    sigma2 = covariance_from_eigensystem([c1, v, c2, v_perp], [120.0, 90.0, 70.0, 10.0])
    sigma3 = covariance_from_eigensystem([w, c2, c1, w_perp], [320.0, 120.0, 80.0, 10.0])
    return [sigma1, sigma2, sigma3], c1, c2


def test_four_dimensional_motivating_example_pattern():
    covariances, a, b = motivating_covariances()

    pool = pool_pca_from_covariances(covariances, n_components=3)
    finite = AnchorPCALambda(n_components=3, lambda_=25.0).fit_covariances(covariances)
    hard = AnchorPCAInfty(n_components=3, block_tol=1e-12).fit_covariances(covariances)

    pool_directions = pool["directions"]
    assert subspace_capture(pool_directions, a) > 0.999
    assert subspace_capture(pool_directions, b) < 1e-8

    for directions in [finite.directions_, hard.directions_]:
        assert subspace_capture(directions, a) > 0.999
        assert subspace_capture(directions, b) > 0.999

    finite_stats = explained_variance(finite.directions_, covariances)
    hard_stats = explained_variance(hard.directions_, covariances)

    assert finite_stats["average"] > hard_stats["average"]
    assert hard_stats["worst_case"] > finite_stats["worst_case"]


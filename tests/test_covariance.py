from __future__ import annotations

import numpy as np
import pytest

from anchorpca import AnchorPCALambda
from anchorpca.covariance import build_covariance_context, environment_weights


def test_uniform_weights_are_default():
    weights, n_obs = environment_weights(3)

    assert np.allclose(weights, np.array([1 / 3, 1 / 3, 1 / 3]))
    assert np.allclose(n_obs, np.ones(3))


def test_observation_weights_use_sample_counts():
    weights, n_obs = environment_weights(
        3,
        weighting="observations",
        n_obs=[2, 3, 5],
    )

    assert np.allclose(weights, np.array([0.2, 0.3, 0.5]))
    assert np.allclose(n_obs, np.array([2.0, 3.0, 5.0]))


def test_observation_weighting_requires_counts_for_covariances():
    covariances = [np.eye(3), 2.0 * np.eye(3)]
    model = AnchorPCALambda(n_components=1, lambda_=1.0, weighting="observations")

    with pytest.raises(ValueError, match="n_obs is required"):
        model.fit_covariances(covariances)


def test_covariance_context_uses_top_rank_equal_to_output_rank():
    covariances = [
        np.diag([3.0, 2.0, 1.0]),
        np.diag([2.0, 3.0, 1.0]),
    ]
    context = build_covariance_context(covariances, n_components=2)

    assert len(context.local_projectors) == 2
    assert all(projector.shape == (3, 3) for projector in context.local_projectors)
    assert np.allclose(np.linalg.eigvalsh(context.local_projectors[0]), [0.0, 1.0, 1.0])


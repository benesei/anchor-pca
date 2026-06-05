from __future__ import annotations

import numpy as np
import pytest

from anchorpca import AnchorPCAInfty, AnchorPCALambda, pool_pca_from_covariances


def projector(directions):
    directions = np.asarray(directions, dtype=float)
    return directions @ directions.T


def subspace_distance(left, right):
    return np.linalg.norm(projector(left) - projector(right), ord="fro")


def canonical_covariances():
    return [
        np.diag([10.0, 9.0, 1.0, 1.0]),
        np.diag([10.0, 1.0, 8.0, 1.0]),
        np.diag([10.0, 1.0, 1.0, 7.0]),
    ]


def test_infty_matches_blockwise_cutoff_construction():
    model = AnchorPCAInfty(n_components=2, block_tol=1e-12)
    model.fit_covariances(canonical_covariances())

    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    assert subspace_distance(model.directions_, expected) < 1e-8
    assert np.allclose(model.rho_by_component_, np.array([1.0, 1.0 / 3.0]))
    assert np.allclose(model.variance_by_component_, np.array([10.0, 11.0 / 3.0]))
    assert model.invariant_dim_estimate_ == 1
    assert model.agreement_blocks_[0]["n_selected"] == 1
    assert model.agreement_blocks_[1]["n_selected"] == 1


def test_infty_invariant_dimension_estimate_is_first_block_dimension():
    model = AnchorPCAInfty(n_components=2, block_tol=0.7)
    model.fit_covariances(canonical_covariances())

    assert model.invariant_dim_estimate_ == model.agreement_blocks_[0]["dimension"]
    assert model.invariant_dim_estimate_ == 4


def test_infty_defaults_to_auto_block_tolerance_for_fit():
    X_envs = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 6.0, 0.0],
            ]
        ),
    ]
    model = AnchorPCAInfty(n_components=1)
    model.fit(X_envs)

    expected = min(0.05, 0.5 * 5.0 ** (-0.4))
    assert model.block_tol == "auto"
    assert model.block_tol_mode_ == "auto"
    assert np.isclose(model.block_tol_, expected)
    assert model.block_tol_alpha_ == 0.4
    assert model.block_tol_c_ == 0.5
    assert model.block_tol_max_ == 0.05


def test_infty_auto_block_tolerance_uses_covariance_observation_counts():
    covariances = canonical_covariances()
    n_obs = np.array([1000.0, 2000.0, 3000.0])
    model = AnchorPCAInfty(n_components=2)
    model.fit_covariances(covariances, n_obs=n_obs)

    expected = min(0.05, 0.5 * 1000.0 ** (-0.4))
    assert model.block_tol_mode_ == "auto"
    assert np.isclose(model.block_tol_, expected)


def test_infty_auto_block_tolerance_requires_real_observation_counts():
    model = AnchorPCAInfty(n_components=2)

    with pytest.raises(ValueError, match="requires observation counts"):
        model.fit_covariances(canonical_covariances())


def test_infty_auto_block_tolerance_validates_observation_counts():
    covariances = canonical_covariances()
    invalid_n_obs = [
        np.array([10.0, 0.0, 10.0]),
        np.array([10.0, np.nan, 10.0]),
        np.array([10.0, np.inf, 10.0]),
    ]

    for n_obs in invalid_n_obs:
        model = AnchorPCAInfty(n_components=2)
        with pytest.raises(ValueError, match="positive finite"):
            model.fit_covariances(covariances, n_obs=n_obs)


def test_infty_manual_block_tolerance_preserves_old_behavior():
    model = AnchorPCAInfty(n_components=2, block_tol=1e-6)
    model.fit_covariances(canonical_covariances())

    assert model.block_tol_ == 1e-6
    assert model.block_tol_mode_ == "manual"


def test_infty_block_tolerance_validation():
    covariances = canonical_covariances()
    invalid_models = [
        AnchorPCAInfty(n_components=2, block_tol=-1e-6),
        AnchorPCAInfty(n_components=2, block_tol="auto", block_tol_alpha=0.0),
        AnchorPCAInfty(n_components=2, block_tol="auto", block_tol_c=0.0),
        AnchorPCAInfty(n_components=2, block_tol="auto", block_tol_max=0.0),
        AnchorPCAInfty(n_components=2, block_tol="bad"),
    ]

    for model in invalid_models:
        with pytest.raises(ValueError):
            model.fit_covariances(covariances, n_obs=np.array([10.0, 10.0, 10.0]))


def test_large_lambda_converges_to_infty():
    covariances = canonical_covariances()
    finite = AnchorPCALambda(n_components=2, lambda_=1_000_000.0)
    hard = AnchorPCAInfty(n_components=2, block_tol=1e-12)

    finite.fit_covariances(covariances)
    hard.fit_covariances(covariances)

    assert subspace_distance(finite.directions_, hard.directions_) < 1e-6


def test_lambda_zero_matches_pool_pca():
    covariances = canonical_covariances()
    finite = AnchorPCALambda(n_components=2, lambda_=0.0)
    pool = pool_pca_from_covariances(covariances, n_components=2)

    finite.fit_covariances(covariances)

    assert subspace_distance(finite.directions_, pool["directions"]) < 1e-8


def test_fit_infers_observation_weights_from_environments():
    X_envs = [
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
    ]
    model = AnchorPCALambda(
        n_components=1,
        lambda_=1.0,
        weighting="observations",
    )
    model.fit(X_envs)

    assert np.allclose(model.n_obs_, np.array([3.0, 4.0]))
    assert np.allclose(model.weights_, np.array([3.0 / 7.0, 4.0 / 7.0]))


def test_public_attributes_do_not_use_old_parameter_names():
    model = AnchorPCALambda(n_components=2, lambda_=1.0)
    model.fit_covariances(canonical_covariances())

    public_names = [name for name in dir(model) if not name.startswith("__")]
    dropped_name = "t" + "au"
    assert not any(dropped_name in name.lower() for name in public_names)

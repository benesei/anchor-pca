from __future__ import annotations

import numpy as np
import pytest
import warnings as warnings_module

from anchorpca import (
    AnchorPCAInfty,
    SStarDimensionWarning,
    estimate_sstar_dimension,
    estimate_sstar_dimension_from_covariances,
)


def covariance_from_top_indices(p, top_indices):
    values = np.asarray([1.0 - 0.1 * j for j in range(p)], dtype=float)
    for rank, index in enumerate(top_indices):
        values[index] = 10.0 - rank
    return np.diag(values)


def sampled_known_m_environments(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    covariances = [
        covariance_from_top_indices(5, [0, 1, 2]),
        covariance_from_top_indices(5, [0, 1, 3]),
        covariance_from_top_indices(5, [0, 1, 4]),
    ]
    return [rng.multivariate_normal(np.zeros(5), covariance, size=n) for covariance in covariances]


def sampled_zero_m_environments(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    covariances = [
        covariance_from_top_indices(4, [0, 1]),
        covariance_from_top_indices(4, [2, 3]),
    ]
    return [rng.multivariate_normal(np.zeros(4), covariance, size=n) for covariance in covariances]


def test_estimate_sstar_dimension_recovers_known_positive_dimension():
    result = estimate_sstar_dimension(
        sampled_known_m_environments(),
        n_components=3,
        assume_gaussian=True,
    )

    assert result.m_hat == 2
    assert result.r_hat == 3
    assert result.tested_s.tolist() == [2, 3]
    assert result.stopped_early


def test_estimate_sstar_dimension_recovers_zero_dimension():
    result = estimate_sstar_dimension(
        sampled_zero_m_environments(),
        n_components=2,
        assume_gaussian=True,
    )

    assert result.m_hat == 0
    assert result.r_hat == 4
    assert not result.stopped_early


def test_dimension_testing_requires_gaussian_assumption():
    with pytest.raises(ValueError, match="assume_gaussian=True"):
        estimate_sstar_dimension(
            sampled_known_m_environments(n=20),
            n_components=3,
        )


def test_covariance_dimension_testing_requires_observation_counts():
    covariances = [
        covariance_from_top_indices(5, [0, 1, 2]),
        covariance_from_top_indices(5, [0, 1, 3]),
    ]

    with pytest.raises(ValueError, match="n_obs is required"):
        estimate_sstar_dimension_from_covariances(
            covariances,
            n_components=3,
            n_obs=None,
            assume_gaussian=True,
        )


def test_covariance_dimension_testing_checks_local_eigengap():
    covariances = [
        np.diag([5.0, 4.0, 4.0, 1.0]),
        np.diag([5.5, 4.2, 4.2, 1.2]),
    ]

    with pytest.raises(ValueError, match="eigengap is too small"):
        estimate_sstar_dimension_from_covariances(
            covariances,
            n_components=2,
            n_obs=[100.0, 100.0],
            assume_gaussian=True,
        )


def test_anchor_pca_infty_default_has_no_sstar_test_attributes():
    model = AnchorPCAInfty(n_components=3, block_tol=0.05).fit(
        sampled_known_m_environments()
    )

    assert model.invariant_dim_estimate_ == 2
    assert not hasattr(model, "sstar_dimension_test_result_")
    assert not hasattr(model, "sstar_dimension_test_mismatch_")


def test_anchor_pca_infty_warn_mode_stores_matching_test_result_without_warning():
    with warnings_module.catch_warnings(record=True) as caught_warnings:
        warnings_module.simplefilter("always")
        model = AnchorPCAInfty(
            n_components=3,
            block_tol=0.05,
            sstar_test_mode="warn",
            assume_gaussian=True,
        ).fit(sampled_known_m_environments(seed=2))

    assert len(caught_warnings) == 0
    assert model.invariant_dim_estimate_ == 2
    assert model.sstar_dimension_test_result_.m_hat == 2
    assert not model.sstar_dimension_test_mismatch_
    assert model.sstar_test_mode_ == "warn"
    assert model.sstar_test_alpha_ == 0.05


def test_anchor_pca_infty_warn_mode_warns_on_dimension_mismatch():
    with pytest.warns(SStarDimensionWarning, match="dim\\(S_star\\)"):
        model = AnchorPCAInfty(
            n_components=3,
            block_tol=1.0,
            sstar_test_mode="warn",
            assume_gaussian=True,
        ).fit(sampled_known_m_environments(seed=2))

    assert model.invariant_dim_estimate_ == 5
    assert model.sstar_dimension_test_result_.m_hat == 2
    assert model.sstar_dimension_test_mismatch_


def test_anchor_pca_infty_calibrate_mode_sets_first_block_to_test_dimension():
    model = AnchorPCAInfty(
        n_components=3,
        block_tol=1.0,
        sstar_test_mode="calibrate",
        assume_gaussian=True,
    ).fit(sampled_known_m_environments(seed=2))

    assert model.invariant_dim_estimate_uncalibrated_ == 5
    assert model.sstar_dimension_test_result_.m_hat == 2
    assert model.invariant_dim_estimate_ == 2
    assert model.block_tol_mode_ == "test_calibrated"
    assert model.sstar_dimension_test_mismatch_


def test_anchor_pca_infty_calibrate_mode_rejects_zero_test_dimension():
    with pytest.raises(ValueError, match="m_hat=0"):
        AnchorPCAInfty(
            n_components=2,
            block_tol=1.0,
            sstar_test_mode="calibrate",
            assume_gaussian=True,
        ).fit(sampled_zero_m_environments())


def test_anchor_pca_infty_test_mode_requires_gaussian_assumption_and_centering():
    with pytest.raises(ValueError, match="assume_gaussian=True"):
        AnchorPCAInfty(n_components=3, sstar_test_mode="warn").fit(
            sampled_known_m_environments(n=20)
        )

    with pytest.raises(ValueError, match="center=True"):
        AnchorPCAInfty(
            n_components=3,
            center=False,
            sstar_test_mode="warn",
            assume_gaussian=True,
        ).fit(sampled_known_m_environments(n=20))


def test_anchor_pca_infty_calibrate_mode_rejects_unseparated_barpi_boundary(monkeypatch):
    class DummyResult:
        m_hat = 1
        alpha = 0.05

    def fake_dimension_test(*args, **kwargs):
        return DummyResult()

    import anchorpca.estimators as estimators

    monkeypatch.setattr(
        estimators,
        "estimate_sstar_dimension_from_covariances",
        fake_dimension_test,
    )
    covariances = [
        covariance_from_top_indices(4, [0, 1]),
        covariance_from_top_indices(4, [0, 1]),
    ]

    with pytest.raises(ValueError, match="does not separate"):
        AnchorPCAInfty(
            n_components=2,
            block_tol=0.0,
            sstar_test_mode="calibrate",
            assume_gaussian=True,
        ).fit_covariances(covariances, n_obs=[100.0, 100.0])

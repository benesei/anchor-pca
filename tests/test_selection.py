from __future__ import annotations

import inspect

import numpy as np
import pytest

from anchorpca import (
    DEFAULT_TWO_E_LAMBDA_GRID,
    AnchorPCAInfty,
    AnchorPCALambda,
    fit_two_e_lambda_from_covariances,
    select_two_e_lambda_from_covariances,
)


def selection_covariances():
    return [
        np.diag([8.0, 5.0, 1.0]),
        np.diag([8.0, 1.0, 5.0]),
        np.diag([8.0, 4.0, 1.0]),
        np.diag([8.0, 1.0, 4.0]),
    ]


def test_default_grid_has_documented_endpoints():
    assert DEFAULT_TWO_E_LAMBDA_GRID[0] == 0.0
    assert np.isinf(DEFAULT_TWO_E_LAMBDA_GRID[-1])
    assert len(DEFAULT_TWO_E_LAMBDA_GRID) == 9


def test_finite_coefficient_uses_number_of_fitted_domains():
    model = fit_two_e_lambda_from_covariances(
        selection_covariances()[:3],
        n_components=1,
        two_e_lambda=12.0,
    )

    assert isinstance(model, AnchorPCALambda)
    assert model.lambda_ == 2.0
    assert model.two_e_lambda_ == 12.0


def test_zero_and_infinity_use_the_intended_estimators():
    covariances = selection_covariances()
    counts = [20, 21, 22, 23]

    zero = fit_two_e_lambda_from_covariances(
        covariances, 1, 0.0, n_obs=counts
    )
    infinity = fit_two_e_lambda_from_covariances(
        covariances, 1, np.inf, n_obs=counts
    )

    assert isinstance(zero, AnchorPCALambda)
    assert zero.lambda_ == 0.0
    assert isinstance(infinity, AnchorPCAInfty)


def test_infinity_auto_tolerance_requires_observation_counts():
    with pytest.raises(ValueError, match="requires observation counts"):
        fit_two_e_lambda_from_covariances(
            selection_covariances(),
            1,
            np.inf,
        )


def test_selection_converts_the_coefficient_within_each_fold(monkeypatch):
    fitted = []
    original_fit = AnchorPCALambda.fit_covariances

    def recording_fit(self, covariances, *, n_obs=None):
        fitted.append((len(covariances), self.lambda_))
        return original_fit(self, covariances, n_obs=n_obs)

    monkeypatch.setattr(AnchorPCALambda, "fit_covariances", recording_fit)
    select_two_e_lambda_from_covariances(
        selection_covariances(),
        1,
        two_e_lambda_grid=(0.0, 8.0, np.inf),
        n_obs=[20, 21, 22, 23],
    )

    fold_fits = [value for n_domains, value in fitted if n_domains == 3]
    refits = [value for n_domains, value in fitted if n_domains == 4]
    assert fold_fits.count(8.0 / 6.0) == 4
    assert refits == [0.0]


def test_selection_returns_scores_and_refits_on_all_sources():
    covariances = selection_covariances()
    counts = np.array([20.0, 30.0, 40.0, 50.0])
    result = select_two_e_lambda_from_covariances(
        covariances,
        1,
        two_e_lambda_grid=(0.0, 1.0, np.inf),
        weighting="observations",
        n_obs=counts,
    )

    assert result.fold_scores.shape == (3, 4)
    assert result.mean_scores.shape == (3,)
    assert result.standard_errors.shape == (3,)
    assert result.candidates == (0.0, 1.0, np.inf)
    assert result.estimator.n_environments_ == 4
    assert np.allclose(result.estimator.n_obs_, counts)
    assert np.allclose(result.estimator.weights_, counts / counts.sum())
    assert result.estimator.two_e_lambda_ == result.selected_two_e_lambda


def test_grid_order_breaks_exact_ties_deterministically():
    covariances = [np.eye(3), np.eye(3), np.eye(3)]
    result = select_two_e_lambda_from_covariances(
        covariances,
        2,
        two_e_lambda_grid=(10.0, 0.0, np.inf),
        n_obs=[20, 20, 20],
        block_tol=1e-12,
    )

    assert np.allclose(result.fold_scores, 2.0 / 3.0)
    assert result.selected_index == 0
    assert result.selected_two_e_lambda == 10.0


@pytest.mark.parametrize(
    "grid, message",
    [
        ((0.0, 1.0), "exactly one infinity"),
        ((0.0, np.inf, np.inf), "exactly one infinity"),
        ((0.0, -1.0, np.inf), "nonnegative"),
        ((0.0, np.nan, np.inf), "nonnegative"),
        ((0.0, 0.0, np.inf), "duplicate"),
    ],
)
def test_grid_validation(grid, message):
    with pytest.raises(ValueError, match=message):
        select_two_e_lambda_from_covariances(
            selection_covariances(),
            1,
            two_e_lambda_grid=grid,
            n_obs=[20, 20, 20, 20],
        )


def test_selection_requires_three_sources_and_valid_inputs():
    with pytest.raises(ValueError, match="at least three"):
        select_two_e_lambda_from_covariances(
            selection_covariances()[:2],
            1,
            n_obs=[20, 20],
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        fit_two_e_lambda_from_covariances(
            [np.diag([1.0, -1.0])],
            1,
            0.0,
        )


def test_selection_interface_has_no_target_data_argument():
    parameters = inspect.signature(select_two_e_lambda_from_covariances).parameters
    assert "target_covariances" not in parameters
    assert "target_data" not in parameters

# `anchorpca` Package API

This package exposes the estimators and utilities used in the paper. All arrays
are converted to `float` NumPy arrays internally. The paper uses the term
`domain`; the Python API uses `environment` in argument names and helper
functions.

## Data Input

Raw-data methods expect a non-empty list of domain/environment matrices:

```python
X_envs = [X_1, X_2, ..., X_E]  # X_e has shape (n_e, p)
```

Covariance methods expect a non-empty list of square covariance matrices with a
common shape `(p, p)`:

```python
covariances = [Sigma_1, Sigma_2, ..., Sigma_E]
n_obs = [n_1, n_2, ..., n_E]  # required for observation weighting and auto tolerance
```

For raw data, empirical covariances use the unbiased denominator `n_e - 1`.

## `AnchorPCALambda`

```python
AnchorPCALambda(n_components, lambda_, *, center=True, weighting="uniform")
```

Finite-penalty Anchor PCA. It computes local top-`k` projectors, forms
`barSigma` and `barPi`, and diagonalizes

```text
M_lambda = barSigma + 2 * E * lambda_ * barPi.
```

Parameters:

- `n_components`: representation dimension `k`; must satisfy `1 <= k <= p`.
- `lambda_`: nonnegative finite-penalty value.
- `center`: whether `.fit(X_envs)` centers each environment before computing its
  covariance. Default: `True`.
- `weighting`: domain/environment weighting scheme. Default: `"uniform"`.
  - `"uniform"`: weights are `1/E`.
  - `"observations"`: weights are `n_e / sum_j n_j`; requires observation
    counts, supplied automatically by `.fit(X_envs)` or explicitly via
    `.fit_covariances(..., n_obs=...)`.

Fit methods:

- `.fit(X_envs)`: computes empirical covariances from raw environments and fits
  the estimator.
- `.fit_covariances(covariances, *, n_obs=None)`: fits from covariance matrices.
- `.fit_transform(X_envs)`: fits and returns transformed environments.

Other methods:

- `.transform(X, mean=None)`: projects rows of `X` onto `directions_`; subtracts
  `mean` first when provided.
- `.explained_variance(covariances=None, weights=None)`: returns per-environment,
  weighted average, and worst-case explained variance for the fitted directions.

## `AnchorPCAInfty`

```python
AnchorPCAInfty(
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
)
```

Python class implementing the paper's hard-agreement estimator
`AnchorPCA_infty`. It groups numerically close eigenvalues of
`barPi` into agreement blocks, visits blocks in decreasing agreement, and
diagonalizes `barSigma` within each selected block.

Shared parameters:

- `n_components`, `center`, and `weighting`: same meaning as for
  `AnchorPCALambda`.

Block-tolerance parameters:

- `block_tol`: either a nonnegative numeric scalar or `"auto"`. Default:
  `"auto"`.
- `block_tol="auto"` requires observation counts and sets
  `min(block_tol_max, block_tol_c * n_min ** (-block_tol_alpha))`, where
  `n_min` is the smallest environment sample size.
- `block_tol_alpha`: positive exponent in the auto tolerance. Default: `0.4`.
- `block_tol_c`: positive multiplier in the auto tolerance. Default: `0.5`.
- `block_tol_max`: positive cap in the auto tolerance. Default: `0.05`.

Optional Gaussian `S_star` dimension diagnostic:

- `sstar_test_mode`: one of `"off"`, `"warn"`, or `"calibrate"`. Default:
  `"off"`.
- `assume_gaussian`: must be `True` whenever `sstar_test_mode != "off"`.
- `sstar_test_alpha`: significance level for the sequential diagnostic.
  Default: `0.05`.
- In `"warn"` mode, the estimator stores the test result and raises
  `SStarDimensionWarning` if the test estimate differs from the first empirical
  agreement-block dimension.
- In `"calibrate"` mode, the test estimate is used to choose a first-block
  tolerance. This mode changes the estimator and raises an error if calibration
  is impossible, for example when the test estimates dimension zero.

Fit and transform methods are the same as for `AnchorPCALambda`. When fitting
raw data with `sstar_test_mode != "off"`, `center=True` is required.

## Fitted Attributes

Both estimators store:

- `n_components_`, `n_features_`, `n_environments_`
- `n_obs_`, `weights_`, `weighting_`
- `covariances_`
- `barSigma_`, `barPi_`
- `local_directions_`, `local_projectors_`
- `directions_`: learned directions as columns, shape `(p, k)`
- `components_`: learned directions as rows, shape `(k, p)`
- `projection_`: projector onto `directions_`

`AnchorPCALambda` additionally stores:

- `M_lambda_`
- `eigenvalues_`

`AnchorPCAInfty` additionally stores:

- `block_tol_`, `block_tol_mode_`
- `block_tol_alpha_`, `block_tol_c_`, `block_tol_max_`
- `agreement_blocks_`
- `invariant_dim_estimate_`: dimension of the first empirical agreement block
- `rho_by_component_`, `variance_by_component_`
- `barPi_eigenvalues_`

If `sstar_test_mode != "off"`, `AnchorPCAInfty` also stores:

- `sstar_dimension_test_result_`
- `sstar_dimension_test_mismatch_`
- `sstar_test_mode_`, `sstar_test_alpha_`
- `invariant_dim_estimate_uncalibrated_` in `"calibrate"` mode

## `FindS_star` and the Sequential Dimension Diagnostic

In the paper, `FindS_star` denotes the full procedure
that first estimates `m = dim(S_star)` and then returns the leading `m_hat`
eigenspace of the empirical projector average `barPi`. The package exposes the
dimension-testing core through the functions below. The experiment scripts use
these functions and then form the corresponding leading eigenspace of `barPi` for
the plotted `FindS_star` subspace diagnostic.

```python
estimate_sstar_dimension(
    X_envs,
    n_components,
    *,
    alpha=0.05,
    center=True,
    assume_gaussian=False,
    check_gaps=True,
    eig_rtol=1e-10,
    eig_atol=1e-12,
)
```

```python
estimate_sstar_dimension_from_covariances(
    covariances,
    n_components,
    *,
    n_obs,
    alpha=0.05,
    assume_gaussian=False,
    check_gaps=True,
    eig_rtol=1e-10,
    eig_atol=1e-12,
)
```

These functions estimate `m = dim(S_star)` using sequential bottom-space
Schott/partial-CPCA tests. They use a Gaussian fixed-`p`, large-sample
calibration and require `assume_gaussian=True`. The covariance-based function
also requires `n_obs` because the test is defined for empirical covariances, not
population covariances without sample sizes.

The returned `SStarDimensionTestResult` contains:

- `m_hat`, `r_hat`
- `alpha`
- `tested_s`, `p_values`, `statistics`
- `stopped_early`
- `schott_results`, a list of one-null `SchottTestResult` objects

## Utilities

- `empirical_covariance(X, center=True)`: returns covariance, mean, and sample
  size for one environment.
- `empirical_covariances_from_envs(X_envs, center=True)`: returns covariances,
  means, and sample sizes for all environments.
- `pool_pca_from_envs(...)` and `pool_pca_from_covariances(...)`: pooled PCA
  baselines using the same covariance and weighting conventions.
- `explained_variance(directions, covariances, weights=None)`: returns
  `per_env`, `average`, `worst_case`, and `worst_env` explained variance.

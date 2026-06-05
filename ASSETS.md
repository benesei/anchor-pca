# Existing Assets and Licenses

This file records the external assets used by the Anchor PCA reproduction code.

## Code in This Repository

- Asset: `anchorpca` package and experiment scripts.
- License: MIT, see `LICENSE`.
- Generated synthetic designs and figures are reproducible outputs created by the
  scripts and are not separately released as a new dataset.

## UCI Gas Sensor Dataset

- Asset: Gas Sensor Array Drift Dataset at Different Concentrations.
- Source: UCI Machine Learning Repository, https://doi.org/10.24432/C5MK6M.
- DOI: `10.24432/C5MK6M`.
- License: CC BY 4.0.
- Use in this repository: downloaded by
  `experiments/real_world/gas_sensor/compute_rolling_split_explained_variance.py`
  if not present locally; the script verifies SHA256
  `98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e`.
- Data type: sensor measurements; no human-subject records are used.
- Redistribution: the code downloads from the original source instead of storing
  a copy of the dataset in this repository.

## External `minPCA` Package

- Asset: external `minPCA` implementation used for wcPCA/`norm-maxRegret`
  baselines.
- Source: `https://github.com/anyafries/minPCA`, pinned at commit
  `9b90ebd0b56fc8eb88a7217e38c49b35a6927f82`.
- License observed from installed package metadata: GNU Affero General Public
  License v3.0.
- Use in this repository: optional external dependency for the perturbation-path
  wcPCA baselines and the gas-sensor `norm-maxRegret` baseline. The package is
  not vendored and no cached baseline outputs are used.

## Principal Python Dependencies

The tested reproduction environment is pinned in `requirements-repro.txt`. The
main Python packages used directly by the repository are:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `pytest`
- `torch` through the external `minPCA` dependency

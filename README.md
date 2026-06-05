# Anchor PCA

This repository contains the Python implementation and experiment code for the paper
**Anchor PCA** by Benedikt Seiter, Anya Fries, Julius von Kügelgen, and Jonas Peters.

Paper: https://arxiv.org/abs/2606.06233

## Overview

Anchor PCA is a covariance-based dimension reduction method for multiple
domains. The Python API uses the name `environment` for a domain, for example
`X_envs = [X_1, ..., X_E]`. The method trades off pooled explained variance
against agreement with the local top principal subspaces of the domains.

The package implements:

- `AnchorPCALambda`: finite projector-agreement penalty `lambda_`;
- `AnchorPCAInfty`: Python class implementing the paper's hard-agreement
  estimator `AnchorPCA_infty`;
- the dimension-testing core of the paper's `FindS_star` procedure via
  `estimate_sstar_dimension(...)` for Gaussian data;
- poolPCA baselines and explained-variance utilities.

Generated experiment data, result CSVs, and figures are ignored by git. They can
be regenerated from the scripts in `experiments/`; see `experiments/README.md`
for the commands and expected outputs.

## Installation

Use Python 3.9 or newer. The code was tested with Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[examples,dev]"
```

For a stricter reproduction environment, install the tested package versions:

```bash
python -m pip install -r requirements-repro.txt
```

The perturbation-path and gas-sensor experiments use the external `minPCA`
implementation for the wcPCA/`norm-maxRegret` baselines. If you did not install
`requirements-repro.txt`, install the pinned version separately:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "git+https://github.com/anyafries/minPCA.git@9b90ebd0b56fc8eb88a7217e38c49b35a6927f82"
```

No GPU is required by the AnchorPCA code. The external baseline uses PyTorch
through its own package implementation.

## Quick Start

```python
from anchorpca import AnchorPCALambda, AnchorPCAInfty

X_envs = [X_env_1, X_env_2, X_env_3]

finite = AnchorPCALambda(n_components=3, lambda_=25.0).fit(X_envs)
hard = AnchorPCAInfty(n_components=3).fit(X_envs)

print(finite.explained_variance())
print(hard.explained_variance())
```

## Package API

The main estimators are:

- `AnchorPCALambda(n_components, lambda_)`, the finite projector-agreement
  penalty estimator;
- `AnchorPCAInfty(n_components)`, the Python class implementing the paper's
  hard-agreement estimator `AnchorPCA_infty`;
- `estimate_sstar_dimension(...)`, the optional Gaussian dimension-testing core
  of the paper's `FindS_star` procedure.

Both estimators accept either raw domain/environment data via `.fit(X_envs)` or
precomputed covariance matrices via `.fit_covariances(...)`. Fitted objects expose
learned directions through `directions_`, row-wise components through
`components_`, the learned projector through `projection_`, and diagnostic
quantities such as `barSigma_`, `barPi_`, and local projectors.

See `src/anchorpca/README.md` for constructor parameters, defaults, fitted
attributes, and utility functions.

## Repository Structure

```text
AnchorPCA/
├── src/anchorpca/              # installable methods package
├── tests/                      # package regression tests
├── experiments/                # paper experiment reproduction scripts
├── ASSETS.md                   # external data/code asset notes
└── requirements-repro.txt      # tested reproduction package versions
```

## Reproducing Paper Results

The detailed experiment entry point is `experiments/README.md`. It maps each
paper figure/table to the script that generates it and gives smoke-test commands
for checking the pipeline.

Minimal package check:

```bash
python -m pytest -q
```

Full reproduction creates ignored `data/`, `results/`, and `figures/` folders
inside `experiments/`.

## Data and Licenses

Synthetic distribution draws are generated deterministically from stored seeds.
The gas sensor experiment downloads the UCI Gas Sensor Array Drift Dataset at
Different Concentrations from https://doi.org/10.24432/C5MK6M, released under CC
BY 4.0. The repository code is MIT licensed. Existing assets and licenses are
summarized in `ASSETS.md`.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@misc{seiter2026anchorpca,
  title         = {Anchor PCA},
  author        = {Seiter, Benedikt and Fries, Anya and von K{\"u}gelgen, Julius and Peters, Jonas},
  year          = {2026},
  eprint        = {2606.06233},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2606.06233}
}
```

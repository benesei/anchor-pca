<h1 align="center">Anchor PCA</h1>

<p align="center">
  <a href="https://benesei.github.io/">Benedikt Seiter</a>, Anya Fries,
  <a href="https://www.juliusvonkugelgen.com/">Julius von Kügelgen</a>, and
  <a href="https://people.math.ethz.ch/~jopeters/">Jonas Peters</a>
</p>

<p align="center">ETH Zürich</p>

<p align="center">
  <a href="https://neurips.cc/virtual/2026/loc/paris/poster/153959"><img src="https://img.shields.io/badge/NeurIPS-2026-68408A" alt="NeurIPS 2026"></a>
  <a href="https://arxiv.org/abs/2606.06233"><img src="https://img.shields.io/badge/arXiv-2606.06233-b31b1b" alt="arXiv:2606.06233"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#video">Video</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#reproduce-the-paper">Reproduction</a> ·
  <a href="ASSETS.md">Data</a> ·
  <a href="#citation">Citation</a>
</p>

Anchor PCA learns a low-dimensional representation from several source domains.
It trades explained variance against agreement of their principal subspaces.

<a name="video"></a>
<details>
<summary>Watch a short visual introduction (86 seconds)</summary>

A visual introduction to Anchor PCA using a simple illustrative example.

https://github.com/user-attachments/assets/53257604-9c41-495d-94a8-ddcf5bb5dc1e

</details>

## Installation

For the paper experiments, use Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install .
```

The core package requires only NumPy and SciPy. For use without the experiment
dependencies, install with `python -m pip install .`.

## Usage

```python
from anchorpca import AnchorPCALambda, AnchorPCAInfty

# Each array contains observations from one source domain.
X_sources = [X_source_1, X_source_2, X_source_3]
finite = AnchorPCALambda(n_components=3, lambda_=25).fit(X_sources)
infinite = AnchorPCAInfty(n_components=3).fit(X_sources)
Z = X_new @ finite.directions_
```

Choose the representation dimension `k` separately. At fixed `k`, the package
can select `2 * E * lambda_` by leave-one-source-domain-out explained variance:

```python
from anchorpca import select_two_e_lambda_from_covariances

selection = select_two_e_lambda_from_covariances(
    source_covariances, n_components=3, n_obs=source_sample_sizes
)
model = selection.estimator
print(selection.selected_two_e_lambda)
```

Target data are not used for selection. When learning preprocessing from raw
data, refit it within each validation fold, as in the gas-sensor experiment.
See the [API notes](src/anchorpca/README.md) for parameters and fitted attributes.

## Reproduce the paper

Regenerate figures and table summaries from the saved paper results:

```bash
python experiments/reproduce.py --from-reference --output-dir outputs/reference
```

Rerun every paper experiment, including the full simulations and classification:

```bash
python experiments/reproduce.py --recompute --output-dir outputs/recomputed
```

Choose an empty output directory for each run. The full run downloads the two
real datasets and takes substantially longer than replotting. No GPU is needed.
The paper experiments used OpenBLAS; other numerical backends may produce minor
numerical differences.

The [experiment guide](experiments/README.md) maps every figure and table to its
code and describes the protocols. [Data and licenses](ASSETS.md) lists sources
and attribution. Fama–French may revise historical returns, so a fresh download
can give different numbers from the original paper data.

## Repository

- `src/anchorpca/`: estimators, selection, and supporting functions.
- `experiments/`: paper experiments, plotting, and compact reference results.
- `tests/`: method and experiment checks (`python -m pytest`).

## Citation

```bibtex
@inproceedings{seiter2026anchor,
  title = {Anchor PCA},
  author = {Seiter, Benedikt and Fries, Anya and von K{\"u}gelgen, Julius and Peters, Jonas},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```

The code is released under the [MIT license](LICENSE). External data and the
minPCA baseline retain their own terms, listed in [ASSETS.md](ASSETS.md).

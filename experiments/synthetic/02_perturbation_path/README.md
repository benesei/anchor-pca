# Perturbation Path

Runtime category: quick to medium with the pinned external `minPCA`
dependency.

This folder reproduces the perturbation-path plot for the four-dimensional
motivating example.

Install `minPCA` before running the default paper pipeline:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  "git+https://github.com/anyafries/minPCA.git@9b90ebd0b56fc8eb88a7217e38c49b35a6927f82"
```

Run from the repository root:

```bash
python experiments/synthetic/02_perturbation_path/run_perturbation_path.py
```

The script rebuilds the population covariances, fits fixed projectors for
`poolPCA`, `AnchorPCA_lambda=25`, `AnchorPCA_infty`, and the wcPCA baselines,
then evaluates average reconstruction error along
`Sigma_e(rho) = Sigma_e + rho Pi_k^(e)`.

Default perturbation and optimizer settings:

- `rho_max = 500`
- `rho_step = 0.25`
- wcPCA fit source: external `minPCA`
- `minpca_restarts = 60`
- `minpca_iters = 1800`
- `minpca_lr = 0.05`
- `minpca_seed = 0`

Output used in the paper:

- `figures/perturbation_path.(png|pdf)`

Result CSVs in `results/` store the fitted line parameters, full plotted
curves, adjacent lower-envelope crossovers, finite pairwise crossovers, and
best-method intervals.

For a package-free diagnostic smoke run, use the deterministic four-dimensional
grid solver:

```bash
python experiments/synthetic/02_perturbation_path/run_perturbation_path.py --wc-source grid
```

The default paper pipeline uses the pinned external dependency above; the grid
solver is documented only as a lightweight diagnostic.

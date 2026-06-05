# Gas Sensor Drift Pipeline

This folder contains the gas-sensor drift experiment from the paper. Temporal
batches of the UCI Gas Sensor Array Drift Dataset are treated as domains
(environments in the Python API); methods are fit on source batches `B1--Bs` and
evaluated on source and held-out target batches `B(s+1)--B10`.

Generated `data/`, `results/`, and `figures/` directories are ignored by git.
Install dependencies from the repository root as described in `../../../README.md`.
The `norm-maxRegret` baseline additionally requires the external `minPCA`
package listed there.

## Data

The compute script downloads the UCI archive if it is not present locally and
verifies SHA256
`98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e` by
default. Dataset page: https://doi.org/10.24432/C5MK6M; license: CC BY 4.0. The
data contain sensor measurements, not human-subject records.

## Full Pipeline

Runtime category: expensive because of the external `norm-maxRegret` baseline.

```bash
python experiments/real_world/gas_sensor/compute_rolling_split_explained_variance.py
python experiments/real_world/gas_sensor/plot_source_target_EV.py
python experiments/real_world/gas_sensor/plot_rolling_split_summary.py
python experiments/real_world/gas_sensor/plot_sstar_poolpca_target_batches.py
```

Publication defaults:

- methods: `poolPCA`, `AnchorPCA_lambda=1`, `AnchorPCA_lambda=10`,
  `AnchorPCA_infty`, and `norm-maxRegret`;
- dimensions: `k = 5, 10, 20, 30, 40`;
- rolling splits: last source batch `s = 3, 4, 5, 6, 7, 8, 9`;
- preprocessing: source-batch standardization only;
- `norm-maxRegret`: `n_restarts=10`, `n_iters=2000`, `lr=0.01`.

Outputs used in the paper:

- `figures/gas_sensor_class_composition_by_batch.(png|pdf)`
- `figures/gas_sensor_publication_b1_b6_source_target_k20.(png|pdf)`
- `figures/rolling_publication_target_ev_combined.(png|pdf)`
- `figures/sstar_poolpca_same_dim_b9_b10_ev.(png|pdf)`
- `results/rolling_publication_explained_variance_all.csv`
- `results/rolling_publication_anchor_infty_sstar_all.csv`
- `results/rolling_publication_metadata.json`

Target batches are used only for explained-variance evaluation and plotting.
The source/target plot defaults to `s=6`, `k=20`; the rolling summary shows
`k = 10, 20, 30` and `s = 3, ..., 8`; the B9/B10 diagnostic compares the first
empirical agreement block of `AnchorPCA_infty` with same-dimensional `poolPCA`.

## Smoke Run

```bash
python experiments/real_world/gas_sensor/compute_rolling_split_explained_variance.py \
  --k-values 10 --last-source-batches 3 --minpca-restarts 1 --minpca-iters 25
python experiments/real_world/gas_sensor/plot_source_target_EV.py \
  --k 10 --last-source-batch 3
```

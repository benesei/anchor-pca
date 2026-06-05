# Experiments

This directory is the main entry point for reproducing the results and plots in
the paper "Anchor PCA". Run commands from the repository root. Generated
`data/`, `results/`, and `figures/` directories are ignored by git.

Install the package and dependencies from the repository root as described in
`../README.md`. Existing data/code assets and licenses are summarized in
`../ASSETS.md`.

## Paper-to-Code Map

The following list contains only outputs used in the current paper. Some scripts
generate extra diagnostics; those are documented in the subfolder READMEs.

- **Motivating example.**
  - Script: `synthetic/01_motivating_example_4d/run_motivating_example.py`
  - Outputs: `synthetic/01_motivating_example_4d/figures/4d_motivation_exp_quotient_view.pdf`,
    `synthetic/01_motivating_example_4d/results/paper_table_check.json`

- **Optimal reconstruction under perturbations.**
  - Script: `synthetic/02_perturbation_path/run_perturbation_path.py`
  - Output: `synthetic/02_perturbation_path/figures/perturbation_path.pdf`

- **Recovering `S_star` in random-subspace simulations.**
  - Folder: `synthetic/03_random_subspace_sstar/`
  - Scripts: `generate_design_bank.py`, `run_infty_sstar_recovery.py`,
    `run_main_seqtest_dimension.py`, `plot_infty_sstar_recovery.py`
  - Main output: `figures/infty_sstar_recovery_gaussian_g5_p10_k5_m2_main.pdf`
  - Appendix outputs: `figures/infty_sstar_recovery_gaussian_mixture_g5_p10_k5_m2_main.pdf`,
    `figures/infty_sstar_recovery_gaussian_g5_p8_k5_m1.pdf`,
    `figures/infty_sstar_recovery_g2_p8_k5_m2_combined_extended_n100000.pdf`

- **Gas-sensor drift experiment.**
  - Folder: `real_world/gas_sensor/`
  - Scripts: `compute_rolling_split_explained_variance.py`,
    `plot_source_target_EV.py`, `plot_rolling_split_summary.py`,
    `plot_sstar_poolpca_target_batches.py`
  - Main output: `figures/gas_sensor_publication_b1_b6_source_target_k20.pdf`
  - Appendix outputs: `figures/gas_sensor_class_composition_by_batch.pdf`,
    `figures/rolling_publication_target_ev_combined.pdf`,
    `figures/sstar_poolpca_same_dim_b9_b10_ev.pdf`

## Smoke Commands

These commands check the full pipeline structure with small synthetic runs.
They do not regenerate the full paper-quality Monte Carlo results.

```bash
python experiments/synthetic/01_motivating_example_4d/run_motivating_example.py
python experiments/synthetic/02_perturbation_path/run_perturbation_path.py --wc-source grid
python experiments/synthetic/03_random_subspace_sstar/generate_design_bank.py --n-draws-per-config 2 --overwrite
python experiments/synthetic/03_random_subspace_sstar/run_infty_sstar_recovery.py --max-designs 2 --sample-reps 2 --n-grid 50 100 --overwrite
python experiments/synthetic/03_random_subspace_sstar/run_main_seqtest_dimension.py --max-designs 2 --sample-reps 2 --n-grid 50 100 --overwrite
python experiments/synthetic/03_random_subspace_sstar/run_g2_low_environment_stress.py --max-designs 2 --sample-reps 2 --n-grid 50 100 --overwrite
python experiments/synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py
```

Gas-sensor smoke run:

```bash
python experiments/real_world/gas_sensor/compute_rolling_split_explained_variance.py \
  --k-values 10 --last-source-batches 3 --minpca-restarts 1 --minpca-iters 25
python experiments/real_world/gas_sensor/plot_source_target_EV.py \
  --k 10 --last-source-batch 3
```

## Reproduction Checks

After a full reproduction run:

```bash
python experiments/check_reproduction_outputs.py --require-generated
python -m pytest -q
```

The output check is non-mutating. It verifies that paper figures and upstream
CSV/metadata files exist and, when possible, checks the expected configurations,
grids, easy/hard settings, and row counts.

## Data Notes

The synthetic experiments generate all distribution draws and empirical samples
from fixed seeds. The gas-sensor experiment downloads the UCI Gas Sensor Array
Drift Dataset at Different Concentrations from https://doi.org/10.24432/C5MK6M,
released under CC BY 4.0. The gas data contain sensor measurements, not
human-subject records.

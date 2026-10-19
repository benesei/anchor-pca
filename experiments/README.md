# Paper experiments

Run commands from the repository root after installing the dependencies in the
[main README](../README.md). The runner reproduces every numbered figure and
table, including the appendix.

```bash
python experiments/reproduce.py --from-reference --output-dir outputs/reference
python experiments/reproduce.py --recompute --output-dir outputs/recomputed
```

The first command uses the compact CSV/JSON files in `reference/` and requires
no data download. The second reruns the complete experiments with the paper's
seeds and sample counts. Each output directory must be empty initially.
Figures are written to `figures/`, table summaries to `tables/`, and computation
outputs to their experiment subdirectories. `verification.json` records the
comparison with the paper results; `reproduction.log` contains execution details.

## Paper-to-code map

Paths below are relative to `experiments/`. The exact output names are listed in
[paper_outputs.json](paper_outputs.json).

| Figure | Content | Code |
| --- | --- | --- |
| 1 | Four-dimensional motivating example | `synthetic/01_motivating_example_4d/run_motivating_example.py` |
| 2 | Covariance perturbation path | `synthetic/02_perturbation_path/run_perturbation_path.py`, `plot_perturbation_path.py` |
| 3, 5, 6 | Gaussian and Gaussian-mixture subspace recovery | `synthetic/03_random_subspace_sstar/run_main_seqtest_dimension.py`, `plot_infty_sstar_recovery.py` |
| 7 | Small-domain recovery and agreement gaps | `synthetic/03_random_subspace_sstar/run_g2_low_environment_stress.py`, `plot_infty_sstar_recovery.py` |
| 4 | Gas source/target explained variance | `real_world/gas_sensor/compute_rolling_split_explained_variance.py`, `select_penalty.py`, `plot_source_target_EV.py` |
| 8 | Gas composition | `real_world/gas_sensor/plot_gas_composition.py` |
| 9 | Gas rolling source/target splits | `real_world/gas_sensor/plot_rolling_split_summary.py` |
| 10 | Gas same-dimensional subspace comparison | `real_world/gas_sensor/plot_sstar_poolpca_target_batches.py` |
| 11 | Fama–French source scree curves | `real_world/fama_french/run_temporal_experiment.py`, `plot_source_scree.py` |

| Table | Content | Computation |
| --- | --- | --- |
| 1 | Motivating-example reconstruction errors | `synthetic/01_motivating_example_4d/run_motivating_example.py` |
| 2 | Independent unseen-domain simulation | `synthetic/03_random_subspace_sstar/run_unseen_target_evaluation.py` |
| 3 | B9 dominant-direction diagnostic | `real_world/gas_sensor/diagnose_b9_shift.py` |
| 4 | Gas weighting sensitivity | `real_world/gas_sensor/evaluate_weighting.py` |
| 5, 6 | Gas classification summary and per-batch accuracy | `real_world/gas_sensor/run_classification.py` |
| 7, 8 | Fama–French target EV and rank sensitivity | `real_world/fama_french/run_temporal_experiment.py` |

`paper_tables.py` exports these numerical summaries. The design bank also
reproduces the population agreement-gap summary in the recovery appendix.

## Individual experiments

Each experiment has a short README with its protocol and standalone commands:

- [Motivating example](synthetic/01_motivating_example_4d/README.md)
- [Perturbation path](synthetic/02_perturbation_path/README.md)
- [Subspace recovery](synthetic/03_random_subspace_sstar/README.md)
- [Gas sensor](real_world/gas_sensor/README.md)
- [Fama–French](real_world/fama_french/README.md)

The scripts accept output paths; use those options to keep large generated
files outside the repository. Reduced runs are covered by the tests and do not
replace full paper reproduction.

## Checks

```bash
python -m pytest
python experiments/check_reproduction_outputs.py --output-dir outputs/recomputed
```

The checker covers output completeness, saved numerical results, source-only
penalty selection, and the paper's configurations. For a revised Fama–French
download, it records differences separately. See [data and licenses](../ASSETS.md)
for provenance and the reason historical returns can change.

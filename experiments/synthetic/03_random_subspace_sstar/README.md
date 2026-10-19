# Invariant-subspace recovery

The paper uses `(E, p, k, m) = (5, 10, 5, 2), (5, 8, 5, 1), (2, 8, 5, 2)`.
Each configuration has easy and hard eigenvalue settings, with 100 independent
population draws per setting (seed 42).
The code calls these settings `balanced` and `threshold_stress`, respectively.

For each draw and sample size, we generate 20 datasets (seed 43). The standard
sample sizes are 50, 100, 200, 500, 1,000, 2,000, and 5,000 observations per
domain. The main configuration uses Gaussian and covariance-preserving
Gaussian-mixture data; the other configurations use Gaussian data. The
small-domain experiment additionally uses 10,000, 30,000, and 100,000 observations.
Anchor PCA and the Gaussian-calibrated `FindS_star` test are evaluated on the
same samples. The test level is 0.05.

```bash
python experiments/synthetic/03_random_subspace_sstar/generate_design_bank.py
python experiments/synthetic/03_random_subspace_sstar/run_main_seqtest_dimension.py
python experiments/synthetic/03_random_subspace_sstar/run_g2_low_environment_stress.py
python experiments/synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py
```

The plotter produces only Figures 3 and 5–7. Sampling and summarization helpers
are shared in `recovery_utils.py`; the two runners do not duplicate experiments.
The design summary also supplies the appendix's population agreement gaps.

## Independent unseen domain

Table 2 uses Gaussian mixtures with five source domains and one independent
target domain, `p=10`, `k=5`, and `m=2`. It uses 50 population draws per easy/hard
setting, five source-sample repetitions per draw, and 500 observations per
source. Each metric is averaged over repetitions within a draw, then over the
50 draws. Target EV uses the target population covariance. Projector error
compares each method's first two directions with the true invariant subspace.

```bash
python experiments/synthetic/03_random_subspace_sstar/run_unseen_target_evaluation.py
```

Use each script's `--output-dir` option to choose another location. Recovery
runners also accept `--design-bank-dir`; the plotter accepts `--results-dir`,
`--main-seqtest-results-dir`, `--stress-results-dir`, and `--figures-dir`.
The complete runner in the experiment guide sets these paths automatically.

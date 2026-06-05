# Random-Subspace Synthetic Experiments

This folder contains the random-subspace distribution-draw bank used for the
empirical `AnchorPCAInfty` recovery experiments in the paper.

Generated `results/` and `figures/` directories are ignored by git.

## Distribution-Draw Bank

```bash
python experiments/synthetic/03_random_subspace_sstar/generate_design_bank.py --overwrite
```

Defaults: seed `42`, 100 distribution draws per feasible `(E,p,k,m,setting)`,
easy and hard settings, and grid points `(3,8,3)`, `(5,8,5)`, `(5,10,5)`,
`(5,10,4)`, and `(2,8,5)`. In code and filenames, the easy setting is stored as
`balanced` and the hard setting as `threshold_stress`. The bank stores the
population covariances, true `S_star` projectors, local top/bottom spaces, and
agreement spectra.

## `AnchorPCAInfty` Recovery

```bash
python experiments/synthetic/03_random_subspace_sstar/run_infty_sstar_recovery.py --overwrite
python experiments/synthetic/03_random_subspace_sstar/run_main_seqtest_dimension.py --overwrite
python experiments/synthetic/03_random_subspace_sstar/run_g2_low_environment_stress.py --overwrite
python experiments/synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py
```

Defaults: sampling seed `43`; for each distribution draw and sample size `N`, 20
independent samples; standard `N` grid `50, 100, 200, 500, 1000, 2000, 5000`;
and an extended small-`E` grid through `100000`. The `FindS_star` diagnostic
uses `alpha = 0.05`; it is overlaid on the main Gaussian, main
Gaussian-mixture, small-`m` Gaussian, and small-`E` Gaussian recovery figures.

Paper figures:

- `figures/infty_sstar_recovery_gaussian_g5_p10_k5_m2_main.(png|pdf)`
- `figures/infty_sstar_recovery_gaussian_mixture_g5_p10_k5_m2_main.(png|pdf)`
- `figures/infty_sstar_recovery_gaussian_g5_p8_k5_m1.(png|pdf)`
- `figures/infty_sstar_recovery_g2_p8_k5_m2_combined_extended_n100000.(png|pdf)`

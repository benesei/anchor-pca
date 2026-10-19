# Gas-sensor drift

We fit on temporal batches B1–Bs and evaluate on later batches B(s+1)–B10,
for `s=3,...,8` and `k=10,20,30`. Feature scales are learned from source
observations only; each batch covariance is centered within that batch. Source
batches have equal fitting weight unless the weighting comparison specifies
sample-size weights. Target EV is the equally weighted mean of batch EV fractions.

The UCI archive is downloaded automatically and its checksum is verified.
See [data and licenses](../../../ASSETS.md).

## Figures 4 and 8–10

```bash
python experiments/real_world/gas_sensor/compute_rolling_split_explained_variance.py
python experiments/real_world/gas_sensor/select_penalty.py
python experiments/real_world/gas_sensor/plot_source_target_EV.py
python experiments/real_world/gas_sensor/plot_rolling_split_summary.py
python experiments/real_world/gas_sensor/plot_sstar_poolpca_target_batches.py
```

The compute script fits poolPCA and infinite-penalty Anchor PCA for all 18
settings. It fits norm-maxRegret only for Figure 4 (`s=6`, `k=20`), using minPCA
with 10 restarts, 2,000 iterations, learning rate 0.01, and seed
`12020 + 10000*s + 101*k`. It also produces the gas-composition figure.

`select_penalty.py` selects `2*E*lambda` over
`{0, 0.001, 0.01, 0.1, 1, 10, 100, 1000, infinity}` by leave-one-source-batch-out
EV. Preprocessing is refitted within each fold, fractions are averaged equally,
and exact ties follow grid order. The selected method is shown as
`AnchorPCA` with a hat-lambda subscript. Figure 10 compares the estimated first
agreement block against poolPCA with the same number of directions.

## Tables 3–6

```bash
python experiments/real_world/gas_sensor/diagnose_b9_shift.py
python experiments/real_world/gas_sensor/evaluate_weighting.py
python experiments/real_world/gas_sensor/run_classification.py
```

The B9 diagnostic fits on B1–B8 at `k=20,30`. The weighting comparison averages
EV over targets within each split, then over the six splits `s=3,...,8`.

Classification fits on B1–B6 and evaluates on B7–B10. It compares the seven
rank-20 or unreduced pipelines in the paper. Preprocessing, representations,
and classifier parameters are fitted within each source-validation fold.
Selection uses equally weighted held-out balanced accuracy; target labels are
used only for final evaluation. The seed is 20260724. All selected final fits
must converge.

Computation scripts accept `--output-dir` and `--data-dir`; plotting scripts
accept `--experiment-dir`, `--results-dir`, and `--figures-dir`. The complete
[reproduction runner](../../README.md) sets these paths automatically.

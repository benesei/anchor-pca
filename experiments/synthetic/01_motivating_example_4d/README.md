# Four-Dimensional Motivating Example

Runtime category: quick.

This folder reproduces the population motivating example from the paper. The
script is deterministic and uses only the covariance matrices defined in the
paper.

Run from the repository root:

```bash
python experiments/synthetic/01_motivating_example_4d/run_motivating_example.py
```

The script fits `poolPCA`, `AnchorPCALambda(lambda_=25)`, and
`AnchorPCAInfty`. Outputs used directly in the paper:

- `figures/4d_motivation_exp_quotient_view.(png|pdf)`
- `results/paper_table_check.json`

`results/paper_table_check.json` records the rounded reconstruction-error
values used in the paper table and is checked by the script. The script also
writes diagnostic nuisance-plane and loading plots, plus full metric CSVs, but
those diagnostics are not included in the current paper. There are no random
seeds in this population example.

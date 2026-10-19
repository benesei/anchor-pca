# Four-dimensional motivating example

Figure 1 and reconstruction errors are computed directly from the three
population covariance matrices in Table 1; no sampling is involved.

```bash
python experiments/synthetic/01_motivating_example_4d/run_motivating_example.py \
  --output-dir outputs/motivating
```

This produces Figure 1 and the reconstruction-error CSV. The script verifies
Table 1's reconstruction values at the reported precision.

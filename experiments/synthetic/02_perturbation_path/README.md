# Covariance perturbation path

Figure 2 evaluates the fixed rank-three subspaces on the motivating example as
perturbation strength varies. The optimal-method bars use the exact crossings
of the reconstruction-error lines.

```bash
python experiments/synthetic/02_perturbation_path/run_perturbation_path.py \
  --output-dir outputs/perturbation
```

The wcPCA baselines use the pinned minPCA package with 60 restarts, 1,800
iterations, learning rate 0.05, and seed 0. The script saves the line parameters,
curves, and crossing points. Create Figure 2 from these saved results:

```bash
python experiments/synthetic/02_perturbation_path/plot_perturbation_path.py \
  --results-dir outputs/perturbation/results --figures-dir outputs/perturbation/figures
```

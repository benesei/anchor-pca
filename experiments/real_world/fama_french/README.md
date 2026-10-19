# Fama–French temporal experiment

We use daily value-weighted returns for the 49 Industry Portfolios from
1970–2024, divided into five-year periods: five source domains from 1970–1994
and six target domains from 1995–2024. Covariances are centered within each
period and source periods have equal fitting weight.

The paper used the May 2026 CRSP revision, downloaded July 26, 2026. The
[official library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library.html)
may revise historical returns, so fresh downloads can give different results.
The script records the downloaded revision and input hashes. Data are not
redistributed; see [data and licenses](../../../ASSETS.md).

```bash
python experiments/real_world/fama_french/run_temporal_experiment.py
python experiments/real_world/fama_french/plot_source_scree.py
```

At each `k=2,...,10`, the script selects `2*E*lambda` using leave-one-source-period-out
EV, then evaluates on the six later periods. It fits norm-maxRegret only at
`k=2`, with 10 restarts, 2,000 iterations, learning rate 0.01, and seed 4159490659.
Target periods are used only for evaluation.

The saved CSVs supply Figure 11 and Tables 7–8. Use `--output-dir` and
`--data-dir` to choose other computation paths, or `--archive` to supply an
existing archive. The plotter accepts `--input-csv` and `--figures-dir`.
The complete runner also accepts `--fama-archive` for an original-vintage run.

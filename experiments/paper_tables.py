"""Export the paper's eight tables and population agreement-gap summary as CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def export_tables(output: Path) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    read = lambda path: pd.read_csv(output / path)
    write = lambda frame, number: frame.to_csv(tables / f"table{number}.csv", index=False)
    write(read("motivating/results/motivating_example_reconstruction_errors.csv"), 1)
    unseen = read("unseen/summary.csv").rename(columns={
        "mean_target_population_ev_percent_total": "target_EV_percent",
        "mean_leading_m_projector_error_op": "projector_error"})
    write(unseen[["regime", "method_id", "target_EV_percent", "projector_error"]], 2)
    diagnostic = read("gas_sensor/results/b9_shift/b9_shift_diagnostics.csv")
    ev = diagnostic.pivot(index=["k", "method_id"], columns="target_batch", values="target_ev_percent_total")
    capture = diagnostic.groupby(["k", "method_id"])["b9_pc1_squared_projection_capture_percent"].first()
    write(ev.rename(columns={9: "B9_EV_percent", 10: "B10_EV_percent"}).join(capture).reset_index(), 3)
    weights = read("gas_sensor/results/weighting/weighting_by_split.csv")
    weights = weights[weights.last_source_batch.between(3, 8)]
    weight_means = weights.groupby(["k", "weighting"])["infinity_minus_poolPCA_pp"].mean()
    write(weight_means.unstack("weighting").reset_index(), 4)
    summary = read("gas_sensor/results/classification/target_metrics_summary.csv")
    selections = read("gas_sensor/results/classification/selections.csv")
    classifiers = summary.merge(selections, on="method_id", validate="one_to_one")
    classifiers["dimension"] = classifiers.method_id.map(lambda method: 20 if method in {
        "poolPCA_l2", "source_selected_Anchor_PCA_l2", "Anchor_PCA_infinity_l2"} else 128)
    classifiers["accuracy_percent"] = 100 * classifiers.accuracy
    classifiers["balanced_accuracy_percent"] = 100 * classifiers.balanced_accuracy
    write(classifiers[["method_id", "dimension", "two_e_lambda", "C", "beta", "l2_penalty",
                       "accuracy_percent", "balanced_accuracy_percent"]], 5)
    batches = read("gas_sensor/results/classification/target_metrics_by_batch.csv")
    write((100 * batches.pivot(index="method_id", columns="target_batch", values="accuracy")).reset_index(), 6)
    periods = read("fama_french/results/target_ev_by_period.csv")
    periods = periods[periods.k == 2]
    period_ev = periods.pivot(index="target_period", columns="method_id", values="target_ev_percent_total")
    period_counts = read("fama_french/results/period_counts.csv").set_index("period_id")
    period_ev["days"] = period_counts.n_observations.reindex(period_ev.index)
    for label, operation in [("Mean", "mean"), ("Minimum", "min")]:
        period_ev.loc[label, period_ev.columns != "days"] = getattr(period_ev.drop(columns="days"), operation)()
    write(period_ev.reset_index(), 7)
    ranks = read("fama_french/results/target_ev_summary.csv")
    pooled = ranks[ranks.method_id == "poolPCA"].set_index("k")
    selected = ranks[ranks.method_id == "source_selected_Anchor_PCA"].set_index("k")
    rank_table = selected[["two_e_lambda", "mean_target_ev_percent_total", "mean_difference_vs_poolPCA_pp"]].rename(
        columns={"mean_target_ev_percent_total": "selected_EV_percent", "mean_difference_vs_poolPCA_pp": "difference_pp"})
    rank_table["poolPCA_EV_percent"] = pooled.mean_target_ev_percent_total
    write(rank_table.reset_index(), 8)
    designs = read("design_bank/design_summary.csv")
    gaps = designs.groupby(["g", "p", "k", "m"])["agreement_gap"].agg(
        median="median", q10=lambda values: values.quantile(.1),
        q90=lambda values: values.quantile(.9), n_draws="size").reset_index()
    gaps.to_csv(tables / "agreement_gaps.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    export_tables(parser.parse_args().output_dir)

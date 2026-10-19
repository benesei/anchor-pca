"""Check all paper outputs against the saved numerical reference results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "paper_outputs.json").read_text())
REFERENCE = HERE / "reference"


def compare_csv(expected_path: Path, actual_path: Path) -> list[str]:
    expected, actual = pd.read_csv(expected_path), pd.read_csv(actual_path)
    if not set(expected.columns).issubset(actual.columns) or len(expected) != len(actual):
        return ["columns or row counts differ"]
    actual = actual[expected.columns]
    keys = [column for column in expected if not pd.api.types.is_float_dtype(expected[column])]
    if keys:
        expected = expected.sort_values(keys, na_position="last").reset_index(drop=True)
        actual = actual.sort_values(keys, na_position="last").reset_index(drop=True)
    differences = []
    for column in expected:
        left, right = expected[column], actual[column]
        if pd.api.types.is_numeric_dtype(left):
            matches = np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                                 rtol=1e-6, atol=1e-8, equal_nan=True)
        else:
            matches = left.fillna("").eq(right.fillna("")).to_numpy()
        if not np.all(matches):
            differences.append(f"{column}: {int((~matches).sum())} different values")
    return differences


def check(output: Path) -> dict[str, object]:
    failures, revised_data = [], []
    for item in MANIFEST["figures"] + MANIFEST["tables"] + MANIFEST["summaries"]:
        path = output / item["path"]
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"Missing {item['label']}: {item['path']}")
    original = json.loads((REFERENCE / "fama_french/protocol.json").read_text())
    current_path = output / MANIFEST["result_groups"]["fama_french"] / "protocol.json"
    current = json.loads(current_path.read_text()) if current_path.exists() else {}
    same_fama_data = current.get("used_periods_sha256") == original["used_periods_sha256"]
    for group, destination in MANIFEST["result_groups"].items():
        for reference in sorted((REFERENCE / group).rglob("*.csv")):
            relative = reference.relative_to(REFERENCE / group)
            actual = output / destination / relative
            if not actual.is_file():
                failures.append(f"Missing numerical output: {destination}/{relative}")
                continue
            differences = compare_csv(reference, actual)
            if differences:
                message = f"{group}/{relative}: " + "; ".join(differences)
                (revised_data if group == "fama_french" and not same_fama_data else failures).append(message)
    for reference in sorted((REFERENCE / "tables").glob("*.csv")):
        actual = output / "tables" / reference.name
        if actual.is_file():
            differences = compare_csv(reference, actual)
            if differences:
                message = f"{reference.name}: " + "; ".join(differences)
                (revised_data if reference.name in {"table7.csv", "table8.csv"} and not same_fama_data else failures).append(message)
    # Check the source-validation means and maxima independently of stored summaries.
    gas = output / MANIFEST["result_groups"]["gas_sensor"] / "penalty_selection"
    folds = pd.read_csv(gas / "penalty_validation_folds.csv")
    summary = pd.read_csv(gas / "penalty_validation_summary.csv")
    keys = ["last_source_batch", "k", "candidate_order", "two_e_lambda"]
    means = folds.groupby(keys, dropna=False)["heldout_total_ev_fraction"].mean().reset_index()
    merged = summary.merge(means, on=keys, validate="one_to_one")
    if len(merged) != 18 * 9 or not np.allclose(merged.mean_total_ev_fraction, merged.heldout_total_ev_fraction, rtol=0, atol=1e-12):
        failures.append("Gas validation summaries do not match equal-fold mean EV fractions.")
    finite = folds[np.isfinite(folds.two_e_lambda)]
    if not np.allclose(finite.raw_lambda_fold, finite.two_e_lambda / (2 * (finite.last_source_batch - 1))):
        failures.append("Gas fold penalties do not preserve the effective coefficient.")
    selection = pd.read_csv(gas / "selection_summary.csv")
    if len(selection) != 18 or set(selection.k) != {10, 20, 30} or set(selection.last_source_batch) != set(range(3, 9)):
        failures.append("Gas selection must cover the 18 paper settings.")
    for (split, rank), candidates in summary.groupby(["last_source_batch", "k"]):
        best = candidates.sort_values(["mean_total_ev_fraction", "candidate_order"], ascending=[False, True]).iloc[0]
        chosen = selection[(selection.last_source_batch == split) & (selection.k == rank)].iloc[0]
        if best.two_e_lambda != chosen.selected_two_e_lambda:
            failures.append(f"Gas selection maximum/tie order differs at s={split}, k={rank}.")
    counts = pd.read_csv(output / "design_bank/design_summary.csv")
    if len(counts) != 600 or set(zip(counts.g, counts.p, counts.k, counts.m)) != {(5, 10, 5, 2), (5, 8, 5, 1), (2, 8, 5, 2)}:
        failures.append("Recovery bank must contain 100 draws for each of the six paper settings.")
    fama = output / MANIFEST["result_groups"]["fama_french"]
    folds = pd.read_csv(fama / "penalty_validation_folds.csv")
    summary = pd.read_csv(fama / "penalty_validation_summary.csv")
    keys = ["k", "candidate_order", "two_e_lambda"]
    means = folds.groupby(keys, dropna=False)["heldout_total_ev_fraction"].mean().reset_index()
    merged = summary.merge(means, on=keys, validate="one_to_one")
    if len(merged) != 9 * 9 or not np.allclose(merged.mean_total_ev_fraction, merged.heldout_total_ev_fraction, rtol=0, atol=1e-12):
        failures.append("Fama validation summaries do not match equal-fold mean EV fractions.")
    for rank, candidates in summary.groupby("k"):
        best = candidates.sort_values(["mean_total_ev_fraction", "candidate_order"], ascending=[False, True]).iloc[0]
        chosen = candidates[candidates.selected == 1]
        if len(chosen) != 1 or chosen.iloc[0].candidate_order != best.candidate_order:
            failures.append(f"Fama selection maximum/tie order differs at k={rank}.")
    periods = pd.read_csv(fama / "period_counts.csv")
    if list(periods.start_year) != list(range(1970, 2025, 5)) or list(periods.end_year) != list(range(1974, 2025, 5)):
        failures.append("Fama periods must cover the paper's five-year blocks from 1970 to 2024.")
    fits = pd.read_csv(output / MANIFEST["result_groups"]["gas_sensor"] / "classification/fit_checks.csv")
    if len(fits) != 7 or not fits.converged.eq(1).all():
        failures.append("All seven selected classification fits must converge.")
    report = {"status": "failed" if failures else "passed", "failures": failures,
              "fama_input_matches_reference": same_fama_data,
              "revised_fama_data_differences": revised_data,
              "figures": len(MANIFEST["figures"]), "tables": len(MANIFEST["tables"])}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError("\n".join(failures))
    print("Checked 11 figures, 8 tables, and the numerical reference results.")
    if not same_fama_data:
        print("Fama–French inputs differ from the paper vintage; differences are recorded in verification.json.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    check(parser.parse_args().output_dir)

"""Check experiment outputs after reproduction.

The script is non-mutating. By default it checks source files and reports the
status of generated outputs. With ``--require-generated`` it fails if paper
figures or upstream CSV/metadata files are missing or inconsistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PaperFigure:
    label: str
    script: str
    output_pdf: str
    upstream: tuple[str, ...] = ()


PAPER_FIGURES = (
    PaperFigure(
        "4d_motivation_exp_quotient_view",
        "experiments/synthetic/01_motivating_example_4d/run_motivating_example.py",
        "experiments/synthetic/01_motivating_example_4d/figures/4d_motivation_exp_quotient_view.pdf",
        ("experiments/synthetic/01_motivating_example_4d/results/paper_table_check.json",),
    ),
    PaperFigure(
        "perturbation_path",
        "experiments/synthetic/02_perturbation_path/run_perturbation_path.py",
        "experiments/synthetic/02_perturbation_path/figures/perturbation_path.pdf",
        ("experiments/synthetic/02_perturbation_path/results/perturbation_path_curves.csv",),
    ),
    PaperFigure(
        "infty_sstar_recovery_gaussian_g5_p10_k5_m2_main",
        "experiments/synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py",
        "experiments/synthetic/03_random_subspace_sstar/figures/infty_sstar_recovery_gaussian_g5_p10_k5_m2_main.pdf",
        (
            "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery/summary_by_curve.csv",
            "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery_main_seqtest/summary_by_curve_seqtest.csv",
        ),
    ),
    PaperFigure(
        "infty_sstar_recovery_gaussian_mixture_g5_p10_k5_m2_main",
        "experiments/synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py",
        "experiments/synthetic/03_random_subspace_sstar/figures/infty_sstar_recovery_gaussian_mixture_g5_p10_k5_m2_main.pdf",
        ("experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery/summary_by_curve.csv",),
    ),
    PaperFigure(
        "infty_sstar_recovery_gaussian_g5_p8_k5_m1",
        "experiments/synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py",
        "experiments/synthetic/03_random_subspace_sstar/figures/infty_sstar_recovery_gaussian_g5_p8_k5_m1.pdf",
        ("experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery/summary_by_curve.csv",),
    ),
    PaperFigure(
        "infty_sstar_recovery_g2_p8_k5_m2_combined_extended_n100000",
        "experiments/synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py",
        "experiments/synthetic/03_random_subspace_sstar/figures/infty_sstar_recovery_g2_p8_k5_m2_combined_extended_n100000.pdf",
        (
            "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery/summary_by_curve.csv",
            "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery_g2_low_environment_stress/summary_by_curve.csv",
            "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery_g2_low_environment_stress/summary_by_curve_seqtest.csv",
        ),
    ),
    PaperFigure(
        "gas_sensor_class_composition_by_batch",
        "experiments/real_world/gas_sensor/compute_rolling_split_explained_variance.py",
        "experiments/real_world/gas_sensor/figures/gas_sensor_class_composition_by_batch.pdf",
        ("experiments/real_world/gas_sensor/results/gas_sensor_class_composition_by_batch.csv",),
    ),
    PaperFigure(
        "gas_sensor_publication_b1_b6_source_target_k20",
        "experiments/real_world/gas_sensor/plot_source_target_EV.py",
        "experiments/real_world/gas_sensor/figures/gas_sensor_publication_b1_b6_source_target_k20.pdf",
        ("experiments/real_world/gas_sensor/results/rolling_publication_explained_variance_all.csv",),
    ),
    PaperFigure(
        "rolling_publication_target_ev_combined",
        "experiments/real_world/gas_sensor/plot_rolling_split_summary.py",
        "experiments/real_world/gas_sensor/figures/rolling_publication_target_ev_combined.pdf",
        ("experiments/real_world/gas_sensor/results/rolling_publication_explained_variance_all.csv",),
    ),
    PaperFigure(
        "sstar_poolpca_same_dim_b9_b10_ev",
        "experiments/real_world/gas_sensor/plot_sstar_poolpca_target_batches.py",
        "experiments/real_world/gas_sensor/figures/sstar_poolpca_same_dim_b9_b10_ev.pdf",
        ("experiments/real_world/gas_sensor/results/rolling_publication_anchor_infty_sstar_all.csv",),
    ),
)

PAPER_BASE_CONFIGS = {"g5_p10_k5_m2", "g5_p8_k5_m1", "g2_p8_k5_m2"}
STANDARD_N = {50, 100, 200, 500, 1000, 2000, 5000}
EXTENDED_N = STANDARD_N | {10000, 30000, 100000}
PROFILES = {"balanced", "threshold_stress"}


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def require_path(path: str, errors: list[str], warnings: list[str], *, require: bool) -> bool:
    full = REPO_ROOT / path
    if full.exists():
        return True
    message = f"Missing generated output: {path}"
    if require:
        errors.append(message)
    else:
        warnings.append(message)
    return False


def check_manifest(require_generated: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for item in PAPER_FIGURES:
        if not (REPO_ROOT / item.script).exists():
            errors.append(f"Missing generating script for {item.label}: {item.script}")
        require_path(item.output_pdf, errors, warnings, require=require_generated)
        for upstream in item.upstream:
            require_path(upstream, errors, warnings, require=require_generated)
    return errors, warnings


def read_csv_if_exists(path: str) -> pd.DataFrame | None:
    full = REPO_ROOT / path
    if not full.exists():
        return None
    return pd.read_csv(full)


def base_config_from_row(row: pd.Series) -> str:
    return f"g{int(row['g'])}_p{int(row['p'])}_k{int(row['k'])}_m{int(row['m'])}"


def check_design_bank(errors: list[str], warnings: list[str], *, require_generated: bool) -> None:
    issues = errors if require_generated else warnings
    path = "experiments/synthetic/03_random_subspace_sstar/results/design_bank/design_summary.csv"
    df = read_csv_if_exists(path)
    if df is None:
        require_path(path, errors, warnings, require=require_generated)
        return
    df = df.copy()
    df["base_config"] = df.apply(base_config_from_row, axis=1)
    for base_config in PAPER_BASE_CONFIGS:
        for profile in PROFILES:
            rows = df[(df["base_config"] == base_config) & (df["profile"] == profile)]
            if require_generated and len(rows) != 100:
                issues.append(
                    f"Design bank has {len(rows)} rows for {profile}_{base_config}; expected 100."
                )
            elif rows.empty:
                warnings.append(f"Design bank does not contain {profile}_{base_config}.")

def check_infty_recovery(errors: list[str], warnings: list[str], *, require_generated: bool) -> None:
    issues = errors if require_generated else warnings
    path = "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery/raw_runs.csv"
    df = read_csv_if_exists(path)
    if df is None:
        require_path(path, errors, warnings, require=require_generated)
        return
    df = df.copy()
    df["base_config"] = df.apply(base_config_from_row, axis=1)
    expected_standard = {"g5_p10_k5_m2", "g5_p8_k5_m1"}
    for base_config in expected_standard:
        rows = df[df["base_config"] == base_config]
        if rows.empty:
            issues.append(f"Missing standard recovery rows for {base_config}.")
            continue
        if set(rows["N"].unique()) != STANDARD_N:
            issues.append(f"Unexpected N grid for {base_config}: {sorted(rows['N'].unique())}.")
        if set(rows["profile"].unique()) != PROFILES:
            issues.append(f"Unexpected profiles for {base_config}: {sorted(rows['profile'].unique())}.")
        if set(rows["distribution"].unique()) != {"gaussian", "gaussian_mixture"}:
            issues.append(
                f"Unexpected distributions for {base_config}: {sorted(rows['distribution'].unique())}."
            )
        if require_generated and len(rows) != 2 * 2 * 7 * 100 * 20:
            issues.append(f"Unexpected row count for {base_config}: {len(rows)}.")
    extra = set(df["base_config"].unique()) - expected_standard
    if extra:
        warnings.append(f"Standard recovery CSV contains optional non-paper configs: {sorted(extra)}.")

    g2_path = (
        "experiments/synthetic/03_random_subspace_sstar/results/"
        "infty_sstar_recovery_g2_low_environment_stress/raw_runs.csv"
    )
    g2 = read_csv_if_exists(g2_path)
    if g2 is None:
        require_path(g2_path, errors, warnings, require=require_generated)
        return
    if set(g2["N"].unique()) != EXTENDED_N:
        issues.append(f"Unexpected g2 extended N grid: {sorted(g2['N'].unique())}.")
    if set(g2["distribution"].unique()) != {"gaussian"}:
        issues.append(f"Unexpected g2 distributions: {sorted(g2['distribution'].unique())}.")
    if require_generated and len(g2) != 2 * 10 * 100 * 20:
        issues.append(f"Unexpected g2 recovery row count: {len(g2)}.")


def check_gas_sensor(errors: list[str], warnings: list[str], *, require_generated: bool) -> None:
    issues = errors if require_generated else warnings
    path = "experiments/real_world/gas_sensor/results/rolling_publication_explained_variance_all.csv"
    df = read_csv_if_exists(path)
    if df is None:
        require_path(path, errors, warnings, require=require_generated)
        return
    expected_methods = {
        "poolPCA",
        "AnchorPCA_lambda=1",
        "AnchorPCA_lambda=10",
        "AnchorPCA_infty",
        "norm-maxRegret",
    }
    if set(df["method_id"].unique()) != expected_methods:
        issues.append(f"Unexpected gas methods: {sorted(df['method_id'].unique())}.")
    if set(df["k"].unique()) != {5, 10, 20, 30, 40}:
        issues.append(f"Unexpected gas k grid: {sorted(df['k'].unique())}.")
    if set(df["last_source_batch"].unique()) != {3, 4, 5, 6, 7, 8, 9}:
        issues.append(
            f"Unexpected gas last-source grid: {sorted(df['last_source_batch'].unique())}."
        )
    if require_generated and len(df) != 1750:
        issues.append(f"Unexpected gas explained-variance row count: {len(df)}; expected 1750.")

    metadata_path = "experiments/real_world/gas_sensor/results/rolling_publication_metadata.json"
    full = REPO_ROOT / metadata_path
    if full.exists():
        metadata = json.loads(full.read_text())
        if metadata.get("archive_sha256") != "98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e":
            issues.append("Gas metadata has unexpected archive SHA256.")
    else:
        require_path(metadata_path, errors, warnings, require=require_generated)


def check_metadata_presence(errors: list[str], warnings: list[str], *, require_generated: bool) -> None:
    issues = errors if require_generated else warnings
    for path in (
        "experiments/synthetic/03_random_subspace_sstar/results/design_bank/metadata.json",
        "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery/metadata.json",
        "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery_main_seqtest/metadata.json",
        "experiments/synthetic/03_random_subspace_sstar/results/infty_sstar_recovery_g2_low_environment_stress/metadata.json",
        "experiments/real_world/gas_sensor/results/rolling_publication_metadata.json",
    ):
        if require_path(path, errors, warnings, require=require_generated):
            metadata = json.loads((REPO_ROOT / path).read_text())
            if "software_versions" not in metadata:
                issues.append(f"Metadata file lacks software_versions: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-generated",
        action="store_true",
        help="Fail when generated paper outputs or full-run row counts are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors, warnings = check_manifest(args.require_generated)
    check_metadata_presence(errors, warnings, require_generated=args.require_generated)
    check_design_bank(errors, warnings, require_generated=args.require_generated)
    check_infty_recovery(errors, warnings, require_generated=args.require_generated)
    check_gas_sensor(errors, warnings, require_generated=args.require_generated)

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Reproduction output check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

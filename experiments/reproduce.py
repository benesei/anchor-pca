"""Reproduce the figures and tables in Anchor PCA (NeurIPS 2026)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "paper_outputs.json").read_text())


def run_step(script: str, *arguments: object, output: Path) -> None:
    print(f"Running {Path(script).name}", flush=True)
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(output / "cache/matplotlib")})
    with (output / "reproduction.log").open("a") as log:
        command = [sys.executable, str(HERE / script), *map(str, arguments)]
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print("\n".join((output / "reproduction.log").read_text().splitlines()[-30:]), file=sys.stderr)
        raise RuntimeError(f"{script} failed; see {output / 'reproduction.log'}")


def compute(output: Path, fama_archive: Path | None) -> None:
    step = lambda script, *args: run_step(script, *args, output=output)
    synthetic = "synthetic/03_random_subspace_sstar/"
    gas = "real_world/gas_sensor/"
    step("synthetic/02_perturbation_path/run_perturbation_path.py", "--output-dir", output / "perturbation")
    step(synthetic + "generate_design_bank.py", "--output-dir", output / "design_bank")
    step(synthetic + "run_main_seqtest_dimension.py", "--design-bank-dir", output / "design_bank",
         "--output-dir", output / "recovery")
    step(synthetic + "run_g2_low_environment_stress.py", "--design-bank-dir", output / "design_bank",
         "--output-dir", output / "stress")
    step(synthetic + "run_unseen_target_evaluation.py", "--output-dir", output / "unseen")
    step(gas + "compute_rolling_split_explained_variance.py", "--output-dir", output / "gas_sensor")
    for script, group in [("select_penalty.py", "penalty_selection"),
                          ("evaluate_weighting.py", "weighting"),
                          ("diagnose_b9_shift.py", "b9_shift"),
                          ("run_classification.py", "classification")]:
        step(gas + script, "--data-dir", output / "gas_sensor/data",
             "--output-dir", output / "gas_sensor/results" / group)
    step(gas + "plot_sstar_poolpca_target_batches.py", "--experiment-dir", output / "gas_sensor",
         "--figures-dir", output / "figures/gas_sensor")
    args = ["--data-dir", output / "fama_french/data", "--output-dir", output / "fama_french/results"]
    if fama_archive is not None:
        args += ["--archive", fama_archive]
    step("real_world/fama_french/run_temporal_experiment.py", *args)


def render(output: Path) -> None:
    step = lambda script, *args: run_step(script, *args, output=output)
    step("synthetic/01_motivating_example_4d/run_motivating_example.py", "--output-dir", output / "motivating",
         "--figures-dir", output / "figures")
    step("synthetic/02_perturbation_path/plot_perturbation_path.py", "--results-dir", output / "perturbation/results",
         "--figures-dir", output / "figures")
    step("synthetic/03_random_subspace_sstar/plot_infty_sstar_recovery.py", "--results-dir", output / "recovery",
         "--main-seqtest-results-dir", output / "recovery", "--stress-results-dir", output / "stress",
         "--figures-dir", output / "figures/AnchorPCAinfty_consistency")
    gas = "real_world/gas_sensor/"
    step(gas + "plot_gas_composition.py", "--input-csv", output / "gas_sensor/results/gas_sensor_class_composition_by_batch.csv",
         "--figures-dir", output / "figures/gas_sensor")
    for script in ["plot_source_target_EV.py", "plot_rolling_split_summary.py"]:
        step(gas + script, "--experiment-dir", output / "gas_sensor", "--figures-dir", output / "figures/gas_sensor")
    step(gas + "plot_sstar_poolpca_target_batches.py", "--plot-data-csv",
         output / "gas_sensor/results/sstar_poolpca_same_dim_b9_b10_ev_plot_data.csv",
         "--figures-dir", output / "figures/gas_sensor")
    step("real_world/fama_french/plot_source_scree.py", "--input-csv", output / "fama_french/results/source_scree.csv",
         "--figures-dir", output / "figures/fama_french")
    step("paper_tables.py", "--output-dir", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-reference", action="store_true", help="Plot the saved paper results without downloading data.")
    mode.add_argument("--recompute", action="store_true", help="Run the complete paper experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--fama-archive", type=Path, help="Optional Fama–French archive, including the original paper vintage.")
    args = parser.parse_args()
    if args.fama_archive and not args.recompute:
        parser.error("--fama-archive is only used with --recompute")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory to keep separate runs independent.")
    output.mkdir(parents=True, exist_ok=True)
    if args.from_reference:
        print("Replotting saved paper results.", flush=True)
        for group, destination in MANIFEST["result_groups"].items():
            shutil.copytree(HERE / "reference" / group, output / destination)
    else:
        compute(output, args.fama_archive.resolve() if args.fama_archive else None)
    render(output)
    run_step("check_reproduction_outputs.py", "--output-dir", output, output=output)
    print(f"Finished. Figures: {output / 'figures'}; tables: {output / 'tables'}", flush=True)


if __name__ == "__main__":
    main()

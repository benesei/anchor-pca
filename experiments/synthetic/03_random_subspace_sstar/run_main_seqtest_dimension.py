"""Run FindS_star comparisons for paper S_star recovery figures.

This script complements ``run_infty_sstar_recovery.py`` for selected paper
figures. It reuses the same distribution draws, sample-size grid, and sampling
seed convention, and adds the FindS_star estimate of ``m = dim(S*)``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_g2_low_environment_stress import (  # noqa: E402
    seqtest_subspace_metrics,
    sequential_test_metrics,
    summarize_seqtest_results,
)
from run_infty_sstar_recovery import (  # noqa: E402
    config_metadata,
    empirical_run_metrics_and_model,
    load_config_arrays,
    population_diagnostics_for_design,
    prepare_output_dir,
    read_design_summary,
    sample_environments,
    stable_seed,
    summarize_results,
    validate_requested_configs,
)
from anchorpca.reproducibility import repo_relative_path, software_versions  # noqa: E402


DEFAULT_SEED = 43
DEFAULT_N_GRID = (50, 100, 200, 500, 1000, 2000, 5000)
DEFAULT_PROFILES = ("balanced", "threshold_stress")
DEFAULT_SAMPLE_REPS = 20
DEFAULT_ALPHA = 0.05
DEFAULT_MEAN_SCALE = 0.75
DEFAULT_W_LOW = 0.5
DEFAULT_W_HIGH = 2.0
VALID_DISTRIBUTIONS = ("gaussian", "gaussian_mixture", "scale_mixture")
DEFAULT_TARGETS = (
    "gaussian:g5_p10_k5_m2",
    "gaussian_mixture:g5_p10_k5_m2",
    "gaussian:g5_p8_k5_m1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FindS_star comparisons for selected S_star recovery figures."
    )
    parser.add_argument(
        "--design-bank-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "design_bank",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "infty_sstar_recovery_main_seqtest",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-grid", type=int, nargs="+", default=list(DEFAULT_N_GRID))
    parser.add_argument("--sample-reps", type=int, default=DEFAULT_SAMPLE_REPS)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_PROFILES),
        choices=["balanced", "threshold_stress"],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help=(
            "Distribution/base-config pairs in the form "
            "'distribution:g{E}_p{p}_k{k}_m{m}'. Defaults are the paper figures "
            "that overlay FindS_star: main Gaussian, main Gaussian mixture, "
            "and small-m Gaussian."
        ),
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--mean-scale", type=float, default=DEFAULT_MEAN_SCALE)
    parser.add_argument("--w-low", type=float, default=DEFAULT_W_LOW)
    parser.add_argument("--w-high", type=float, default=DEFAULT_W_HIGH)
    parser.add_argument(
        "--max-designs",
        type=int,
        default=None,
        help="Limit number of stored distribution draws per code-level profile config.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_target(target: str) -> tuple[str, str]:
    pieces = str(target).split(":", maxsplit=1)
    if len(pieces) != 2:
        raise ValueError(
            "Each --targets value must have the form "
            "'distribution:g{E}_p{p}_k{k}_m{m}', e.g. 'gaussian:g5_p10_k5_m2'."
        )
    distribution, base_config = pieces
    if distribution not in VALID_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Expected one of {list(VALID_DISTRIBUTIONS)}."
        )
    if not base_config.startswith("g") or "_p" not in base_config or "_k" not in base_config or "_m" not in base_config:
        raise ValueError(f"Invalid base config label in --targets: {base_config!r}.")
    return distribution, base_config


def target_specs(profiles: list[str], targets: list[str]) -> list[dict[str, str]]:
    specs = []
    seen = set()
    for target in targets:
        distribution, base_config = parse_target(target)
        for profile in profiles:
            config_id = f"{profile}_{base_config}"
            key = (distribution, config_id)
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                {
                    "distribution": distribution,
                    "base_config": base_config,
                    "config_id": config_id,
                }
            )
    return specs


def write_metadata(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    config_ids: list[str],
    raw_rows: int,
) -> None:
    metadata = {
        "script": repo_relative_path(__file__),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": software_versions(),
        "seed": int(args.seed),
        "design_bank_dir": repo_relative_path(args.design_bank_dir),
        "config_ids": config_ids,
        "targets": list(args.targets),
        "n_grid": [int(value) for value in args.n_grid],
        "sample_reps": int(args.sample_reps),
        "alpha": float(args.alpha),
        "mean_scale": float(args.mean_scale),
        "w_low": float(args.w_low),
        "w_high": float(args.w_high),
        "max_designs": args.max_designs,
        "findsstar_assume_gaussian": True,
        "raw_rows": int(raw_rows),
        "note": (
            "FindS_star comparisons for selected AnchorPCA_infty S_star recovery "
            "figures. The same Gaussian-calibrated test is intentionally evaluated "
            "also on requested non-Gaussian covariance-preserving samples."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if any(int(value) <= 1 for value in args.n_grid):
        raise ValueError("All --n-grid values must be greater than one.")
    if args.sample_reps <= 0:
        raise ValueError("--sample-reps must be positive.")
    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError("--alpha must lie in (0, 1).")
    if args.max_designs is not None and args.max_designs <= 0:
        raise ValueError("--max-designs must be positive when supplied.")

    design_bank_dir = args.design_bank_dir.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    design_summary = read_design_summary(design_bank_dir)
    specs = target_specs(list(args.profiles), list(args.targets))
    config_ids = sorted({spec["config_id"] for spec in specs})
    validate_requested_configs(design_summary, config_ids)

    raw_rows = []
    for spec in specs:
        config_id = spec["config_id"]
        distribution = spec["distribution"]
        config = config_metadata(design_summary, config_id)
        arrays = load_config_arrays(design_bank_dir, config_id)
        n_designs_available = int(arrays["covariances"].shape[0])
        n_designs = (
            min(n_designs_available, int(args.max_designs))
            if args.max_designs is not None
            else n_designs_available
        )
        print(
            f"Running {config_id}: {n_designs} designs, "
            f"N={list(args.n_grid)}, distribution={distribution}, alpha={float(args.alpha)}"
        )
        for design_replicate in range(n_designs):
            population_diagnostics = population_diagnostics_for_design(
                design_summary,
                config_id=config_id,
                design_replicate=design_replicate,
            )
            covariances = np.asarray(arrays["covariances"][design_replicate], dtype=float)
            for n_obs in args.n_grid:
                for sample_replicate in range(int(args.sample_reps)):
                    sample_seed = stable_seed(
                        int(args.seed),
                        config_id,
                        distribution,
                        design_replicate,
                        int(n_obs),
                        sample_replicate,
                    )
                    rng = np.random.default_rng(sample_seed)
                    X_envs = sample_environments(
                        rng=rng,
                        covariances=covariances,
                        n_obs=int(n_obs),
                        distribution=distribution,
                        mean_scale=float(args.mean_scale),
                        w_low=float(args.w_low),
                        w_high=float(args.w_high),
                    )
                    anchor_metrics, model = empirical_run_metrics_and_model(
                        X_envs=X_envs,
                        arrays=arrays,
                        config=config,
                        design_replicate=design_replicate,
                    )
                    seq_metrics = sequential_test_metrics(
                        X_envs=X_envs,
                        k=int(config["k"]),
                        m=int(config["m"]),
                        alpha=float(args.alpha),
                        enabled=True,
                    )
                    seq_subspace_metrics = seqtest_subspace_metrics(
                        barPi=model.barPi_,
                        sstar_projector=np.asarray(
                            arrays["sstar_projectors"][design_replicate],
                            dtype=float,
                        ),
                        m_hat=seq_metrics["m_hat_seqtest"],
                    )
                    raw_rows.append(
                        {
                            **config,
                            "distribution": distribution,
                            "design_replicate": int(design_replicate),
                            "sample_replicate": int(sample_replicate),
                            "N": int(n_obs),
                            "sample_seed": int(sample_seed),
                            **population_diagnostics,
                            **anchor_metrics,
                            **seq_metrics,
                            **seq_subspace_metrics,
                        }
                    )

    raw = pd.DataFrame(raw_rows)
    design_anchor, curve_anchor = summarize_results(raw)
    design_seq, curve_seq = summarize_seqtest_results(raw)
    raw.to_csv(output_dir / "raw_runs.csv", index=False)
    design_anchor.to_csv(output_dir / "summary_by_design.csv", index=False)
    curve_anchor.to_csv(output_dir / "summary_by_curve.csv", index=False)
    design_seq.to_csv(output_dir / "summary_by_design_seqtest.csv", index=False)
    curve_seq.to_csv(output_dir / "summary_by_curve_seqtest.csv", index=False)
    write_metadata(
        output_dir=output_dir,
        args=args,
        config_ids=config_ids,
        raw_rows=len(raw),
    )
    print(f"Wrote {len(raw)} raw runs to {output_dir}")


if __name__ == "__main__":
    main()

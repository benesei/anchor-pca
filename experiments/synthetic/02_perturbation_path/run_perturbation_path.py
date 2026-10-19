"""Reproduce the perturbation-path plot for the 4D motivating example.

The script uses the population model from the paper's motivating example. It
fits each fixed rank-3 method on the original covariances and evaluates the
average reconstruction error along

    Sigma_e(rho) = Sigma_e + rho * Pi_k^(e),  rho >= 0.

All plotted curves are recomputed from fitted projectors and the population
covariances. By default, the wcPCA baselines are fitted with the external
``minPCA`` package. No curve values are read from a cached CSV or hardcoded.

Run from this directory or from the repository root:

    python experiments/synthetic/02_perturbation_path/run_perturbation_path.py

Numerical results are written to ``results/`` under the output directory.
Use ``plot_perturbation_path.py`` to create Figure 2 from those results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def find_repo_root() -> Path:
    """Find the repository root so the local package can be imported."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "anchorpca").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing src/anchorpca.")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))

from anchorpca import AnchorPCAInfty, AnchorPCALambda, pool_pca_from_covariances  # noqa: E402
from anchorpca.reproducibility import software_versions  # noqa: E402


LAMBDA_VALUE = 25.0
N_COMPONENTS = 3
N_ENVIRONMENTS = 3
RHO_DESIGN = 2.0 * N_ENVIRONMENTS * LAMBDA_VALUE

METHOD_ORDER = [
    "poolPCA",
    "maxRCS",
    "maxRegret",
    "norm-maxRegret",
    "AnchorPCA_lambda=25",
    "AnchorPCA_infty",
]

METHOD_LABELS = {
    "poolPCA": "poolPCA",
    "maxRCS": "maxRCS",
    "maxRegret": "maxRegret",
    "norm-maxRegret": "norm-maxRegret",
    "AnchorPCA_lambda=25": r"AnchorPCA$_{\lambda = 25}$",
    "AnchorPCA_infty": r"AnchorPCA$_{\infty}$",
}

MINPCA_INSTALL_MESSAGE = (
    "The perturbation-path script uses the external AGPL minPCA package for "
    "wcPCA baselines by default, but minPCA is not importable in this Python "
    "environment. Install the pinned dependency from requirements-repro.txt."
)

MINPCA_METHOD_SPECS = {
    "maxRCS": ("maxrcs", False),
    "maxRegret": ("maxregret", False),
    "norm-maxRegret": ("maxregret", True),
}


@dataclass(frozen=True)
class FittedMethod:
    method_id: str
    directions: np.ndarray
    objective_value: float | None = None
    fit_source: str = "anchorpca"
    package_function: str | None = None
    package_norm: bool | None = None

    @property
    def projection(self) -> np.ndarray:
        return self.directions @ self.directions.T


@dataclass(frozen=True)
class LineParameters:
    method_id: str
    intercept: float
    slope: float

    def evaluate(self, rho: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * rho


def rotated_pair(phi_deg: float) -> tuple[np.ndarray, np.ndarray]:
    phi = np.deg2rad(phi_deg)
    direction = np.array([0.0, 0.0, np.cos(phi), np.sin(phi)])
    complement = np.array([0.0, 0.0, -np.sin(phi), np.cos(phi)])
    return direction, complement


def covariance_from_eigensystem(
    directions: list[np.ndarray],
    eigenvalues: list[float],
) -> np.ndarray:
    covariance = np.zeros((4, 4), dtype=float)
    for direction, value in zip(directions, eigenvalues):
        covariance += float(value) * np.outer(direction, direction)
    return covariance


def build_population_model() -> dict[str, object]:
    """Build the exact 4D population model used in the motivating example."""
    c1 = np.array([1.0, 0.0, 0.0, 0.0])
    c2 = np.array([0.0, 1.0, 0.0, 0.0])
    c3 = np.array([0.0, 0.0, 1.0, 0.0])
    c4 = np.array([0.0, 0.0, 0.0, 1.0])

    a = c1
    b = c2
    u, u_perp = rotated_pair(0.0)
    v, v_perp = rotated_pair(50.0)
    w, w_perp = rotated_pair(100.0)

    covariances = [
        covariance_from_eigensystem([u, a, b, u_perp], [220.0, 140.0, 90.0, 25.0]),
        covariance_from_eigensystem([a, v, b, v_perp], [120.0, 90.0, 70.0, 10.0]),
        covariance_from_eigensystem([w, b, a, w_perp], [320.0, 120.0, 80.0, 10.0]),
    ]
    local_projectors = [
        np.eye(4) - np.outer(u_perp, u_perp),
        np.eye(4) - np.outer(v_perp, v_perp),
        np.eye(4) - np.outer(w_perp, w_perp),
    ]

    return {
        "basis": {"a": a, "b": b, "c3": c3, "c4": c4},
        "covariances": covariances,
        "local_projectors": local_projectors,
    }


def wc_objective_value(
    method_id: str,
    projection: np.ndarray,
    covariances: list[np.ndarray],
) -> float:
    identity = np.eye(projection.shape[0])
    losses = np.asarray(
        [np.trace((identity - projection) @ covariance) for covariance in covariances],
        dtype=float,
    )
    if method_id == "maxRCS":
        objective = losses
    elif method_id == "maxRegret":
        best_rank3_loss = np.asarray(
            [np.linalg.eigvalsh(covariance).min() for covariance in covariances],
            dtype=float,
        )
        objective = losses - best_rank3_loss
    elif method_id == "norm-maxRegret":
        best_rank3_loss = np.asarray(
            [np.linalg.eigvalsh(covariance).min() for covariance in covariances],
            dtype=float,
        )
        traces = np.asarray([np.trace(covariance) for covariance in covariances], dtype=float)
        objective = (losses - best_rank3_loss) / traces
    else:
        raise ValueError(f"Unknown wcPCA baseline: {method_id}")
    return float(np.max(objective))


def load_minpca():
    try:
        import torch
        from minPCA.minpca import minPCA
    except ImportError as exc:
        raise RuntimeError(MINPCA_INSTALL_MESSAGE) from exc
    return torch, minPCA


def fit_minpca_wc_baseline(
    method_id: str,
    covariances: list[np.ndarray],
    *,
    n_restarts: int,
    n_iters: int,
    lr: float,
    seed: int,
) -> FittedMethod:
    """Fit a wcPCA baseline with the upstream minPCA package."""
    if method_id not in MINPCA_METHOD_SPECS:
        raise ValueError(f"Unknown minPCA-backed wcPCA baseline: {method_id}")

    torch, minPCA = load_minpca()
    function, norm = MINPCA_METHOD_SPECS[method_id]
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    covariances_float32 = [np.asarray(covariance, dtype=np.float32) for covariance in covariances]
    model = minPCA(n_components=N_COMPONENTS, function=function, norm=norm)
    model.fit(
        covariances_float32,
        n_restarts=int(n_restarts),
        n_iters=int(n_iters),
        lr=float(lr),
    )

    directions = np.asarray(model.v_.detach().cpu().numpy(), dtype=float)
    directions, _ = np.linalg.qr(directions)
    projection = directions @ directions.T
    return FittedMethod(
        method_id=method_id,
        directions=directions,
        objective_value=wc_objective_value(method_id, projection, covariances),
        fit_source="minPCA",
        package_function=function,
        package_norm=norm,
    )


def fit_methods(
    covariances: list[np.ndarray],
    *,
    minpca_restarts: int,
    minpca_iters: int,
    minpca_lr: float,
    minpca_seed: int,
) -> dict[str, FittedMethod]:
    pool = pool_pca_from_covariances(covariances, n_components=N_COMPONENTS)
    anchor_lambda = AnchorPCALambda(
        n_components=N_COMPONENTS,
        lambda_=LAMBDA_VALUE,
    ).fit_covariances(covariances)
    anchor_infty = AnchorPCAInfty(
        n_components=N_COMPONENTS,
        block_tol=1e-12,
    ).fit_covariances(covariances)

    methods = {
        "poolPCA": FittedMethod("poolPCA", pool["directions"]),
        "AnchorPCA_lambda=25": FittedMethod("AnchorPCA_lambda=25", anchor_lambda.directions_),
        "AnchorPCA_infty": FittedMethod("AnchorPCA_infty", anchor_infty.directions_),
    }

    for method_id in ["maxRCS", "maxRegret", "norm-maxRegret"]:
        methods[method_id] = fit_minpca_wc_baseline(
            method_id, covariances, n_restarts=minpca_restarts,
            n_iters=minpca_iters, lr=minpca_lr, seed=minpca_seed,
        )

    return methods


def reconstruction_line_parameters(
    projection: np.ndarray,
    covariances: list[np.ndarray],
    local_projectors: list[np.ndarray],
) -> tuple[float, float]:
    identity = np.eye(projection.shape[0])
    intercept = float(
        np.mean([np.trace((identity - projection) @ covariance) for covariance in covariances])
    )
    slope = float(
        np.mean(
            [np.trace((identity - projection) @ projector) for projector in local_projectors]
        )
    )
    return intercept, slope


def build_line_parameters(
    methods: dict[str, FittedMethod],
    covariances: list[np.ndarray],
    local_projectors: list[np.ndarray],
) -> dict[str, LineParameters]:
    line_parameters = {}
    for method_id, method in methods.items():
        intercept, slope = reconstruction_line_parameters(
            method.projection,
            covariances,
            local_projectors,
        )
        line_parameters[method_id] = LineParameters(method_id, intercept, slope)
    return line_parameters


def crossover_rho(
    left_method_id: str,
    right_method_id: str,
    line_parameters: dict[str, LineParameters],
) -> float:
    left = line_parameters[left_method_id]
    right = line_parameters[right_method_id]
    denominator = left.slope - right.slope
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float((right.intercept - left.intercept) / denominator)


def pairwise_crossovers(
    method_ids: list[str],
    line_parameters: dict[str, LineParameters],
    *,
    rho_min: float,
    rho_max: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_index, left_method_id in enumerate(method_ids):
        for right_method_id in method_ids[left_index + 1 :]:
            rho_star = crossover_rho(left_method_id, right_method_id, line_parameters)
            if not np.isfinite(rho_star):
                continue
            if float(rho_min) <= rho_star <= float(rho_max):
                rows.append(
                    {
                        "left_method_id": left_method_id,
                        "right_method_id": right_method_id,
                        "rho_star": rho_star,
                    }
                )
    return sorted(rows, key=lambda row: float(row["rho_star"]))


def lower_envelope_intervals(
    method_ids: list[str],
    line_parameters: dict[str, LineParameters],
    *,
    rho_min: float,
    rho_max: float,
) -> list[dict[str, object]]:
    candidates = [float(rho_min), float(rho_max)]
    candidates.extend(
        float(row["rho_star"])
        for row in pairwise_crossovers(
            method_ids,
            line_parameters,
            rho_min=rho_min,
            rho_max=rho_max,
        )
    )
    candidates = sorted(set(round(value, 12) for value in candidates))

    intervals: list[dict[str, object]] = []
    for start, end in zip(candidates[:-1], candidates[1:]):
        if np.isclose(start, end):
            continue
        midpoint = 0.5 * (start + end)
        best_method_id = min(
            method_ids,
            key=lambda method_id: line_parameters[method_id].intercept
            + line_parameters[method_id].slope * midpoint,
        )
        if intervals and intervals[-1]["best_method_id"] == best_method_id:
            intervals[-1]["rho_end"] = end
        else:
            intervals.append(
                {
                    "rho_start": start,
                    "rho_end": end,
                    "best_method_id": best_method_id,
                    "best_method_label": METHOD_LABELS[best_method_id],
                }
            )
    return intervals


def lower_envelope_crossovers(
    intervals: list[dict[str, object]],
) -> list[tuple[str, str, float]]:
    crossovers: list[tuple[str, str, float]] = []
    for left, right in zip(intervals[:-1], intervals[1:]):
        crossovers.append(
            (
                str(left["best_method_id"]),
                str(right["best_method_id"]),
                float(right["rho_start"]),
            )
        )
    return crossovers


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    results_dir: Path,
    methods: dict[str, FittedMethod],
    line_parameters: dict[str, LineParameters],
    rho_grid: np.ndarray,
    curves: dict[str, np.ndarray],
    crossovers: list[tuple[str, str, float]],
    envelope_intervals: list[dict[str, object]],
    all_pairwise_crossovers: list[dict[str, object]],
    *,
    minpca_restarts: int,
    minpca_iters: int,
    minpca_lr: float,
    minpca_seed: int,
) -> None:
    line_rows = [
        {
            "method_id": method_id,
            "method_label": METHOD_LABELS[method_id],
            "intercept_rho0": line_parameters[method_id].intercept,
            "slope": line_parameters[method_id].slope,
            "rho_design": RHO_DESIGN if method_id == "AnchorPCA_lambda=25" else "",
            "fit_source": methods[method_id].fit_source,
            "package_function": methods[method_id].package_function or "",
            "package_norm": methods[method_id].package_norm
            if methods[method_id].package_norm is not None
            else "",
            "wc_objective_value": methods[method_id].objective_value
            if methods[method_id].objective_value is not None
            else "",
        }
        for method_id in METHOD_ORDER
    ]
    write_csv(results_dir / "perturbation_path_line_parameters.csv", line_rows)

    curve_rows = []
    for method_id in METHOD_ORDER:
        for rho, value in zip(rho_grid, curves[method_id]):
            curve_rows.append(
                {
                    "method_id": method_id,
                    "method_label": METHOD_LABELS[method_id],
                    "rho": float(rho),
                    "avg_reconstruction_error": float(value),
                }
            )
    write_csv(results_dir / "perturbation_path_curves.csv", curve_rows)

    crossover_rows = [
        {
            "left_method_id": left,
            "right_method_id": right,
            "rho_star": rho_star,
        }
        for left, right, rho_star in crossovers
    ]
    write_csv(results_dir / "perturbation_path_crossovers.csv", crossover_rows)
    write_csv(results_dir / "perturbation_path_pairwise_crossovers.csv", all_pairwise_crossovers)
    write_csv(results_dir / "perturbation_path_best_intervals.csv", envelope_intervals)

    metadata = {
        "software_versions": software_versions(),
        "data_source": (
            "All plotted y-values are recomputed from the 4D population covariances, "
            "the local rank-3 projectors, fitted AnchorPCA projectors, and fitted wcPCA "
            "baseline projectors fitted with minPCA. The crossover CSV records "
            "adjacent lower-envelope transitions; the pairwise crossover CSV records all "
            "finite pairwise intersections in the plotted rho range."
        ),
        "lambda_value": LAMBDA_VALUE,
        "rho_design": RHO_DESIGN,
        "rho_grid_min": float(rho_grid.min()),
        "rho_grid_max": float(rho_grid.max()),
        "rho_grid_size": int(rho_grid.size),
        "wc_source": "minpca",
        "minpca_restarts": int(minpca_restarts),
        "minpca_iters": int(minpca_iters),
        "minpca_lr": float(minpca_lr),
        "minpca_seed": int(minpca_seed),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "perturbation_path_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def print_summary(
    line_parameters: dict[str, LineParameters],
    crossovers: list[tuple[str, str, float]],
) -> None:
    print("Perturbation-path line parameters")
    print("---------------------------------")
    print("wcPCA fit source: minPCA")
    for method_id in METHOD_ORDER:
        params = line_parameters[method_id]
        print(
            "{label:28s} intercept={intercept:8.3f}  slope={slope:7.4f}".format(
                label=method_id,
                intercept=params.intercept,
                slope=params.slope,
            )
        )

    print("\nAdjacent lower-envelope crossovers")
    print("----------------------------------")
    for left, right, rho_star in crossovers:
        print(f"{left:22s} -> {right:22s} rho={rho_star:8.3f}")


def run(
    output_dir: Path,
    *,
    rho_max: float,
    rho_step: float,
    minpca_restarts: int,
    minpca_iters: int,
    minpca_lr: float,
    minpca_seed: int,
) -> None:
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"

    model = build_population_model()
    covariances = model["covariances"]
    local_projectors = model["local_projectors"]

    methods = fit_methods(
        covariances,
        minpca_restarts=minpca_restarts,
        minpca_iters=minpca_iters,
        minpca_lr=minpca_lr,
        minpca_seed=minpca_seed,
    )
    line_parameters = build_line_parameters(methods, covariances, local_projectors)

    rho_grid = np.arange(0.0, float(rho_max) + 0.5 * float(rho_step), float(rho_step))
    curves = {
        method_id: line_parameters[method_id].evaluate(rho_grid)
        for method_id in METHOD_ORDER
    }

    all_pairwise_crossovers = pairwise_crossovers(
        METHOD_ORDER,
        line_parameters,
        rho_min=0.0,
        rho_max=float(rho_max),
    )
    envelope_intervals = lower_envelope_intervals(
        METHOD_ORDER,
        line_parameters,
        rho_min=0.0,
        rho_max=float(rho_max),
    )
    crossovers = lower_envelope_crossovers(envelope_intervals)

    write_outputs(
        results_dir,
        methods,
        line_parameters,
        rho_grid,
        curves,
        crossovers,
        envelope_intervals,
        all_pairwise_crossovers,
        minpca_restarts=minpca_restarts,
        minpca_iters=minpca_iters,
        minpca_lr=minpca_lr,
        minpca_seed=minpca_seed,
    )
    from plot_perturbation_path import plot_from_results
    plot_from_results(results_dir, figures_dir)
    print_summary(line_parameters, crossovers)
    print(f"\nWrote figure to {figures_dir}")
    print(f"Wrote tables to {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory under which figures/ and results/ are written.",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=500.0,
        help="Maximum perturbation strength shown on the main axis.",
    )
    parser.add_argument(
        "--rho-step",
        type=float,
        default=0.25,
        help="Grid spacing for saved main-axis curve values.",
    )
    parser.add_argument(
        "--minpca-restarts",
        type=int,
        default=60,
        help="Number of minPCA random restarts.",
    )
    parser.add_argument(
        "--minpca-iters",
        type=int,
        default=1800,
        help="Number of minPCA optimization iterations.",
    )
    parser.add_argument(
        "--minpca-lr",
        type=float,
        default=0.05,
        help="minPCA optimizer learning rate.",
    )
    parser.add_argument(
        "--minpca-seed",
        type=int,
        default=0,
        help="Random seed for minPCA initialization and restarts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.output_dir.resolve(),
        rho_max=args.rho_max,
        rho_step=args.rho_step,
        minpca_restarts=args.minpca_restarts,
        minpca_iters=args.minpca_iters,
        minpca_lr=args.minpca_lr,
        minpca_seed=args.minpca_seed,
    )


if __name__ == "__main__":
    main()

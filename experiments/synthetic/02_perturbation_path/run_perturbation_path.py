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

Generated files are written to local ``figures/`` and ``results/`` folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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

METHOD_STYLES = {
    "poolPCA": {"color": "#666666", "linestyle": "-", "linewidth": 2.3},
    "maxRCS": {"color": "#2ca02c", "linestyle": "--", "linewidth": 2.1},
    "maxRegret": {"color": "#d62728", "linestyle": "--", "linewidth": 2.1},
    "norm-maxRegret": {"color": "#9467bd", "linestyle": "-.", "linewidth": 2.3},
    "AnchorPCA_lambda=25": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.8},
    "AnchorPCA_infty": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2.8},
}

MINPCA_INSTALL_MESSAGE = (
    "The perturbation-path script uses the external AGPL minPCA package for "
    "wcPCA baselines by default, but minPCA is not importable in this Python "
    "environment. Install it from https://github.com/anyafries/minPCA, or rerun "
    "this script with `--wc-source grid` to use the deterministic 4D grid "
    "solver instead."
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


def complement_directions_from_omitted_direction(
    omitted_direction: np.ndarray,
    bar_sigma: np.ndarray,
) -> np.ndarray:
    """Return a rank-3 basis omitting ``omitted_direction``.

    The basis is ordered by variance under the average covariance. This affects
    display and summaries, not the rank-3 projector used in the plotted losses.
    """
    omitted_direction = np.asarray(omitted_direction, dtype=float)
    omitted_direction = omitted_direction / np.linalg.norm(omitted_direction)
    projector = np.eye(omitted_direction.size) - np.outer(omitted_direction, omitted_direction)
    values, vectors = np.linalg.eigh(projector)
    complement = vectors[:, np.argsort(values)[::-1][:N_COMPONENTS]]

    restricted = complement.T @ bar_sigma @ complement
    restricted = 0.5 * (restricted + restricted.T)
    values_restricted, rotations = np.linalg.eigh(restricted)
    order = np.argsort(values_restricted)[::-1]
    return complement @ rotations[:, order]


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


def solve_grid_wc_baseline(
    method_id: str,
    covariances: list[np.ndarray],
    basis: dict[str, np.ndarray],
    *,
    n_b: int,
    n_theta: int,
) -> FittedMethod:
    """Solve a wcPCA baseline in this example's omitted-direction chart.

    Every rank-3 candidate considered by these baselines contains the invariant
    direction ``a``. Its orthogonal complement can therefore be parameterized by
    one unit vector in span(b, c3, c4). This is kept as a transparent,
    package-free fallback and sanity check. The default script path uses the
    external minPCA package instead.
    """
    b_grid = np.linspace(-1.0, 1.0, int(n_b))
    theta_grid = np.linspace(0.0, np.pi, int(n_theta), endpoint=False)
    cos_theta = np.cos(theta_grid)
    sin_theta = np.sin(theta_grid)

    chart_basis = np.column_stack([basis["b"], basis["c3"], basis["c4"]])
    block_covariances = np.asarray(
        [chart_basis.T @ covariance @ chart_basis for covariance in covariances],
        dtype=float,
    )
    trace_by_environment = np.asarray([np.trace(covariance) for covariance in covariances])
    best_rank3_loss = np.asarray(
        [np.linalg.eigvalsh(covariance).min() for covariance in covariances],
        dtype=float,
    )

    best_value = np.inf
    best_b = 0.0
    best_theta = 0.0

    for b_coef in b_grid:
        radius = np.sqrt(max(0.0, 1.0 - float(b_coef) ** 2))
        candidates = np.vstack(
            [
                np.full_like(theta_grid, b_coef),
                radius * cos_theta,
                radius * sin_theta,
            ]
        )
        losses = np.einsum(
            "it,eij,jt->et",
            candidates,
            block_covariances,
            candidates,
            optimize=True,
        )

        if method_id == "maxRCS":
            objective = losses.max(axis=0)
        elif method_id == "maxRegret":
            objective = (losses - best_rank3_loss[:, None]).max(axis=0)
        elif method_id == "norm-maxRegret":
            objective = ((losses - best_rank3_loss[:, None]) / trace_by_environment[:, None]).max(axis=0)
        else:
            raise ValueError(f"Unknown wcPCA baseline: {method_id}")

        candidate_index = int(np.argmin(objective))
        candidate_value = float(objective[candidate_index])
        if candidate_value < best_value:
            best_value = candidate_value
            best_b = float(b_coef)
            best_theta = float(theta_grid[candidate_index])

    best_radius = np.sqrt(max(0.0, 1.0 - best_b**2))
    omitted_direction = chart_basis @ np.array(
        [best_b, best_radius * np.cos(best_theta), best_radius * np.sin(best_theta)]
    )
    omitted_direction = omitted_direction / np.linalg.norm(omitted_direction)
    orientation_index = int(np.argmax(np.abs(omitted_direction)))
    if omitted_direction[orientation_index] < 0:
        omitted_direction = -omitted_direction

    bar_sigma = np.mean(covariances, axis=0)
    directions = complement_directions_from_omitted_direction(omitted_direction, bar_sigma)
    return FittedMethod(
        method_id=method_id,
        directions=directions,
        objective_value=best_value,
        fit_source="grid",
    )


def fit_methods(
    covariances: list[np.ndarray],
    basis: dict[str, np.ndarray],
    *,
    wc_source: str,
    wc_grid_b: int,
    wc_grid_theta: int,
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
        if wc_source == "minpca":
            methods[method_id] = fit_minpca_wc_baseline(
                method_id,
                covariances,
                n_restarts=minpca_restarts,
                n_iters=minpca_iters,
                lr=minpca_lr,
                seed=minpca_seed,
            )
        elif wc_source == "grid":
            methods[method_id] = solve_grid_wc_baseline(
                method_id,
                covariances,
                basis,
                n_b=wc_grid_b,
                n_theta=wc_grid_theta,
            )
        else:
            raise ValueError(f"Unknown wc_source: {wc_source}")

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
    wc_source: str,
    wc_grid_b: int,
    wc_grid_theta: int,
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
            f"baseline projectors from wc_source={wc_source!r}. The crossover CSV records "
            "adjacent lower-envelope transitions; the pairwise crossover CSV records all "
            "finite pairwise intersections in the plotted rho range."
        ),
        "lambda_value": LAMBDA_VALUE,
        "rho_design": RHO_DESIGN,
        "rho_grid_min": float(rho_grid.min()),
        "rho_grid_max": float(rho_grid.max()),
        "rho_grid_size": int(rho_grid.size),
        "wc_source": wc_source,
        "wc_grid_b": int(wc_grid_b),
        "wc_grid_theta": int(wc_grid_theta),
        "minpca_restarts": int(minpca_restarts),
        "minpca_iters": int(minpca_iters),
        "minpca_lr": float(minpca_lr),
        "minpca_seed": int(minpca_seed),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "perturbation_path_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def plot_perturbation_path(
    figures_dir: Path,
    line_parameters: dict[str, LineParameters],
    rho_grid: np.ndarray,
    rho_zoom: np.ndarray,
    curves: dict[str, np.ndarray],
    curves_zoom: dict[str, np.ndarray],
    crossovers: list[tuple[str, str, float]],
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for method_id in METHOD_ORDER:
        ax.plot(
            rho_grid,
            curves[method_id],
            label=METHOD_LABELS[method_id],
            **METHOD_STYLES[method_id],
        )

    y_min = min(float(curves[method_id].min()) for method_id in METHOD_ORDER)
    y_max = max(float(curves[method_id].max()) for method_id in METHOD_ORDER)
    y_padding = 0.035 * (y_max - y_min)
    ax.set_xlim(0.0, 500.0)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    y0, y1 = ax.get_ylim()
    reference_line_top = min(500.0, y1)
    ax.axvline(
        RHO_DESIGN,
        ymin=0.0,
        ymax=(reference_line_top - y0) / (y1 - y0),
        color="black",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.text(
        RHO_DESIGN,
        y0 + 0.035 * (y1 - y0),
        rf"$\rho_0 = 2E\lambda = {RHO_DESIGN:.0f}$",
        va="bottom",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )

    ax.set_xlabel(r"Perturbation strength $\rho$", labelpad=8)
    ax.set_ylabel("Avg. reconstruction error")

    axins = ax.inset_axes([0.06, 0.57, 0.48, 0.42])
    for method_id in METHOD_ORDER:
        axins.plot(rho_zoom, curves_zoom[method_id], **METHOD_STYLES[method_id])

    for _, _, rho_star in crossovers:
        if rho_zoom[0] <= rho_star <= rho_zoom[-1]:
            axins.axvline(rho_star, color="black", linestyle=":", linewidth=1.0, alpha=0.55)

    zoom_values = np.concatenate([curves_zoom[method_id] for method_id in METHOD_ORDER])
    axins.set_xlim(0.0, 25.0)
    axins.set_ylim(float(zoom_values.min()) - 1.0, float(zoom_values.max()) + 1.0)
    axins.set_xticks([0, 5, 10, 15, 20, 25])
    axins.set_xticklabels(["0", "5", "10", "15", "20", ""])
    axins.tick_params(labelsize=8, pad=2, length=2)
    ax.indicate_inset_zoom(axins, edgecolor="0.35", linewidth=1.0)

    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.94, 0.04),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.78",
        fontsize=8,
        columnspacing=1.2,
        handlelength=2.6,
        handletextpad=0.55,
        labelspacing=0.35,
        borderpad=0.35,
    )

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.965)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figures_dir / "perturbation_path.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        figures_dir / "perturbation_path.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def print_summary(
    line_parameters: dict[str, LineParameters],
    crossovers: list[tuple[str, str, float]],
    *,
    wc_source: str,
) -> None:
    print("Perturbation-path line parameters")
    print("---------------------------------")
    print(f"wcPCA fit source: {wc_source}")
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
    wc_source: str,
    wc_grid_b: int,
    wc_grid_theta: int,
    minpca_restarts: int,
    minpca_iters: int,
    minpca_lr: float,
    minpca_seed: int,
) -> None:
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"

    model = build_population_model()
    basis = model["basis"]
    covariances = model["covariances"]
    local_projectors = model["local_projectors"]

    methods = fit_methods(
        covariances,
        basis,
        wc_source=wc_source,
        wc_grid_b=wc_grid_b,
        wc_grid_theta=wc_grid_theta,
        minpca_restarts=minpca_restarts,
        minpca_iters=minpca_iters,
        minpca_lr=minpca_lr,
        minpca_seed=minpca_seed,
    )
    line_parameters = build_line_parameters(methods, covariances, local_projectors)

    rho_grid = np.arange(0.0, float(rho_max) + 0.5 * float(rho_step), float(rho_step))
    rho_zoom = np.linspace(0.0, 25.0, 1001)
    curves = {
        method_id: line_parameters[method_id].evaluate(rho_grid)
        for method_id in METHOD_ORDER
    }
    curves_zoom = {
        method_id: line_parameters[method_id].evaluate(rho_zoom)
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
        wc_source=wc_source,
        wc_grid_b=wc_grid_b,
        wc_grid_theta=wc_grid_theta,
        minpca_restarts=minpca_restarts,
        minpca_iters=minpca_iters,
        minpca_lr=minpca_lr,
        minpca_seed=minpca_seed,
    )
    plot_perturbation_path(
        figures_dir,
        line_parameters,
        rho_grid,
        rho_zoom,
        curves,
        curves_zoom,
        crossovers,
    )
    print_summary(line_parameters, crossovers, wc_source=wc_source)
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
        "--wc-source",
        choices=["minpca", "grid"],
        default="minpca",
        help=(
            "How to fit wcPCA baselines. 'minpca' uses the external minPCA package; "
            "'grid' uses the deterministic 4D omitted-direction solver."
        ),
    )
    parser.add_argument(
        "--wc-grid-b",
        type=int,
        default=5001,
        help="Number of b-coordinate grid points when --wc-source grid.",
    )
    parser.add_argument(
        "--wc-grid-theta",
        type=int,
        default=7201,
        help="Number of angular grid points when --wc-source grid.",
    )
    parser.add_argument(
        "--minpca-restarts",
        type=int,
        default=60,
        help="Number of minPCA random restarts when --wc-source minpca.",
    )
    parser.add_argument(
        "--minpca-iters",
        type=int,
        default=1800,
        help="Number of minPCA optimization iterations when --wc-source minpca.",
    )
    parser.add_argument(
        "--minpca-lr",
        type=float,
        default=0.05,
        help="minPCA optimizer learning rate when --wc-source minpca.",
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
        wc_source=args.wc_source,
        wc_grid_b=args.wc_grid_b,
        wc_grid_theta=args.wc_grid_theta,
        minpca_restarts=args.minpca_restarts,
        minpca_iters=args.minpca_iters,
        minpca_lr=args.minpca_lr,
        minpca_seed=args.minpca_seed,
    )


if __name__ == "__main__":
    main()

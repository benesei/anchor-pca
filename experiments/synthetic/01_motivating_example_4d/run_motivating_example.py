"""Reproduce the four-dimensional motivating example.

The script builds the population covariance model from the paper, fits
poolPCA, AnchorPCA_lambda with lambda=25, and AnchorPCA_infty, then writes:

- the quotient-space geometry figure,
- the nuisance-plane figure,
- the squared-loading figure,
- method metrics and reconstruction-error tables.

Run from this directory or from the repository root:

    python experiments/synthetic/01_motivating_example_4d/run_motivating_example.py

Generated files are written to local ``figures/`` and ``results/`` folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def find_repo_root() -> Path:
    """Find the repository root so the local package can be imported."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "anchorpca").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing src/anchorpca.")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))

from anchorpca import (  # noqa: E402
    AnchorPCAInfty,
    AnchorPCALambda,
    explained_variance,
    pool_pca_from_covariances,
)
from anchorpca.reproducibility import software_versions  # noqa: E402


LAMBDA_VALUE = 25.0
N_COMPONENTS = 3
N_ENVIRONMENTS = 3
RHO_DESIGN = 2.0 * N_ENVIRONMENTS * LAMBDA_VALUE

PAPER_RECONSTRUCTION_TABLE = {
    "poolPCA": (93.3, 243.3),
    "AnchorPCA_lambda=25": (98.5, 163.4),
    "AnchorPCA_infty": (113.8, 172.5),
}

COLORS = {
    "u": "#59A14F",
    "v": "#9C755F",
    "w": "#E15759",
    "pool": "#7f7f7f",
    "anchor": "#1f77b4",
    "lambda": "#1f77b4",
    "infty": "#ff7f0e",
    "invariant": "#000000",
}

FRAME_LS = ":"
SSTAR_LABEL = r"$\mathcal{S}_\star$"
ANCHOR_PLOT_LABEL = r"AnchorPCA$_{\lambda=25}$"


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


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return vector / norm


def build_population_model() -> dict[str, object]:
    """Build the exact population model from the paper appendix."""
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
        "nuisance": {
            "u": u,
            "v": v,
            "w": w,
            "u_perp": u_perp,
            "v_perp": v_perp,
            "w_perp": w_perp,
        },
        "covariances": covariances,
        "local_projectors": local_projectors,
    }


def fit_methods(covariances: list[np.ndarray]) -> dict[str, np.ndarray]:
    pool = pool_pca_from_covariances(covariances, n_components=N_COMPONENTS)
    finite = AnchorPCALambda(
        n_components=N_COMPONENTS,
        lambda_=LAMBDA_VALUE,
    ).fit_covariances(covariances)
    hard = AnchorPCAInfty(
        n_components=N_COMPONENTS,
        block_tol=1e-12,
    ).fit_covariances(covariances)
    return {
        "poolPCA": pool["directions"],
        "AnchorPCA_lambda=25": finite.directions_,
        "AnchorPCA_infty": hard.directions_,
    }


def projector_from_directions(directions: np.ndarray) -> np.ndarray:
    return directions @ directions.T


def subspace_capture(directions: np.ndarray, vector: np.ndarray) -> float:
    return float(np.linalg.norm(directions.T @ vector) ** 2)


def nuisance_angle(directions: np.ndarray) -> float:
    nuisance_part = normalize(directions[:, 2][2:4])
    angle = float(np.degrees(np.arctan2(nuisance_part[1], nuisance_part[0])))
    if angle < 0:
        angle += 180.0
    return angle


def build_metric_rows(
    methods: dict[str, np.ndarray],
    covariances: list[np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method, directions in methods.items():
        stats = explained_variance(directions, covariances)
        rows.append(
            {
                "method": method,
                "domain_1_ev": stats["per_env"][0],
                "domain_2_ev": stats["per_env"][1],
                "domain_3_ev": stats["per_env"][2],
                "average_ev": stats["average"],
                "worst_case_ev": stats["worst_case"],
                "capture_a": subspace_capture(directions, a),
                "capture_b": subspace_capture(directions, b),
                "third_direction_angle_deg": nuisance_angle(directions),
            }
        )
    return rows


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
            [np.trace((identity - projection) @ local_projector) for local_projector in local_projectors]
        )
    )
    return intercept, slope


def build_reconstruction_rows(
    methods: dict[str, np.ndarray],
    covariances: list[np.ndarray],
    local_projectors: list[np.ndarray],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method, directions in methods.items():
        intercept, slope = reconstruction_line_parameters(
            projector_from_directions(directions),
            covariances,
            local_projectors,
        )
        rows.append(
            {
                "method": method,
                "unperturbed_average_reconstruction_error": intercept,
                "perturbed_average_reconstruction_error": intercept + slope * RHO_DESIGN,
                "perturbation_slope": slope,
            }
        )
    return rows


def check_paper_reconstruction_table(rows: list[dict[str, float | str]]) -> None:
    rounded = {
        str(row["method"]): (
            round(float(row["unperturbed_average_reconstruction_error"]), 1),
            round(float(row["perturbed_average_reconstruction_error"]), 1),
        )
        for row in rows
    }
    if rounded != PAPER_RECONSTRUCTION_TABLE:
        raise AssertionError(
            "Reconstruction table does not match the manuscript values.\n"
            f"Computed: {rounded}\n"
            f"Expected: {PAPER_RECONSTRUCTION_TABLE}"
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reconstruction_markdown(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Method | Original covariances | Perturbed covariances |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {original:.1f} | {perturbed:.1f} |".format(
                method=row["method"],
                original=float(row["unperturbed_average_reconstruction_error"]),
                perturbed=float(row["perturbed_average_reconstruction_error"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def nuisance_xy(vector4: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(vector4, dtype=float)[2:4])


def save_figure(
    fig,
    figures_dir: Path,
    stem: str,
    pad_inches: float | None = None,
    dpi: int = 220,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"}
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    fig.savefig(figures_dir / f"{stem}.png", dpi=dpi, **kwargs)
    fig.savefig(figures_dir / f"{stem}.pdf", **kwargs)


def plane_quad_from_direction(
    direction_xy: np.ndarray,
    z_half: float = 1.1,
    xy_half: float = 1.0,
) -> list[np.ndarray]:
    direction_xy = normalize(direction_xy)
    return [
        np.array([-xy_half * direction_xy[0], -xy_half * direction_xy[1], -z_half]),
        np.array([xy_half * direction_xy[0], xy_half * direction_xy[1], -z_half]),
        np.array([xy_half * direction_xy[0], xy_half * direction_xy[1], z_half]),
        np.array([-xy_half * direction_xy[0], -xy_half * direction_xy[1], z_half]),
    ]


def plot_plane(ax, quad, color, alpha, linewidth=1.5, frame_ls=FRAME_LS):
    poly = Poly3DCollection(
        [quad],
        alpha=alpha,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
    )
    ax.add_collection3d(poly)
    outline = np.array([*quad, quad[0]])
    ax.plot(
        outline[:, 0],
        outline[:, 1],
        outline[:, 2],
        color=color,
        lw=linewidth,
        ls=frame_ls,
        alpha=0.95,
    )


def plot_direction_3d(
    ax,
    direction_xy,
    color,
    label,
    *,
    length=1.08,
    lw=1.9,
    ls="--",
    alpha=0.85,
    anchor_scale=0.90,
    label_radial=0.08,
    label_perp=0.18,
    label_vertical=0.10,
    fontsize=9.5,
    fontweight="bold",
):
    direction_xy = normalize(direction_xy)
    direction = np.array([direction_xy[0], direction_xy[1], 0.0])
    perpendicular = np.array([-direction[1], direction[0], 0.0])
    segment = np.vstack([-length * direction, length * direction])
    ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=lw, ls=ls, alpha=alpha)
    label_pos = (
        anchor_scale * direction
        + label_radial * direction
        + label_perp * perpendicular
        + np.array([0.0, 0.0, label_vertical])
    )
    ax.text(
        label_pos[0],
        label_pos[1],
        label_pos[2],
        label,
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
        fontweight=fontweight,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.88,
            "linewidth": 0.85,
        },
    )


def style_legend(legend, mapping, emphasized_labels=()):
    for text in legend.get_texts():
        label = text.get_text()
        if label in mapping:
            text.set_color(mapping[label])
        if label in emphasized_labels:
            text.set_fontsize(11.5)
            text.set_fontweight("bold")


def style_geometry_axes(ax):
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_xlabel(r"$c_3$", fontsize=9, labelpad=1)
    ax.set_ylabel(r"$c_4$", fontsize=9, labelpad=1)
    ax.set_zlabel(r"$b$", fontsize=9, labelpad=0)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=7, pad=0)
    ax.tick_params(axis="y", labelsize=7, pad=0)
    ax.tick_params(axis="z", labelsize=7, pad=0)
    ax.set_box_aspect((1, 1, 0.70))
    ax.view_init(elev=20, azim=-55)


def plot_quotient_geometry(
    figures_dir: Path,
    nuisance: dict[str, np.ndarray],
    methods: dict[str, np.ndarray],
) -> None:
    u_xy = nuisance_xy(nuisance["u"])
    v_xy = nuisance_xy(nuisance["v"])
    w_xy = nuisance_xy(nuisance["w"])
    anchor_xy = nuisance_xy(methods["AnchorPCA_lambda=25"][:, 2])
    infty_xy = nuisance_xy(methods["AnchorPCA_infty"][:, 2])

    fig = plt.figure(figsize=(9.6, 3.20))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in [ax_left, ax_right]:
        style_geometry_axes(ax)
        ax.plot([0, 0], [0, 0], [-1.15, 1.15], color=COLORS["invariant"], lw=2.3)
    ax_right.set_zlabel("")
    ax_right.text2D(
        0.92,
        0.51,
        r"$b$",
        transform=ax_right.transAxes,
        fontsize=9,
        ha="center",
        va="center",
    )

    label_specs = {
        "u": {"label_radial": 0.08, "label_perp": 0.20, "label_vertical": 0.10},
        "v": {"label_radial": 0.10, "label_perp": -0.18, "label_vertical": 0.18},
        "w": {"label_radial": 0.10, "label_perp": 0.18, "label_vertical": 0.04},
    }
    left_handles = []
    for domain_label, name, direction_xy in [
        ("Domain 1", "u", u_xy),
        ("Domain 2", "v", v_xy),
        ("Domain 3", "w", w_xy),
    ]:
        plot_plane(
            ax_left,
            plane_quad_from_direction(direction_xy),
            COLORS[name],
            alpha=0.18,
            linewidth=1.35,
        )
        plot_direction_3d(ax_left, direction_xy, COLORS[name], rf"${name}$", **label_specs[name])
        left_handles.append(
            Patch(
                facecolor=COLORS[name],
                edgecolor=COLORS[name],
                alpha=0.18,
                linestyle=FRAME_LS,
                label=domain_label,
            )
        )

    pool_quad = [
        np.array([-1.05, -1.05, 0.0]),
        np.array([1.05, -1.05, 0.0]),
        np.array([1.05, 1.05, 0.0]),
        np.array([-1.05, 1.05, 0.0]),
    ]
    plot_plane(ax_right, pool_quad, COLORS["pool"], alpha=0.18)

    for direction_xy, color in [
        (anchor_xy, COLORS["anchor"]),
        (infty_xy, COLORS["infty"]),
    ]:
        plot_plane(
            ax_right,
            plane_quad_from_direction(direction_xy),
            color,
            alpha=0.20,
            linewidth=1.5,
        )

    right_label_specs = {
        "u": {"label_radial": 0.08, "label_perp": 0.22, "label_vertical": 0.10},
        "v": {
            "anchor_scale": 0.82,
            "label_radial": 0.18,
            "label_perp": -0.28,
            "label_vertical": 0.20,
        },
        "w": {"label_radial": 0.10, "label_perp": 0.18, "label_vertical": 0.04},
    }
    for name, direction_xy in [("u", u_xy), ("v", v_xy), ("w", w_xy)]:
        plot_direction_3d(
            ax_right,
            direction_xy,
            COLORS[name],
            rf"${name}$",
            alpha=0.78,
            **right_label_specs[name],
        )

    invariant_handle = Line2D([0], [0], color=COLORS["invariant"], lw=2.3, label=SSTAR_LABEL)
    legend_left = ax_left.legend(
        handles=[*left_handles, invariant_handle],
        loc="upper left",
        bbox_to_anchor=(0.07, 0.84),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.78",
        fontsize=9,
        borderaxespad=0.0,
        handlelength=1.3,
        handletextpad=0.42,
        labelspacing=0.24,
        borderpad=0.25,
    )
    style_legend(
        legend_left,
        {
            "Domain 1": COLORS["u"],
            "Domain 2": COLORS["v"],
            "Domain 3": COLORS["w"],
            SSTAR_LABEL: COLORS["invariant"],
        },
        emphasized_labels={SSTAR_LABEL},
    )

    legend_right = ax_right.legend(
        handles=[
            Patch(
                facecolor=COLORS["pool"],
                edgecolor=COLORS["pool"],
                alpha=0.18,
                linestyle=FRAME_LS,
                label="poolPCA",
            ),
            Patch(
                facecolor=COLORS["anchor"],
                edgecolor=COLORS["anchor"],
                alpha=0.20,
                linestyle=FRAME_LS,
                label=ANCHOR_PLOT_LABEL,
            ),
            Patch(
                facecolor=COLORS["infty"],
                edgecolor=COLORS["infty"],
                alpha=0.20,
                linestyle=FRAME_LS,
                label=r"AnchorPCA$_{\infty}$",
            ),
            invariant_handle,
        ],
        loc="upper left",
        bbox_to_anchor=(0.51, 0.84),
        bbox_transform=fig.transFigure,
        frameon=True,
        fancybox=False,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.78",
        fontsize=9,
        borderaxespad=0.0,
        handlelength=1.3,
        handletextpad=0.42,
        labelspacing=0.24,
        borderpad=0.25,
    )
    style_legend(
        legend_right,
        {
            "poolPCA": COLORS["pool"],
            r"AnchorPCA$_{\lambda = 25}$": COLORS["anchor"],
            r"AnchorPCA$_{\infty}$": COLORS["infty"],
            SSTAR_LABEL: COLORS["invariant"],
        },
        emphasized_labels={SSTAR_LABEL},
    )

    fig.subplots_adjust(top=0.998, wspace=0.025, left=0.005, right=0.998, bottom=0.00)
    position = ax_left.get_position()
    ax_left.set_position([position.x0 + 0.022, position.y0, position.width, position.height])
    save_figure(fig, figures_dir, "4d_motivation_exp_quotient_view", pad_inches=0.01, dpi=200)
    plt.close(fig)


def draw_arrow_2d(ax, vector_xy, color, label, offset=(0.02, 0.02), lw=2.4):
    ax.arrow(
        0,
        0,
        vector_xy[0],
        vector_xy[1],
        length_includes_head=True,
        head_width=0.035,
        head_length=0.06,
        linewidth=lw,
        color=color,
        zorder=3,
    )
    ax.text(
        vector_xy[0] + offset[0],
        vector_xy[1] + offset[1],
        label,
        color=color,
        fontsize=12,
        weight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.0,
            "alpha": 0.85,
        },
    )


def plot_nuisance_plane(
    figures_dir: Path,
    nuisance: dict[str, np.ndarray],
    methods: dict[str, np.ndarray],
) -> None:
    u_xy = nuisance_xy(nuisance["u"])
    v_xy = nuisance_xy(nuisance["v"])
    w_xy = nuisance_xy(nuisance["w"])
    lambda_xy = nuisance_xy(methods["AnchorPCA_lambda=25"][:, 2])
    infty_xy = nuisance_xy(methods["AnchorPCA_infty"][:, 2])

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.add_patch(Circle((0, 0), radius=1.0, edgecolor="#bbbbbb", facecolor="none", lw=1.3))
    ax.axhline(0, color="#bbbbbb", lw=1.1)
    ax.axvline(0, color="#bbbbbb", lw=1.1)

    draw_arrow_2d(ax, u_xy, COLORS["u"], r"$u$", offset=(0.03, -0.12))
    draw_arrow_2d(ax, v_xy, COLORS["v"], r"$v$", offset=(0.04, 0.03))
    draw_arrow_2d(ax, w_xy, COLORS["w"], r"$w$", offset=(-0.18, 0.04))
    draw_arrow_2d(
        ax,
        lambda_xy,
        COLORS["lambda"],
        r"AnchorPCA$_{\lambda=25}$",
        offset=(-0.28, 0.10),
        lw=3.0,
    )
    draw_arrow_2d(
        ax,
        infty_xy,
        COLORS["infty"],
        r"AnchorPCA$_{\infty}$",
        offset=(0.03, -0.12),
        lw=3.0,
    )
    ax.text(
        -1.16,
        -1.10,
        r"poolPCA quotient plane = span$(c_3,c_4)$",
        color=COLORS["pool"],
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": COLORS["pool"],
            "linewidth": 1.0,
            "alpha": 0.85,
        },
    )
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$c_3$")
    ax.set_ylabel(r"$c_4$")
    ax.set_title("Nuisance-plane view", fontsize=18)
    ax.grid(alpha=0.25)
    save_figure(fig, figures_dir, "4d_motivation_exp_nuisance_plane")
    plt.close(fig)


def plot_squared_loadings(figures_dir: Path, methods: dict[str, np.ndarray]) -> None:
    order = ["poolPCA", "AnchorPCA_lambda=25", "AnchorPCA_infty"]
    titles = {
        "poolPCA": "poolPCA",
        "AnchorPCA_lambda=25": r"AnchorPCA$_{\lambda=25}$",
        "AnchorPCA_infty": r"AnchorPCA$_{\infty}$",
    }
    title_colors = {
        "poolPCA": COLORS["pool"],
        "AnchorPCA_lambda=25": COLORS["lambda"],
        "AnchorPCA_infty": COLORS["infty"],
    }
    row_labels = [r"$a$", r"$b$", r"$c_3$", r"$c_4$"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    image = None
    for ax, method in zip(axes, order):
        loadings = methods[method] ** 2
        image = ax.imshow(loadings, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(N_COMPONENTS))
        ax.set_xticklabels([r"$d_1$", r"$d_2$", r"$d_3$"], fontsize=12)
        ax.set_yticks(range(4))
        ax.set_yticklabels(row_labels, fontsize=12)
        ax.set_title(titles[method], color=title_colors[method], fontsize=18, weight="bold")
        for i in range(loadings.shape[0]):
            for j in range(loadings.shape[1]):
                ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)

    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.05)
    colorbar.set_label(r"$|\langle \mathrm{coordinate}, d_j\rangle|^2$", fontsize=12)
    save_figure(fig, figures_dir, "4d_motivation_exp_loadings")
    plt.close(fig)


def print_reconstruction_table(rows: list[dict[str, float | str]]) -> None:
    print("Average reconstruction error table")
    print("----------------------------------")
    for row in rows:
        print(
            "{method:22s} original={original:6.1f}  perturbed={perturbed:6.1f}".format(
                method=str(row["method"]),
                original=float(row["unperturbed_average_reconstruction_error"]),
                perturbed=float(row["perturbed_average_reconstruction_error"]),
            )
        )


def run(output_dir: Path, *, include_diagnostics: bool = False) -> None:
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    model = build_population_model()
    basis = model["basis"]
    nuisance = model["nuisance"]
    covariances = model["covariances"]
    local_projectors = model["local_projectors"]

    methods = fit_methods(covariances)
    metric_rows = build_metric_rows(methods, covariances, basis["a"], basis["b"])
    reconstruction_rows = build_reconstruction_rows(methods, covariances, local_projectors)
    check_paper_reconstruction_table(reconstruction_rows)

    write_csv(results_dir / "motivating_example_metrics.csv", metric_rows)
    write_csv(results_dir / "motivating_example_reconstruction_errors.csv", reconstruction_rows)
    write_reconstruction_markdown(
        results_dir / "motivating_example_reconstruction_errors.md",
        reconstruction_rows,
    )
    (results_dir / "paper_table_check.json").write_text(
        json.dumps(
            {
                "software_versions": software_versions(),
                "rho_design": RHO_DESIGN,
                "paper_values_checked_at_one_decimal": PAPER_RECONSTRUCTION_TABLE,
                "status": "passed",
            },
            indent=2,
        )
        + "\n"
    )

    plot_quotient_geometry(figures_dir, nuisance, methods)
    if include_diagnostics:
        plot_nuisance_plane(figures_dir, nuisance, methods)
        plot_squared_loadings(figures_dir, methods)
    print_reconstruction_table(reconstruction_rows)
    print(f"\nWrote figures to {figures_dir}")
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
        "--include-diagnostics",
        action="store_true",
        help="Also write auxiliary diagnostic figures not used in the paper.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir.resolve(), include_diagnostics=bool(args.include_diagnostics))


if __name__ == "__main__":
    main()

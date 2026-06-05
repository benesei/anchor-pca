"""Generate random-subspace distribution draws with exact S_star.

This script creates the reusable distribution-draw bank used in the paper. It
stores covariances, true S_star bases/projectors, local top/bottom eigenspaces,
and population diagnostics. It deliberately does not sample finite datasets;
downstream simulation scripts load these population objects and sample fresh data
for their chosen sample sizes.

Run from the repository root:

    python experiments/synthetic/03_random_subspace_sstar/generate_design_bank.py

Generated files are written to ``results/design_bank`` next to this script by
default. The default seed is 42.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

from anchorpca.reproducibility import repo_relative_path, software_versions


CORE_GRID = (
    (3, 8, 3),
    (5, 8, 5),
    (5, 10, 5),
    (5, 10, 4),
)

EXTRA_GPKM_CONFIGS = (
    # Small-E robustness case: E < k, with minimal feasible m.
    (2, 8, 5, 2),
)

DEFAULT_SEED = 42
DEFAULT_N_DESIGNS_PER_CONFIG = 100
DEFAULT_RANK_TOL = 1e-10
DEFAULT_MIN_SPAN_SINGULAR_VALUE = 1e-8
DEFAULT_MAX_ATTEMPTS = 1000
LOCAL_EIGENGAP_MIN = 0.5


@dataclass(frozen=True)
class EigenvalueProfile:
    stable_top_range: tuple[float, float]
    environment_top_range: tuple[float, float]
    bottom_range: tuple[float, float]


PROFILES = {
    "balanced": EigenvalueProfile(
        stable_top_range=(5.0, 8.0),
        environment_top_range=(5.0, 8.0),
        bottom_range=(0.5, 3.0),
    ),
    "threshold_stress": EigenvalueProfile(
        stable_top_range=(3.2, 4.2),
        environment_top_range=(6.0, 9.0),
        bottom_range=(0.5, 2.5),
    ),
}


SUMMARY_FIELDNAMES = [
    "config_id",
    "replicate_id",
    "profile",
    "g",
    "p",
    "k",
    "m",
    "q",
    "design_seed",
    "sample_seed_base",
    "local_min_eigengap",
    "span_min_singular_value",
    "agreement_gap",
    "barPi_top_m_min",
    "barPi_after_m",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate random-subspace distribution draws with exact S_star."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "design_bank",
        help="Directory for metadata, summary CSV, and per-configuration npz files.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-draws-per-config",
        dest="n_draws_per_config",
        type=int,
        default=DEFAULT_N_DESIGNS_PER_CONFIG,
        help="Number of independent distribution draws per code-level (g,p,k,m,setting).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing new generated files.",
    )
    parser.add_argument("--rank-tol", type=float, default=DEFAULT_RANK_TOL)
    parser.add_argument(
        "--min-span-singular-value",
        type=float,
        default=DEFAULT_MIN_SPAN_SINGULAR_VALUE,
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser.parse_args()


def feasible_m_values(g: int, p: int, k: int) -> list[int]:
    if not (1 <= k < p):
        raise ValueError(f"Need 1 <= k < p, got k={k}, p={p}.")
    q = p - k
    return list(range(max(0, p - g * q), k + 1))


def config_id_for(g: int, p: int, k: int, m: int, profile: str) -> str:
    return f"{profile}_g{g}_p{p}_k{k}_m{m}"


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = ":".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32 - 1)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def projector_from_basis(basis: np.ndarray) -> np.ndarray:
    return symmetrize(np.asarray(basis, dtype=float) @ np.asarray(basis, dtype=float).T)


def eigvalsh_desc(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(symmetrize(matrix))[::-1]


def eigh_desc(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=float), vectors[:, order]


def haar_basis(rng: np.random.Generator, n_rows: int, n_cols: int | None = None) -> np.ndarray:
    if n_cols is None:
        n_cols = n_rows
    gaussian = rng.normal(size=(n_rows, n_cols))
    q, r = np.linalg.qr(gaussian, mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def draw_bottom_coordinate_bases(
    rng: np.random.Generator,
    *,
    g: int,
    d: int,
    q: int,
    rank_tol: float,
    min_span_singular_value: float,
    max_attempts: int,
) -> tuple[list[np.ndarray], float]:
    for _ in range(max_attempts):
        bases = [haar_basis(rng, d, q) for _ in range(g)]
        concatenated = np.concatenate(bases, axis=1)
        singular_values = np.linalg.svd(concatenated, compute_uv=False)
        numerical_rank = int(np.sum(singular_values > rank_tol))
        min_singular_value = float(singular_values[d - 1]) if singular_values.size >= d else 0.0
        if numerical_rank == d and min_singular_value >= min_span_singular_value:
            return bases, min_singular_value
    raise RuntimeError(
        "Could not draw environment-specific bottom spaces spanning S_star^perp "
        f"after {max_attempts} attempts."
    )


def draw_eigenvalues(
    rng: np.random.Generator,
    *,
    profile: EigenvalueProfile,
    m: int,
    k: int,
    q: int,
) -> tuple[np.ndarray, np.ndarray]:
    stable_count = m
    environment_count = k - m
    stable = rng.uniform(*profile.stable_top_range, size=stable_count)
    environment = rng.uniform(*profile.environment_top_range, size=environment_count)
    top = np.concatenate([stable, environment])
    bottom = rng.uniform(*profile.bottom_range, size=q)

    if top.size != k or bottom.size != q:
        raise RuntimeError("Internal eigenvalue draw produced inconsistent dimensions.")
    if float(np.min(top) - np.max(bottom)) <= LOCAL_EIGENGAP_MIN:
        raise RuntimeError("Generated eigenvalues violate the required local eigengap.")
    return top, bottom


def build_covariance(top_basis: np.ndarray, bottom_basis: np.ndarray, top_values: np.ndarray, bottom_values: np.ndarray) -> np.ndarray:
    covariance = (
        top_basis @ np.diag(top_values) @ top_basis.T
        + bottom_basis @ np.diag(bottom_values) @ bottom_basis.T
    )
    return symmetrize(covariance)


def validate_design(
    *,
    covariances: np.ndarray,
    sstar_projector: np.ndarray,
    top_bases: np.ndarray,
    bottom_bases: np.ndarray,
    top_projectors: np.ndarray,
    barPi: np.ndarray,
    agreement_eigenvalues: np.ndarray,
    agreement_gap: float,
    span_min_singular_value: float,
    g: int,
    p: int,
    k: int,
    m: int,
    rank_tol: float,
    min_span_singular_value: float,
) -> None:
    q = p - k
    d = p - m
    identity_k = np.eye(k)
    identity_q = np.eye(q)
    zero_kq = np.zeros((k, q))

    if not np.allclose(sstar_projector, sstar_projector.T, atol=1e-8):
        raise RuntimeError("S_star projector is not symmetric.")
    if span_min_singular_value < min_span_singular_value:
        raise RuntimeError("Bottom spaces do not span S_star^perp robustly enough.")

    bottom_concat = np.concatenate([bottom_bases[e] for e in range(g)], axis=1)
    bottom_rank = int(np.linalg.matrix_rank(bottom_concat, tol=rank_tol))
    if bottom_rank != d:
        raise RuntimeError(f"Bottom-space span rank is {bottom_rank}, expected {d}.")

    for e in range(g):
        top_basis = top_bases[e]
        bottom_basis = bottom_bases[e]
        covariance = covariances[e]

        if not np.allclose(covariance, covariance.T, atol=1e-8):
            raise RuntimeError(f"covariances[{e}] is not symmetric.")
        if float(np.min(np.linalg.eigvalsh(covariance))) <= rank_tol:
            raise RuntimeError(f"covariances[{e}] is not positive definite.")
        if not np.allclose(top_basis.T @ top_basis, identity_k, atol=1e-8):
            raise RuntimeError(f"top_bases[{e}] is not orthonormal.")
        if not np.allclose(bottom_basis.T @ bottom_basis, identity_q, atol=1e-8):
            raise RuntimeError(f"bottom_bases[{e}] is not orthonormal.")
        if not np.allclose(top_basis.T @ bottom_basis, zero_kq, atol=1e-8):
            raise RuntimeError(f"top_bases[{e}] is not orthogonal to bottom_bases[{e}].")

        _, eigenvectors = eigh_desc(covariance)
        empirical_top_projector = projector_from_basis(eigenvectors[:, :k])
        empirical_bottom_projector = projector_from_basis(eigenvectors[:, k:])
        if np.linalg.norm(empirical_top_projector - top_projectors[e], ord=2) > 1e-7:
            raise RuntimeError(f"top-k eigenspace of covariances[{e}] does not match U_e.")
        if np.linalg.norm(empirical_bottom_projector - projector_from_basis(bottom_basis), ord=2) > 1e-7:
            raise RuntimeError(f"bottom eigenspace of covariances[{e}] does not match B_e.")

    if not np.allclose(barPi, np.mean(top_projectors, axis=0), atol=1e-10):
        raise RuntimeError("barPi does not match the mean local top projectors.")

    if m > 0:
        if float(np.min(agreement_eigenvalues[:m])) < 1.0 - 1e-8:
            raise RuntimeError("barPi does not have eigenvalue 1 with multiplicity at least m.")
        if float(agreement_eigenvalues[m]) >= 1.0 - max(rank_tol, 1e-12):
            raise RuntimeError("barPi has an extra near-invariant direction outside S_star.")
    else:
        if float(agreement_eigenvalues[0]) >= 1.0 - max(rank_tol, 1e-12):
            raise RuntimeError("barPi is numerically invariant although m=0.")

    if not agreement_gap > max(rank_tol, 1e-12):
        raise RuntimeError("agreement_gap is not positive.")


def build_single_design(
    *,
    design_seed: int,
    g: int,
    p: int,
    k: int,
    m: int,
    profile_name: str,
    rank_tol: float,
    min_span_singular_value: float,
    max_attempts: int,
) -> dict[str, object]:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown eigenvalue profile: {profile_name}.")
    if m not in feasible_m_values(g, p, k):
        raise ValueError(f"m={m} is infeasible for g={g}, p={p}, k={k}.")

    profile = PROFILES[profile_name]
    q = p - k
    d = p - m
    rng = np.random.default_rng(design_seed)

    last_error = None
    for _ in range(max_attempts):
        try:
            Q = haar_basis(rng, p, p)
            sstar_basis = Q[:, :m]
            sstar_perp_basis = Q[:, m:]
            sstar_projector = projector_from_basis(sstar_basis)

            coordinate_bottom_bases, span_min_singular_value = draw_bottom_coordinate_bases(
                rng,
                g=g,
                d=d,
                q=q,
                rank_tol=rank_tol,
                min_span_singular_value=min_span_singular_value,
                max_attempts=max_attempts,
            )

            covariances = []
            top_bases = []
            bottom_bases = []
            top_projectors = []
            top_eigenvalues = []
            bottom_eigenvalues = []
            local_gaps = []

            for coordinate_bottom_basis in coordinate_bottom_bases:
                bottom_basis = sstar_perp_basis @ coordinate_bottom_basis
                coordinate_top_complement = null_space(coordinate_bottom_basis.T)
                if coordinate_top_complement.shape != (d, k - m):
                    raise RuntimeError(
                        "Unexpected nullspace shape while constructing environment-specific top space."
                    )
                top_basis = np.concatenate(
                    [sstar_basis, sstar_perp_basis @ coordinate_top_complement],
                    axis=1,
                )

                top_values, bottom_values = draw_eigenvalues(
                    rng,
                    profile=profile,
                    m=m,
                    k=k,
                    q=q,
                )
                covariance = build_covariance(top_basis, bottom_basis, top_values, bottom_values)

                covariances.append(covariance)
                top_bases.append(top_basis)
                bottom_bases.append(bottom_basis)
                top_projectors.append(projector_from_basis(top_basis))
                top_eigenvalues.append(top_values)
                bottom_eigenvalues.append(bottom_values)
                local_gaps.append(float(np.min(top_values) - np.max(bottom_values)))

            covariances_array = np.asarray(covariances, dtype=float)
            top_bases_array = np.asarray(top_bases, dtype=float)
            bottom_bases_array = np.asarray(bottom_bases, dtype=float)
            top_projectors_array = np.asarray(top_projectors, dtype=float)
            top_eigenvalues_array = np.asarray(top_eigenvalues, dtype=float)
            bottom_eigenvalues_array = np.asarray(bottom_eigenvalues, dtype=float)
            barPi = symmetrize(np.mean(top_projectors_array, axis=0))
            barSigma = symmetrize(np.mean(covariances_array, axis=0))
            agreement_eigenvalues = eigvalsh_desc(barPi)
            agreement_gap = float(1.0 - agreement_eigenvalues[m])
            local_min_eigengap = float(np.min(local_gaps))

            validate_design(
                covariances=covariances_array,
                sstar_projector=sstar_projector,
                top_bases=top_bases_array,
                bottom_bases=bottom_bases_array,
                top_projectors=top_projectors_array,
                barPi=barPi,
                agreement_eigenvalues=agreement_eigenvalues,
                agreement_gap=agreement_gap,
                span_min_singular_value=span_min_singular_value,
                g=g,
                p=p,
                k=k,
                m=m,
                rank_tol=rank_tol,
                min_span_singular_value=min_span_singular_value,
            )

            return {
                "covariances": covariances_array,
                "sstar_basis": sstar_basis,
                "sstar_projector": sstar_projector,
                "top_bases": top_bases_array,
                "bottom_bases": bottom_bases_array,
                "top_projectors": top_projectors_array,
                "barPi": barPi,
                "barSigma": barSigma,
                "top_eigenvalues": top_eigenvalues_array,
                "bottom_eigenvalues": bottom_eigenvalues_array,
                "agreement_eigenvalues": agreement_eigenvalues,
                "agreement_gap": agreement_gap,
                "local_min_eigengap": local_min_eigengap,
                "span_min_singular_value": span_min_singular_value,
            }
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to generate a valid design after {max_attempts} attempts. "
        f"Last validation error: {last_error}"
    )


def build_config_designs(
    *,
    master_seed: int,
    config_id: str,
    g: int,
    p: int,
    k: int,
    m: int,
    profile_name: str,
    n_designs: int,
    rank_tol: float,
    min_span_singular_value: float,
    max_attempts: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    designs = []
    summary_rows = []

    for replicate_id in range(n_designs):
        design_seed = stable_seed(master_seed, "design", config_id, replicate_id)
        sample_seed_base = stable_seed(master_seed, "sample", config_id, replicate_id)
        design = build_single_design(
            design_seed=design_seed,
            g=g,
            p=p,
            k=k,
            m=m,
            profile_name=profile_name,
            rank_tol=rank_tol,
            min_span_singular_value=min_span_singular_value,
            max_attempts=max_attempts,
        )
        design["design_seed"] = int(design_seed)
        design["sample_seed_base"] = int(sample_seed_base)
        designs.append(design)

        agreement_eigenvalues = np.asarray(design["agreement_eigenvalues"], dtype=float)
        row = {
            "config_id": config_id,
            "replicate_id": replicate_id,
            "profile": profile_name,
            "g": g,
            "p": p,
            "k": k,
            "m": m,
            "q": p - k,
            "design_seed": int(design_seed),
            "sample_seed_base": int(sample_seed_base),
            "local_min_eigengap": float(design["local_min_eigengap"]),
            "span_min_singular_value": float(design["span_min_singular_value"]),
            "agreement_gap": float(design["agreement_gap"]),
            "barPi_top_m_min": (
                float(np.min(agreement_eigenvalues[:m])) if m > 0 else np.nan
            ),
            "barPi_after_m": float(agreement_eigenvalues[m]),
        }
        summary_rows.append(row)

    arrays = {
        "covariances": np.asarray([design["covariances"] for design in designs], dtype=float),
        "sstar_bases": np.asarray([design["sstar_basis"] for design in designs], dtype=float),
        "sstar_projectors": np.asarray([design["sstar_projector"] for design in designs], dtype=float),
        "top_bases": np.asarray([design["top_bases"] for design in designs], dtype=float),
        "bottom_bases": np.asarray([design["bottom_bases"] for design in designs], dtype=float),
        "top_projectors": np.asarray([design["top_projectors"] for design in designs], dtype=float),
        "barPi": np.asarray([design["barPi"] for design in designs], dtype=float),
        "barSigma": np.asarray([design["barSigma"] for design in designs], dtype=float),
        "top_eigenvalues": np.asarray([design["top_eigenvalues"] for design in designs], dtype=float),
        "bottom_eigenvalues": np.asarray([design["bottom_eigenvalues"] for design in designs], dtype=float),
        "agreement_eigenvalues": np.asarray(
            [design["agreement_eigenvalues"] for design in designs], dtype=float
        ),
        "agreement_gap": np.asarray([design["agreement_gap"] for design in designs], dtype=float),
        "local_min_eigengap": np.asarray(
            [design["local_min_eigengap"] for design in designs], dtype=float
        ),
        "span_min_singular_value": np.asarray(
            [design["span_min_singular_value"] for design in designs], dtype=float
        ),
        "design_seed": np.asarray([design["design_seed"] for design in designs], dtype=np.uint32),
        "sample_seed_base": np.asarray(
            [design["sample_seed_base"] for design in designs], dtype=np.uint32
        ),
    }
    return arrays, summary_rows


def write_metadata(
    *,
    output_dir: Path,
    seed: int,
    n_draws_per_config: int,
    rank_tol: float,
    min_span_singular_value: float,
    max_attempts: int,
    configs: list[dict[str, object]],
) -> None:
    metadata = {
        "script": repo_relative_path(__file__),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": software_versions(),
        "master_seed": int(seed),
        "n_draws_per_config": int(n_draws_per_config),
        "rank_tol": float(rank_tol),
        "min_span_singular_value": float(min_span_singular_value),
        "max_attempts": int(max_attempts),
        "local_eigengap_min": float(LOCAL_EIGENGAP_MIN),
        "core_grid": [{"g": g, "p": p, "k": k} for g, p, k in CORE_GRID],
        "extra_gpkm_configs": [
            {"g": g, "p": p, "k": k, "m": m}
            for g, p, k, m in EXTRA_GPKM_CONFIGS
        ],
        "profiles": {
            name: {
                "stable_top_range": list(profile.stable_top_range),
                "environment_top_range": list(profile.environment_top_range),
                "bottom_range": list(profile.bottom_range),
            }
            for name, profile in PROFILES.items()
        },
        "configs": configs,
        "output_schema": {
            "npz_per_config": "designs/{config_id}.npz",
            "summary_csv": "design_summary.csv",
            "note": "No finite-sample data are stored; downstream simulations resample from covariances.",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty. "
                "Use --overwrite or choose a different --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "designs").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.n_draws_per_config <= 0:
        raise ValueError("--n-draws-per-config must be positive.")
    if args.rank_tol <= 0:
        raise ValueError("--rank-tol must be positive.")
    if args.min_span_singular_value <= 0:
        raise ValueError("--min-span-singular-value must be positive.")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")

    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    configs = []
    for g, p, k in CORE_GRID:
        for m in feasible_m_values(g, p, k):
            for profile_name in PROFILES:
                config_id = config_id_for(g, p, k, m, profile_name)
                configs.append(
                    {
                        "config_id": config_id,
                        "g": g,
                        "p": p,
                        "k": k,
                        "m": m,
                        "q": p - k,
                        "profile": profile_name,
                    }
                )
    existing_config_ids = {str(config["config_id"]) for config in configs}
    for g, p, k, m in EXTRA_GPKM_CONFIGS:
        if m not in feasible_m_values(g, p, k):
            raise ValueError(f"Extra config m={m} is infeasible for g={g}, p={p}, k={k}.")
        for profile_name in PROFILES:
            config_id = config_id_for(g, p, k, m, profile_name)
            if config_id in existing_config_ids:
                continue
            configs.append(
                {
                    "config_id": config_id,
                    "g": g,
                    "p": p,
                    "k": k,
                    "m": m,
                    "q": p - k,
                    "profile": profile_name,
                }
            )
            existing_config_ids.add(config_id)

    summary_rows = []
    for config in configs:
        config_id = str(config["config_id"])
        print(f"Generating {config_id} ({args.n_draws_per_config} distribution draws)")
        arrays, rows = build_config_designs(
            master_seed=int(args.seed),
            config_id=config_id,
            g=int(config["g"]),
            p=int(config["p"]),
            k=int(config["k"]),
            m=int(config["m"]),
            profile_name=str(config["profile"]),
            n_designs=int(args.n_draws_per_config),
            rank_tol=float(args.rank_tol),
            min_span_singular_value=float(args.min_span_singular_value),
            max_attempts=int(args.max_attempts),
        )
        np.savez_compressed(output_dir / "designs" / f"{config_id}.npz", **arrays)
        summary_rows.extend(rows)

    with (output_dir / "design_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_metadata(
        output_dir=output_dir,
        seed=int(args.seed),
        n_draws_per_config=int(args.n_draws_per_config),
        rank_tol=float(args.rank_tol),
        min_span_singular_value=float(args.min_span_singular_value),
        max_attempts=int(args.max_attempts),
        configs=configs,
    )
    print(
        f"Wrote {len(summary_rows)} designs across {len(configs)} configs to {output_dir}"
    )


if __name__ == "__main__":
    main()

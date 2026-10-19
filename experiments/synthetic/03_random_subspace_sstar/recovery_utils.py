"""Sampling and summaries shared by the paper recovery experiments."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "src" / "anchorpca").exists():
            return candidate
    raise RuntimeError("Could not locate repository root containing src/anchorpca.")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))

from anchorpca import AnchorPCAInfty, pool_pca_from_covariances  # noqa: E402


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = ":".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32 - 1)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def projector_from_basis(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, dtype=float)
    return symmetrize(basis @ basis.T)


def eig_desc(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(symmetrize(matrix))
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


def first_agreement_block_projector(
    barPi: np.ndarray,
    block_tol: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    rho, vectors = eig_desc(barPi)
    first_block = [0]
    for j in range(1, rho.size):
        if abs(float(rho[j]) - float(rho[first_block[0]])) <= float(block_tol):
            first_block.append(j)
        else:
            break
    basis = vectors[:, first_block]
    return projector_from_basis(basis), len(first_block), rho


def operator_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=float), ord=2))


def projector_metric_norm(matrix: np.ndarray) -> float:
    value = operator_norm(matrix)
    if value < 0.0 and value > -1e-12:
        return 0.0
    if value > 1.0 and value < 1.0 + 1e-10:
        return 1.0
    return value


def sample_gaussian(
    rng: np.random.Generator,
    covariance: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    p = covariance.shape[0]
    return rng.multivariate_normal(np.zeros(p), covariance, size=n_obs)


def min_eigval(matrix: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(symmetrize(matrix)).min())


def sample_gaussian_mixture_same_covariance(
    rng: np.random.Generator,
    covariance: np.ndarray,
    n_obs: int,
    *,
    mean_scale: float,
    max_tries: int = 50,
) -> np.ndarray:
    covariance = symmetrize(covariance)
    p = covariance.shape[0]
    values, vectors = eig_desc(covariance)
    direction = vectors[:, 0]
    base_scale = float(mean_scale) * np.sqrt(max(float(values[0]), 1e-12))

    for shrink in range(max_tries):
        scale = base_scale * (0.8**shrink)
        mu = scale * direction
        sigma0 = symmetrize(covariance - np.outer(mu, mu))
        if min_eigval(sigma0) >= -1e-10:
            labels = rng.binomial(1, 0.5, size=n_obs)
            means = np.where(labels[:, None] == 1, mu[None, :], -mu[None, :])
            noise = rng.multivariate_normal(np.zeros(p), sigma0, size=n_obs)
            return noise + means

    raise RuntimeError("Could not construct PSD within-component covariance.")


def sample_environments(
    *,
    rng: np.random.Generator,
    covariances: np.ndarray,
    n_obs: int,
    distribution: str,
    mean_scale: float,
) -> list[np.ndarray]:
    output = []
    for covariance in covariances:
        if distribution == "gaussian":
            output.append(sample_gaussian(rng, covariance, n_obs))
        elif distribution == "gaussian_mixture":
            output.append(
                sample_gaussian_mixture_same_covariance(
                    rng,
                    covariance,
                    n_obs,
                    mean_scale=mean_scale,
                )
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}.")
    return output


def read_design_summary(design_bank_dir: Path) -> pd.DataFrame:
    path = design_bank_dir / "design_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find design summary: {path}")
    return pd.read_csv(path)


def validate_requested_configs(summary: pd.DataFrame, config_ids: list[str]) -> None:
    available = set(summary["config_id"].astype(str))
    missing = sorted(set(config_ids).difference(available))
    if missing:
        raise ValueError(f"Requested configs are not in the distribution-draw bank: {missing}")
    bad = (
        summary[summary["config_id"].isin(config_ids)]
        .groupby("config_id", sort=False)["m"]
        .first()
    )
    m_zero = [config for config, m_value in bad.items() if int(m_value) == 0]
    if m_zero:
        raise ValueError(f"m=0 configs are excluded from this simulation: {m_zero}")


def load_config_arrays(design_bank_dir: Path, config_id: str) -> dict[str, np.ndarray]:
    path = design_bank_dir / "designs" / f"{config_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Could not find design file: {path}")
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def config_metadata(summary: pd.DataFrame, config_id: str) -> dict[str, object]:
    rows = summary[summary["config_id"] == config_id]
    if rows.empty:
        raise ValueError(f"No summary rows for config {config_id}.")
    first = rows.iloc[0]
    return {
        "config_id": config_id,
        "profile": str(first["profile"]),
        "g": int(first["g"]),
        "p": int(first["p"]),
        "k": int(first["k"]),
        "m": int(first["m"]),
        "q": int(first["q"]),
    }


def population_diagnostics_for_design(
    summary: pd.DataFrame,
    *,
    config_id: str,
    design_replicate: int,
) -> dict[str, float]:
    match = summary[
        (summary["config_id"] == config_id)
        & (summary["replicate_id"].astype(int) == int(design_replicate))
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one summary row for {config_id} replicate {design_replicate}, found {len(match)}."
        )
    row = match.iloc[0]
    return {
        "agreement_gap": float(row["agreement_gap"]),
        "local_min_eigengap": float(row["local_min_eigengap"]),
        "span_min_singular_value": float(row["span_min_singular_value"]),
    }


def empirical_run_metrics_and_model(
    *,
    X_envs: list[np.ndarray],
    arrays: dict[str, np.ndarray],
    config: dict[str, object],
    design_replicate: int,
) -> tuple[dict[str, float | int], AnchorPCAInfty]:
    k = int(config["k"])
    m = int(config["m"])
    p = int(config["p"])

    sstar_projector = np.asarray(arrays["sstar_projectors"][design_replicate], dtype=float)
    population_barPi = np.asarray(arrays["barPi"][design_replicate], dtype=float)
    population_local_projectors = np.asarray(arrays["top_projectors"][design_replicate], dtype=float)

    model = AnchorPCAInfty(n_components=k, block_tol="auto").fit(X_envs)
    first_projector, m_hat_first_block, rho = first_agreement_block_projector(
        model.barPi_,
        model.block_tol_,
    )
    projection_k = model.projection_
    identity = np.eye(p)

    local_errors = [
        projector_metric_norm(empirical - population)
        for empirical, population in zip(model.local_projectors_, population_local_projectors)
    ]

    pool_m = pool_pca_from_covariances(
        model.covariances_,
        n_components=m,
        n_obs=model.n_obs_,
    )
    pool_k = pool_pca_from_covariances(
        model.covariances_,
        n_components=k,
        n_obs=model.n_obs_,
    )

    spread_m = float(rho[0] - rho[m - 1])
    gap_m = float(rho[0] - rho[m])
    tolerance_separates_oracle = bool(spread_m <= model.block_tol_ < gap_m)

    metrics: dict[str, float | int] = {
        "block_tol": float(model.block_tol_),
        "m_hat_first_block": int(m_hat_first_block),
        "dim_correct": int(m_hat_first_block == m),
        "first_block_projector_error_op": projector_metric_norm(first_projector - sstar_projector),
        "first_block_projector_error_fro": float(
            np.linalg.norm(first_projector - sstar_projector, ord="fro")
        ),
        "rank_k_containment_loss": projector_metric_norm((identity - projection_k) @ sstar_projector),
        "barPi_error_op": operator_norm(model.barPi_ - population_barPi),
        "local_projector_error_mean": float(np.mean(local_errors)),
        "local_projector_error_max": float(np.max(local_errors)),
        "spread_m": spread_m,
        "gap_m": gap_m,
        "tolerance_separates_oracle": int(tolerance_separates_oracle),
        "pool_top_m_projector_error_op": projector_metric_norm(pool_m["projection"] - sstar_projector),
        "pool_rank_k_containment_loss": projector_metric_norm(
            (identity - pool_k["projection"]) @ sstar_projector
        ),
    }
    for index, value in enumerate(rho, start=1):
        metrics[f"rho_{index}"] = float(value)
    return metrics, model


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty. "
                "Use --overwrite or choose a different --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def summarize_results(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    design_group_cols = [
        "config_id",
        "profile",
        "distribution",
        "g",
        "p",
        "k",
        "m",
        "q",
        "N",
        "design_replicate",
    ]
    metric_cols = [
        "dim_correct",
        "first_block_projector_error_op",
        "first_block_projector_error_fro",
        "rank_k_containment_loss",
        "barPi_error_op",
        "local_projector_error_mean",
        "local_projector_error_max",
        "spread_m",
        "gap_m",
        "block_tol",
        "tolerance_separates_oracle",
        "pool_top_m_projector_error_op",
        "pool_rank_k_containment_loss",
    ]

    aggregations = {column: ["mean", "median"] for column in metric_cols}
    aggregations["sample_seed"] = "count"
    for column in [
        "agreement_gap",
        "local_min_eigengap",
        "span_min_singular_value",
    ]:
        aggregations[column] = "first"
    design_summary = raw.groupby(design_group_cols, sort=False).agg(aggregations)
    design_summary.columns = [
        "_".join(part for part in column if part) for column in design_summary.columns
    ]
    design_summary = design_summary.rename(
        columns={
            "sample_seed_count": "n_runs",
            "agreement_gap_first": "agreement_gap",
            "local_min_eigengap_first": "local_min_eigengap",
            "span_min_singular_value_first": "span_min_singular_value",
        }
    ).reset_index()

    curve_group_cols = [
        "config_id",
        "profile",
        "distribution",
        "g",
        "p",
        "k",
        "m",
        "q",
        "N",
    ]
    rows = []
    for key, group in design_summary.groupby(curve_group_cols, sort=False):
        row = dict(zip(curve_group_cols, key))
        row["n_designs"] = int(len(group))
        row["n_runs"] = int(group["n_runs"].sum())

        for source, output_prefix in [
            ("dim_correct_mean", "dim_correct"),
            ("first_block_projector_error_op_median", "first_block_projector_error_op"),
            ("pool_top_m_projector_error_op_median", "pool_top_m_projector_error_op"),
            ("rank_k_containment_loss_median", "rank_k_containment_loss"),
            ("barPi_error_op_median", "barPi_error_op"),
            ("spread_m_median", "spread_m"),
            ("gap_m_median", "gap_m"),
            ("block_tol_median", "block_tol"),
            ("tolerance_separates_oracle_mean", "tolerance_separates_oracle"),
        ]:
            values = group[source].astype(float)
            row[f"{output_prefix}_mean"] = float(values.mean())
            row[f"{output_prefix}_q10"] = float(values.quantile(0.10))
            row[f"{output_prefix}_median"] = float(values.median())
            row[f"{output_prefix}_q90"] = float(values.quantile(0.90))
        rows.append(row)

    curve_summary = pd.DataFrame(rows).sort_values(curve_group_cols).reset_index(drop=True)
    return design_summary, curve_summary


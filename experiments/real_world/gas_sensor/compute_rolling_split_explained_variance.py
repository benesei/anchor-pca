"""Compute publication gas-sensor rolling-split explained variance.

This script is the single calculation entry point for the gas-sensor figures.
For each last source batch s in 3, ..., 9, it fits representations on source
batches B1--Bs only and evaluates them on both source batches and held-out
target batches B(s+1)--B10. Feature standardization is re-fit within each split
using source batches only.

Computed full-representation methods:

* poolPCA
* AnchorPCA_lambda=1
* AnchorPCA_lambda=10
* AnchorPCA_infty
* norm-maxRegret

The script also evaluates the AnchorPCA_infty directions selected from the
first empirical agreement block. This is a tolerance-dependent estimate of the
invariant subspace S_star, not an oracle population quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
from anchorpca.linalg import symmetrize  # noqa: E402
from anchorpca.reproducibility import software_versions  # noqa: E402


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/270/"
    "gas+sensor+array+drift+dataset+at+different+concentrations.zip"
)
EXPECTED_SHA256 = "98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e"

N_FEATURES = 128
ALL_BATCHES = tuple(range(1, 11))
DEFAULT_LAST_SOURCE_BATCHES = tuple(range(3, 10))
DEFAULT_K_VALUES = (5, 10, 20, 30, 40)
ANCHOR_LAMBDAS = (1.0, 10.0)

PUBLICATION_METHOD_ORDER = (
    "poolPCA",
    "AnchorPCA_lambda=1",
    "AnchorPCA_lambda=10",
    "AnchorPCA_infty",
    "norm-maxRegret",
)

METHOD_LABELS = {
    "poolPCA": "poolPCA",
    "AnchorPCA_lambda=1": r"AnchorPCA$_{\lambda=1}$",
    "AnchorPCA_lambda=10": r"AnchorPCA$_{\lambda=10}$",
    "AnchorPCA_infty": r"AnchorPCA$_{\infty}$",
    "norm-maxRegret": "norm-maxRegret",
    "AnchorPCA_infty_Sstar_first_block": r"AnchorPCA$_{\infty}$ estimated $S_\star$",
}

REPORTED_GAS_ORDER = (
    "Ethanol",
    "Ethylene",
    "Ammonia",
    "Acetaldehyde",
    "Acetone",
    "Toluene",
)

# The .dat files store numeric labels rather than names. This mapping is the
# unique mapping that makes the observed class counts match the composition
# table reported with the dataset.
GAS_CLASS_LABELS = {
    1: "Acetone",
    2: "Acetaldehyde",
    3: "Ethanol",
    4: "Ethylene",
    5: "Ammonia",
    6: "Toluene",
}

REPORTED_GAS_COMPOSITION_COUNTS = {
    1: {"Ethanol": 83, "Ethylene": 30, "Ammonia": 70, "Acetaldehyde": 98, "Acetone": 90, "Toluene": 74},
    2: {"Ethanol": 100, "Ethylene": 109, "Ammonia": 532, "Acetaldehyde": 334, "Acetone": 164, "Toluene": 5},
    3: {"Ethanol": 216, "Ethylene": 240, "Ammonia": 275, "Acetaldehyde": 490, "Acetone": 365, "Toluene": 0},
    4: {"Ethanol": 12, "Ethylene": 30, "Ammonia": 12, "Acetaldehyde": 43, "Acetone": 64, "Toluene": 0},
    5: {"Ethanol": 20, "Ethylene": 46, "Ammonia": 63, "Acetaldehyde": 40, "Acetone": 28, "Toluene": 0},
    6: {"Ethanol": 110, "Ethylene": 29, "Ammonia": 606, "Acetaldehyde": 574, "Acetone": 514, "Toluene": 467},
    7: {"Ethanol": 360, "Ethylene": 744, "Ammonia": 630, "Acetaldehyde": 662, "Acetone": 649, "Toluene": 568},
    8: {"Ethanol": 40, "Ethylene": 33, "Ammonia": 143, "Acetaldehyde": 30, "Acetone": 30, "Toluene": 18},
    9: {"Ethanol": 100, "Ethylene": 75, "Ammonia": 78, "Acetaldehyde": 55, "Acetone": 61, "Toluene": 101},
    10: {"Ethanol": 600, "Ethylene": 600, "Ammonia": 600, "Acetaldehyde": 600, "Acetone": 600, "Toluene": 600},
}

GAS_COMPOSITION_COLORS = {
    "Ethanol": "#0072B2",
    "Ethylene": "#E69F00",
    "Ammonia": "#009E73",
    "Acetaldehyde": "#D55E00",
    "Acetone": "#CC79A7",
    "Toluene": "#56B4E9",
}


@dataclass(frozen=True)
class Dataset:
    X_raw: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]


@dataclass(frozen=True)
class BatchStats:
    batch: int
    covariance: np.ndarray
    centered: np.ndarray
    n_obs: int
    total_variance: float


@dataclass(frozen=True)
class FittedRepresentation:
    method_id: str
    directions: np.ndarray
    fit_source: str
    details: dict[str, object]


@dataclass(frozen=True)
class SplitData:
    last_source_batch: int
    source_batches: tuple[int, ...]
    target_batches: tuple[int, ...]
    all_batches: tuple[int, ...]
    X: np.ndarray
    preprocessing: dict[str, object]
    batch_stats: dict[int, BatchStats]
    source_covariances: list[np.ndarray]
    source_n_obs: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(data_dir: Path, *, force: bool, skip_sha256_check: bool) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "gas_sensor_array_drift_dataset.zip"

    if force or not zip_path.exists():
        print(f"Downloading UCI gas sensor archive to {zip_path}")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    else:
        print(f"Using cached archive {zip_path}")

    if not skip_sha256_check:
        observed = sha256_file(zip_path)
        if observed != EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded archive checksum mismatch. "
                f"Observed {observed}, expected {EXPECTED_SHA256}."
            )
    return zip_path


def extract_archive(zip_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    batch_paths = [extract_dir / f"batch{i}.dat" for i in ALL_BATCHES]
    if all(path.exists() for path in batch_paths):
        return batch_paths

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        expected_names = {f"batch{i}.dat" for i in ALL_BATCHES}
        missing = sorted(expected_names.difference(names))
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {missing}")

        for name in sorted(expected_names):
            target = extract_dir / name
            with archive.open(name) as source, target.open("wb") as dest:
                dest.write(source.read())

    return batch_paths


def parse_batch_file(path: Path, batch: int) -> tuple[np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    with path.open("r") as handle:
        for row_index, line in enumerate(handle):
            parts = line.strip().split()
            if not parts:
                continue

            label_token = parts[0]
            try:
                gas_class_text, concentration_text = label_token.split(";")
                gas_class = int(gas_class_text)
                concentration = float(concentration_text)
            except ValueError as exc:
                raise ValueError(f"Could not parse label token {label_token!r} in {path}") from exc

            vector = np.zeros(N_FEATURES, dtype=float)
            seen_indices: set[int] = set()
            for token in parts[1:]:
                try:
                    feature_index_text, value_text = token.split(":")
                    feature_index = int(feature_index_text)
                    value = float(value_text)
                except ValueError as exc:
                    raise ValueError(f"Could not parse feature token {token!r} in {path}") from exc

                if not 1 <= feature_index <= N_FEATURES:
                    raise ValueError(
                        f"Feature index {feature_index} in {path} is outside 1..{N_FEATURES}."
                    )
                if feature_index in seen_indices:
                    raise ValueError(f"Duplicate feature index {feature_index} in {path}.")
                seen_indices.add(feature_index)
                vector[feature_index - 1] = value

            if len(seen_indices) != N_FEATURES:
                raise ValueError(
                    f"Row {row_index} in {path} has {len(seen_indices)} features; "
                    f"expected {N_FEATURES}."
                )

            rows.append(vector)
            records.append(
                {
                    "batch": int(batch),
                    "row_in_batch": int(row_index),
                    "gas_class": gas_class,
                    "concentration": concentration,
                }
            )

    if not rows:
        raise ValueError(f"No rows parsed from {path}")

    return np.vstack(rows), pd.DataFrame.from_records(records)


def load_dataset(data_dir: Path, *, force_download: bool, skip_sha256_check: bool) -> Dataset:
    zip_path = download_archive(
        data_dir,
        force=force_download,
        skip_sha256_check=skip_sha256_check,
    )
    batch_paths = extract_archive(zip_path, data_dir / "raw")

    arrays = []
    frames = []
    for path in sorted(batch_paths, key=lambda item: int(re.search(r"batch(\d+)", item.name).group(1))):
        batch = int(re.search(r"batch(\d+)", path.name).group(1))
        X_batch, metadata_batch = parse_batch_file(path, batch)
        arrays.append(X_batch)
        frames.append(metadata_batch)

    X_raw = np.vstack(arrays)
    metadata = pd.concat(frames, ignore_index=True)
    feature_names = [f"feature_{j:03d}" for j in range(1, N_FEATURES + 1)]
    return Dataset(X_raw=X_raw, metadata=metadata, feature_names=feature_names)


def source_standardize(
    X_raw: np.ndarray,
    metadata: pd.DataFrame,
    source_batches: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """Standardize features using only observations from source batches."""
    source_mask = metadata["batch"].isin(source_batches).to_numpy()
    if not source_mask.any():
        raise ValueError("No source observations found for standardization.")
    source_X = np.asarray(X_raw[source_mask], dtype=float)
    mean = source_X.mean(axis=0)
    scale = source_X.std(axis=0, ddof=1)
    zero_scale = scale <= 0
    scale = scale.copy()
    scale[zero_scale] = 1.0
    X = (np.asarray(X_raw, dtype=float) - mean) / scale
    return X, {
        "mode": "source-standard",
        "source_batches": list(source_batches),
        "n_source_observations": int(source_X.shape[0]),
        "zero_scale_feature_count": int(zero_scale.sum()),
    }


def preprocess_features(
    X_raw: np.ndarray,
    metadata: pd.DataFrame,
    source_batches: tuple[int, ...],
    *,
    scale_mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if scale_mode == "source-standard":
        return source_standardize(X_raw, metadata, source_batches)
    if scale_mode == "none":
        return np.asarray(X_raw, dtype=float).copy(), {"mode": "none"}
    raise ValueError(f"Unknown scale mode: {scale_mode}")


def build_batch_stats(
    X: np.ndarray,
    metadata: pd.DataFrame,
    batches: tuple[int, ...],
) -> dict[int, BatchStats]:
    stats: dict[int, BatchStats] = {}
    metadata_batches = set(int(batch) for batch in metadata["batch"].unique())
    missing = sorted(set(batches).difference(metadata_batches))
    if missing:
        raise ValueError(f"Requested batches not found in metadata: {missing}")

    for batch in batches:
        mask = metadata["batch"].to_numpy() == batch
        X_batch = np.asarray(X[mask], dtype=float)
        if X_batch.shape[0] < 2:
            raise ValueError(f"Batch {batch} has fewer than two observations.")
        centered = X_batch - X_batch.mean(axis=0)
        covariance = symmetrize((centered.T @ centered) / (X_batch.shape[0] - 1))
        total_variance = float(np.trace(covariance))
        if total_variance <= 0:
            raise ValueError(f"Batch {batch} has non-positive total variance.")
        stats[int(batch)] = BatchStats(
            batch=int(batch),
            covariance=covariance,
            centered=centered,
            n_obs=int(X_batch.shape[0]),
            total_variance=total_variance,
        )
    return stats


def make_source_target_batches(last_source_batch: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return source B1--Bs and target B(s+1)--B10 batch tuples."""
    s = int(last_source_batch)
    if s < 1 or s >= 10:
        raise ValueError("last_source_batch must satisfy 1 <= s < 10.")
    return tuple(range(1, s + 1)), tuple(range(s + 1, 11))


def format_batch_label(batches: tuple[int, ...]) -> str:
    ordered = list(batches)
    if not ordered:
        return ""
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"B{ordered[0]}-B{ordered[-1]}"
    return ", ".join(f"B{batch}" for batch in ordered)


def prepare_split_data(
    X_raw: np.ndarray,
    metadata: pd.DataFrame,
    *,
    last_source_batch: int,
    scale_mode: str,
) -> SplitData:
    """Prepare one split with source-only preprocessing and source covariances."""
    source_batches, target_batches = make_source_target_batches(last_source_batch)
    all_batches = tuple(sorted(set(source_batches).union(target_batches)))
    if set(source_batches).intersection(target_batches):
        raise RuntimeError("Source and target batches overlap.")

    X, preprocessing = preprocess_features(
        X_raw,
        metadata,
        source_batches,
        scale_mode=scale_mode,
    )
    batch_stats = build_batch_stats(X, metadata, all_batches)
    source_covariances = [batch_stats[batch].covariance for batch in source_batches]
    source_n_obs = np.asarray([batch_stats[batch].n_obs for batch in source_batches], dtype=float)

    return SplitData(
        last_source_batch=int(last_source_batch),
        source_batches=source_batches,
        target_batches=target_batches,
        all_batches=all_batches,
        X=X,
        preprocessing=preprocessing,
        batch_stats=batch_stats,
        source_covariances=source_covariances,
        source_n_obs=source_n_obs,
    )


def load_minpca():
    try:
        import torch
        from minPCA.minpca import minPCA
    except ImportError as exc:
        raise RuntimeError(
            "The publication gas-sensor pipeline requires the external minPCA "
            "package for norm-maxRegret. Install it from "
            "https://github.com/anyafries/minPCA."
        ) from exc
    return torch, minPCA


def orient_qr(directions: np.ndarray, k: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(directions, dtype=float))
    q = q[:, :k]
    for j in range(q.shape[1]):
        pivot = int(np.argmax(np.abs(q[:, j])))
        if q[pivot, j] < 0:
            q[:, j] *= -1.0
    return q


def minpca_seed_for(*, base_seed: int, last_source_batch: int, k: int) -> int:
    """Deterministic publication seed tied to the split and dimension."""
    return int(base_seed) + 10_000 * int(last_source_batch) + 101 * int(k)


def fit_minpca_representation(
    method_id: str,
    covariances: list[np.ndarray],
    k: int,
    *,
    n_restarts: int,
    n_iters: int,
    lr: float,
    seed: int,
    verbose: bool,
) -> FittedRepresentation:
    if method_id != "norm-maxRegret":
        raise ValueError(f"Unknown publication minPCA method: {method_id}")

    torch, minPCA = load_minpca()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    covariances_float32 = [np.asarray(covariance, dtype=np.float32) for covariance in covariances]
    model = minPCA(n_components=int(k), function="maxregret", norm=True)
    model.fit(
        covariances_float32,
        n_restarts=int(n_restarts),
        n_iters=int(n_iters),
        lr=float(lr),
        verbose=bool(verbose),
    )
    directions = orient_qr(model.v_.detach().cpu().numpy(), k)
    return FittedRepresentation(
        method_id=method_id,
        directions=directions,
        fit_source="minPCA",
        details={
            "package_function": "maxregret",
            "package_norm": True,
            "n_restarts": int(n_restarts),
            "n_iters": int(n_iters),
            "lr": float(lr),
            "seed": int(seed),
            "minvar": float(model.minvar_) if hasattr(model, "minvar_") else None,
            "pooled_var": float(model.pooled_var_) if hasattr(model, "pooled_var_") else None,
        },
    )


def fit_selected_representations(
    source_covariances: list[np.ndarray],
    source_n_obs: np.ndarray,
    *,
    k: int,
    last_source_batch: int,
    block_tol: float | str,
    minpca_restarts: int,
    minpca_iters: int,
    minpca_lr: float,
    minpca_base_seed: int,
    minpca_verbose: bool,
) -> dict[str, FittedRepresentation]:
    """Fit publication methods using source covariances only."""
    fitted: dict[str, FittedRepresentation] = {}

    pool = pool_pca_from_covariances(source_covariances, n_components=k)
    fitted["poolPCA"] = FittedRepresentation(
        method_id="poolPCA",
        directions=np.asarray(pool["directions"], dtype=float),
        fit_source="anchorpca",
        details={},
    )

    for lambda_value in ANCHOR_LAMBDAS:
        model = AnchorPCALambda(n_components=k, lambda_=lambda_value).fit_covariances(
            source_covariances
        )
        method_id = f"AnchorPCA_lambda={int(lambda_value)}"
        fitted[method_id] = FittedRepresentation(
            method_id=method_id,
            directions=np.asarray(model.directions_, dtype=float),
            fit_source="anchorpca",
            details={"lambda": float(lambda_value)},
        )

    hard = AnchorPCAInfty(n_components=k, block_tol=block_tol).fit_covariances(
        source_covariances,
        n_obs=source_n_obs,
    )
    fitted["AnchorPCA_infty"] = FittedRepresentation(
        method_id="AnchorPCA_infty",
        directions=np.asarray(hard.directions_, dtype=float),
        fit_source="anchorpca",
        details={
            "block_tol_requested": block_tol,
            "block_tol": float(hard.block_tol_),
            "block_tol_mode": hard.block_tol_mode_,
            "block_tol_alpha": float(hard.block_tol_alpha_),
            "block_tol_c": float(hard.block_tol_c_),
            "block_tol_max": float(hard.block_tol_max_),
            "invariant_dim_estimate": int(hard.invariant_dim_estimate_),
            "invariant_n_selected": int(hard.agreement_blocks_[0]["n_selected"]),
            "invariant_block_rho": float(hard.agreement_blocks_[0]["rho"]),
            "source_n_obs": np.asarray(source_n_obs, dtype=float),
            "agreement_blocks": hard.agreement_blocks_,
            "barPi_eigenvalues": np.asarray(hard.barPi_eigenvalues_, dtype=float),
        },
    )

    minpca_seed = minpca_seed_for(
        base_seed=minpca_base_seed,
        last_source_batch=last_source_batch,
        k=k,
    )
    # This official minPCA norm-maxRegret fit is the runtime bottleneck of the
    # publication script. The defaults below use deterministic publication
    # settings: n_restarts=10, n_iters=2000, lr=0.01, with seed tied to (s, k).
    fitted["norm-maxRegret"] = fit_minpca_representation(
        "norm-maxRegret",
        source_covariances,
        k,
        n_restarts=minpca_restarts,
        n_iters=minpca_iters,
        lr=minpca_lr,
        seed=minpca_seed,
        verbose=minpca_verbose,
    )

    return fitted


def explained_variance_row(
    representation: FittedRepresentation,
    batch_stats: BatchStats,
    *,
    k: int,
    split: str,
) -> dict[str, object]:
    directions = np.asarray(representation.directions, dtype=float)
    if directions.shape != (batch_stats.covariance.shape[0], k):
        raise ValueError(
            f"{representation.method_id} directions have shape {directions.shape}; "
            f"expected {(batch_stats.covariance.shape[0], k)}."
        )

    gram = directions.T @ directions
    if not np.allclose(gram, np.eye(k), atol=1e-5):
        raise ValueError(f"{representation.method_id} directions are not orthonormal.")

    trace_value = float(np.trace(directions.T @ batch_stats.covariance @ directions))
    projected = batch_stats.centered @ directions
    projection_value = float(np.sum(np.var(projected, axis=0, ddof=1)))
    if not np.isclose(trace_value, projection_value, rtol=1e-7, atol=1e-7):
        raise RuntimeError(
            "Explained-variance cross-check failed for "
            f"{representation.method_id}, batch {batch_stats.batch}, k={k}: "
            f"trace={trace_value}, projection={projection_value}."
        )

    percent = 100.0 * projection_value / batch_stats.total_variance
    if percent < -1e-8 or percent > 100.0 + 1e-6:
        raise RuntimeError(
            f"Explained variance percentage outside [0, 100] for "
            f"{representation.method_id}, batch {batch_stats.batch}, k={k}: {percent}."
        )

    return {
        "k": int(k),
        "split": split,
        "batch": int(batch_stats.batch),
        "method_id": representation.method_id,
        "method_label": METHOD_LABELS[representation.method_id],
        "fit_source": representation.fit_source,
        "n_obs": int(batch_stats.n_obs),
        "total_variance": batch_stats.total_variance,
        "explained_variance": projection_value,
        "explained_variance_trace_check": trace_value,
        "percent_explained_variance": percent,
    }


def evaluate_representations(
    fitted: dict[str, FittedRepresentation],
    batch_stats: dict[int, BatchStats],
    source_batches: tuple[int, ...],
    target_batches: tuple[int, ...],
    *,
    k: int,
) -> pd.DataFrame:
    rows = []
    for method_id in PUBLICATION_METHOD_ORDER:
        representation = fitted[method_id]
        for batch in source_batches:
            rows.append(
                explained_variance_row(
                    representation,
                    batch_stats[batch],
                    k=k,
                    split="source",
                )
            )
        for batch in target_batches:
            rows.append(
                explained_variance_row(
                    representation,
                    batch_stats[batch],
                    k=k,
                    split="target",
                )
            )
    return pd.DataFrame(rows)


def attach_method_details(rows: pd.DataFrame, fitted: dict[str, FittedRepresentation]) -> pd.DataFrame:
    rows = rows.copy()
    rows["lambda"] = np.nan
    rows["block_tol"] = np.nan
    rows["block_tol_mode"] = ""
    rows["invariant_dim_estimate"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    rows["invariant_n_selected"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    rows["invariant_block_rho"] = np.nan
    rows["minpca_n_restarts"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    rows["minpca_n_iters"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    rows["minpca_lr"] = np.nan
    rows["minpca_seed"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")

    for method_id, representation in fitted.items():
        details = representation.details
        mask = rows["method_id"] == method_id
        if "lambda" in details:
            rows.loc[mask, "lambda"] = float(details["lambda"])
        if method_id == "AnchorPCA_infty":
            rows.loc[mask, "block_tol"] = float(details["block_tol"])
            rows.loc[mask, "block_tol_mode"] = str(details["block_tol_mode"])
            rows.loc[mask, "invariant_dim_estimate"] = int(details["invariant_dim_estimate"])
            rows.loc[mask, "invariant_n_selected"] = int(details["invariant_n_selected"])
            rows.loc[mask, "invariant_block_rho"] = float(details["invariant_block_rho"])
        if method_id == "norm-maxRegret":
            rows.loc[mask, "minpca_n_restarts"] = int(details["n_restarts"])
            rows.loc[mask, "minpca_n_iters"] = int(details["n_iters"])
            rows.loc[mask, "minpca_lr"] = float(details["lr"])
            rows.loc[mask, "minpca_seed"] = int(details["seed"])

    return rows


def evaluate_anchor_infty_sstar(
    fitted: dict[str, FittedRepresentation],
    batch_stats: dict[int, BatchStats],
    source_batches: tuple[int, ...],
    target_batches: tuple[int, ...],
    *,
    k: int,
) -> pd.DataFrame:
    """Evaluate directions selected from AnchorPCA_infty's first block."""
    representation = fitted["AnchorPCA_infty"]
    details = representation.details
    n_selected = int(details["invariant_n_selected"])
    if n_selected <= 0:
        raise RuntimeError("AnchorPCA_infty selected no directions from the first block.")

    directions = np.asarray(representation.directions, dtype=float)[:, :n_selected]
    gram = directions.T @ directions
    if not np.allclose(gram, np.eye(n_selected), atol=1e-5):
        raise ValueError("AnchorPCA_infty first-block directions are not orthonormal.")

    rows = []
    for split, batches in [("source", source_batches), ("target", target_batches)]:
        for batch in batches:
            stats = batch_stats[batch]
            trace_value = float(np.trace(directions.T @ stats.covariance @ directions))
            projected = stats.centered @ directions
            projection_value = float(np.sum(np.var(projected, axis=0, ddof=1)))
            if not np.isclose(trace_value, projection_value, rtol=1e-7, atol=1e-7):
                raise RuntimeError(
                    "S_star explained-variance cross-check failed for "
                    f"batch {batch}, k={k}: trace={trace_value}, "
                    f"projection={projection_value}."
                )

            percent = 100.0 * projection_value / stats.total_variance
            if percent < -1e-8 or percent > 100.0 + 1e-6:
                raise RuntimeError(
                    "S_star explained variance percentage outside [0, 100] for "
                    f"batch {batch}, k={k}: {percent}."
                )

            rows.append(
                {
                    "k": int(k),
                    "split": split,
                    "batch": int(batch),
                    "method_id": "AnchorPCA_infty_Sstar_first_block",
                    "method_label": METHOD_LABELS["AnchorPCA_infty_Sstar_first_block"],
                    "fit_source": representation.fit_source,
                    "n_obs": int(stats.n_obs),
                    "total_variance": stats.total_variance,
                    "explained_variance": projection_value,
                    "explained_variance_trace_check": trace_value,
                    "percent_explained_variance": percent,
                    "block_tol": float(details["block_tol"]),
                    "block_tol_mode": str(details["block_tol_mode"]),
                    "invariant_dim_estimate": int(details["invariant_dim_estimate"]),
                    "invariant_n_selected": n_selected,
                    "invariant_block_rho": float(details["invariant_block_rho"]),
                    "note": (
                        "Uses only the AnchorPCA_infty directions selected from "
                        "the first tolerance-grouped empirical barPi block. "
                        "invariant_dim_estimate is a finite-sample estimate."
                    ),
                }
            )

    return pd.DataFrame(rows)


def add_split_columns(
    rows: pd.DataFrame,
    *,
    split_data: SplitData,
    preprocessing_mode: str,
) -> pd.DataFrame:
    rows = rows.copy()
    rows["last_source_batch"] = int(split_data.last_source_batch)
    rows["n_source_batches"] = len(split_data.source_batches)
    rows["source_batches_label"] = format_batch_label(split_data.source_batches)
    rows["target_batches_label"] = format_batch_label(split_data.target_batches)
    rows["preprocessing_mode"] = preprocessing_mode
    return rows


def run_single_split_calculation(
    X_raw: np.ndarray,
    metadata: pd.DataFrame,
    *,
    last_source_batch: int,
    k_values: tuple[int, ...],
    scale_mode: str,
    block_tol: float | str,
    minpca_restarts: int,
    minpca_iters: int,
    minpca_lr: float,
    minpca_base_seed: int,
    minpca_verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    split_data = prepare_split_data(
        X_raw,
        metadata,
        last_source_batch=last_source_batch,
        scale_mode=scale_mode,
    )

    min_source_n = min(split_data.batch_stats[batch].n_obs for batch in split_data.source_batches)
    max_rank = min(N_FEATURES, min_source_n - 1)
    if max(k_values) > max_rank:
        raise ValueError(
            f"max(k)={max(k_values)} exceeds rank budget {max_rank} for "
            f"source batches {list(split_data.source_batches)}."
        )

    frames: list[pd.DataFrame] = []
    sstar_frames: list[pd.DataFrame] = []
    method_details_by_k: dict[str, object] = {}

    for k in k_values:
        print(
            f"  fitting s={last_source_batch}, k={k} "
            f"(source {format_batch_label(split_data.source_batches)}, "
            f"target {format_batch_label(split_data.target_batches)})"
        )
        fitted = fit_selected_representations(
            split_data.source_covariances,
            split_data.source_n_obs,
            k=k,
            last_source_batch=last_source_batch,
            block_tol=block_tol,
            minpca_restarts=minpca_restarts,
            minpca_iters=minpca_iters,
            minpca_lr=minpca_lr,
            minpca_base_seed=minpca_base_seed,
            minpca_verbose=minpca_verbose,
        )
        rows = evaluate_representations(
            fitted,
            split_data.batch_stats,
            split_data.source_batches,
            split_data.target_batches,
            k=k,
        )
        rows = attach_method_details(rows, fitted)
        sstar_rows = evaluate_anchor_infty_sstar(
            fitted,
            split_data.batch_stats,
            split_data.source_batches,
            split_data.target_batches,
            k=k,
        )
        rows = add_split_columns(
            rows,
            split_data=split_data,
            preprocessing_mode=split_data.preprocessing["mode"],
        )
        sstar_rows = add_split_columns(
            sstar_rows,
            split_data=split_data,
            preprocessing_mode=split_data.preprocessing["mode"],
        )
        frames.append(rows)
        sstar_frames.append(sstar_rows)
        method_details_by_k[str(k)] = {
            method_id: representation.details
            for method_id, representation in fitted.items()
        }

    metadata_for_split = {
        "last_source_batch": int(last_source_batch),
        "source_batches": list(split_data.source_batches),
        "target_batches": list(split_data.target_batches),
        "preprocessing": split_data.preprocessing,
        "source_n_obs": split_data.source_n_obs.tolist(),
        "method_details_by_k": method_details_by_k,
    }
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(sstar_frames, ignore_index=True),
        metadata_for_split,
    )


def build_target_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    target = all_results[all_results["split"] == "target"].copy()
    summary = (
        target
        .groupby(["last_source_batch", "n_source_batches", "k", "method_id"], sort=False)
        .agg(
            mean_target_ev=("percent_explained_variance", "mean"),
            min_target_ev=("percent_explained_variance", "min"),
            max_target_ev=("percent_explained_variance", "max"),
            n_target_batches=("batch", "nunique"),
        )
        .reset_index()
    )
    return summary


def build_sstar_summary(sstar_results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["last_source_batch", "n_source_batches", "k", "split"]
    for column in ["invariant_dim_estimate", "invariant_n_selected", "block_tol"]:
        nunique = sstar_results.groupby(group_cols, sort=False)[column].nunique(dropna=False)
        if (nunique > 1).any():
            bad = nunique[nunique > 1].index.tolist()
            raise RuntimeError(
                f"Unexpected non-constant {column} values in S_star groups: {bad}"
            )

    summary = (
        sstar_results
        .groupby(group_cols, sort=False)
        .agg(
            mean_sstar_ev=("percent_explained_variance", "mean"),
            min_sstar_ev=("percent_explained_variance", "min"),
            max_sstar_ev=("percent_explained_variance", "max"),
            invariant_dim_estimate=("invariant_dim_estimate", "first"),
            invariant_n_selected=("invariant_n_selected", "first"),
            block_tol=("block_tol", "first"),
            block_tol_mode=("block_tol_mode", "first"),
            n_batches=("batch", "nunique"),
        )
        .reset_index()
    )
    summary["invariant_dim_estimate"] = summary["invariant_dim_estimate"].astype("Int64")
    summary["invariant_n_selected"] = summary["invariant_n_selected"].astype("Int64")
    return summary


def write_outputs(
    all_results: pd.DataFrame,
    target_summary: pd.DataFrame,
    sstar_results: pd.DataFrame,
    sstar_summary: pd.DataFrame,
    metadata: dict[str, object],
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(
        results_dir / "rolling_publication_explained_variance_all.csv",
        index=False,
    )
    target_summary.to_csv(
        results_dir / "rolling_publication_target_summary.csv",
        index=False,
    )
    sstar_results.to_csv(
        results_dir / "rolling_publication_anchor_infty_sstar_all.csv",
        index=False,
    )
    sstar_summary.to_csv(
        results_dir / "rolling_publication_anchor_infty_sstar_summary.csv",
        index=False,
    )
    with (results_dir / "rolling_publication_metadata.json").open("w") as handle:
        json.dump(to_jsonable(metadata), handle, indent=2)


def save_figure(fig, figures_dir: Path, stem: str, *, dpi: int = 600) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")


def reported_gas_composition_table() -> pd.DataFrame:
    table = pd.DataFrame.from_dict(
        REPORTED_GAS_COMPOSITION_COUNTS,
        orient="index",
    )
    table.index.name = "batch"
    return table.reindex(index=ALL_BATCHES, columns=REPORTED_GAS_ORDER).astype(int)


def build_gas_composition_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return observed counts by batch and gas name."""
    unknown_labels = sorted(set(metadata["gas_class"].unique()).difference(GAS_CLASS_LABELS))
    if unknown_labels:
        raise ValueError(f"Unknown gas_class labels in data: {unknown_labels}")

    counts_by_label = (
        metadata.groupby(["batch", "gas_class"])
        .size()
        .rename("n_obs")
        .reset_index()
    )
    counts = (
        counts_by_label.pivot(index="batch", columns="gas_class", values="n_obs")
        .reindex(index=ALL_BATCHES, columns=sorted(GAS_CLASS_LABELS))
        .fillna(0)
        .astype(int)
    )
    counts = counts.rename(columns=GAS_CLASS_LABELS)
    return counts.reindex(columns=REPORTED_GAS_ORDER).astype(int)


def validate_reported_gas_composition(observed_counts: pd.DataFrame) -> None:
    """Check that the parsed data match the published composition table."""
    expected_counts = reported_gas_composition_table()
    observed = observed_counts.reindex(index=ALL_BATCHES, columns=REPORTED_GAS_ORDER).astype(int)
    if not observed.equals(expected_counts):
        diff = observed.subtract(expected_counts, fill_value=0).astype(int)
        raise RuntimeError(
            "Parsed gas-class counts do not match the published composition table. "
            "This means the gas_class-to-name mapping or the local data files are inconsistent.\n"
            f"Observed counts:\n{observed.to_string()}\n\n"
            f"Expected counts:\n{expected_counts.to_string()}\n\n"
            f"Observed - expected:\n{diff.to_string()}"
        )


def write_gas_composition_csv(observed_counts: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    totals = observed_counts.sum(axis=1)
    percentages = observed_counts.div(totals, axis=0) * 100.0
    rows = []
    inverse_labels = {name: label for label, name in GAS_CLASS_LABELS.items()}
    for batch in observed_counts.index:
        for gas in REPORTED_GAS_ORDER:
            rows.append(
                {
                    "batch": int(batch),
                    "gas": gas,
                    "gas_class_label": int(inverse_labels[gas]),
                    "n_obs": int(observed_counts.loc[batch, gas]),
                    "batch_n_obs": int(totals.loc[batch]),
                    "percent_of_batch": float(percentages.loc[batch, gas]),
                }
            )
    composition = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    composition.to_csv(
        results_dir / "gas_sensor_class_composition_by_batch.csv",
        index=False,
    )
    return composition


def plot_gas_composition(
    observed_counts: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Plot batch-wise gas composition with published gas names in the legend."""
    validate_reported_gas_composition(observed_counts)
    totals = observed_counts.sum(axis=1)
    percentages = observed_counts.div(totals, axis=0) * 100.0

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Palatino", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.9,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 2.85))
    x = np.arange(len(percentages.index))
    bottom = np.zeros(len(percentages), dtype=float)
    for gas in REPORTED_GAS_ORDER:
        values = percentages[gas].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=GAS_COMPOSITION_COLORS[gas],
            edgecolor="white",
            linewidth=0.5,
            label=gas,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([f"B{int(batch)}" for batch in percentages.index])
    ax.tick_params(axis="x", pad=3)
    for x_pos, batch in zip(x, percentages.index):
        ax.text(
            x_pos,
            -0.105,
            f"{int(totals.loc[batch])}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.8,
            clip_on=False,
        )
    ax.text(
        -0.62,
        -0.105,
        r"$n_{B_i}$",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7.0,
        clip_on=False,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Gas composition (%)")
    ax.grid(False)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
    )
    fig.tight_layout(rect=(0.0, 0.10, 0.84, 1.0))
    save_figure(fig, figures_dir, "gas_sensor_class_composition_by_batch")
    plt.close(fig)


def write_and_plot_gas_composition(
    metadata: pd.DataFrame,
    *,
    figures_dir: Path,
    results_dir: Path,
) -> pd.DataFrame:
    observed_counts = build_gas_composition_table(metadata)
    validate_reported_gas_composition(observed_counts)
    composition = write_gas_composition_csv(observed_counts, results_dir)
    plot_gas_composition(observed_counts, figures_dir)
    return composition


def print_data_diagnostics(dataset: Dataset) -> None:
    class_table = build_gas_composition_table(dataset.metadata)
    validate_reported_gas_composition(class_table)
    print("\nData diagnostics")
    print("=" * 72)
    print(f"Rows: {dataset.X_raw.shape[0]:,}")
    print(f"Features: {dataset.X_raw.shape[1]:,}")
    print(f"Batches: {sorted(dataset.metadata['batch'].unique().tolist())}")
    print(f"Gas classes: {sorted(dataset.metadata['gas_class'].unique().tolist())}")
    print(f"Gas label mapping: {GAS_CLASS_LABELS}")
    print("\nGas composition counts by batch")
    print(class_table.to_string())


def print_result_summary(target_summary: pd.DataFrame) -> None:
    print("\nTarget % explained variance summary (mean across target batches)")
    print("=" * 72)
    for k in sorted(target_summary["k"].unique()):
        sub = target_summary[target_summary["k"] == k]
        print(f"\n  k = {k}")
        for _, row in sub.iterrows():
            print(
                f"    s={int(row['last_source_batch']):d}  "
                f"{row['method_id']:.<28s} "
                f"mean={row['mean_target_ev']:6.2f}%  "
                f"min={row['min_target_ev']:6.2f}%  "
                f"max={row['max_target_ev']:6.2f}%"
            )


def validate_k_values(k_values: Iterable[int]) -> tuple[int, ...]:
    parsed = tuple(int(k) for k in k_values)
    if not parsed:
        raise ValueError("At least one k value is required.")
    if any(k <= 0 or k > N_FEATURES for k in parsed):
        raise ValueError(f"All k values must satisfy 1 <= k <= {N_FEATURES}.")
    return parsed


def validate_last_source_batches(values: Iterable[int]) -> tuple[int, ...]:
    parsed = tuple(int(s) for s in values)
    if not parsed:
        raise ValueError("At least one last source batch is required.")
    invalid = [s for s in parsed if s < 3 or s > 9]
    if invalid:
        raise ValueError(f"Publication rolling splits require s in 3..9; got {invalid}.")
    return parsed


def run_publication_calculation(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else output_dir / "data"
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"

    k_values = validate_k_values(args.k_values)
    last_source_batches = validate_last_source_batches(args.last_source_batches)

    dataset = load_dataset(
        data_dir,
        force_download=args.force_download,
        skip_sha256_check=args.skip_sha256_check,
    )
    missing = set(ALL_BATCHES).difference(dataset.metadata["batch"].unique())
    if missing:
        raise ValueError(f"Batches missing from data: {sorted(missing)}")

    print_data_diagnostics(dataset)
    gas_composition = write_and_plot_gas_composition(
        dataset.metadata,
        figures_dir=figures_dir,
        results_dir=results_dir,
    )
    print("\nPublication rolling-split calculation")
    print("=" * 72)
    print(f"Last source batches s: {list(last_source_batches)}")
    print(f"k values: {list(k_values)}")
    print(f"Preprocessing: {args.scale} (fit separately on source batches only)")
    print(
        "Runtime note: this full script can take a long time primarily because "
        "norm-maxRegret is computed with the official minPCA package. "
        f"Current deterministic settings are n_restarts={args.minpca_restarts}, "
        f"n_iters={args.minpca_iters}, lr={args.minpca_lr}, with seeds tied to (s, k)."
    )
    print(
        "Leakage check: targets are never used for standardization, "
        "covariance fitting, or method fitting; they enter only EV evaluation."
    )

    all_frames: list[pd.DataFrame] = []
    sstar_frames: list[pd.DataFrame] = []
    split_metadata: list[dict[str, object]] = []

    for s in last_source_batches:
        print(f"\nSplit s={s}: source B1-B{s}, target B{s + 1}-B10")
        rows, sstar_rows, metadata_for_split = run_single_split_calculation(
            dataset.X_raw,
            dataset.metadata,
            last_source_batch=s,
            k_values=k_values,
            scale_mode=args.scale,
            block_tol=args.block_tol,
            minpca_restarts=args.minpca_restarts,
            minpca_iters=args.minpca_iters,
            minpca_lr=args.minpca_lr,
            minpca_base_seed=args.minpca_base_seed,
            minpca_verbose=args.minpca_verbose,
        )
        all_frames.append(rows)
        sstar_frames.append(sstar_rows)
        split_metadata.append(metadata_for_split)

    all_results = pd.concat(all_frames, ignore_index=True)
    sstar_results = pd.concat(sstar_frames, ignore_index=True)
    target_summary = build_target_summary(all_results)
    sstar_summary = build_sstar_summary(sstar_results)

    metadata = {
        "experiment": "gas_sensor_rolling_publication",
        "software_versions": software_versions(),
        "data_url": DATA_URL,
        "archive_sha256": EXPECTED_SHA256,
        "n_features": N_FEATURES,
        "gas_class_labels": GAS_CLASS_LABELS,
        "gas_composition_checked_against_reported_table": True,
        "gas_composition_total_rows": int(gas_composition["n_obs"].sum()),
        "last_source_batches": list(last_source_batches),
        "splits": [
            {
                "last_source_batch": int(s),
                "source_batches": list(make_source_target_batches(s)[0]),
                "target_batches": list(make_source_target_batches(s)[1]),
            }
            for s in last_source_batches
        ],
        "k_values": list(k_values),
        "methods": list(PUBLICATION_METHOD_ORDER),
        "preprocessing": args.scale,
        "preprocessing_note": (
            "Feature standardization is re-fit for each split using only "
            "source batches B1--Bs."
        ),
        "evaluation_note": (
            "Each percentage is 100 * trace(V.T @ Sigma_batch @ V) / "
            "trace(Sigma_batch), where Sigma_batch is the centered covariance "
            "of that batch after source-only preprocessing. The script "
            "cross-checks this against the sample variance of centered "
            "projected observations."
        ),
        "sstar_note": (
            "S_star rows evaluate only the AnchorPCA_infty directions selected "
            "from the first tolerance-grouped empirical barPi block. "
            "invariant_dim_estimate is the size of that first empirical block "
            "and is a finite-sample estimate."
        ),
        "block_tol_requested": args.block_tol,
        "minpca": {
            "method": "norm-maxRegret",
            "n_restarts": int(args.minpca_restarts),
            "n_iters": int(args.minpca_iters),
            "lr": float(args.minpca_lr),
            "base_seed": int(args.minpca_base_seed),
            "seed_rule": "base_seed + 10000 * last_source_batch + 101 * k",
        },
        "split_metadata": split_metadata,
    }

    write_outputs(
        all_results,
        target_summary,
        sstar_results,
        sstar_summary,
        metadata,
        results_dir,
    )
    print_result_summary(target_summary)
    print(f"\nWrote publication CSVs to {results_dir}")


def parse_block_tol_arg(value: str) -> float | str:
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        numeric_value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "block tolerance must be 'auto' or a nonnegative numeric scalar."
        ) from exc
    if not np.isfinite(numeric_value) or numeric_value < 0:
        raise argparse.ArgumentTypeError("block tolerance must be a nonnegative finite value.")
    return numeric_value


def to_jsonable(value):
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Experiment directory under which data/ and results/ are used.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional explicit data directory. Defaults to OUTPUT_DIR/data.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the UCI archive even if a cached copy exists.",
    )
    parser.add_argument(
        "--skip-sha256-check",
        action="store_true",
        help="Skip the archive SHA256 verification.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Representation dimensions to evaluate.",
    )
    parser.add_argument(
        "--last-source-batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAST_SOURCE_BATCHES),
        help="Last source batches s. Publication default is 3 4 5 6 7 8 9.",
    )
    parser.add_argument(
        "--scale",
        choices=["source-standard", "none"],
        default="source-standard",
        help="Feature preprocessing. source-standard uses source batches only.",
    )
    parser.add_argument(
        "--block-tol",
        type=parse_block_tol_arg,
        default="auto",
        help="AnchorPCA_infty block tolerance: 'auto' or a nonnegative scalar.",
    )
    parser.add_argument(
        "--minpca-restarts",
        type=int,
        default=10,
        help="norm-maxRegret random restarts.",
    )
    parser.add_argument(
        "--minpca-iters",
        type=int,
        default=2000,
        help="norm-maxRegret optimization iterations per restart.",
    )
    parser.add_argument(
        "--minpca-lr",
        type=float,
        default=0.01,
        help="norm-maxRegret optimizer learning rate.",
    )
    parser.add_argument(
        "--minpca-base-seed",
        type=int,
        default=12020,
        help="Base seed; actual seed is deterministic in (s, k).",
    )
    parser.add_argument(
        "--minpca-verbose",
        action="store_true",
        help="Print minPCA optimizer progress.",
    )
    return parser


def main() -> None:
    run_publication_calculation(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

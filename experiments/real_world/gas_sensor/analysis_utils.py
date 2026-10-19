"""Shared computation helpers for the gas-sensor extensions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import compute_rolling_split_explained_variance as gas


@dataclass(frozen=True)
class SourceFold:
    heldout_batch: int
    training_batches: tuple[int, ...]
    X_standardized: np.ndarray
    training_mask: np.ndarray
    validation_mask: np.ndarray
    training_covariances: tuple[np.ndarray, ...]
    training_n_obs: np.ndarray
    heldout_covariance: np.ndarray


def load_dataset(data_dir: Path, *, force_download: bool = False) -> gas.Dataset:
    return gas.load_dataset(
        Path(data_dir),
        force_download=force_download,
        skip_sha256_check=False,
    )


def source_validation_folds(
    dataset: gas.Dataset,
    source_batches: tuple[int, ...],
) -> tuple[SourceFold, ...]:
    """Build source folds with standardization refitted in every fold."""
    if len(source_batches) < 3:
        raise ValueError("At least three source batches are required.")
    batch_labels = dataset.metadata["batch"].to_numpy(dtype=int)
    folds = []
    for heldout in source_batches:
        training = tuple(batch for batch in source_batches if batch != heldout)
        transformed, _ = gas.preprocess_features(
            dataset.X_raw,
            dataset.metadata,
            training,
            scale_mode="source-standard",
        )
        stats = gas.build_batch_stats(transformed, dataset.metadata, source_batches)
        folds.append(
            SourceFold(
                heldout_batch=heldout,
                training_batches=training,
                X_standardized=np.asarray(transformed, dtype=float),
                training_mask=np.isin(batch_labels, training),
                validation_mask=batch_labels == heldout,
                training_covariances=tuple(stats[batch].covariance for batch in training),
                training_n_obs=np.asarray(
                    [stats[batch].n_obs for batch in training], dtype=float
                ),
                heldout_covariance=stats[heldout].covariance,
            )
        )
    return tuple(folds)


def total_ev_fraction(projection: np.ndarray, covariance: np.ndarray) -> float:
    total = float(np.trace(covariance))
    if total <= 0:
        raise ValueError("Covariance must have positive total variance.")
    value = float(np.trace(np.asarray(projection) @ covariance)) / total
    if value < -1e-10 or value > 1.0 + 1e-8:
        raise RuntimeError(f"Explained-variance fraction outside [0, 1]: {value}.")
    return value


def standard_error(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def write_protocol(path: Path, contents: dict[str, object]) -> None:
    path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(frame: pd.DataFrame, path: Path, sort_by: tuple[str, ...]) -> None:
    if sort_by:
        frame = frame.sort_values(list(sort_by), kind="mergesort")
    frame.to_csv(path, index=False)

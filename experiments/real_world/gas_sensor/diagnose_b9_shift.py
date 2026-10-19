"""Diagnose the B9 gas-sensor shift for the B1--B8 source fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "src" / "anchorpca").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anchorpca import fit_two_e_lambda_from_covariances  # noqa: E402
from anchorpca.reproducibility import software_versions  # noqa: E402
from analysis_utils import (  # noqa: E402
    load_dataset,
    total_ev_fraction,
    write_frame,
    write_protocol,
)
import compute_rolling_split_explained_variance as gas  # noqa: E402


def run(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.data_dir, force_download=args.force_download)
    split = gas.prepare_split_data(
        dataset.X_raw,
        dataset.metadata,
        last_source_batch=8,
        scale_mode="source-standard",
    )
    b9_covariance = split.batch_stats[9].covariance
    eigenvalues, eigenvectors = np.linalg.eigh(b9_covariance)
    pc1 = eigenvectors[:, -1]
    pc1_ev_percent = 100.0 * float(eigenvalues[-1] / eigenvalues.sum())
    rows = []
    for rank in args.k_values:
        for method_id, coefficient in (("poolPCA", 0.0), ("Anchor_PCA_infinity", np.inf)):
            model = fit_two_e_lambda_from_covariances(
                split.source_covariances,
                rank,
                coefficient,
                n_obs=split.source_n_obs,
                weighting="uniform",
            )
            squared_capture = 100.0 * float(pc1 @ model.projection_ @ pc1)
            for batch in (9, 10):
                rows.append(
                    {
                        "last_source_batch": 8,
                        "k": rank,
                        "method_id": method_id,
                        "target_batch": batch,
                        "target_ev_percent_total": 100.0
                        * total_ev_fraction(
                            model.projection_, split.batch_stats[batch].covariance
                        ),
                        "b9_pc1_ev_percent_total": pc1_ev_percent,
                        "b9_pc1_squared_projection_capture_percent": squared_capture,
                    }
                )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_frame(
        pd.DataFrame(rows),
        output / "b9_shift_diagnostics.csv",
        ("k", "method_id", "target_batch"),
    )
    write_protocol(
        output / "protocol.json",
        {
            "dataset": "UCI Gas Sensor Array Drift Dataset",
            "archive_sha256": gas.EXPECTED_SHA256,
            "source_batches": list(range(1, 9)),
            "target_batches": [9, 10],
            "ranks": list(args.k_values),
            "preprocessing": "standardized using B1--B8 only",
            "source_weighting": "uniform domain",
            "squared_capture_definition": "100 times v_B9_PC1 transpose P v_B9_PC1",
            "target_used_for_fitting": False,
            "software_versions": software_versions(),
        },
    )
    print(f"Wrote B9 diagnostic table to {output}")


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=directory / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=directory / "results" / "b9_shift",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=(20, 30))
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

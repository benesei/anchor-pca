from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fama_scree_year_labels_and_values(tmp_path, monkeypatch):
    module = load_script(
        "fama_scree_plot", "experiments/real_world/fama_french/plot_source_scree.py"
    )
    rows = pd.DataFrame({
        "source_period": ["source_1"] * 3,
        "start_year": [1970] * 3, "end_year": [1974] * 3,
        "component": [3, 1, 2], "explained_variance_fraction": [.1, .6, .3],
    })
    figures = []
    original_close = module.plt.close
    monkeypatch.setattr(module.plt, "close", lambda fig: figures.append(fig))
    module.plot_source_scree(rows, tmp_path)
    figure = figures[-1]
    try:
        left, right = figure.axes
        assert left.get_legend().get_texts()[0].get_text() == "1970-1974"
        np.testing.assert_allclose(left.lines[0].get_ydata(), [60, 30, 10])
        np.testing.assert_allclose(right.lines[0].get_ydata(), [60, 90, 100])
        assert (tmp_path / "fama_french_49_source_scree.pdf").is_file()
    finally:
        original_close(figure)


def fake_gas_dataset(n_features: int = 32, n_per_batch: int = 40):
    gas = load_script(
        "gas_compute_for_smoke",
        "experiments/real_world/gas_sensor/compute_rolling_split_explained_variance.py",
    )
    rng = np.random.default_rng(42)
    arrays = []
    records = []
    for batch in range(1, 11):
        labels = np.resize(np.arange(1, 7), n_per_batch)
        values = rng.normal(size=(n_per_batch, n_features))
        values += 0.05 * batch * np.linspace(-1.0, 1.0, n_features)
        values[:, :6] += np.eye(6)[labels - 1]
        arrays.append(values)
        records.extend(
            {
                "batch": batch,
                "row_in_batch": row,
                "gas_class": int(label),
                "concentration": 1.0,
            }
            for row, label in enumerate(labels)
        )
    return gas.Dataset(
        X_raw=np.vstack(arrays),
        metadata=pd.DataFrame(records),
        feature_names=[f"feature_{index}" for index in range(n_features)],
    )


def test_fama_french_pipeline_smoke(tmp_path, monkeypatch):
    module = load_script(
        "fama_french_smoke",
        "experiments/real_world/fama_french/run_temporal_experiment.py",
    )
    rng = np.random.default_rng(3)
    periods = []
    definitions = [
        *(('source', index, 1970 + 5 * index) for index in range(5)),
        *(('target', index, 1995 + 5 * index) for index in range(6)),
    ]
    for role, index, start in definitions:
        periods.append(
            SimpleNamespace(
                period_id=f"{role}_{index + 1}",
                role=role,
                start_year=start,
                end_year=start + 4,
                values=rng.normal(size=(70, 49)),
                raw_rows=70,
                incomplete_rows_removed=0,
            )
        )
    monkeypatch.setattr(
        module,
        "resolve_archive",
        lambda archive, data_dir: tmp_path / "asset.zip",
    )
    monkeypatch.setattr(module, "load_periods", lambda archive: tuple(periods))
    monkeypatch.setattr(module, "archive_provenance", lambda archive: {"original_archive": True})
    monkeypatch.setattr(
        module,
        "fit_norm_maxregret",
        lambda covariances, rank, **kwargs: np.diag([1.0] * rank + [0.0] * (49 - rank)),
    )
    output = tmp_path / "fama"
    module.run(
        argparse.Namespace(
            archive=None,
            data_dir=tmp_path,
            output_dir=output,
            ranks=(2,),
            minpca_restarts=1,
            minpca_iterations=2,
        )
    )
    expected = {
        "period_counts.csv",
        "source_scree.csv",
        "penalty_validation_folds.csv",
        "penalty_validation_summary.csv",
        "target_ev_by_period.csv",
        "target_ev_summary.csv",
        "protocol.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert set(pd.read_csv(output / "target_ev_summary.csv")["method_id"]) == {
        "poolPCA",
        "source_selected_Anchor_PCA",
        "norm-maxRegret",
    }


def test_gas_ev_pipeline_smokes(tmp_path, monkeypatch):
    dataset = fake_gas_dataset()
    selection = load_script(
        "gas_selection_smoke",
        "experiments/real_world/gas_sensor/select_penalty.py",
    )
    weighting = load_script(
        "gas_weighting_smoke",
        "experiments/real_world/gas_sensor/evaluate_weighting.py",
    )
    b9 = load_script(
        "gas_b9_smoke",
        "experiments/real_world/gas_sensor/diagnose_b9_shift.py",
    )
    for module in (selection, weighting, b9):
        monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: dataset)

    selection_output = tmp_path / "selection"
    selection.run(
        argparse.Namespace(
            data_dir=tmp_path,
            output_dir=selection_output,
            s_values=(3,),
            k_values=(10,),
            force_download=False,
        )
    )
    assert len(pd.read_csv(selection_output / "selection_summary.csv")) == 1
    assert len(pd.read_csv(selection_output / "penalty_validation_folds.csv")) == 27
    selected_ev = pd.read_csv(selection_output / "selected_ev_by_batch.csv")
    assert len(selected_ev) == 30
    assert set(selected_ev[selected_ev.split == "source"].batch) == {1, 2, 3}
    assert set(selected_ev[selected_ev.split == "target"].batch) == set(range(4, 11))
    assert selected_ev.percent_explained_variance.between(0, 100).all()

    weighting_output = tmp_path / "weighting"
    weighting.run(
        argparse.Namespace(
            data_dir=tmp_path,
            output_dir=weighting_output,
            s_values=(3,),
            k_values=(10,),
            force_download=False,
        )
    )
    assert len(pd.read_csv(weighting_output / "weighting_summary.csv")) == 2

    b9_output = tmp_path / "b9"
    b9.run(
        argparse.Namespace(
            data_dir=tmp_path,
            output_dir=b9_output,
            k_values=(20, 30),
            force_download=False,
        )
    )
    diagnostics = pd.read_csv(b9_output / "b9_shift_diagnostics.csv")
    assert len(diagnostics) == 8
    assert set(diagnostics["target_batch"]) == {9, 10}


def test_gas_source_validation_refits_standardization():
    dataset = fake_gas_dataset()
    utilities = load_script(
        "gas_analysis_utils_smoke",
        "experiments/real_world/gas_sensor/analysis_utils.py",
    )
    folds = utilities.source_validation_folds(dataset, (1, 2, 3))
    for fold in folds:
        training = fold.X_standardized[fold.training_mask]
        assert np.allclose(training.mean(axis=0), 0.0, atol=1e-12)
        assert fold.heldout_batch not in fold.training_batches


class DummyClassifier:
    converged = True
    n_iter = 1
    n_nonzero_coefficients = 1
    optimizer_success = True
    gradient_norm = 0.0
    constraint_violation = 0.0

    def predict(self, features):
        return np.resize(np.arange(1, 7), np.asarray(features).shape[0])


def test_gas_classification_pipeline_schema_smoke(tmp_path, monkeypatch):
    module = load_script(
        "gas_classification_smoke",
        "experiments/real_world/gas_sensor/run_classification.py",
    )
    dataset = fake_gas_dataset(n_features=20, n_per_batch=30)
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(module, "fit_logistic", lambda *args, **kwargs: DummyClassifier())
    monkeypatch.setattr(module, "fit_exact_group_dro", lambda *args, **kwargs: DummyClassifier())
    monkeypatch.setattr(module, "fit_vrex_candidate", lambda *args, **kwargs: DummyClassifier())
    output = tmp_path / "classification"
    module.run(
        argparse.Namespace(
            data_dir=tmp_path,
            output_dir=output,
            c_values=(0.1,),
            finite_two_e_lambda=(0.0, 10.0),
            dg_l2_values=(0.001,),
            beta_values=(0.0,),
            dg_max_iter=2,
            n_jobs=1,
            force_download=False,
        )
    )
    assert {path.name for path in output.iterdir()} == {
        "source_validation.csv",
        "selections.csv",
        "target_metrics_by_batch.csv",
        "target_metrics_summary.csv",
        "fit_checks.csv",
        "protocol.json",
    }
    assert len(pd.read_csv(output / "target_metrics_summary.csv")) == 7
    assert len(pd.read_csv(output / "target_metrics_by_batch.csv")) == 28


def test_unseen_target_pipeline_smoke(tmp_path):
    module = load_script(
        "unseen_target_smoke",
        "experiments/synthetic/03_random_subspace_sstar/run_unseen_target_evaluation.py",
    )
    output = tmp_path / "unseen"
    module.run(
        argparse.Namespace(
            output_dir=output,
            profiles=("balanced", "threshold_stress"),
            n_designs=1,
            sample_repetitions=1,
            n_source=50,
        )
    )
    assert {path.name for path in output.iterdir()} == {
        "runs.csv",
        "by_design.csv",
        "summary.csv",
        "design_checks.csv",
        "protocol.json",
    }
    assert len(pd.read_csv(output / "runs.csv")) == 4
    assert len(pd.read_csv(output / "summary.csv")) == 4

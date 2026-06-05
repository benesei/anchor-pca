from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


GAS_SENSOR_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "real_world"
    / "gas_sensor"
)
sys.path.insert(0, str(GAS_SENSOR_DIR))

import compute_rolling_split_explained_variance as compute  # noqa: E402
import plot_loading_diagnostics as loading_plot  # noqa: E402
import plot_rolling_split_summary as rolling_plot  # noqa: E402
import plot_source_target_EV as source_plot  # noqa: E402
import plot_sstar_poolpca_target_batches as sstar_target_plot  # noqa: E402
import plot_sstar_poolpca_tradeoff as sstar_tradeoff  # noqa: E402


def synthetic_metadata(n_batches=10, n_per_batch=4):
    records = []
    for batch in range(1, n_batches + 1):
        for row in range(n_per_batch):
            records.append(
                {
                    "batch": batch,
                    "row_in_batch": row,
                    "gas_class": 1 + (row % 2),
                    "concentration": float(row + 1),
                }
            )
    return pd.DataFrame.from_records(records)


def reported_composition_metadata():
    inverse_labels = {name: label for label, name in compute.GAS_CLASS_LABELS.items()}
    records = []
    for batch, gas_counts in compute.REPORTED_GAS_COMPOSITION_COUNTS.items():
        row = 0
        for gas in compute.REPORTED_GAS_ORDER:
            for _ in range(gas_counts[gas]):
                records.append(
                    {
                        "batch": batch,
                        "row_in_batch": row,
                        "gas_class": inverse_labels[gas],
                        "concentration": 1.0,
                    }
                )
                row += 1
    return pd.DataFrame.from_records(records)


def test_split_construction_for_publication_s_values():
    for s in range(3, 10):
        source_batches, target_batches = compute.make_source_target_batches(s)

        assert source_batches == tuple(range(1, s + 1))
        assert target_batches == tuple(range(s + 1, 11))
        assert set(source_batches).isdisjoint(target_batches)
        assert tuple(sorted(set(source_batches).union(target_batches))) == tuple(range(1, 11))


def test_gas_label_mapping_matches_reported_composition_table():
    metadata = reported_composition_metadata()

    observed = compute.build_gas_composition_table(metadata)
    expected = compute.reported_gas_composition_table()

    compute.validate_reported_gas_composition(observed)
    assert observed.equals(expected)
    assert compute.GAS_CLASS_LABELS == {
        1: "Acetone",
        2: "Acetaldehyde",
        3: "Ethanol",
        4: "Ethylene",
        5: "Ammonia",
        6: "Toluene",
    }


def test_gas_composition_plot_and_csv_are_written(tmp_path):
    metadata = reported_composition_metadata()

    composition = compute.write_and_plot_gas_composition(
        metadata,
        figures_dir=tmp_path,
        results_dir=tmp_path,
    )

    assert (tmp_path / "gas_sensor_class_composition_by_batch.png").exists()
    assert (tmp_path / "gas_sensor_class_composition_by_batch.pdf").exists()
    assert (tmp_path / "gas_sensor_class_composition_by_batch.csv").exists()
    assert int(composition["n_obs"].sum()) == 13_910
    assert set(composition["gas"]) == set(compute.REPORTED_GAS_ORDER)


def test_source_standardization_uses_source_batches_only():
    metadata = pd.DataFrame(
        {
            "batch": [1, 1, 2, 2, 3, 3],
            "row_in_batch": [0, 1, 0, 1, 0, 1],
            "gas_class": [1, 1, 1, 1, 1, 1],
            "concentration": [1.0] * 6,
        }
    )
    X_raw = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [1_000.0, 2_000.0],
            [3_000.0, 4_000.0],
        ]
    )

    X, info = compute.source_standardize(X_raw, metadata, (1, 2))
    source_mask = metadata["batch"].isin((1, 2)).to_numpy()

    assert info["source_batches"] == [1, 2]
    assert info["n_source_observations"] == 4
    assert np.allclose(X[source_mask].mean(axis=0), np.zeros(2))
    assert np.allclose(X[source_mask].std(axis=0, ddof=1), np.ones(2))


def test_prepare_split_data_keeps_targets_out_of_preprocessing_and_fitting_inputs():
    metadata = synthetic_metadata(n_batches=10, n_per_batch=4)
    X_raw = np.vstack(
        [
            np.column_stack(
                [
                    np.arange(4, dtype=float) + batch,
                    2.0 * np.arange(4, dtype=float) + batch,
                ]
            )
            for batch in range(1, 11)
        ]
    )
    target_mask = metadata["batch"] > 3
    X_raw[target_mask.to_numpy()] += 10_000.0

    split = compute.prepare_split_data(
        X_raw,
        metadata,
        last_source_batch=3,
        scale_mode="source-standard",
    )
    source_mask = metadata["batch"].isin(split.source_batches).to_numpy()

    assert split.source_batches == (1, 2, 3)
    assert split.target_batches == (4, 5, 6, 7, 8, 9, 10)
    assert len(split.source_covariances) == 3
    assert split.source_n_obs.tolist() == [4.0, 4.0, 4.0]
    assert np.allclose(split.X[source_mask].mean(axis=0), np.zeros(2))
    assert np.allclose(split.X[source_mask].std(axis=0, ddof=1), np.ones(2))


def test_explained_variance_trace_and_projection_checks_agree():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    centered = X - X.mean(axis=0)
    covariance = (centered.T @ centered) / (X.shape[0] - 1)
    stats = compute.BatchStats(
        batch=1,
        covariance=covariance,
        centered=centered,
        n_obs=X.shape[0],
        total_variance=float(np.trace(covariance)),
    )
    representation = compute.FittedRepresentation(
        method_id="poolPCA",
        directions=np.array([[1.0], [0.0]]),
        fit_source="test",
        details={},
    )

    row = compute.explained_variance_row(representation, stats, k=1, split="source")

    assert np.isclose(row["explained_variance"], row["explained_variance_trace_check"])
    assert np.isclose(row["percent_explained_variance"], 100.0)


def make_mock_explained_variance_csv():
    rows = []
    methods = (
        "poolPCA",
        "AnchorPCA_lambda=1",
        "AnchorPCA_infty",
        "norm-maxRegret",
    )
    for s in (5, 6):
        for k in (20, 30):
            for batch in range(1, 11):
                split = "source" if batch <= s else "target"
                for method in methods:
                    value = 50.0 + batch + 0.1 * k
                    if method == "AnchorPCA_lambda=1":
                        value += 1.0
                    if method == "AnchorPCA_infty":
                        value += 5.0
                    if method == "norm-maxRegret":
                        value += 2.0
                    rows.append(
                        {
                            "last_source_batch": s,
                            "n_source_batches": s,
                            "k": k,
                            "split": split,
                            "batch": batch,
                            "method_id": method,
                            "percent_explained_variance": value,
                        }
                    )
    return pd.DataFrame(rows)


def test_source_target_plot_extracts_exact_rows_for_selected_s_and_k():
    all_results = make_mock_explained_variance_csv()

    rows = source_plot.select_source_target_rows(
        all_results,
        last_source_batch=6,
        k=20,
    )

    assert set(rows["last_source_batch"]) == {6}
    assert set(rows["k"]) == {20}
    assert set(rows["method_id"]) == set(source_plot.PLOT_METHOD_ORDER)
    assert len(rows) == 10 * len(source_plot.PLOT_METHOD_ORDER)
    assert set(rows.loc[rows["split"] == "source", "batch"]) == set(range(1, 7))
    assert set(rows.loc[rows["split"] == "target", "batch"]) == set(range(7, 11))


def test_green_arrow_uses_anchor_infty_max_relative_target_gain_over_poolpca():
    rows = []
    for batch, pool, anchor in [
        (7, 50.0, 60.0),
        (8, 80.0, 100.0),
        (9, 20.0, 22.0),
        (10, 90.0, 95.0),
    ]:
        rows.extend(
            [
                {
                    "last_source_batch": 6,
                    "k": 20,
                    "split": "target",
                    "batch": batch,
                    "method_id": "poolPCA",
                    "percent_explained_variance": pool,
                },
                {
                    "last_source_batch": 6,
                    "k": 20,
                    "split": "target",
                    "batch": batch,
                    "method_id": "AnchorPCA_infty",
                    "percent_explained_variance": anchor,
                },
                {
                    "last_source_batch": 6,
                    "k": 20,
                    "split": "target",
                    "batch": batch,
                    "method_id": "norm-maxRegret",
                    "percent_explained_variance": anchor + 1.0,
                },
            ]
        )
    improvement = source_plot.compute_anchor_infty_target_improvement(pd.DataFrame(rows))

    assert improvement["batch"] == 8
    assert np.isclose(improvement["relative_improvement_percent"], 25.0)


def test_source_target_plot_writes_figure_from_selected_csv_values(tmp_path):
    all_results = make_mock_explained_variance_csv()
    rows = source_plot.select_source_target_rows(
        all_results,
        last_source_batch=6,
        k=20,
    )
    source_batches, target_batches = compute.make_source_target_batches(6)

    improvement = source_plot.plot_source_target(
        rows,
        tmp_path,
        source_batches=source_batches,
        target_batches=target_batches,
        plot_stem="mock_source_target",
        arrow_label_position="below",
        target_spacing_scale=1.0,
    )

    assert (tmp_path / "mock_source_target.png").exists()
    assert (tmp_path / "mock_source_target.pdf").exists()
    assert improvement == source_plot.compute_anchor_infty_target_improvement(rows)


def test_source_target_plot_parser_can_disable_green_arrow():
    parser = source_plot.build_arg_parser()

    default_args = parser.parse_args([])
    no_arrow_args = parser.parse_args(["--no-plot-arrow"])
    arrow_args = parser.parse_args(["--plot-arrow"])

    assert default_args.plot_arrow is True
    assert no_arrow_args.plot_arrow is False
    assert arrow_args.plot_arrow is True


def make_mock_rolling_explained_variance_csv():
    rows = []
    methods = (
        "poolPCA",
        "AnchorPCA_lambda=1",
        "AnchorPCA_infty",
        "AnchorPCA_lambda=10",
    )
    method_offsets = {
        "poolPCA": 0.0,
        "AnchorPCA_lambda=1": 1.0,
        "AnchorPCA_infty": 2.0,
        "AnchorPCA_lambda=10": 3.0,
    }
    for k in (5, 10, 20, 30, 40):
        for s in range(3, 10):
            for batch in range(1, 11):
                split = "source" if batch <= s else "target"
                for method in methods:
                    rows.append(
                        {
                            "last_source_batch": s,
                            "n_source_batches": s,
                            "k": k,
                            "split": split,
                            "batch": batch,
                            "method_id": method,
                            "percent_explained_variance": (
                                50.0
                                + k / 10.0
                                + s
                                + 0.1 * batch
                                + method_offsets[method]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_rolling_plot_filters_exclude_s9_k5_k40_and_anchor_lambda_10():
    all_results = make_mock_rolling_explained_variance_csv()

    plot_rows = rolling_plot.select_rolling_plot_data(all_results)

    assert len(plot_rows) == 2 * 3 * 6 * 3
    assert set(plot_rows["k"]) == {10, 20, 30}
    assert set(plot_rows["split"]) == {"source", "target"}
    assert set(plot_rows["last_source_batch"]) == set(range(3, 9))
    assert "AnchorPCA_lambda=10" not in set(plot_rows["method_id"])
    assert set(plot_rows["method_id"]) == {
        "poolPCA",
        "AnchorPCA_lambda=1",
        "AnchorPCA_infty",
    }

    s6_source = plot_rows[
        (plot_rows["last_source_batch"] == 6)
        & (plot_rows["k"] == 20)
        & (plot_rows["split"] == "source")
        & (plot_rows["method_id"] == "poolPCA")
    ].iloc[0]
    s6_target = plot_rows[
        (plot_rows["last_source_batch"] == 6)
        & (plot_rows["k"] == 20)
        & (plot_rows["split"] == "target")
        & (plot_rows["method_id"] == "poolPCA")
    ].iloc[0]

    assert int(s6_source["n_batches"]) == 6
    assert int(s6_target["n_batches"]) == 4
    assert np.isclose(
        s6_source["mean_ev"],
        np.mean([50.0 + 2.0 + 6 + 0.1 * batch for batch in range(1, 7)]),
    )
    assert np.isclose(
        s6_target["mean_ev"],
        np.mean([50.0 + 2.0 + 6 + 0.1 * batch for batch in range(7, 11)]),
    )


def test_rolling_summary_plot_writes_figure_from_selected_csv_values(tmp_path):
    all_results = make_mock_rolling_explained_variance_csv()
    plot_rows = rolling_plot.select_rolling_plot_data(all_results)

    rolling_plot.plot_rolling_summary(
        plot_rows,
        tmp_path,
        stem="mock_rolling_summary",
    )

    assert (tmp_path / "mock_rolling_summary.png").exists()
    assert (tmp_path / "mock_rolling_summary.pdf").exists()


def make_mock_sstar_and_pool_same_dim_rows():
    sstar_rows = []
    pool_rows = []
    for k, d in [(10, 3), (20, 5)]:
        for s in (3, 4):
            for batch in range(1, 11):
                split = "source" if batch <= s else "target"
                pool_value = 70.0 + k / 10.0 + 0.2 * batch
                sstar_value = pool_value - 4.0 if split == "source" else pool_value + 6.0
                common = {
                    "last_source_batch": s,
                    "n_source_batches": s,
                    "k": k,
                    "split": split,
                    "batch": batch,
                    "invariant_dim_estimate": d,
                    "invariant_n_selected": d,
                    "block_tol": 0.04,
                    "block_tol_mode": "auto",
                    "preprocessing_mode": "source-standard",
                }
                sstar_rows.append(
                    {
                        **common,
                        "percent_explained_variance": sstar_value,
                    }
                )
                pool_rows.append(
                    {
                        **common,
                        "comparison_dim": d,
                        "percent_explained_variance": pool_value,
                    }
                )
    return pd.DataFrame(sstar_rows), pd.DataFrame(pool_rows)


def test_sstar_tradeoff_summary_uses_same_dimension_batch_values():
    sstar_rows, pool_rows = make_mock_sstar_and_pool_same_dim_rows()

    selected = sstar_tradeoff.select_sstar_rows(
        sstar_rows,
        k_values=(10, 20),
        last_source_batches=(3, 4),
    )
    batch_table = sstar_tradeoff.build_same_dim_batch_table(selected, pool_rows)
    summary = sstar_tradeoff.build_tradeoff_summary(batch_table)

    assert len(batch_table) == 2 * 2 * 10
    assert len(summary) == 2 * 2
    assert set(summary["invariant_n_selected"]) == {3, 5}
    assert np.allclose(summary["source_difference_pp"], -4.0)
    assert np.allclose(summary["target_difference_pp"], 6.0)
    assert (summary["n_target_wins"] == summary["n_target_batches"]).all()


def test_sstar_tradeoff_plot_writes_figure_from_summary_values(tmp_path):
    sstar_rows, pool_rows = make_mock_sstar_and_pool_same_dim_rows()
    selected = sstar_tradeoff.select_sstar_rows(
        sstar_rows,
        k_values=(10, 20),
        last_source_batches=(3, 4),
    )
    batch_table = sstar_tradeoff.build_same_dim_batch_table(selected, pool_rows)
    summary = sstar_tradeoff.build_tradeoff_summary(batch_table)

    sstar_tradeoff.plot_tradeoff(summary, tmp_path, stem="mock_sstar_tradeoff")

    assert (tmp_path / "mock_sstar_tradeoff.png").exists()
    assert (tmp_path / "mock_sstar_tradeoff.pdf").exists()


def test_sstar_target_batch_plot_extracts_b9_b10_same_dim_values():
    sstar_rows, pool_rows = make_mock_sstar_and_pool_same_dim_rows()
    selected = sstar_tradeoff.select_sstar_rows(
        sstar_rows,
        k_values=(10, 20),
        last_source_batches=(3, 4),
    )
    batch_table = sstar_tradeoff.build_same_dim_batch_table(selected, pool_rows)

    plot_rows = sstar_target_plot.select_b9_b10_plot_data(
        batch_table,
        target_batches=(9, 10),
        k_values=(10, 20),
        last_source_batches=(3, 4),
    )

    assert len(plot_rows) == 2 * 2 * 2 * 2
    assert set(plot_rows["target_batch"]) == {9, 10}
    assert set(plot_rows["method_id"]) == {
        "poolPCA_top_same_dim",
        "AnchorPCA_infty_Sstar_first_block",
    }
    assert set(plot_rows["invariant_n_selected"]) == {3, 5}

    b9_s3_k10 = plot_rows[
        (plot_rows["target_batch"] == 9)
        & (plot_rows["last_source_batch"] == 3)
        & (plot_rows["k"] == 10)
    ]
    sstar_value = b9_s3_k10.loc[
        b9_s3_k10["method_id"] == "AnchorPCA_infty_Sstar_first_block",
        "percent_explained_variance",
    ].iloc[0]
    pool_value = b9_s3_k10.loc[
        b9_s3_k10["method_id"] == "poolPCA_top_same_dim",
        "percent_explained_variance",
    ].iloc[0]
    assert np.isclose(sstar_value - pool_value, 6.0)


def test_sstar_target_batch_plot_writes_figure_from_selected_values(tmp_path):
    sstar_rows, pool_rows = make_mock_sstar_and_pool_same_dim_rows()
    selected = sstar_tradeoff.select_sstar_rows(
        sstar_rows,
        k_values=(10, 20),
        last_source_batches=(3, 4),
    )
    batch_table = sstar_tradeoff.build_same_dim_batch_table(selected, pool_rows)
    plot_rows = sstar_target_plot.select_b9_b10_plot_data(
        batch_table,
        target_batches=(9, 10),
        k_values=(10, 20),
        last_source_batches=(3, 4),
    )

    sstar_target_plot.plot_b9_b10_same_dim_ev(
        plot_rows,
        tmp_path,
        stem="mock_sstar_b9_b10",
    )

    assert (tmp_path / "mock_sstar_b9_b10.png").exists()
    assert (tmp_path / "mock_sstar_b9_b10.pdf").exists()


def test_loading_diagnostic_leverage_tables_are_normalized():
    feature_table = loading_plot.feature_index_table()
    rng = np.random.default_rng(123)
    q, _ = np.linalg.qr(rng.normal(size=(compute.N_FEATURES, 4)))

    heatmap = loading_plot.direction_feature_type_leverage(
        q,
        method_id="mock",
        top_directions=3,
        feature_table=feature_table,
    )
    sensor = loading_plot.subspace_sensor_importance(
        q,
        method_id="mock",
        feature_table=feature_table,
    )
    feature_type = loading_plot.subspace_feature_type_importance(
        q,
        method_id="mock",
        feature_table=feature_table,
    )

    by_direction = heatmap.groupby("direction")["leverage_percent"].sum()
    assert np.allclose(by_direction.to_numpy(), np.full(3, 100.0))
    assert np.isclose(sensor["importance_percent"].sum(), 100.0)
    assert np.isclose(feature_type["importance_percent"].sum(), 100.0)
    assert set(sensor["sensor"]) == set(range(1, 17))
    assert set(feature_type["feature_type"]) == set(loading_plot.FEATURE_TYPES)


def test_loading_diagnostic_plots_write_figures(tmp_path):
    feature_table = loading_plot.feature_index_table()
    q_pool = np.eye(compute.N_FEATURES, 4)
    q_anchor = np.roll(q_pool, shift=1, axis=0)
    fitted = {
        "poolPCA": {"directions": q_pool},
        "AnchorPCA_infty": {"directions": q_anchor},
    }
    heatmap, sensor, feature_type = loading_plot.build_diagnostic_tables(
        fitted,
        k=4,
        top_directions=3,
        feature_table=feature_table,
    )

    loading_plot.plot_top_direction_feature_heatmaps(
        heatmap,
        tmp_path,
        stem="mock_loading_heatmap",
        m_hat=2,
    )
    loading_plot.plot_importance_summaries(
        sensor,
        feature_type,
        tmp_path,
        stem="mock_loading_importance",
    )

    assert (tmp_path / "mock_loading_heatmap.png").exists()
    assert (tmp_path / "mock_loading_heatmap.pdf").exists()
    assert (tmp_path / "mock_loading_importance.png").exists()
    assert (tmp_path / "mock_loading_importance.pdf").exists()

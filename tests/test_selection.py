"""Tests for rolling validation, search generation, and residual forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_pipeline import assign_split
from shared.models import SerializableHuber, build_lstm
from shared.selection import (
    ROLLING_FOLDS,
    aggregate_validation_results,
    generate_lstm_candidates,
    select_lstm_candidate,
    validate_search_completeness,
)


def test_rolling_folds_are_expanding_and_stop_before_frozen_test():
    previous_validation_end = None
    for fold in ROLLING_FOLDS:
        assert pd.Timestamp(fold.train_start) < pd.Timestamp(fold.validation_start)
        assert pd.Timestamp(fold.validation_start) < pd.Timestamp(fold.validation_end)
        if previous_validation_end is not None:
            assert pd.Timestamp(fold.validation_start) == previous_validation_end
        previous_validation_end = pd.Timestamp(fold.validation_end)
    assert previous_validation_end == pd.Timestamp("2016-08-01")


def test_split_is_assigned_by_target_timestamp_for_a_fold():
    fold = ROLLING_FOLDS[0]
    values = pd.Series(
        pd.to_datetime(
            [
                "2014-12-31 23:00",
                "2015-01-01 00:00",
                "2015-06-30 23:00",
                "2015-07-01 00:00",
            ]
        )
    )
    split = assign_split(
        values,
        train_start=fold.train_start,
        validation_start=fold.validation_start,
        test_start=fold.validation_end,
        data_end_exclusive=fold.validation_end,
    )
    assert split.tolist()[:3] == ["train", "validation", "validation"]
    assert pd.isna(split.iloc[3])


def test_candidate_search_is_deterministic_and_stratified():
    first = generate_lstm_candidates(42)
    second = generate_lstm_candidates(42)
    assert first == second
    assert len(first) == 16
    assert sum(item.formulation == "direct" for item in first) == 8
    assert sum(item.formulation == "residual" for item in first) == 8
    assert first[0].source == "basic_anchor"
    assert first[1].source == "previous_best_anchor"


def test_one_standard_error_selection_prefers_simpler_direct_model():
    summary = pd.DataFrame(
        [
            {"candidate_id": "large", "mean_mae": 8.00, "se_mae": 0.10, "parameters": 1000, "num_layers": 2, "formulation": "residual"},
            {"candidate_id": "small", "mean_mae": 8.08, "se_mae": 0.02, "parameters": 500, "num_layers": 1, "formulation": "direct"},
        ]
    )
    assert select_lstm_candidate(summary)["candidate_id"] == "small"


def test_zero_initialised_residual_model_begins_as_persistence():
    model = build_lstm(
        (4, 1),
        units=2,
        dense_units=2,
        dropout=0.0,
        formulation="residual",
        pm25_feature_index=0,
        pm25_mean=10.0,
        pm25_scale=2.0,
    )
    inputs = np.zeros((3, 4, 1), dtype=np.float32)
    inputs[:, -1, 0] = np.array([-1.0, 0.0, 2.0])
    predictions = model.predict(inputs, verbose=0).reshape(-1)
    np.testing.assert_allclose(predictions, [8.0, 10.0, 14.0], atol=1e-5)


def test_candidate_standard_error_uses_three_fold_means_not_six_fits():
    rows = []
    for fold, values in zip(("fold_1", "fold_2", "fold_3"), ((0.9, 1.1), (1.9, 2.1), (2.9, 3.1)), strict=True):
        for seed, mae in zip((42, 52), values, strict=True):
            rows.append(
                {
                    "candidate_id": "candidate_00",
                    "fold": fold,
                    "seed": seed,
                    "mae": mae,
                    "best_epoch": 4,
                    "parameters": 100,
                    "num_layers": 1,
                    "formulation": "direct",
                }
            )
    summary = aggregate_validation_results(pd.DataFrame(rows)).iloc[0]
    assert summary["folds"] == 3
    assert summary["evaluations"] == 6
    assert np.isclose(summary["mean_mae"], 2.0)
    assert np.isclose(summary["std_mae"], 1.0)
    assert np.isclose(summary["se_mae"], 1 / np.sqrt(3))
    assert summary["mean_seed_std_mae"] > 0


def test_huber_delta_survives_save_load_and_resumed_training(tmp_path):
    from tensorflow import keras

    model = build_lstm((4, 1), units=2, dense_units=2, dropout=0.0, huber_delta=10.0)
    model.train_on_batch(
        np.zeros((2, 4, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
    )
    path = tmp_path / "model.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    assert isinstance(restored.loss, SerializableHuber)
    assert restored.loss.delta == 10.0
    loss = restored.train_on_batch(
        np.zeros((2, 4, 1), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
    )
    assert np.isfinite(np.asarray(loss, dtype=float)).all()


def test_full_search_contract_requires_exactly_96_unique_fits():
    candidates = [item.candidate_id for item in generate_lstm_candidates(42)]
    folds = [fold.name for fold in ROLLING_FOLDS]
    rows = [
        {
            "candidate_id": candidate,
            "fold": fold,
            "seed": seed,
            "source_fingerprint": "source",
        }
        for candidate in candidates
        for fold in folds
        for seed in (42, 52)
    ]
    validated = validate_search_completeness(
        pd.DataFrame(rows), candidates, folds, (42, 52), source_fingerprint="source"
    )
    assert len(validated) == 96
    with np.testing.assert_raises(ValueError):
        validate_search_completeness(
            pd.DataFrame(rows[:-1]), candidates, folds, (42, 52), source_fingerprint="source"
        )

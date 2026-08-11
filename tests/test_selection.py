"""Tests for rolling validation, search generation, and residual forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_pipeline import assign_split
from shared.models import build_lstm
from shared.selection import ROLLING_FOLDS, generate_lstm_candidates, select_lstm_candidate


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

"""Tests for rolling validation, search generation, and residual forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_pipeline import assign_split
from shared.models import SerializableHuber, SerializablePinball, build_lstm
from shared.selection import (
    ROLLING_FOLDS,
    aggregate_validation_results,
    generate_optimization_recipes,
    generate_lstm_candidates,
    paired_comparison_table,
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
    assert len(first) == 12
    assert sum(item.formulation == "direct" for item in first) == 3
    assert sum(item.formulation == "residual" for item in first) == 9
    assert first[0].source == "basic_anchor"
    assert first[1].source == "candidate13_anchor"
    assert {item.units for item in first[2:]} == {64, 96, 128}
    assert sum(item.batch_size == 128 for item in first[2:]) == 5


def test_paired_selection_does_not_use_unrelated_between_fold_variance():
    rows = []
    for fold, base in zip(("fold_1", "fold_2", "fold_3"), (7.0, 9.0, 11.0), strict=True):
        for seed in (42, 52):
            rows.extend(
                [
                    {
                        "candidate_id": "large",
                        "fold": fold,
                        "seed": seed,
                        "mae": base,
                        "best_epoch": 4,
                        "parameters": 1000,
                        "num_layers": 2,
                        "formulation": "residual",
                    },
                    {
                        "candidate_id": "small",
                        "fold": fold,
                        "seed": seed,
                        "mae": base + 0.08,
                        "best_epoch": 4,
                        "parameters": 500,
                        "num_layers": 1,
                        "formulation": "direct",
                    },
                ]
            )
    results = pd.DataFrame(rows)
    summary = aggregate_validation_results(results)
    selected, comparisons, audit = select_lstm_candidate(results, summary)
    assert selected["candidate_id"] == "large"
    assert comparisons.loc[
        comparisons["candidate_id"].eq("small"), "paired_one_se_eligible"
    ].item() is False
    assert audit["raw_winner_by_mean_mae"] == "large"


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


def test_pinball_tau_half_matches_mae_and_survives_serialization(tmp_path):
    from tensorflow import keras

    y_true = np.array([[1.0], [5.0]], dtype=np.float32)
    y_pred = np.array([[3.0], [2.0]], dtype=np.float32)
    pinball = SerializablePinball(tau=0.5)
    assert np.isclose(float(pinball(y_true, y_pred)), np.abs(y_true - y_pred).mean())
    model = build_lstm(
        (4, 1),
        units=2,
        dense_units=2,
        dropout=0.0,
        loss_name="pinball",
        pinball_tau=0.55,
        clipnorm=1.0,
        use_ema=True,
    )
    path = tmp_path / "pinball.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    assert isinstance(restored.loss, SerializablePinball)
    assert restored.loss.tau == 0.55
    assert restored.optimizer.clipnorm == 1.0
    assert restored.optimizer.use_ema is True


def test_full_search_contract_requires_exactly_72_unique_fits():
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
    assert len(validated) == 72
    with np.testing.assert_raises(ValueError):
        validate_search_completeness(
            pd.DataFrame(rows[:-1]), candidates, folds, (42, 52), source_fingerprint="source"
        )


def test_optimization_recipe_contract_is_complete_and_preregistered():
    recipes = generate_optimization_recipes()
    assert len(recipes) == 6
    assert recipes[0].source == "candidate13_reproduction"
    assert recipes[4].loss_name == "pinball" and recipes[4].pinball_tau == 0.55
    assert recipes[5].optimizer_name == "adamw" and recipes[5].weight_decay == 1e-5
    rows = [
        {
            "candidate_id": recipe.candidate_id,
            "fold": fold.name,
            "seed": seed,
            "source_fingerprint": "source",
        }
        for recipe in recipes
        for fold in ROLLING_FOLDS
        for seed in (42, 52)
    ]
    validated = validate_search_completeness(
        pd.DataFrame(rows),
        [recipe.candidate_id for recipe in recipes],
        [fold.name for fold in ROLLING_FOLDS],
        (42, 52),
        source_fingerprint="source",
    )
    assert len(validated) == 36


def test_paired_comparison_averages_seeds_before_fold_uncertainty():
    rows = []
    for fold, difference in zip(("fold_1", "fold_2", "fold_3"), (0.01, 0.02, 0.03), strict=True):
        for seed in (42, 52):
            rows.append({"candidate_id": "best", "fold": fold, "seed": seed, "mae": 8.0})
            rows.append(
                {"candidate_id": "other", "fold": fold, "seed": seed, "mae": 8.0 + difference}
            )
    table = paired_comparison_table(pd.DataFrame(rows))
    other = table.loc[table["candidate_id"].eq("other")].iloc[0]
    assert np.isclose(other["mean_paired_degradation"], 0.02)
    assert np.isclose(other["paired_standard_error"], 0.01 / np.sqrt(3))

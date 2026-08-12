"""Unit tests for causal preprocessing and split construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.config import EXPECTED_STATIONS, ProjectPaths
from shared.data_pipeline import (
    PreprocessingPolicy,
    create_model_table,
    fit_preprocessor,
    load_frozen_benchmark,
    load_and_validate_raw,
    transform_model_table,
)


def synthetic_raw() -> pd.DataFrame:
    timestamps = pd.date_range("2015-12-29", "2016-01-03 23:00", freq="h")
    frames = []
    for station_index, station in enumerate(EXPECTED_STATIONS):
        size = len(timestamps)
        values = np.arange(size, dtype=float) + 10 + station_index
        frame = pd.DataFrame(
            {
                "No": np.arange(size),
                "year": timestamps.year,
                "month": timestamps.month,
                "day": timestamps.day,
                "hour": timestamps.hour,
                "PM2.5": values,
                "PM10": values + 1,
                "SO2": values + 2,
                "NO2": values + 3,
                "CO": values + 4,
                "O3": values + 5,
                "TEMP": values + 6,
                "PRES": values + 7,
                "DEWP": values + 8,
                "RAIN": np.zeros(size),
                "wd": ["N"] * size,
                "WSPM": values + 9,
                "station": station,
                "source_file": f"{station}.csv",
                "timestamp": timestamps,
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["station", "timestamp"]).reset_index(drop=True)


def test_load_raw_rejects_duplicate_key(tmp_path):
    frame = synthetic_raw().iloc[:10].copy()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    csv_path = tmp_path / "PRSA_Data_test.csv"
    frame.drop(columns=["timestamp", "source_file"]).to_csv(csv_path, index=False)
    paths = ProjectPaths(tmp_path)
    try:
        load_and_validate_raw(paths, [csv_path], strict=False)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Duplicate station-timestamp keys were not rejected.")


def test_preprocessing_is_causal_and_train_fitted(tmp_path):
    raw = synthetic_raw()
    station = EXPECTED_STATIONS[0]
    station_rows = raw.index[raw["station"].eq(station)]
    previous_index, missing_index = station_rows[30], station_rows[31]
    previous_value = raw.loc[previous_index, "PM2.5"]
    raw.loc[missing_index, "PM2.5"] = np.nan

    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    table, artifacts, metadata = create_model_table(raw, paths)

    assert table.loc[missing_index, "PM2.5_input_raw"] == previous_value
    assert table.loc[missing_index, "missing__PM2.5"] == 1.0
    assert table.loc[missing_index, "PM2.5_target_original"] != table.loc[missing_index, "PM2.5_target_original"]
    assert not table[artifacts.feature_columns].isna().any().any()
    assert metadata["lookback_hours"] == 24

    train_mask = table["timestamp"] < pd.Timestamp("2016-01-01")
    scaled_train_mean = table.loc[train_mask, artifacts.scaled_columns].mean().abs().max()
    assert scaled_train_mean < 1e-5


def test_missing_target_is_not_a_valid_anchor(tmp_path):
    raw = synthetic_raw()
    station_rows = raw.index[raw["station"].eq(EXPECTED_STATIONS[0])]
    missing_target_row = station_rows[40]
    raw.loc[missing_target_row, "PM2.5"] = np.nan

    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    table, _, _ = create_model_table(raw, paths)
    anchor = missing_target_row - 1
    assert np.isnan(table.loc[anchor, "target_pm25"])
    assert not bool(table.loc[anchor, "valid_anchor"])


def test_forward_fill_is_capped_before_train_fitted_median_fallback(tmp_path):
    raw = synthetic_raw()
    station = EXPECTED_STATIONS[0]
    rows = raw.index[raw["station"].eq(station)]
    raw.loc[rows, "PM2.5"] = 100.0
    raw.loc[rows[20], "PM2.5"] = 10.0
    raw.loc[rows[21:26], "PM2.5"] = np.nan

    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    table, artifacts, _ = create_model_table(
        raw,
        paths,
        forward_fill_limit_hours=2,
    )

    assert table.loc[rows[21], "PM2.5_input_raw"] == 10.0
    assert table.loc[rows[22], "PM2.5_input_raw"] == 10.0
    assert table.loc[rows[23], "PM2.5_input_raw"] == artifacts.station_medians[station]["PM2.5"]
    assert table.loc[rows[23], "missing__PM2.5"] == 1.0


def test_fit_then_transform_matches_convenience_interface(tmp_path):
    raw = synthetic_raw()
    policy = PreprocessingPolicy()
    artifacts = fit_preprocessor(raw, policy, source_fingerprint="test-source")
    transformed, metadata = transform_model_table(raw, artifacts, policy)
    wrapped, wrapped_artifacts, wrapped_metadata = create_model_table(
        raw,
        ProjectPaths(tmp_path),
        persist=False,
        source_fingerprint="test-source",
    )
    np.testing.assert_allclose(
        transformed[artifacts.feature_columns],
        wrapped[wrapped_artifacts.feature_columns],
    )
    assert metadata == wrapped_metadata
    assert artifacts.source_fingerprint == "test-source"


def test_frozen_benchmark_cannot_load_before_selection_is_locked(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    benchmark = synthetic_raw().iloc[:4].copy()
    benchmark["timestamp"] = pd.date_range("2016-08-01", periods=4, freq="h")
    benchmark.to_parquet(paths.frozen_benchmark_parquet, index=False)
    selection = paths.models / "final_model_selection_smoke.json"

    with np.testing.assert_raises(PermissionError):
        load_frozen_benchmark(paths, selection)

    selection.write_text('{"selection_locked_before_benchmark": false}', encoding="utf-8")
    with np.testing.assert_raises(PermissionError):
        load_frozen_benchmark(paths, selection)

    selection.write_text(
        '{"selection_locked_before_benchmark": true, "source_fingerprint": "abc"}',
        encoding="utf-8",
    )
    loaded = load_frozen_benchmark(paths, selection, expected_source_fingerprint="abc")
    assert len(loaded) == 4

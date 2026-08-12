"""Unit tests for station-safe lazy sequence generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.sequence_pipeline import StationWindowSequence, prediction_frame


def sequence_table() -> pd.DataFrame:
    timestamps = pd.date_range("2015-01-01", periods=30, freq="h")
    return pd.DataFrame(
        {
            "feature": np.arange(30, dtype=np.float32),
            "target_pm25": np.arange(1, 31, dtype=np.float32),
            "station_code": np.zeros(30, dtype=np.int16),
            "station": ["A"] * 30,
            "timestamp": timestamps,
            "target_timestamp": timestamps + pd.Timedelta(hours=1),
            "PM2.5_input_raw": np.arange(30, dtype=float),
            "valid_anchor": [False] * 23 + [True] * 7,
            "split": ["train"] * 30,
        }
    )


def test_sequence_window_and_target_alignment():
    table = sequence_table()
    sequence = StationWindowSequence(
        table,
        ["feature"],
        split="train",
        lookback=24,
        batch_size=2,
        shuffle=False,
    )
    x_batch, y_batch = sequence[0]
    np.testing.assert_array_equal(x_batch[0, :, 0], np.arange(24, dtype=np.float32))
    assert y_batch[0, 0] == 24.0
    assert sequence.ordered_anchor_indices()[0] == 23


def test_cross_station_anchor_is_rejected():
    table = sequence_table()
    table.loc[:22, "station_code"] = 0
    table.loc[23:, "station_code"] = 1
    table["valid_anchor"] = False
    table.loc[23, "valid_anchor"] = True
    with pytest.raises(ValueError, match="station boundary"):
        StationWindowSequence(
            table,
            ["feature"],
            split="train",
            lookback=24,
            batch_size=1,
            shuffle=False,
        )


def test_daily_persistence_is_exactly_24_hours_before_target():
    table = sequence_table()
    sequence = StationWindowSequence(
        table,
        ["feature"],
        split="train",
        lookback=24,
        batch_size=10,
        shuffle=False,
    )
    frame = prediction_frame(table, sequence, np.zeros(len(sequence.ordered_anchor_indices())))
    np.testing.assert_array_equal(frame["daily_persistence"], np.arange(7, dtype=float))
    target_time = pd.to_datetime(frame.loc[0, "target_timestamp"])
    source_time = table.loc[table["PM2.5_input_raw"].eq(frame.loc[0, "daily_persistence"]), "timestamp"].iloc[0]
    assert target_time - source_time == pd.Timedelta(hours=24)

"""Unit tests for the assignment's regression metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.evaluation import paired_moving_block_bootstrap, regression_metrics


def test_regression_metrics_known_values():
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([0.0, 12.0, 16.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["n"] == 3
    assert np.isclose(metrics["mae"], 2.0)
    assert np.isclose(metrics["rmse"], np.sqrt(20 / 3))
    assert np.isfinite(metrics["smape_percent"])


def test_calendar_moving_block_bootstrap_is_deterministic_and_clustered():
    days = pd.date_range("2017-01-01", periods=14, freq="D")
    frame = pd.DataFrame(
        [
            {
                "station": station,
                "target_timestamp": day + pd.Timedelta(hours=hour),
                "y_true": float(day.day + hour),
                "model_a": float(day.day + hour) + (0.2 if day.day % 2 else -0.2),
                "model_b": float(day.day + hour) + 1.0,
            }
            for day in days
            for station in ("A", "B")
            for hour in range(2)
        ]
    )
    first = paired_moving_block_bootstrap(
        frame, "model_a", "model_b", block_days=7, n_boot=200, seed=42
    )
    second = paired_moving_block_bootstrap(
        frame, "model_a", "model_b", block_days=7, n_boot=200, seed=42
    )
    assert first == second
    assert first["block_length_days"] == 7
    assert first["n_calendar_days"] == 14
    assert first["n_rows"] == len(frame)
    assert first["ci95_upper"] < 0

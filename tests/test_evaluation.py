"""Unit tests for the assignment's regression metrics."""

from __future__ import annotations

import numpy as np

from shared.evaluation import regression_metrics


def test_regression_metrics_known_values():
    actual = np.array([0.0, 10.0, 20.0])
    predicted = np.array([0.0, 12.0, 16.0])
    metrics = regression_metrics(actual, predicted)
    assert metrics["n"] == 3
    assert np.isclose(metrics["mae"], 2.0)
    assert np.isclose(metrics["rmse"], np.sqrt(20 / 3))
    assert np.isfinite(metrics["smape_percent"])


"""Regression metrics, grouped error analysis, and temporal bootstrap tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return metrics in the original PM2.5 unit scale."""

    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("Metric inputs have different lengths.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs contain missing or non-finite values.")
    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.zeros_like(denominator, dtype=float)
    np.divide(
        2.0 * np.abs(predicted - actual),
        denominator,
        out=smape_terms,
        where=denominator != 0,
    )
    smape = np.mean(smape_terms)
    return {
        "n": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "smape_percent": float(smape * 100.0),
    }


def add_analysis_groups(predictions: pd.DataFrame, training_targets: pd.Series) -> pd.DataFrame:
    """Add season and mutually exclusive training-defined pollution bands."""

    result = predictions.copy()
    month = pd.to_datetime(result["target_timestamp"]).dt.month
    result["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )
    q25, q75, q90 = training_targets.quantile([0.25, 0.75, 0.90]).to_numpy(dtype=float)
    result["pollution_band"] = pd.cut(
        result["y_true"],
        bins=[-np.inf, q25, q75, q90, np.inf],
        labels=["low", "moderate", "high", "extreme"],
        include_lowest=True,
    )
    return result


def metrics_by_group(
    predictions: pd.DataFrame,
    group_column: str,
    prediction_column: str = "y_pred",
) -> pd.DataFrame:
    """Calculate the standard metric set for each subgroup."""

    rows: list[dict] = []
    for group_value, group in predictions.groupby(group_column, observed=True, sort=True):
        metrics = regression_metrics(group["y_true"].to_numpy(), group[prediction_column].to_numpy())
        rows.append({group_column: group_value, **metrics})
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    prediction_a: str,
    prediction_b: str,
    n_boot: int = 1_000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired station-day MAE differences as 24-hour blocks.

    A negative difference means model A has lower MAE than model B.
    """

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["target_timestamp"]).dt.date
    frame["absolute_error_a"] = np.abs(frame["y_true"] - frame[prediction_a])
    frame["absolute_error_b"] = np.abs(frame["y_true"] - frame[prediction_b])
    block_difference = (
        frame.groupby(["station", "date"], observed=True)[["absolute_error_a", "absolute_error_b"]]
        .mean()
        .eval("absolute_error_a - absolute_error_b")
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draw = rng.choice(block_difference, size=len(block_difference), replace=True)
        samples[index] = draw.mean()
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_mae_difference_a_minus_b": float(block_difference.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "n_station_day_blocks": len(block_difference),
        "bootstrap_samples": int(n_boot),
    }

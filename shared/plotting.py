"""Consistent plotting utilities for notebook and report-ready outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "axes.titleweight": "bold"})


def save_figure(path: Path) -> None:
    """Save the active figure and release it for long headless runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_history(history: pd.DataFrame, path: Path, title: str) -> None:
    """Plot training metrics, including validation only when values exist."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.index + 1, history["mae"], label="train")
    if "val_mae" in history and history["val_mae"].notna().any():
        axes[0].plot(history.index + 1, history["val_mae"], label="validation")
    axes[0].set(title="MAE", xlabel="Epoch", ylabel="MAE")
    axes[0].legend()
    axes[1].plot(history.index + 1, history["loss"], label="train")
    if "val_loss" in history and history["val_loss"].notna().any():
        axes[1].plot(history.index + 1, history["val_loss"], label="validation")
    axes[1].set(title="Huber loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    fig.suptitle(title)
    save_figure(path)


def plot_actual_vs_predicted(predictions: pd.DataFrame, path: Path, title: str) -> None:
    sample = predictions.sample(min(20_000, len(predictions)), random_state=42)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=sample, x="y_true", y="y_pred", alpha=0.25, s=12, edgecolor=None)
    limit = float(max(sample["y_true"].max(), sample["y_pred"].max()))
    plt.plot([0, limit], [0, limit], "--", color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("Observed PM2.5")
    plt.ylabel("Predicted PM2.5")
    save_figure(path)


def plot_station_mae(metrics: pd.DataFrame, path: Path) -> None:
    ordered = metrics.sort_values("mae", ascending=True)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=ordered, x="mae", y="station", color="#3b82f6")
    plt.title("Benchmark MAE by monitoring station")
    plt.xlabel("MAE (µg/m³)")
    plt.ylabel("Station")
    save_figure(path)


def plot_test_week(
    predictions: pd.DataFrame,
    station: str,
    start: str,
    path: Path,
    hours: int = 168,
) -> None:
    start_time = pd.Timestamp(start)
    subset = predictions[
        predictions["station"].eq(station)
        & predictions["target_timestamp"].between(start_time, start_time + pd.Timedelta(hours=hours))
    ]
    plt.figure(figsize=(14, 4))
    plt.plot(subset["target_timestamp"], subset["y_true"], label="observed", linewidth=1.5)
    plt.plot(subset["target_timestamp"], subset["y_pred"], label="predicted", linewidth=1.2)
    plt.title(f"One-week forecast example: {station}")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    save_figure(path)

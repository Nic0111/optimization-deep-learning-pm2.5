"""Resumable assignment stages shared by the CLI and local notebooks."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    CONFIG,
    CONTINUOUS_INPUT_COLUMNS,
    EXPECTED_STATIONS,
    PATHS,
    ProjectPaths,
)
from .data_pipeline import (
    create_model_table,
    download_dataset,
    extract_dataset,
    load_and_validate_raw,
    missing_summary,
    save_model_table,
    schema_summary,
)
from .evaluation import (
    add_analysis_groups,
    metrics_by_group,
    paired_block_bootstrap,
    regression_metrics,
)
from .plotting import (
    plot_actual_vs_predicted,
    plot_history,
    plot_station_mae,
    plot_test_week,
    save_figure,
    set_plot_style,
)
from .runtime import RunStateStore, RuntimeOptions, tracked_stage, write_json_atomic
from .selection import (
    LSTMConfig,
    ROLLING_FOLDS,
    aggregate_validation_results,
    generate_lstm_candidates,
    select_lstm_candidate,
    select_one_standard_error,
)
from .sequence_pipeline import build_sequences, prediction_frame

StageFunction = Callable[[RuntimeOptions, ProjectPaths, RunStateStore], tuple[dict[str, Any], list[Path]]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def _smoke_table(model_table: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Keep two complete stations while rebuilding their physical anchor indices."""

    result = model_table[model_table["station"].isin(EXPECTED_STATIONS[:2])].reset_index(drop=True)
    result["valid_anchor"] = False
    for indices in result.groupby("station", observed=True).groups.values():
        station_indices = np.asarray(list(indices), dtype=np.int64)
        candidates = station_indices[lookback - 1 :]
        result.loc[candidates, "valid_anchor"] = (
            result.loc[candidates, "target_pm25"].notna()
            & result.loc[candidates, "split"].notna()
        )
    return result


def _load_model_inputs(
    options: RuntimeOptions,
    paths: ProjectPaths,
    splits: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if not paths.model_table.exists():
        raise FileNotFoundError("Prepared model data is missing; run the prepare stage first.")
    model_table = pd.read_parquet(paths.model_table)
    model_table["timestamp"] = pd.to_datetime(model_table["timestamp"])
    model_table["target_timestamp"] = pd.to_datetime(model_table["target_timestamp"])
    feature_columns = _read_json(paths.feature_metadata)["features"]
    lookback = int(_read_json(paths.split_metadata)["lookback_hours"])
    if options.mode == "smoke":
        model_table = _smoke_table(model_table, lookback)
    if splits is None:
        splits = tuple(
            split
            for split in ("train", "validation", "test")
            if bool((model_table["valid_anchor"] & model_table["split"].eq(split)).any())
        )
    sequences = build_sequences(
        model_table,
        feature_columns,
        lookback=lookback,
        batch_size=options.batch_size,
        seed=CONFIG.seed,
        splits=splits,
    )
    return model_table, feature_columns, sequences


def _component_summary(store: RunStateStore, key: str) -> dict[str, Any]:
    return store.load().get("components", {}).get(key, {}).get("summary", {})


def acquire_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    _: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Download, extract, validate, and persist the official UCI source."""

    manifest = download_dataset(paths, force=False)
    station_files = extract_dataset(paths, force=False)
    air_quality = load_and_validate_raw(paths, station_files, strict=True)
    air_quality.to_parquet(paths.combined_parquet, index=False)

    station_audit = (
        air_quality.groupby("station", observed=True)
        .agg(rows=("timestamp", "size"), start=("timestamp", "min"), end=("timestamp", "max"))
        .reset_index()
    )
    schema = schema_summary(air_quality)
    missing = missing_summary(air_quality)
    station_missing = (
        air_quality.groupby("station", observed=True)[list(CONTINUOUS_INPUT_COLUMNS) + ["wd"]]
        .agg(lambda values: int(values.isna().sum()))
        .reset_index()
    )
    _write_frame(station_audit, paths.assignment1_tables / "station_source_audit.csv")
    _write_frame(schema, paths.assignment1_tables / "raw_schema.csv")
    _write_frame(missing, paths.assignment1_tables / "raw_missing_summary.csv")
    _write_frame(station_missing, paths.assignment1_tables / "raw_missing_by_station.csv")

    summary = {
        "rows": len(air_quality),
        "columns": int(air_quality.shape[1]),
        "stations": int(air_quality["station"].nunique()),
        "start": str(air_quality["timestamp"].min()),
        "end": str(air_quality["timestamp"].max()),
        "archive_sha256": manifest.sha256,
        "archive_bytes": manifest.bytes,
        "source_url": manifest.url,
        "dataset_doi": manifest.doi,
        "run_mode": options.mode,
    }
    return summary, [
        paths.source_manifest,
        paths.combined_parquet,
        paths.assignment1_tables / "station_source_audit.csv",
        paths.assignment1_tables / "raw_schema.csv",
        paths.assignment1_tables / "raw_missing_summary.csv",
        paths.assignment1_tables / "raw_missing_by_station.csv",
    ]


def prepare_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    _: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Produce EDA evidence and leakage-safe train-fitted model inputs."""

    if not paths.combined_parquet.exists():
        raise FileNotFoundError("Validated source data is missing; run the acquire stage first.")
    air_quality = pd.read_parquet(paths.combined_parquet)
    air_quality["timestamp"] = pd.to_datetime(air_quality["timestamp"])
    set_plot_style()

    analysis_columns = list(CONTINUOUS_INPUT_COLUMNS)
    descriptive = air_quality[analysis_columns].describe(
        percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
    ).T
    _write_frame(
        descriptive,
        paths.assignment1_tables / "descriptive_statistics.csv",
        index=True,
    )

    missing_station = (
        air_quality.groupby("station", observed=True)[analysis_columns]
        .agg(lambda values: values.isna().mean() * 100)
    )
    _write_frame(
        missing_station,
        paths.assignment1_tables / "missing_percent_by_station.csv",
        index=True,
    )
    missing_run_rows: list[dict[str, Any]] = []
    for station, station_data in air_quality.groupby("station", observed=True):
        for column in analysis_columns + ["wd"]:
            mask = station_data[column].isna()
            run_ids = mask.ne(mask.shift(fill_value=False)).cumsum()
            lengths = mask.groupby(run_ids).sum()
            lengths = lengths[lengths > 0]
            missing_run_rows.append(
                {
                    "station": station,
                    "column": column,
                    "missing_runs": len(lengths),
                    "maximum_run_hours": int(lengths.max()) if len(lengths) else 0,
                    "median_run_hours": float(lengths.median()) if len(lengths) else 0.0,
                }
            )
    missing_runs = pd.DataFrame(missing_run_rows)
    _write_frame(missing_runs, paths.assignment1_tables / "missing_run_summary.csv")
    plt.figure(figsize=(11, 6))
    sns.heatmap(missing_station, cmap="Reds", linewidths=0.2)
    plt.title("Missing measurements by station and variable (%)")
    save_figure(paths.assignment1_figures / "missingness_heatmap.png")

    _, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(air_quality["PM2.5"].dropna(), bins=80, ax=axes[0], color="#2563eb")
    axes[0].set_title("PM2.5 distribution")
    axes[0].set_xlabel("PM2.5 (µg/m³)")
    sns.boxplot(data=air_quality, x="PM2.5", y="station", ax=axes[1], showfliers=False)
    axes[1].set_title("PM2.5 by monitoring station")
    save_figure(paths.assignment1_figures / "pm25_distribution_and_station_boxplots.png")
    station_statistics = (
        air_quality.groupby("station", observed=True)["PM2.5"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    _write_frame(station_statistics, paths.assignment1_tables / "pm25_station_statistics.csv")

    eda = air_quality.assign(
        year_value=air_quality["timestamp"].dt.year,
        month_value=air_quality["timestamp"].dt.month,
        hour_value=air_quality["timestamp"].dt.hour,
    )
    hourly = eda.groupby("hour_value", observed=True)["PM2.5"].mean()
    monthly = eda.groupby("month_value", observed=True)["PM2.5"].mean()
    yearly = eda.groupby("year_value", observed=True)["PM2.5"].mean()
    _, axes = plt.subplots(1, 3, figsize=(15, 4))
    hourly.plot(ax=axes[0], marker="o", title="Average PM2.5 by hour")
    monthly.plot(ax=axes[1], marker="o", title="Average PM2.5 by month")
    yearly.plot(ax=axes[2], marker="o", title="Average PM2.5 by year")
    for axis in axes:
        axis.set_ylabel("PM2.5 (µg/m³)")
    save_figure(paths.assignment1_figures / "pm25_temporal_patterns.png")
    temporal_summary = pd.concat(
        [hourly.rename("hourly_mean"), monthly.rename("monthly_mean"), yearly.rename("yearly_mean")],
        axis=1,
    )
    _write_frame(
        temporal_summary,
        paths.assignment1_tables / "pm25_temporal_summary.csv",
        index=True,
    )

    correlation = air_quality[analysis_columns].corr()
    _write_frame(
        correlation,
        paths.assignment1_tables / "continuous_correlation.csv",
        index=True,
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, cmap="vlag", center=0, square=True)
    plt.title("Continuous-variable correlation")
    save_figure(paths.assignment1_figures / "continuous_correlation_heatmap.png")

    relationship_columns = ["PM2.5", "PM10", "NO2", "TEMP", "WSPM"]
    complete_relationships = air_quality[relationship_columns].dropna()
    relationship_sample = complete_relationships.sample(
        min(20_000, len(complete_relationships)), random_state=CONFIG.seed
    )
    _, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, column in zip(axes.ravel(), relationship_columns[1:], strict=True):
        sns.scatterplot(
            data=relationship_sample,
            x=column,
            y="PM2.5",
            alpha=0.15,
            s=10,
            edgecolor=None,
            ax=axis,
        )
        axis.set_title(f"PM2.5 versus {column}")
    save_figure(paths.assignment1_figures / "pm25_feature_relationships.png")

    station_series = {
        station: group.set_index("timestamp")["PM2.5"]
        for station, group in air_quality.groupby("station", observed=True)
    }
    autocorrelation = pd.DataFrame(
        {
            "lag_hours": range(1, 169),
            "mean_station_autocorrelation": [
                np.nanmean([series.autocorr(lag=lag) for series in station_series.values()])
                for lag in range(1, 169)
            ],
        }
    )
    _write_frame(autocorrelation, paths.assignment1_tables / "pm25_autocorrelation.csv")
    plt.figure(figsize=(11, 4))
    sns.lineplot(data=autocorrelation, x="lag_hours", y="mean_station_autocorrelation")
    plt.axvline(24, color="black", linestyle="--", label="24-hour lookback")
    plt.legend()
    plt.title("Mean station PM2.5 autocorrelation")
    save_figure(paths.assignment1_figures / "pm25_autocorrelation.png")

    daily_pm25 = (
        air_quality.assign(date=air_quality["timestamp"].dt.floor("D"))
        .groupby(["station", "date"], observed=True)["PM2.5"]
        .mean()
        .reset_index(name="daily_mean_pm25")
        .sort_values("daily_mean_pm25", ascending=False)
    )
    _write_frame(daily_pm25.head(20), paths.assignment1_tables / "highest_daily_pm25_episodes.csv")
    episode = daily_pm25.iloc[0]
    episode_start = pd.Timestamp(episode["date"]) - pd.Timedelta(days=3)
    episode_end = pd.Timestamp(episode["date"]) + pd.Timedelta(days=4)
    episode_data = air_quality[
        air_quality["station"].eq(episode["station"])
        & air_quality["timestamp"].between(episode_start, episode_end)
    ]
    plt.figure(figsize=(13, 4))
    plt.plot(episode_data["timestamp"], episode_data["PM2.5"], label="PM2.5")
    plt.plot(episode_data["timestamp"], episode_data["PM10"], label="PM10", alpha=0.7)
    plt.title(f"Highest daily PM2.5 episode: {episode['station']}")
    plt.ylabel("Concentration (µg/m³)")
    plt.legend()
    save_figure(paths.assignment1_figures / "selected_pollution_episode.png")

    model_table, preprocessing, split_metadata = create_model_table(air_quality, paths)
    save_model_table(model_table, paths)
    split_distribution = (
        model_table.loc[model_table["valid_anchor"]]
        .groupby("split", observed=True)["target_pm25"]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )
    _write_frame(
        split_distribution,
        paths.assignment1_tables / "target_distribution_by_split.csv",
        index=True,
    )
    split_groups = model_table.loc[
        model_table["valid_anchor"], ["split", "target_pm25"]
    ].groupby("split", observed=True)
    split_plot_data = pd.concat(
        [
            frame.sample(min(20_000, len(frame)), random_state=CONFIG.seed)
            for _, frame in split_groups
        ],
        ignore_index=True,
    )
    upper_limit = model_table.loc[model_table["valid_anchor"], "target_pm25"].quantile(0.99)
    plt.figure(figsize=(10, 4))
    sns.kdeplot(
        data=split_plot_data[split_plot_data["target_pm25"] <= upper_limit],
        x="target_pm25",
        hue="split",
        common_norm=False,
    )
    plt.title("PM2.5 target distribution across chronological splits (≤99th percentile)")
    save_figure(paths.assignment1_figures / "target_distribution_by_split.png")

    sample_sequence = build_sequences(
        model_table,
        preprocessing.feature_columns,
        CONFIG.lookback_hours,
        batch_size=4,
        seed=CONFIG.seed,
    )["train"]
    x_sample, y_sample = sample_sequence[0]
    if x_sample.shape[1] != CONFIG.lookback_hours or y_sample.shape[1] != 1:
        raise RuntimeError("Sequence shape verification failed.")

    summary = {
        "rows": len(model_table),
        "features": len(preprocessing.feature_columns),
        "split_counts": split_metadata["split_counts"],
        "sequence_shape": list(x_sample.shape),
        "target_shape": list(y_sample.shape),
        "run_mode": options.mode,
    }
    return summary, [
        paths.model_table,
        paths.feature_metadata,
        paths.split_metadata,
        paths.preprocessor_file,
        paths.assignment1_tables / "descriptive_statistics.csv",
        paths.assignment1_tables / "missing_percent_by_station.csv",
        paths.assignment1_tables / "missing_run_summary.csv",
        paths.assignment1_tables / "pm25_station_statistics.csv",
        paths.assignment1_tables / "pm25_temporal_summary.csv",
        paths.assignment1_tables / "continuous_correlation.csv",
        paths.assignment1_tables / "pm25_autocorrelation.csv",
        paths.assignment1_tables / "highest_daily_pm25_episodes.csv",
        paths.assignment1_tables / "target_distribution_by_split.csv",
        paths.assignment1_figures / "missingness_heatmap.png",
        paths.assignment1_figures / "pm25_distribution_and_station_boxplots.png",
        paths.assignment1_figures / "pm25_temporal_patterns.png",
        paths.assignment1_figures / "continuous_correlation_heatmap.png",
        paths.assignment1_figures / "pm25_feature_relationships.png",
        paths.assignment1_figures / "pm25_autocorrelation.png",
        paths.assignment1_figures / "selected_pollution_episode.png",
        paths.assignment1_figures / "target_distribution_by_split.png",
    ]


def _modeling_data(options: RuntimeOptions, paths: ProjectPaths) -> pd.DataFrame:
    data = pd.read_parquet(paths.combined_parquet)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if options.mode == "smoke":
        data = data[data["station"].isin(EXPECTED_STATIONS[:2])].copy()
    return data


def _folds(options: RuntimeOptions) -> tuple[Any, ...]:
    return ROLLING_FOLDS if options.mode == "full" else ROLLING_FOLDS[:1]


def _fold_inputs(
    raw_data: pd.DataFrame,
    fold: Any,
    fill_limit: int | None,
    lookback: int,
    options: RuntimeOptions,
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, Any, dict[str, Any], dict[str, Any]]:
    subset = raw_data[raw_data["timestamp"] < pd.Timestamp(fold.validation_end)].copy()
    table, artifacts, metadata = create_model_table(
        subset,
        paths,
        train_start=fold.train_start,
        validation_start=fold.validation_start,
        test_start=fold.validation_end,
        data_end_exclusive=fold.validation_end,
        lookback_hours=lookback,
        forward_fill_limit_hours=fill_limit,
        persist=False,
    )
    sequences = build_sequences(
        table,
        artifacts.feature_columns,
        lookback,
        options.batch_size,
        CONFIG.seed,
        splits=("train", "validation"),
    )
    return table, artifacts, metadata, sequences


def _scaler_values(artifacts: Any) -> tuple[int, float, float]:
    index = artifacts.continuous_columns.index("PM2.5")
    feature_index = artifacts.feature_columns.index("z__PM2.5")
    return feature_index, float(artifacts.scaler.mean_[index]), float(artifacts.scaler.scale_[index])


def _fit_validation(
    model: Any,
    sequences: dict[str, Any],
    targets: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    from tensorflow import keras

    started = time.perf_counter()
    history = model.fit(
        sequences["train"],
        validation_data=sequences["validation"],
        epochs=max_epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_mae",
                mode="min",
                patience=patience,
                restore_best_weights=True,
            ),
            keras.callbacks.TerminateOnNaN(),
        ],
        verbose=0,
    )
    seconds = time.perf_counter() - started
    predictions = np.maximum(model.predict(sequences["validation"], verbose=0).reshape(-1), 0.0)
    best_epoch = int(np.argmin(history.history["val_mae"]) + 1)
    return {
        **regression_metrics(targets, predictions),
        "best_epoch": best_epoch,
        "epochs_recorded": len(history.history["loss"]),
        "parameters": int(model.count_params()),
        "training_seconds": seconds,
    }


def _append_progress(path: Path, row: dict[str, Any], keys: list[str]) -> pd.DataFrame:
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    combined = combined.drop_duplicates(keys, keep="last")
    _write_frame(combined, path)
    return combined


def _already_done(frame: pd.DataFrame, **values: Any) -> bool:
    if frame.empty:
        return False
    mask = pd.Series(True, index=frame.index)
    for column, value in values.items():
        mask &= frame[column].astype(str).eq(str(value))
    return bool(mask.any())


def _ablation_summary(results: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        results.groupby(column, observed=True)["mae"]
        .agg([("mean_mae", "mean"), ("std_mae", "std"), ("evaluations", "size")])
        .reset_index()
    )
    summary["se_mae"] = summary["std_mae"].fillna(0.0) / np.sqrt(summary["evaluations"])
    return summary


def _train_final_model(
    model: Any,
    train_sequence: Any,
    *,
    epochs: int,
    checkpoint: Path,
    history_path: Path,
) -> tuple[Any, float, pd.DataFrame]:
    from tensorflow import keras

    started = time.perf_counter()
    history = model.fit(
        train_sequence,
        epochs=max(1, int(epochs)),
        callbacks=[keras.callbacks.TerminateOnNaN()],
        verbose=2,
    )
    seconds = time.perf_counter() - started
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    frame = pd.DataFrame(history.history)
    frame.index.name = "epoch"
    _write_frame(frame.reset_index(), history_path)
    return model, seconds, frame


def baselines_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    _: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Select preprocessing by rolling validation and fit fair basic models."""

    from tensorflow import keras

    from .models import build_cnn, build_lstm, set_global_determinism

    suffix = options.artifact_suffix
    raw_data = _modeling_data(options, paths)
    folds = _folds(options)
    max_epochs = 1 if options.mode == "smoke" else CONFIG.max_epochs

    fill_path = paths.assignment2_tables / f"forward_fill_ablation{suffix}.csv"
    fill_results = pd.read_csv(fill_path) if fill_path.exists() else pd.DataFrame()
    fill_candidates: list[int | None] = [24] if options.mode == "smoke" else [6, 24, None]
    for fill_limit in fill_candidates:
        label = "unlimited" if fill_limit is None else str(fill_limit)
        for fold in folds:
            if _already_done(fill_results, fill_limit=label, fold=fold.name):
                continue
            table, artifacts, _, sequences = _fold_inputs(
                raw_data, fold, fill_limit, 24, options, paths
            )
            anchors = sequences["validation"].ordered_anchor_indices()
            set_global_determinism(CONFIG.seed)
            model = build_lstm((24, len(artifacts.feature_columns)))
            result = _fit_validation(
                model,
                sequences,
                table.loc[anchors, "target_pm25"].to_numpy(),
                max_epochs=max_epochs,
                patience=CONFIG.early_stopping_patience,
            )
            fill_results = _append_progress(
                fill_path,
                {"fill_limit": label, "fold": fold.name, "seed": CONFIG.seed, **result},
                ["fill_limit", "fold", "seed"],
            )
            keras.backend.clear_session()
    fill_summary = _ablation_summary(fill_results, "fill_limit")
    fill_summary["preference_rank"] = fill_summary["fill_limit"].map(
        {"24": 0, "6": 1, "unlimited": 2}
    )
    selected_fill_row = select_one_standard_error(
        fill_summary, simplicity_columns=("preference_rank",)
    )
    selected_fill_label = str(selected_fill_row["fill_limit"])
    selected_fill = None if selected_fill_label == "unlimited" else int(selected_fill_label)
    fill_summary_path = paths.assignment2_tables / f"forward_fill_ablation_summary{suffix}.csv"
    _write_frame(fill_summary, fill_summary_path)

    lookback_path = paths.assignment2_tables / f"lookback_ablation{suffix}.csv"
    lookback_results = pd.read_csv(lookback_path) if lookback_path.exists() else pd.DataFrame()
    lookback_candidates = [24] if options.mode == "smoke" else [12, 24, 48]
    for lookback in lookback_candidates:
        for fold in folds:
            if _already_done(lookback_results, lookback=lookback, fold=fold.name):
                continue
            table, artifacts, _, sequences = _fold_inputs(
                raw_data, fold, selected_fill, lookback, options, paths
            )
            anchors = sequences["validation"].ordered_anchor_indices()
            set_global_determinism(CONFIG.seed)
            model = build_lstm((lookback, len(artifacts.feature_columns)))
            result = _fit_validation(
                model,
                sequences,
                table.loc[anchors, "target_pm25"].to_numpy(),
                max_epochs=max_epochs,
                patience=CONFIG.early_stopping_patience,
            )
            lookback_results = _append_progress(
                lookback_path,
                {"lookback": lookback, "fold": fold.name, "seed": CONFIG.seed, **result},
                ["lookback", "fold", "seed"],
            )
            keras.backend.clear_session()
    lookback_summary = _ablation_summary(lookback_results, "lookback")
    selected_lookback_row = select_one_standard_error(
        lookback_summary, simplicity_columns=("lookback",)
    )
    selected_lookback = int(selected_lookback_row["lookback"])
    lookback_summary_path = paths.assignment2_tables / f"lookback_ablation_summary{suffix}.csv"
    _write_frame(lookback_summary, lookback_summary_path)

    rolling_path = paths.assignment2_tables / f"basic_rolling_results{suffix}.csv"
    rolling_results = pd.read_csv(rolling_path) if rolling_path.exists() else pd.DataFrame()
    persistence_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for fold in folds:
        table, artifacts, metadata, sequences = _fold_inputs(
            raw_data, fold, selected_fill, selected_lookback, options, paths
        )
        fold_metadata.append(
            {
                **fold.to_dict(),
                "lookback_hours": selected_lookback,
                "forward_fill_limit_hours": selected_fill,
                "split_counts": metadata["split_counts"],
                "preprocessing_fit": metadata["preprocessing_fit"],
            }
        )
        anchors = sequences["validation"].ordered_anchor_indices()
        targets = table.loc[anchors, "target_pm25"].to_numpy()
        baseline = prediction_frame(
            table,
            sequences["validation"],
            table.loc[anchors, "PM2.5_input_raw"].to_numpy(),
        )
        for name in ("persistence", "seasonal_persistence"):
            persistence_rows.append(
                {"fold": fold.name, "model": name, **regression_metrics(targets, baseline[name])}
            )
        for name in ("lstm", "cnn"):
            if _already_done(rolling_results, model=name, fold=fold.name):
                continue
            set_global_determinism(CONFIG.seed)
            input_shape = (selected_lookback, len(artifacts.feature_columns))
            model = build_lstm(input_shape) if name == "lstm" else build_cnn(input_shape)
            result = _fit_validation(
                model,
                sequences,
                targets,
                max_epochs=(1 if options.mode == "smoke" else (CONFIG.max_epochs if name == "lstm" else CONFIG.cnn_max_epochs)),
                patience=(CONFIG.early_stopping_patience if name == "lstm" else CONFIG.cnn_early_stopping_patience),
            )
            rolling_results = _append_progress(
                rolling_path,
                {"model": name, "fold": fold.name, "seed": CONFIG.seed, **result},
                ["model", "fold", "seed"],
            )
            keras.backend.clear_session()

    baseline_metrics_path = paths.assignment2_tables / f"validation_baseline_metrics{suffix}.csv"
    baseline_metrics = (
        pd.DataFrame(persistence_rows)
        .groupby("model", observed=True)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), r2=("r2", "mean"), smape_percent=("smape_percent", "mean"))
        .reset_index()
    )
    _write_frame(baseline_metrics, baseline_metrics_path)
    comparison = (
        rolling_results.groupby("model", observed=True)
        .agg(
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            smape_percent=("smape_percent", "mean"),
            parameters=("parameters", "first"),
            training_seconds=("training_seconds", "sum"),
            median_best_epoch=("best_epoch", "median"),
        )
        .reset_index()
        .sort_values("mae")
    )
    comparison_path = paths.assignment2_tables / f"basic_model_comparison{suffix}.csv"
    _write_frame(comparison, comparison_path)
    selected_model = str(comparison.iloc[0]["model"])
    selection_path = paths.models / f"selected_architecture{suffix}.json"
    write_json_atomic(
        selection_path,
        {
            "selected_model": selected_model,
            "selection_metric": "mean_rolling_validation_mae",
            "run_mode": options.mode,
            "config_hash": options.stable_hash(),
        },
    )
    selected_payload = {
        "forward_fill_limit_hours": selected_fill,
        "lookback_hours": selected_lookback,
        "selection_rule": "lowest mean rolling MAE with one-standard-error simplicity rule",
        "run_mode": options.mode,
        "config_hash": options.stable_hash(),
    }
    write_json_atomic(paths.selected_preprocessing, selected_payload)
    write_json_atomic(
        paths.rolling_validation_metadata,
        {"folds": fold_metadata, "frozen_test_start": CONFIG.test_start},
    )

    final_table, final_artifacts, _, = create_model_table(
        raw_data,
        paths,
        train_start=CONFIG.train_start,
        validation_start=CONFIG.test_start,
        test_start=CONFIG.test_start,
        data_end_exclusive=CONFIG.data_end_exclusive,
        lookback_hours=selected_lookback,
        forward_fill_limit_hours=selected_fill,
        persist=True,
    )
    save_model_table(final_table, paths)
    final_sequences = build_sequences(
        final_table,
        final_artifacts.feature_columns,
        selected_lookback,
        options.batch_size,
        CONFIG.seed,
        splits=("train",),
    )
    final_seconds: dict[str, float] = {}
    for name in ("lstm", "cnn"):
        epochs = int(max(1, np.ceil(comparison.loc[comparison["model"].eq(name), "median_best_epoch"].iloc[0])))
        set_global_determinism(CONFIG.seed)
        input_shape = (selected_lookback, len(final_artifacts.feature_columns))
        model = build_lstm(input_shape) if name == "lstm" else build_cnn(input_shape)
        checkpoint = paths.models / f"basic_{name}{suffix}.keras"
        history_path = paths.histories / f"basic_{name}{suffix}_epochs.csv"
        _, seconds, history = _train_final_model(
            model,
            final_sequences["train"],
            epochs=epochs,
            checkpoint=checkpoint,
            history_path=history_path,
        )
        final_seconds[name] = seconds
        plot_history(
            history.assign(val_mae=np.nan, val_loss=np.nan),
            paths.assignment2_figures / f"basic_{name}_training{suffix}.png",
            f"Basic {name.upper()} final pre-test training",
        )
        keras.backend.clear_session()
    outputs = [
        fill_path,
        fill_summary_path,
        lookback_path,
        lookback_summary_path,
        rolling_path,
        baseline_metrics_path,
        comparison_path,
        selection_path,
        paths.selected_preprocessing,
        paths.rolling_validation_metadata,
        paths.model_table,
        paths.feature_metadata,
        paths.split_metadata,
        paths.preprocessor_file,
        paths.models / f"basic_lstm{suffix}.keras",
        paths.models / f"basic_cnn{suffix}.keras",
        paths.histories / f"basic_lstm{suffix}_epochs.csv",
        paths.histories / f"basic_cnn{suffix}_epochs.csv",
        paths.assignment2_figures / f"basic_lstm_training{suffix}.png",
        paths.assignment2_figures / f"basic_cnn_training{suffix}.png",
    ]
    return {
        "selected_model": selected_model,
        "selected_forward_fill_limit_hours": selected_fill,
        "selected_lookback_hours": selected_lookback,
        "rolling_validation_mae": float(comparison.iloc[0]["mae"]),
        "final_training_seconds": final_seconds,
        "folds": len(folds),
        "run_mode": options.mode,
    }, outputs


def _build_candidate_model(
    candidate: LSTMConfig,
    input_shape: tuple[int, int],
    artifacts: Any,
) -> Any:
    from .models import build_lstm

    feature_index, pm25_mean, pm25_scale = _scaler_values(artifacts)
    return build_lstm(
        input_shape,
        units=candidate.units,
        dense_units=candidate.dense_units,
        dropout=candidate.dropout,
        learning_rate=candidate.learning_rate,
        l2_strength=candidate.l2_strength,
        num_layers=candidate.num_layers,
        huber_delta=candidate.huber_delta,
        formulation=candidate.formulation,
        pm25_feature_index=feature_index,
        pm25_mean=pm25_mean,
        pm25_scale=pm25_scale,
    )


def tune_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    _: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Run repeated-seed stratified random search over rolling folds."""

    import joblib
    from tensorflow import keras

    from .models import set_global_determinism

    suffix = options.artifact_suffix
    selected_preprocessing = _read_json(paths.selected_preprocessing)
    selected_lookback = int(selected_preprocessing["lookback_hours"])
    selected_fill = selected_preprocessing["forward_fill_limit_hours"]
    raw_data = _modeling_data(options, paths)
    all_candidates = generate_lstm_candidates(CONFIG.seed, CONFIG.random_search_candidates)
    candidates = all_candidates if options.mode == "full" else [all_candidates[0], all_candidates[8]]
    seeds = (42, 52) if options.mode == "full" else (42,)
    folds = _folds(options)
    max_epochs = 1 if options.mode == "smoke" else CONFIG.max_epochs
    search_path = paths.assignment2_tables / f"random_search_rolling_results{suffix}.csv"
    results = pd.read_csv(search_path) if search_path.exists() else pd.DataFrame()
    for fold in folds:
        table, artifacts, _, sequences = _fold_inputs(
            raw_data, fold, selected_fill, selected_lookback, options, paths
        )
        anchors = sequences["validation"].ordered_anchor_indices()
        targets = table.loc[anchors, "target_pm25"].to_numpy()
        for candidate in candidates:
            for seed in seeds:
                if _already_done(
                    results,
                    candidate_id=candidate.candidate_id,
                    fold=fold.name,
                    seed=seed,
                ):
                    continue
                set_global_determinism(seed)
                model = _build_candidate_model(
                    candidate,
                    (selected_lookback, len(artifacts.feature_columns)),
                    artifacts,
                )
                fit = _fit_validation(
                    model,
                    sequences,
                    targets,
                    max_epochs=max_epochs,
                    patience=CONFIG.early_stopping_patience,
                )
                results = _append_progress(
                    search_path,
                    {
                        **candidate.to_dict(),
                        "fold": fold.name,
                        "seed": seed,
                        **fit,
                    },
                    ["candidate_id", "fold", "seed"],
                )
                keras.backend.clear_session()
    summary = aggregate_validation_results(results)
    summary_path = paths.assignment2_tables / f"random_search_summary{suffix}.csv"
    _write_frame(summary, summary_path)
    selected_row = select_lstm_candidate(summary)
    selected_candidate = next(
        candidate for candidate in candidates if candidate.candidate_id == selected_row["candidate_id"]
    )
    selected_results = results[results["candidate_id"].eq(selected_candidate.candidate_id)]
    epoch_budget = int(max(1, np.ceil(selected_results["best_epoch"].median())))
    best_path = paths.models / f"best_hyperparameters{suffix}.json"
    write_json_atomic(
        best_path,
        {
            "architecture": "lstm",
            "objective": "mean_rolling_validation_mae",
            "values": selected_candidate.to_dict(),
            "epoch_budget": epoch_budget,
            "selection_rule": "one-standard-error; parameters, layers, direct formulation",
            "run_mode": options.mode,
            "config_hash": options.stable_hash(),
        },
    )

    model_table, feature_columns, sequences = _load_model_inputs(
        options, paths, splits=("train",)
    )
    artifacts = joblib.load(paths.preprocessor_file)
    set_global_determinism(CONFIG.seed)
    model = _build_candidate_model(
        selected_candidate,
        (selected_lookback, len(feature_columns)),
        artifacts,
    )
    final_checkpoint = paths.models / f"tuned_final{suffix}.keras"
    final_history = paths.histories / f"tuned_final{suffix}_epochs.csv"
    model, training_seconds, _ = _train_final_model(
        model,
        sequences["train"],
        epochs=epoch_budget,
        checkpoint=final_checkpoint,
        history_path=final_history,
    )
    del model_table, model
    final_selection_path = paths.models / f"final_model_selection{suffix}.json"
    write_json_atomic(
        final_selection_path,
        {
            "architecture": "lstm",
            "selected_candidate": selected_candidate.to_dict(),
            "selected_seed": CONFIG.seed,
            "epoch_budget": epoch_budget,
            "selection_metric": "mean_rolling_validation_mae",
            "mean_validation_mae": float(selected_row["mean_mae"]),
            "validation_standard_error": float(selected_row["se_mae"]),
            "final_checkpoint": str(final_checkpoint),
            "frozen_test_used_for_selection": False,
            "run_mode": options.mode,
            "config_hash": options.stable_hash(),
        },
    )
    outputs = [search_path, summary_path, best_path, final_checkpoint, final_history, final_selection_path]
    return {
        "architecture": "lstm",
        "candidate_count": len(candidates),
        "validation_fits": len(results),
        "selected_candidate": selected_candidate.candidate_id,
        "formulation": selected_candidate.formulation,
        "validation_mae": float(selected_row["mean_mae"]),
        "epoch_budget": epoch_budget,
        "training_seconds": training_seconds,
        "run_mode": options.mode,
    }, outputs


def evaluate_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    _: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Evaluate frozen models on the previously inspected chronological benchmark."""

    from tensorflow import keras

    # Import registers the custom residual layer before loading a residual model.
    from .models import PersistenceResidual

    del PersistenceResidual

    suffix = options.artifact_suffix
    required = [
        paths.models / f"basic_lstm{suffix}.keras",
        paths.models / f"basic_cnn{suffix}.keras",
        paths.models / f"tuned_final{suffix}.keras",
        paths.models / f"final_model_selection{suffix}.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required trained artifacts are missing: {missing}")
    model_table, feature_columns, sequences = _load_model_inputs(
        options, paths, splits=("test",)
    )
    del feature_columns
    final_selection = _read_json(paths.models / f"final_model_selection{suffix}.json")
    if final_selection.get("run_mode") != options.mode:
        raise RuntimeError("The tuned model was produced in a different run mode.")
    set_plot_style()

    models = {
        "basic_lstm": keras.models.load_model(paths.models / f"basic_lstm{suffix}.keras"),
        "basic_cnn": keras.models.load_model(paths.models / f"basic_cnn{suffix}.keras"),
        "tuned_final": keras.models.load_model(paths.models / f"tuned_final{suffix}.keras"),
    }
    prediction_values: dict[str, np.ndarray] = {}
    inference_rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        started = time.perf_counter()
        prediction_values[model_name] = np.maximum(
            model.predict(sequences["test"], verbose=0).reshape(-1),
            0.0,
        )
        inference_rows.append(
            {
                "model": model_name,
                "inference_seconds": time.perf_counter() - started,
                "parameters": model.count_params(),
            }
        )
    inference_path = paths.assignment2_tables / f"inference_timing{suffix}.csv"
    _write_frame(pd.DataFrame(inference_rows), inference_path)

    base_frame = prediction_frame(model_table, sequences["test"], prediction_values["tuned_final"])
    base_frame = base_frame.rename(columns={"y_pred": "tuned_final"})
    base_frame["basic_lstm"] = prediction_values["basic_lstm"]
    base_frame["basic_cnn"] = prediction_values["basic_cnn"]
    predictions_path = paths.assignment2_tables / f"final_test_predictions{suffix}.parquet"
    base_frame.to_parquet(predictions_path, index=False)

    prediction_columns = [
        "persistence",
        "seasonal_persistence",
        "basic_lstm",
        "basic_cnn",
        "tuned_final",
    ]
    overall_metrics = pd.DataFrame(
        [
            {"model": column, **regression_metrics(base_frame["y_true"], base_frame[column])}
            for column in prediction_columns
        ]
    ).sort_values("mae")
    overall_path = paths.assignment2_tables / f"final_overall_metrics{suffix}.csv"
    _write_frame(overall_metrics, overall_path)

    training_targets = model_table.loc[
        model_table["valid_anchor"] & model_table["split"].eq("train"), "target_pm25"
    ]
    grouped_frame = add_analysis_groups(base_frame, training_targets)
    group_outputs: dict[str, pd.DataFrame] = {}
    for group_column in ("station", "season", "pollution_band"):
        rows: list[pd.DataFrame] = []
        for model_name in prediction_columns:
            metrics = metrics_by_group(
                grouped_frame,
                group_column,
                prediction_column=model_name,
            )
            metrics.insert(0, "model", model_name)
            rows.append(metrics)
        group_outputs[group_column] = pd.concat(rows, ignore_index=True)
    station_metrics = group_outputs["station"]
    season_metrics = group_outputs["season"]
    pollution_metrics = group_outputs["pollution_band"]
    _write_frame(
        station_metrics,
        paths.assignment2_tables / f"final_metrics_by_station{suffix}.csv",
    )
    _write_frame(
        season_metrics,
        paths.assignment2_tables / f"final_metrics_by_season{suffix}.csv",
    )
    _write_frame(
        pollution_metrics,
        paths.assignment2_tables / f"final_metrics_by_pollution_band{suffix}.csv",
    )

    bootstrap_results = pd.DataFrame(
        [
            {
                "comparison": f"tuned_final_vs_{comparison_model}",
                **paired_block_bootstrap(
                    base_frame,
                    "tuned_final",
                    comparison_model,
                    n_boot=(100 if options.mode == "smoke" else 1_000),
                    seed=CONFIG.seed,
                ),
            }
            for comparison_model in ("basic_lstm", "basic_cnn", "persistence")
        ]
    )
    _write_frame(
        bootstrap_results,
        paths.assignment2_tables / f"paired_block_bootstrap{suffix}.csv",
    )

    tuned_grouped = grouped_frame.assign(y_pred=grouped_frame["tuned_final"])
    plot_actual_vs_predicted(
        tuned_grouped,
        paths.assignment2_figures / f"final_actual_vs_predicted{suffix}.png",
        "Tuned model: observed versus predicted PM2.5",
    )
    plot_station_mae(
        station_metrics[station_metrics["model"].eq("tuned_final")],
        paths.assignment2_figures / f"final_station_mae{suffix}.png",
    )
    plot_test_week(
        tuned_grouped,
        station=str(grouped_frame["station"].iloc[0]),
        start=str(grouped_frame["target_timestamp"].min().date()),
        path=paths.assignment2_figures / f"final_example_week{suffix}.png",
    )
    residual_rows = []
    for model_name in prediction_columns:
        residual = grouped_frame["y_true"] - grouped_frame[model_name]
        extreme = grouped_frame["pollution_band"].astype(str).eq("extreme")
        residual_rows.append(
            {
                "model": model_name,
                "mean_residual": float(residual.mean()),
                "median_residual": float(residual.median()),
                "underprediction_percent": float((residual > 0).mean() * 100),
                "extreme_mean_residual": float(residual.loc[extreme].mean()),
                "extreme_underprediction_percent": float((residual.loc[extreme] > 0).mean() * 100),
            }
        )
    residual_diagnostics = pd.DataFrame(residual_rows)
    residual_path = paths.assignment2_tables / f"residual_diagnostics{suffix}.csv"
    _write_frame(residual_diagnostics, residual_path)
    residual_frame = tuned_grouped.copy()
    residual_frame["residual"] = residual_frame["y_true"] - residual_frame["y_pred"]
    _, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.histplot(residual_frame["residual"], bins=80, ax=axes[0], color="#7c3aed")
    axes[0].set_title("Test residual distribution")
    residual_sample = residual_frame.iloc[:: max(1, len(residual_frame) // 10_000)]
    sns.scatterplot(
        data=residual_sample,
        x="target_timestamp",
        y="residual",
        hue="station",
        legend=False,
        alpha=0.35,
        s=12,
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residuals over the future test period")
    save_figure(paths.assignment2_figures / f"final_residual_analysis{suffix}.png")

    tuned = overall_metrics.loc[overall_metrics["model"].eq("tuned_final")].iloc[0]
    persistence = overall_metrics.loc[overall_metrics["model"].eq("persistence")].iloc[0]
    improvement = (persistence["mae"] - tuned["mae"]) / persistence["mae"] * 100
    basic_bootstrap = bootstrap_results.loc[
        bootstrap_results["comparison"].eq("tuned_final_vs_basic_lstm")
    ].iloc[0]
    significant = bool(float(basic_bootstrap["ci95_upper"]) < 0)
    claim_path = paths.assignment2_tables / f"optimization_claim{suffix}.json"
    write_json_atomic(
        claim_path,
        {
            "statistically_superior_to_basic_lstm": significant,
            "rule": "claim superiority only when the paired block-bootstrap CI is below zero",
            "test_description": "frozen chronological benchmark previously inspected",
            "test_used_for_model_selection": False,
        },
    )
    summary = {
        "winner": str(overall_metrics.iloc[0]["model"]),
        "winner_mae": float(overall_metrics.iloc[0]["mae"]),
        "tuned_mae": float(tuned["mae"]),
        "persistence_mae": float(persistence["mae"]),
        "improvement_over_persistence_percent": float(improvement),
        "test_anchors": len(base_frame),
        "statistically_superior_to_basic_lstm": significant,
        "run_mode": options.mode,
    }
    outputs = [
        predictions_path,
        overall_path,
        inference_path,
        paths.assignment2_tables / f"final_metrics_by_station{suffix}.csv",
        paths.assignment2_tables / f"final_metrics_by_season{suffix}.csv",
        paths.assignment2_tables / f"final_metrics_by_pollution_band{suffix}.csv",
        paths.assignment2_tables / f"paired_block_bootstrap{suffix}.csv",
        residual_path,
        claim_path,
        paths.assignment2_figures / f"final_actual_vs_predicted{suffix}.png",
        paths.assignment2_figures / f"final_station_mae{suffix}.png",
        paths.assignment2_figures / f"final_example_week{suffix}.png",
        paths.assignment2_figures / f"final_residual_analysis{suffix}.png",
    ]
    return summary, outputs


STAGE_FUNCTIONS: dict[str, StageFunction] = {
    "acquire": acquire_stage,
    "prepare": prepare_stage,
    "baselines": baselines_stage,
    "tune": tune_stage,
    "evaluate": evaluate_stage,
}


def run_stage(
    stage: str,
    options: RuntimeOptions | None = None,
    paths: ProjectPaths = PATHS,
) -> dict[str, Any]:
    """Run or safely reuse one pipeline stage."""

    options = options or RuntimeOptions.from_environment()
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"Unknown workflow stage: {stage}")
    store = RunStateStore(paths, options)
    if options.resume and store.is_complete(stage):
        return store.summary(stage)
    if options.reuse_only:
        raise RuntimeError(
            f"The {stage!r} stage is not complete for {options.mode} mode; "
            "notebook rendering is not allowed to start expensive work."
        )
    function = STAGE_FUNCTIONS[stage]
    expected_summary: dict[str, Any]
    outputs: list[Path]
    # The function determines its output contract after resolving the run suffix.
    with tracked_stage(store, stage, outputs=[]) as tracked_summary:
        expected_summary, outputs = function(options, paths, store)
        tracked_summary.update(expected_summary)
    # Replace the empty output list written by the context manager with the real contract.
    state = store.load()
    state["stages"][stage]["outputs"] = [str(path) for path in outputs]
    store.save(state)
    return expected_summary


def run_pipeline(
    options: RuntimeOptions,
    paths: ProjectPaths = PATHS,
    stages: tuple[str, ...] = ("acquire", "prepare", "baselines", "tune", "evaluate"),
) -> dict[str, dict[str, Any]]:
    """Execute the ordered experiment stages with stage-level resume."""

    summaries: dict[str, dict[str, Any]] = {}
    for stage in stages:
        print(f"\n=== ODL stage: {stage} ({options.mode}) ===", flush=True)
        summaries[stage] = run_stage(stage, options=options, paths=paths)
        print(json.dumps(summaries[stage], indent=2, default=str), flush=True)
    return summaries

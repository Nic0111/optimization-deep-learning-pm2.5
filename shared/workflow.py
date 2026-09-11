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
    FEATURE_POLICIES,
    FeaturePolicyName,
    PreprocessingPolicy,
    create_model_table,
    download_dataset,
    extract_dataset,
    fit_preprocessor,
    load_development_data,
    load_frozen_benchmark,
    load_and_validate_raw,
    missing_summary,
    partition_frozen_benchmark,
    schema_summary,
    transform_model_table,
)
from .evaluation import (
    add_analysis_groups,
    metrics_by_group,
    paired_moving_block_bootstrap,
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
from .runtime import RunStateStore, RuntimeOptions, write_json_atomic
from .selection import (
    LSTMConfig,
    ROLLING_FOLDS,
    aggregate_validation_results,
    generate_optimization_recipes,
    generate_lstm_candidates,
    select_paired_one_standard_error,
    select_lstm_candidate,
    select_one_standard_error,
    validate_search_completeness,
)
from .sequence_pipeline import build_sequences, prediction_frame

StageFunction = Callable[[RuntimeOptions, ProjectPaths, RunStateStore], tuple[dict[str, Any], list[Path]]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)


def _selected_preprocessing_path(paths: ProjectPaths, suffix: str) -> Path:
    return paths.processed / f"selected_preprocessing{suffix}.json"


def _rolling_metadata_path(paths: ProjectPaths, suffix: str) -> Path:
    return paths.processed / f"rolling_validation_metadata{suffix}.json"


def _model_input_paths(paths: ProjectPaths, suffix: str) -> tuple[Path, Path, Path, Path]:
    return (
        paths.processed / f"model_table{suffix}.parquet",
        paths.processed / f"feature_metadata{suffix}.json",
        paths.processed / f"split_metadata{suffix}.json",
        paths.preprocessing / f"preprocessing{suffix}.joblib",
    )


def _benchmark_input_paths(paths: ProjectPaths, suffix: str) -> tuple[Path, Path]:
    return (
        paths.processed / f"frozen_benchmark_model_table{suffix}.parquet",
        paths.processed / f"frozen_benchmark_split_metadata{suffix}.json",
    )


def _persist_model_inputs(
    table: pd.DataFrame,
    artifacts: Any,
    metadata: dict[str, Any],
    paths: ProjectPaths,
    suffix: str,
) -> tuple[Path, Path, Path, Path]:
    import joblib

    model_path, feature_path, split_path, preprocessor_path = _model_input_paths(paths, suffix)
    table.to_parquet(model_path, index=False)
    joblib.dump(artifacts, preprocessor_path)
    write_json_atomic(
        feature_path,
        {
            "features": artifacts.feature_columns,
            "count": len(artifacts.feature_columns),
            "feature_policy": artifacts.feature_policy,
            "engineered_features": artifacts.engineered_columns,
        },
    )
    write_json_atomic(split_path, metadata)
    return model_path, feature_path, split_path, preprocessor_path


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
    model_path, feature_path, split_path, _ = _model_input_paths(paths, options.artifact_suffix)
    if not model_path.exists():
        raise FileNotFoundError("Prepared model data is missing; run the prepare stage first.")
    model_table = pd.read_parquet(model_path)
    model_table["timestamp"] = pd.to_datetime(model_table["timestamp"])
    model_table["target_timestamp"] = pd.to_datetime(model_table["target_timestamp"])
    feature_columns = _read_json(feature_path)["features"]
    lookback = int(_read_json(split_path)["lookback_hours"])
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
    development_rows, benchmark_rows = partition_frozen_benchmark(air_quality, paths)

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
        "development_rows": development_rows,
        "frozen_benchmark_rows": benchmark_rows,
        "run_mode": options.mode,
    }
    return summary, [
        paths.source_manifest,
        paths.combined_parquet,
        paths.development_parquet,
        paths.frozen_benchmark_parquet,
        paths.assignment1_tables / "station_source_audit.csv",
        paths.assignment1_tables / "raw_schema.csv",
        paths.assignment1_tables / "raw_missing_summary.csv",
        paths.assignment1_tables / "raw_missing_by_station.csv",
    ]


def prepare_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    store: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Produce development-only EDA and leakage-safe train-fitted model inputs."""

    air_quality = load_development_data(paths)
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
    plt.axvline(24, color="black", linestyle="--", label="24-hour ablation candidate")
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

    model_table, preprocessing, split_metadata = create_model_table(
        air_quality,
        paths,
        data_end_exclusive=CONFIG.test_start,
        persist=False,
        source_fingerprint=store.source_fingerprint,
    )
    model_input_outputs = _persist_model_inputs(
        model_table,
        preprocessing,
        split_metadata,
        paths,
        options.artifact_suffix,
    )
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
        splits=("train",),
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
        *model_input_outputs,
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
    data = load_development_data(paths)
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
    source_fingerprint: str | None = None,
    feature_policy: FeaturePolicyName = "base",
    batch_size: int | None = None,
    seed: int = CONFIG.seed,
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
        feature_policy=feature_policy,
        persist=False,
        source_fingerprint=source_fingerprint,
    )
    sequences = build_sequences(
        table,
        artifacts.feature_columns,
        lookback,
        batch_size or options.batch_size,
        seed,
        splits=("train", "validation"),
    )
    return table, artifacts, metadata, sequences


def _scaler_values(artifacts: Any) -> tuple[int, float, float]:
    index = artifacts.scaled_source_columns.index("PM2.5")
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


def _initial_progress(path: Path, *, resume: bool) -> pd.DataFrame:
    if not resume:
        path.unlink(missing_ok=True)
        return pd.DataFrame()
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _already_done(frame: pd.DataFrame, **values: Any) -> bool:
    if frame.empty:
        return False
    mask = pd.Series(True, index=frame.index)
    for column, value in values.items():
        if column not in frame.columns:
            return False
        mask &= frame[column].astype(str).eq(str(value))
    return bool(mask.any())


def _validate_named_completeness(
    results: pd.DataFrame,
    *,
    item_column: str,
    item_ids: tuple[str, ...],
    fold_names: tuple[str, ...],
    seeds: tuple[int, ...],
    source_fingerprint: str,
) -> pd.DataFrame:
    """Validate a named fit grid without retaining an unrelated candidate identifier."""

    working = results.copy()
    if item_column != "candidate_id":
        working = working.drop(columns=["candidate_id"], errors="ignore").rename(
            columns={item_column: "candidate_id"}
        )
    validated = validate_search_completeness(
        working,
        item_ids,
        fold_names,
        seeds,
        source_fingerprint=source_fingerprint,
    )
    if item_column != "candidate_id":
        validated = validated.rename(columns={"candidate_id": item_column})
    return validated


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
    store: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Select causal features by paired validation and fit fair basic models."""

    from tensorflow import keras

    from .models import build_cnn, build_lstm, set_global_determinism

    suffix = options.artifact_suffix
    raw_data = _modeling_data(options, paths)
    folds = _folds(options)
    seeds = (42, 52) if options.mode == "full" else (42,)
    selected_fill = 24
    selected_lookback = 12
    max_epochs = 1 if options.mode == "smoke" else CONFIG.max_epochs
    policies = list(FEATURE_POLICIES)

    preprocessing_lock_path = paths.processed / f"preprocessing_lock{suffix}.json"
    write_json_atomic(
        preprocessing_lock_path,
        {
            "forward_fill_limit_hours": selected_fill,
            "lookback_hours": selected_lookback,
            "basis": "V2.1 rolling-validation ablations; held fixed for focused V2.2",
            "source_fingerprint": store.source_fingerprint,
        },
    )

    feature_path = paths.assignment2_tables / f"feature_policy_ablation{suffix}.csv"
    feature_results = _initial_progress(feature_path, resume=options.resume)
    for fold in folds:
        for policy_name in policies:
            table, artifacts, _, _ = _fold_inputs(
                raw_data,
                fold,
                selected_fill,
                selected_lookback,
                options,
                paths,
                store.source_fingerprint,
                feature_policy=policy_name,
            )
            for seed in seeds:
                if _already_done(
                    feature_results,
                    candidate_id=policy_name,
                    fold=fold.name,
                    seed=seed,
                    source_fingerprint=store.source_fingerprint,
                ):
                    continue
                sequences = build_sequences(
                    table,
                    artifacts.feature_columns,
                    selected_lookback,
                    options.batch_size,
                    seed,
                    splits=("train", "validation"),
                )
                anchors = sequences["validation"].ordered_anchor_indices()
                set_global_determinism(seed)
                model = build_lstm((selected_lookback, len(artifacts.feature_columns)))
                result = _fit_validation(
                    model,
                    sequences,
                    table.loc[anchors, "target_pm25"].to_numpy(),
                    max_epochs=max_epochs,
                    patience=CONFIG.early_stopping_patience,
                )
                feature_results = _append_progress(
                    feature_path,
                    {
                        "candidate_id": policy_name,
                        "feature_policy": policy_name,
                        "complexity_rank": policies.index(policy_name),
                        "fold": fold.name,
                        "seed": seed,
                        "num_layers": 1,
                        "formulation": "direct",
                        "source_fingerprint": store.source_fingerprint,
                        **result,
                    },
                    ["candidate_id", "fold", "seed"],
                )
                keras.backend.clear_session()
    feature_results = validate_search_completeness(
        feature_results,
        policies,
        [fold.name for fold in folds],
        seeds,
        source_fingerprint=store.source_fingerprint,
    )
    _write_frame(feature_results, feature_path)
    feature_summary = aggregate_validation_results(feature_results)
    feature_summary["feature_policy"] = feature_summary["candidate_id"]
    feature_summary["complexity_rank"] = feature_summary["feature_policy"].map(
        {name: rank for rank, name in enumerate(policies)}
    )
    selected_feature_row, feature_comparisons, feature_audit = select_paired_one_standard_error(
        feature_results,
        feature_summary,
        item_column="candidate_id",
        simplicity_columns=("complexity_rank",),
    )
    selected_feature = str(selected_feature_row["feature_policy"])
    feature_summary_path = paths.assignment2_tables / f"feature_policy_ablation_summary{suffix}.csv"
    feature_audit_path = paths.assignment2_tables / f"feature_policy_paired_audit{suffix}.csv"
    _write_frame(feature_summary, feature_summary_path)
    _write_frame(feature_comparisons, feature_audit_path)

    rolling_path = paths.assignment2_tables / f"basic_rolling_results{suffix}.csv"
    rolling_results = _initial_progress(rolling_path, resume=options.resume)
    selected_feature_results = feature_results.loc[
        feature_results["candidate_id"].eq(selected_feature)
    ]
    for row in selected_feature_results.to_dict(orient="records"):
        row.pop("candidate_id", None)
        rolling_results = _append_progress(
            rolling_path,
            {**row, "model": "lstm"},
            ["model", "fold", "seed"],
        )

    persistence_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for fold in folds:
        table, artifacts, metadata, _ = _fold_inputs(
            raw_data,
            fold,
            selected_fill,
            selected_lookback,
            options,
            paths,
            store.source_fingerprint,
            feature_policy=selected_feature,
        )
        fold_metadata.append(
            {
                **fold.to_dict(),
                "lookback_hours": selected_lookback,
                "forward_fill_limit_hours": selected_fill,
                "feature_policy": selected_feature,
                "split_counts": metadata["split_counts"],
                "preprocessing_fit": metadata["preprocessing_fit"],
            }
        )
        reference_sequences = build_sequences(
            table,
            artifacts.feature_columns,
            selected_lookback,
            options.batch_size,
            CONFIG.seed,
            splits=("train", "validation"),
        )
        anchors = reference_sequences["validation"].ordered_anchor_indices()
        targets = table.loc[anchors, "target_pm25"].to_numpy()
        baseline = prediction_frame(
            table,
            reference_sequences["validation"],
            table.loc[anchors, "PM2.5_input_raw"].to_numpy(),
        )
        for name in ("persistence", "daily_persistence"):
            persistence_rows.append(
                {"fold": fold.name, "model": name, **regression_metrics(targets, baseline[name])}
            )
        for seed in seeds:
            if _already_done(
                rolling_results,
                model="cnn",
                fold=fold.name,
                seed=seed,
                source_fingerprint=store.source_fingerprint,
            ):
                continue
            sequences = build_sequences(
                table,
                artifacts.feature_columns,
                selected_lookback,
                options.batch_size,
                seed,
                splits=("train", "validation"),
            )
            set_global_determinism(seed)
            model = build_cnn((selected_lookback, len(artifacts.feature_columns)))
            result = _fit_validation(
                model,
                sequences,
                targets,
                max_epochs=(1 if options.mode == "smoke" else CONFIG.cnn_max_epochs),
                patience=CONFIG.cnn_early_stopping_patience,
            )
            rolling_results = _append_progress(
                rolling_path,
                {
                    "model": "cnn",
                    "fold": fold.name,
                    "seed": seed,
                    "feature_policy": selected_feature,
                    "source_fingerprint": store.source_fingerprint,
                    **result,
                },
                ["model", "fold", "seed"],
            )
            keras.backend.clear_session()

    expected_models = ("lstm", "cnn")
    validated_rolling = _validate_named_completeness(
        rolling_results,
        item_column="model",
        item_ids=expected_models,
        fold_names=tuple(fold.name for fold in folds),
        seeds=seeds,
        source_fingerprint=store.source_fingerprint,
    )
    rolling_results = validated_rolling
    _write_frame(rolling_results, rolling_path)
    baseline_metrics_path = paths.assignment2_tables / f"validation_baseline_metrics{suffix}.csv"
    baseline_metrics = (
        pd.DataFrame(persistence_rows)
        .groupby("model", observed=True)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            smape_percent=("smape_percent", "mean"),
        )
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
            "source_fingerprint": store.source_fingerprint,
        },
    )
    selected_payload = {
        "forward_fill_limit_hours": selected_fill,
        "lookback_hours": selected_lookback,
        "feature_policy": selected_feature,
        "selection_rule": "paired fold-level one-standard-error feature selection",
        "feature_selection_audit": feature_audit,
        "run_mode": options.mode,
        "config_hash": options.stable_hash(),
        "source_fingerprint": store.source_fingerprint,
    }
    selected_preprocessing_path = _selected_preprocessing_path(paths, suffix)
    rolling_metadata_path = _rolling_metadata_path(paths, suffix)
    write_json_atomic(selected_preprocessing_path, selected_payload)
    write_json_atomic(
        rolling_metadata_path,
        {
            "folds": fold_metadata,
            "frozen_test_start": CONFIG.test_start,
            "source_fingerprint": store.source_fingerprint,
        },
    )

    final_table, final_artifacts, final_metadata = create_model_table(
        raw_data,
        paths,
        train_start=CONFIG.train_start,
        validation_start=CONFIG.test_start,
        test_start=CONFIG.test_start,
        data_end_exclusive=CONFIG.test_start,
        lookback_hours=selected_lookback,
        forward_fill_limit_hours=selected_fill,
        feature_policy=selected_feature,
        persist=False,
        source_fingerprint=store.source_fingerprint,
    )
    final_model_input_outputs = _persist_model_inputs(
        final_table,
        final_artifacts,
        final_metadata,
        paths,
        suffix,
    )
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
        epochs = int(
            max(
                1,
                np.ceil(
                    comparison.loc[comparison["model"].eq(name), "median_best_epoch"].iloc[0]
                ),
            )
        )
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
            f"Basic {name.upper()} final pre-benchmark training",
        )
        keras.backend.clear_session()
    outputs = [
        preprocessing_lock_path,
        feature_path,
        feature_summary_path,
        feature_audit_path,
        rolling_path,
        baseline_metrics_path,
        comparison_path,
        selection_path,
        selected_preprocessing_path,
        rolling_metadata_path,
        *final_model_input_outputs,
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
        "selected_feature_policy": selected_feature,
        "feature_validation_fits": len(feature_results),
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
        loss_name=candidate.loss_name,
        pinball_tau=candidate.pinball_tau,
        optimizer_name=candidate.optimizer_name,
        weight_decay=candidate.weight_decay,
        clipnorm=candidate.clipnorm,
        use_ema=candidate.use_ema,
        ema_momentum=candidate.ema_momentum,
        formulation=candidate.formulation,
        pm25_feature_index=feature_index,
        pm25_mean=pm25_mean,
        pm25_scale=pm25_scale,
    )


def tune_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    store: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Compare optimization recipes, run the focused search, and enforce the gate."""

    import joblib
    from tensorflow import keras

    from .models import set_global_determinism

    suffix = options.artifact_suffix
    selected_preprocessing_path = _selected_preprocessing_path(paths, suffix)
    selected_preprocessing = _read_json(selected_preprocessing_path)
    if selected_preprocessing.get("source_fingerprint") != store.source_fingerprint:
        raise RuntimeError("Selected preprocessing belongs to different methodology source.")
    selected_lookback = int(selected_preprocessing["lookback_hours"])
    selected_fill = selected_preprocessing["forward_fill_limit_hours"]
    selected_feature: FeaturePolicyName = selected_preprocessing["feature_policy"]
    raw_data = _modeling_data(options, paths)
    seeds = (42, 52) if options.mode == "full" else (42,)
    folds = _folds(options)
    max_epochs = 1 if options.mode == "smoke" else CONFIG.focused_max_epochs

    recipes = generate_optimization_recipes()
    recipe_path = paths.assignment2_tables / f"optimization_recipe_results{suffix}.csv"
    recipe_results = _initial_progress(recipe_path, resume=options.resume)
    for fold in folds:
        table, artifacts, _, _ = _fold_inputs(
            raw_data,
            fold,
            selected_fill,
            selected_lookback,
            options,
            paths,
            store.source_fingerprint,
            feature_policy=selected_feature,
        )
        for recipe in recipes:
            for seed in seeds:
                if _already_done(
                    recipe_results,
                    candidate_id=recipe.candidate_id,
                    fold=fold.name,
                    seed=seed,
                    source_fingerprint=store.source_fingerprint,
                ):
                    continue
                sequences = build_sequences(
                    table,
                    artifacts.feature_columns,
                    selected_lookback,
                    recipe.batch_size,
                    seed,
                    splits=("train", "validation"),
                )
                anchors = sequences["validation"].ordered_anchor_indices()
                set_global_determinism(seed)
                model = _build_candidate_model(
                    recipe,
                    (selected_lookback, len(artifacts.feature_columns)),
                    artifacts,
                )
                fit = _fit_validation(
                    model,
                    sequences,
                    table.loc[anchors, "target_pm25"].to_numpy(),
                    max_epochs=max_epochs,
                    patience=CONFIG.focused_early_stopping_patience,
                )
                recipe_results = _append_progress(
                    recipe_path,
                    {
                        **recipe.to_dict(),
                        "fold": fold.name,
                        "seed": seed,
                        "feature_policy": selected_feature,
                        "source_fingerprint": store.source_fingerprint,
                        **fit,
                    },
                    ["candidate_id", "fold", "seed"],
                )
                keras.backend.clear_session()
    recipe_results = validate_search_completeness(
        recipe_results,
        [recipe.candidate_id for recipe in recipes],
        [fold.name for fold in folds],
        seeds,
        source_fingerprint=store.source_fingerprint,
    )
    _write_frame(recipe_results, recipe_path)
    recipe_summary = aggregate_validation_results(recipe_results)
    recipe_summary["symmetric_loss_rank"] = recipe_summary["candidate_id"].map(
        {recipe.candidate_id: int(recipe.loss_name == "pinball") for recipe in recipes}
    )
    recipe_summary["optimizer_rank"] = recipe_summary["candidate_id"].map(
        {recipe.candidate_id: int(recipe.optimizer_name == "adamw") for recipe in recipes}
    )
    recipe_summary["mechanism_rank"] = recipe_summary["candidate_id"].map(
        {
            recipe.candidate_id: int(recipe.clipnorm is not None)
            + int(recipe.use_ema)
            + int(recipe.weight_decay > 0)
            for recipe in recipes
        }
    )
    selected_recipe_row, recipe_comparisons, recipe_audit = select_paired_one_standard_error(
        recipe_results,
        recipe_summary,
        item_column="candidate_id",
        simplicity_columns=(),
        preference_columns=("symmetric_loss_rank", "optimizer_rank", "mechanism_rank"),
    )
    selected_recipe = next(
        recipe
        for recipe in recipes
        if recipe.candidate_id == selected_recipe_row["candidate_id"]
    )
    recipe_summary_path = paths.assignment2_tables / f"optimization_recipe_summary{suffix}.csv"
    recipe_audit_path = paths.assignment2_tables / f"optimization_recipe_paired_audit{suffix}.csv"
    _write_frame(recipe_summary, recipe_summary_path)
    _write_frame(recipe_comparisons, recipe_audit_path)

    all_candidates = generate_lstm_candidates(
        CONFIG.seed,
        CONFIG.random_search_candidates,
        recipe=selected_recipe,
    )
    candidates = all_candidates if options.mode == "full" else all_candidates[:2]
    search_path = paths.assignment2_tables / f"focused_lstm_search_results{suffix}.csv"
    results = _initial_progress(search_path, resume=options.resume)
    for fold in folds:
        table, artifacts, _, _ = _fold_inputs(
            raw_data,
            fold,
            selected_fill,
            selected_lookback,
            options,
            paths,
            store.source_fingerprint,
            feature_policy=selected_feature,
        )
        for candidate in candidates:
            for seed in seeds:
                if _already_done(
                    results,
                    candidate_id=candidate.candidate_id,
                    fold=fold.name,
                    seed=seed,
                    source_fingerprint=store.source_fingerprint,
                ):
                    continue
                sequences = build_sequences(
                    table,
                    artifacts.feature_columns,
                    selected_lookback,
                    candidate.batch_size,
                    seed,
                    splits=("train", "validation"),
                )
                anchors = sequences["validation"].ordered_anchor_indices()
                set_global_determinism(seed)
                model = _build_candidate_model(
                    candidate,
                    (selected_lookback, len(artifacts.feature_columns)),
                    artifacts,
                )
                fit = _fit_validation(
                    model,
                    sequences,
                    table.loc[anchors, "target_pm25"].to_numpy(),
                    max_epochs=max_epochs,
                    patience=CONFIG.focused_early_stopping_patience,
                )
                results = _append_progress(
                    search_path,
                    {
                        **candidate.to_dict(),
                        "fold": fold.name,
                        "seed": seed,
                        "feature_policy": selected_feature,
                        "selected_recipe_id": selected_recipe.candidate_id,
                        "source_fingerprint": store.source_fingerprint,
                        **fit,
                    },
                    ["candidate_id", "fold", "seed"],
                )
                keras.backend.clear_session()
    results = validate_search_completeness(
        results,
        [candidate.candidate_id for candidate in candidates],
        [fold.name for fold in folds],
        seeds,
        source_fingerprint=store.source_fingerprint,
    )
    _write_frame(results, search_path)
    summary = aggregate_validation_results(results)
    summary_path = paths.assignment2_tables / f"focused_lstm_search_summary{suffix}.csv"
    selected_row, candidate_comparisons, candidate_audit = select_lstm_candidate(results, summary)
    candidate_audit_path = paths.assignment2_tables / f"focused_lstm_paired_audit{suffix}.csv"
    _write_frame(summary, summary_path)
    _write_frame(candidate_comparisons, candidate_audit_path)
    selected_candidate = next(
        candidate for candidate in candidates if candidate.candidate_id == selected_row["candidate_id"]
    )

    basic_path = paths.assignment2_tables / f"basic_rolling_results{suffix}.csv"
    basic_results = pd.read_csv(basic_path)
    basic_results = basic_results.loc[
        basic_results["model"].eq("lstm")
        & basic_results["source_fingerprint"].eq(store.source_fingerprint)
    ].copy()
    tuned_fold = results.loc[results["candidate_id"].eq(selected_candidate.candidate_id)].groupby(
        "fold", observed=True
    )["mae"].mean()
    basic_fold = basic_results.groupby("fold", observed=True)["mae"].mean()
    paired_gate = pd.concat(
        [tuned_fold.rename("tuned_mae"), basic_fold.rename("basic_mae")], axis=1
    ).dropna()
    wins = int((paired_gate["tuned_mae"] < paired_gate["basic_mae"]).sum())
    required_wins = 2 if options.mode == "full" else 1
    gate_checks = {
        "non_basic_configuration": selected_candidate.source != "basic_anchor",
        "lower_mean_mae_than_basic_lstm": float(tuned_fold.mean()) < float(basic_fold.mean()),
        "wins_required_fold_averages": wins >= required_wins,
        "complete_search": len(results) == len(candidates) * len(folds) * len(seeds),
    }
    validation_gate_passed = bool(all(gate_checks.values()))
    gate_bypassed_for_smoke = options.mode == "smoke"
    gate_path = paths.models / f"focused_lstm_validation_gate{suffix}.json"
    write_json_atomic(
        gate_path,
        {
            "passed": validation_gate_passed,
            "bypassed_for_smoke": gate_bypassed_for_smoke,
            "checks": gate_checks,
            "selected_candidate": selected_candidate.candidate_id,
            "selected_source": selected_candidate.source,
            "selected_mean_mae": float(tuned_fold.mean()),
            "basic_lstm_mean_mae": float(basic_fold.mean()),
            "fold_wins": wins,
            "required_fold_wins": required_wins,
            "paired_fold_metrics": paired_gate.reset_index().to_dict(orient="records"),
            "source_fingerprint": store.source_fingerprint,
        },
    )
    if options.mode == "full" and not validation_gate_passed:
        failed = [name for name, passed in gate_checks.items() if not passed]
        raise RuntimeError(
            "Focused tuned-LSTM validation gate failed before benchmark access: "
            + ", ".join(failed)
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
            "selected_feature_policy": selected_feature,
            "selected_optimization_recipe": selected_recipe.to_dict(),
            "epoch_budget": epoch_budget,
            "selection_rule": "paired one-standard-error; parameters, layers, direct formulation",
            "run_mode": options.mode,
            "config_hash": options.stable_hash(),
            "source_fingerprint": store.source_fingerprint,
            "recipe_selection_audit": recipe_audit,
            "candidate_selection_audit": candidate_audit,
        },
    )

    model_table, feature_columns, _ = _load_model_inputs(
        options, paths, splits=("train",)
    )
    _, _, _, preprocessor_path = _model_input_paths(paths, suffix)
    artifacts = joblib.load(preprocessor_path)
    set_global_determinism(CONFIG.seed)
    model = _build_candidate_model(
        selected_candidate,
        (selected_lookback, len(feature_columns)),
        artifacts,
    )
    final_sequences = build_sequences(
        model_table,
        feature_columns,
        selected_lookback,
        selected_candidate.batch_size,
        CONFIG.seed,
        splits=("train",),
    )
    final_checkpoint = paths.models / f"tuned_final{suffix}.keras"
    final_history = paths.histories / f"tuned_final{suffix}_epochs.csv"
    model, training_seconds, _ = _train_final_model(
        model,
        final_sequences["train"],
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
            "selected_feature_policy": selected_feature,
            "selected_optimization_recipe": selected_recipe.to_dict(),
            "final_checkpoint": str(final_checkpoint),
            "frozen_test_used_for_selection": False,
            "selection_locked_before_benchmark": True,
            "validation_gate_passed": validation_gate_passed,
            "validation_gate_bypassed_for_smoke": gate_bypassed_for_smoke,
            "run_mode": options.mode,
            "config_hash": options.stable_hash(),
            "source_fingerprint": store.source_fingerprint,
            "recipe_selection_audit": recipe_audit,
            "candidate_selection_audit": candidate_audit,
        },
    )
    outputs = [
        recipe_path,
        recipe_summary_path,
        recipe_audit_path,
        search_path,
        summary_path,
        candidate_audit_path,
        gate_path,
        best_path,
        final_checkpoint,
        final_history,
        final_selection_path,
    ]
    return {
        "architecture": "lstm",
        "selected_feature_policy": selected_feature,
        "selected_recipe": selected_recipe.candidate_id,
        "recipe_validation_fits": len(recipe_results),
        "candidate_count": len(candidates),
        "validation_fits": len(results),
        "selected_candidate": selected_candidate.candidate_id,
        "formulation": selected_candidate.formulation,
        "validation_mae": float(selected_row["mean_mae"]),
        "validation_gate_passed": validation_gate_passed,
        "validation_gate_bypassed_for_smoke": gate_bypassed_for_smoke,
        "epoch_budget": epoch_budget,
        "training_seconds": training_seconds,
        "run_mode": options.mode,
    }, outputs


def evaluate_stage(
    options: RuntimeOptions,
    paths: ProjectPaths,
    store: RunStateStore,
) -> tuple[dict[str, Any], list[Path]]:
    """Evaluate frozen models on the previously inspected chronological benchmark."""

    import joblib
    from tensorflow import keras

    # Import registers custom objects before loading any saved model.
    from .models import PersistenceResidual, SerializableHuber, SerializablePinball

    del PersistenceResidual, SerializableHuber, SerializablePinball

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
    final_selection_path = paths.models / f"final_model_selection{suffix}.json"
    final_selection = _read_json(final_selection_path)
    if final_selection.get("run_mode") != options.mode:
        raise RuntimeError("The tuned model was produced in a different run mode.")
    if final_selection.get("source_fingerprint") != store.source_fingerprint:
        raise RuntimeError("The tuned model belongs to different methodology source.")
    gate_passed = bool(final_selection.get("validation_gate_passed"))
    smoke_bypass = bool(final_selection.get("validation_gate_bypassed_for_smoke"))
    if not gate_passed and not (options.mode == "smoke" and smoke_bypass):
        raise PermissionError("Frozen benchmark access requires a passed tuned-LSTM validation gate.")
    development = load_development_data(paths)
    benchmark = load_frozen_benchmark(
        paths,
        final_selection_path,
        expected_source_fingerprint=store.source_fingerprint,
    )
    if options.mode == "smoke":
        stations = EXPECTED_STATIONS[:2]
        development = development.loc[development["station"].isin(stations)].copy()
        benchmark = benchmark.loc[benchmark["station"].isin(stations)].copy()
    raw_evaluation = pd.concat([development, benchmark], ignore_index=True)
    _, _, _, preprocessor_path = _model_input_paths(paths, suffix)
    artifacts = joblib.load(preprocessor_path)
    if artifacts.source_fingerprint != store.source_fingerprint:
        raise RuntimeError("The final preprocessor belongs to different methodology source.")
    selected_preprocessing = _read_json(_selected_preprocessing_path(paths, suffix))
    selected_lookback = int(selected_preprocessing["lookback_hours"])
    policy = PreprocessingPolicy(
        train_start=CONFIG.train_start,
        fit_end_exclusive=CONFIG.test_start,
        validation_start=CONFIG.test_start,
        test_start=CONFIG.test_start,
        data_end_exclusive=CONFIG.data_end_exclusive,
        lookback_hours=selected_lookback,
        forward_fill_limit_hours=selected_preprocessing["forward_fill_limit_hours"],
        feature_policy=selected_preprocessing["feature_policy"],
    )
    model_table, benchmark_metadata = transform_model_table(raw_evaluation, artifacts, policy)
    benchmark_model_path, benchmark_metadata_path = _benchmark_input_paths(paths, suffix)
    model_table.to_parquet(benchmark_model_path, index=False)
    write_json_atomic(benchmark_metadata_path, benchmark_metadata)
    sequences = build_sequences(
        model_table,
        artifacts.feature_columns,
        selected_lookback,
        int(final_selection["selected_candidate"]["batch_size"]),
        CONFIG.seed,
        splits=("test",),
    )
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
    if base_frame.duplicated(["station", "target_timestamp"]).any():
        raise RuntimeError("Frozen benchmark predictions contain duplicate station-target rows.")
    model_prediction_columns = ["basic_lstm", "basic_cnn", "tuned_final"]
    prediction_matrix = base_frame[model_prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(prediction_matrix).all() or (prediction_matrix < 0).any():
        raise RuntimeError("Model predictions must be finite and non-negative.")
    if options.mode == "full" and len(base_frame) != 59_917:
        raise RuntimeError(f"Expected 59,917 common benchmark rows, found {len(base_frame):,}.")
    predictions_path = paths.assignment2_tables / f"final_test_predictions{suffix}.parquet"
    base_frame.to_parquet(predictions_path, index=False)

    prediction_columns = [
        "persistence",
        "daily_persistence",
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
                "analysis": "official" if block_days == 7 else "sensitivity",
                **paired_moving_block_bootstrap(
                    base_frame,
                    "tuned_final",
                    comparison_model,
                    block_days=block_days,
                    n_boot=(100 if options.mode == "smoke" else 10_000),
                    seed=CONFIG.seed,
                ),
            }
            for comparison_model in ("basic_lstm", "basic_cnn", "persistence")
            for block_days in (1, 7)
        ]
    )
    bootstrap_path = paths.assignment2_tables / f"paired_moving_block_bootstrap{suffix}.csv"
    _write_frame(
        bootstrap_results,
        bootstrap_path,
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
        & bootstrap_results["block_length_days"].eq(7)
    ].iloc[0]
    significant = bool(float(basic_bootstrap["ci95_upper"]) < 0)
    observed_winner = str(overall_metrics.iloc[0]["model"])
    tuned_is_winner = observed_winner == "tuned_final"
    claim_path = paths.assignment2_tables / f"optimization_claim{suffix}.json"
    write_json_atomic(
        claim_path,
        {
            "statistically_superior_to_basic_lstm": significant,
            "tuned_is_observed_mae_winner": tuned_is_winner,
            "observed_mae_winner": observed_winner,
            "rule": "claim superiority only when the paired seven-day moving-block CI is below zero",
            "official_block_length_days": 7,
            "sensitivity_block_length_days": 1,
            "bootstrap_samples": int(basic_bootstrap["bootstrap_samples"]),
            "test_description": "frozen chronological benchmark previously inspected",
            "test_used_for_model_selection": False,
        },
    )
    summary = {
        "winner": observed_winner,
        "winner_mae": float(overall_metrics.iloc[0]["mae"]),
        "tuned_mae": float(tuned["mae"]),
        "persistence_mae": float(persistence["mae"]),
        "improvement_over_persistence_percent": float(improvement),
        "test_anchors": len(base_frame),
        "statistically_superior_to_basic_lstm": significant,
        "tuned_is_observed_mae_winner": tuned_is_winner,
        "run_mode": options.mode,
    }
    outputs = [
        predictions_path,
        overall_path,
        inference_path,
        paths.assignment2_tables / f"final_metrics_by_station{suffix}.csv",
        paths.assignment2_tables / f"final_metrics_by_season{suffix}.csv",
        paths.assignment2_tables / f"final_metrics_by_pollution_band{suffix}.csv",
        bootstrap_path,
        residual_path,
        claim_path,
        benchmark_model_path,
        benchmark_metadata_path,
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
    store.begin(stage)
    try:
        summary, outputs = function(options, paths, store)
        missing_outputs = [str(path) for path in outputs if not path.exists()]
        if missing_outputs:
            raise RuntimeError(f"Stage {stage!r} did not produce declared outputs: {missing_outputs}")
        store.complete(stage, summary=summary, outputs=outputs)
        return summary
    except BaseException as error:
        store.fail(stage, error)
        raise


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

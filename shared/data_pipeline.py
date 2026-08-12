"""Download, validate, combine, and preprocess the Beijing air-quality data."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    CONFIG,
    CONTINUOUS_INPUT_COLUMNS,
    EXPECTED_STATIONS,
    ProjectPaths,
)


@dataclass(frozen=True)
class SourceManifest:
    """Provenance recorded for the downloaded UCI archive."""

    url: str
    doi: str
    license: str
    retrieved_at_utc: str
    archive_name: str
    bytes: int
    sha256: str


@dataclass
class PreprocessingArtifacts:
    """Fitted preprocessing objects and the model feature contract."""

    scaler: StandardScaler
    continuous_columns: list[str]
    scaled_columns: list[str]
    feature_columns: list[str]
    station_medians: dict[str, dict[str, float]]
    global_medians: dict[str, float]
    station_columns: list[str]
    forward_fill_limit_hours: int | None
    fit_start: str
    fit_end_exclusive: str
    methodology_version: str = CONFIG.methodology_version
    source_fingerprint: str | None = None


@dataclass(frozen=True)
class PreprocessingPolicy:
    """All temporal choices needed to fit and apply preprocessing safely."""

    train_start: str = CONFIG.train_start
    fit_end_exclusive: str = CONFIG.validation_start
    validation_start: str = CONFIG.validation_start
    test_start: str = CONFIG.test_start
    data_end_exclusive: str = CONFIG.data_end_exclusive
    lookback_hours: int = CONFIG.lookback_hours
    forward_fill_limit_hours: int | None = CONFIG.forward_fill_limit_hours


COMPASS_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def download_dataset(paths: ProjectPaths, force: bool = False) -> SourceManifest:
    """Download the official UCI archive and record a reproducibility manifest.

    Existing archives are reused unless ``force`` is true.  Downloads first go
    to a ``.part`` file so an interrupted transfer cannot masquerade as a valid
    source archive.
    """

    paths.ensure_directories()
    archive = paths.source_archive
    if force or not archive.exists():
        part = archive.with_suffix(archive.suffix + ".part")
        if part.exists():
            part.unlink()
        request = Request(CONFIG.source_url, headers={"User-Agent": "ODL-assignment/1.0"})
        with urlopen(request, timeout=120) as response, part.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if part.stat().st_size < 1_000_000:
            part.unlink(missing_ok=True)
            raise RuntimeError("Downloaded archive is unexpectedly small.")
        part.replace(archive)

    try:
        with ZipFile(archive) as zipped:
            corrupt_member = zipped.testzip()
    except BadZipFile as exc:
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {archive}") from exc
    if corrupt_member:
        raise RuntimeError(f"Corrupt member in source ZIP: {corrupt_member}")

    manifest = SourceManifest(
        url=CONFIG.source_url,
        doi=CONFIG.dataset_doi,
        license=CONFIG.dataset_license,
        retrieved_at_utc=datetime.now(UTC).isoformat(),
        archive_name=archive.name,
        bytes=archive.stat().st_size,
        sha256=_sha256(archive),
    )
    _write_json(paths.source_manifest, asdict(manifest))
    return manifest


def _csv_members(zipped: ZipFile) -> list[str]:
    return [
        name
        for name in zipped.namelist()
        if name.lower().endswith(".csv") and "PRSA_Data_" in Path(name).name
    ]


def extract_dataset(paths: ProjectPaths, force: bool = False) -> list[Path]:
    """Extract the 12 station CSVs from either a flat or nested UCI ZIP."""

    paths.ensure_directories()
    existing = sorted(paths.raw_extract.glob("PRSA_Data_*.csv"))
    if len(existing) == len(EXPECTED_STATIONS) and not force:
        return existing

    if not paths.source_archive.exists():
        raise FileNotFoundError("Source archive is missing; run download_dataset first.")

    paths.raw_extract.mkdir(parents=True, exist_ok=True)
    if force:
        for old_csv in paths.raw_extract.glob("PRSA_Data_*.csv"):
            old_csv.unlink()

    with ZipFile(paths.source_archive) as outer:
        members = _csv_members(outer)
        source_zip = outer
        nested_zip: ZipFile | None = None
        if not members:
            nested_names = [name for name in outer.namelist() if name.lower().endswith(".zip")]
            if not nested_names:
                raise RuntimeError("UCI archive contains neither station CSVs nor a nested ZIP.")
            nested_zip = ZipFile(BytesIO(outer.read(nested_names[0])))
            source_zip = nested_zip
            members = _csv_members(source_zip)

        try:
            for member in members:
                destination = paths.raw_extract / Path(member).name
                with source_zip.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        finally:
            if nested_zip is not None:
                nested_zip.close()

    extracted = sorted(paths.raw_extract.glob("PRSA_Data_*.csv"))
    if len(extracted) != len(EXPECTED_STATIONS):
        raise RuntimeError(f"Expected 12 station CSVs, found {len(extracted)}.")
    return extracted


def load_and_validate_raw(
    paths: ProjectPaths,
    csv_paths: Iterable[Path] | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Load all station files and validate row counts and hourly continuity."""

    files = list(csv_paths or sorted(paths.raw_extract.glob("PRSA_Data_*.csv")))
    if not files:
        raise FileNotFoundError("No extracted station CSVs were found.")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, na_values=["NA", ""])
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(
        data[["year", "month", "day", "hour"]], errors="raise"
    )
    data = data.sort_values(["station", "timestamp"], kind="stable").reset_index(drop=True)

    duplicate_count = int(data.duplicated(["station", "timestamp"]).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate station-timestamp rows.")

    stations = tuple(sorted(data["station"].dropna().unique()))
    counts = data.groupby("station", observed=True).size()
    if strict:
        if stations != tuple(sorted(EXPECTED_STATIONS)):
            raise ValueError(f"Unexpected station set: {stations}")
        if len(data) != CONFIG.expected_total_rows:
            raise ValueError(f"Expected {CONFIG.expected_total_rows} rows, found {len(data)}.")
        if not counts.eq(CONFIG.expected_rows_per_station).all():
            raise ValueError(f"Unexpected rows per station:\n{counts}")
        if data["timestamp"].min() != pd.Timestamp(CONFIG.expected_start):
            raise ValueError("Unexpected dataset start timestamp.")
        if data["timestamp"].max() != pd.Timestamp(CONFIG.expected_end):
            raise ValueError("Unexpected dataset end timestamp.")

        expected_range = pd.date_range(CONFIG.expected_start, CONFIG.expected_end, freq="h")
        for station, group in data.groupby("station", sort=True, observed=True):
            observed = pd.DatetimeIndex(group["timestamp"])
            if not observed.equals(expected_range):
                raise ValueError(f"Hourly timestamp grid is incomplete for {station}.")

    return data


def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values for every column."""

    counts = data.isna().sum()
    return (
        pd.DataFrame(
            {
                "column": counts.index,
                "missing_count": counts.values,
                "missing_percent": (counts.values / len(data)) * 100.0,
            }
        )
        .sort_values("missing_count", ascending=False)
        .reset_index(drop=True)
    )


def schema_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Produce a compact schema table suitable for notebook display/export."""

    return pd.DataFrame(
        {
            "column": data.columns,
            "dtype": [str(data[column].dtype) for column in data.columns],
            "non_null": [int(data[column].notna().sum()) for column in data.columns],
            "unique": [int(data[column].nunique(dropna=True)) for column in data.columns],
        }
    )


def assign_split(
    target_timestamp: pd.Series,
    *,
    train_start: str = CONFIG.train_start,
    validation_start: str = CONFIG.validation_start,
    test_start: str = CONFIG.test_start,
    data_end_exclusive: str = CONFIG.data_end_exclusive,
) -> pd.Series:
    """Assign split labels according to the target time, not the input anchor."""

    result = pd.Series(pd.NA, index=target_timestamp.index, dtype="string")
    result.loc[
        (target_timestamp >= pd.Timestamp(train_start))
        & (target_timestamp < pd.Timestamp(validation_start))
    ] = "train"
    result.loc[
        (target_timestamp >= pd.Timestamp(validation_start))
        & (target_timestamp < pd.Timestamp(test_start))
    ] = "validation"
    result.loc[
        (target_timestamp >= pd.Timestamp(test_start))
        & (target_timestamp < pd.Timestamp(data_end_exclusive))
    ] = "test"
    return result


def _ordered_copy(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data.sort_values(["station", "timestamp"], kind="stable").reset_index(drop=True).copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    unknown = set(data["station"].dropna().unique()) - set(EXPECTED_STATIONS)
    if unknown:
        raise ValueError(f"Unknown monitoring stations: {sorted(unknown)}")
    return data


def _feature_contract() -> tuple[list[str], list[str], list[str]]:
    scaled = [f"z__{column}" for column in CONTINUOUS_INPUT_COLUMNS]
    missing = [f"missing__{column}" for column in CONTINUOUS_INPUT_COLUMNS]
    temporal = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    wind = ["wind_sin", "wind_cos", "wind_missing"]
    stations = [f"station__{station}" for station in EXPECTED_STATIONS]
    return scaled, stations, scaled + missing + temporal + wind + stations


def _causal_fill(
    data: pd.DataFrame,
    *,
    fill_limit: int | None,
    station_medians: dict[str, dict[str, float]],
    global_medians: dict[str, float],
) -> pd.DataFrame:
    result = data.copy()
    continuous = list(CONTINUOUS_INPUT_COLUMNS)
    result[continuous] = result.groupby("station", sort=False, observed=True)[continuous].ffill(
        limit=fill_limit
    )
    for station, indices in result.groupby("station", sort=False, observed=True).groups.items():
        medians = station_medians.get(str(station), global_medians)
        result.loc[indices, continuous] = result.loc[indices, continuous].fillna(medians)
    result[continuous] = result[continuous].fillna(global_medians)
    return result


def fit_preprocessor(
    training_data: pd.DataFrame,
    policy: PreprocessingPolicy,
    *,
    source_fingerprint: str | None = None,
) -> PreprocessingArtifacts:
    """Fit every learned preprocessing value using only the declared fit period."""

    data = _ordered_copy(training_data)
    fit_mask = (
        (data["timestamp"] >= pd.Timestamp(policy.train_start))
        & (data["timestamp"] < pd.Timestamp(policy.fit_end_exclusive))
    )
    if not fit_mask.any():
        raise ValueError("The preprocessing fit period contains no rows.")
    fit_data = data.loc[fit_mask].copy()
    continuous = list(CONTINUOUS_INPUT_COLUMNS)
    fit_data[continuous] = fit_data.groupby("station", sort=False, observed=True)[continuous].ffill(
        limit=policy.forward_fill_limit_hours
    )
    station_frame = fit_data.groupby("station", observed=True)[continuous].median()
    global_series = fit_data[continuous].median()
    if global_series.isna().any():
        missing = global_series.index[global_series.isna()].tolist()
        raise ValueError(f"Training data cannot provide fallback medians for: {missing}")
    global_medians = {column: float(global_series[column]) for column in continuous}
    station_medians: dict[str, dict[str, float]] = {}
    for station in fit_data["station"].dropna().unique():
        medians = station_frame.loc[station].fillna(global_series)
        station_medians[str(station)] = {column: float(medians[column]) for column in continuous}
    imputed_fit = _causal_fill(
        fit_data,
        fill_limit=policy.forward_fill_limit_hours,
        station_medians=station_medians,
        global_medians=global_medians,
    )
    scaler = StandardScaler().fit(imputed_fit[continuous])
    scaled_columns, station_columns, feature_columns = _feature_contract()
    return PreprocessingArtifacts(
        scaler=scaler,
        continuous_columns=continuous,
        scaled_columns=scaled_columns,
        feature_columns=feature_columns,
        station_medians=station_medians,
        global_medians=global_medians,
        station_columns=station_columns,
        forward_fill_limit_hours=policy.forward_fill_limit_hours,
        fit_start=policy.train_start,
        fit_end_exclusive=policy.fit_end_exclusive,
        source_fingerprint=source_fingerprint,
    )


def transform_model_table(
    raw_data: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
    policy: PreprocessingPolicy,
) -> tuple[pd.DataFrame, dict]:
    """Apply a fitted preprocessor and construct station-safe forecast anchors."""

    if artifacts.methodology_version != CONFIG.methodology_version:
        raise RuntimeError("The fitted preprocessor belongs to a different methodology version.")
    data = _ordered_copy(raw_data)
    data["PM2.5_target_original"] = data["PM2.5"]
    for column in CONTINUOUS_INPUT_COLUMNS:
        data[f"missing__{column}"] = data[column].isna().astype("float32")
    data = _causal_fill(
        data,
        fill_limit=artifacts.forward_fill_limit_hours,
        station_medians=artifacts.station_medians,
        global_medians=artifacts.global_medians,
    )
    data["PM2.5_input_raw"] = data["PM2.5"]

    wind_missing = data["wd"].isna() | ~data["wd"].isin(COMPASS_DEGREES)
    wind_radians = np.deg2rad(data["wd"].map(COMPASS_DEGREES).fillna(0.0).astype(float))
    data["wind_sin"] = np.sin(wind_radians).astype("float32")
    data["wind_cos"] = np.cos(wind_radians).astype("float32")
    data["wind_missing"] = wind_missing.astype("float32")
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0).astype("float32")
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0).astype("float32")
    day_of_year = data["timestamp"].dt.dayofyear
    days_in_year = np.where(data["timestamp"].dt.is_leap_year, 366.0, 365.0)
    data["doy_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year).astype("float32")
    data["doy_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year).astype("float32")
    for station, column in zip(EXPECTED_STATIONS, artifacts.station_columns, strict=True):
        data[column] = data["station"].eq(station).astype("float32")
    data[artifacts.scaled_columns] = artifacts.scaler.transform(
        data[artifacts.continuous_columns]
    ).astype("float32")

    data["target_pm25"] = data.groupby("station", sort=False, observed=True)[
        "PM2.5_target_original"
    ].shift(-CONFIG.forecast_horizon)
    data["target_timestamp"] = data.groupby("station", sort=False, observed=True)["timestamp"].shift(
        -CONFIG.forecast_horizon
    )
    data["split"] = assign_split(
        data["target_timestamp"],
        train_start=policy.train_start,
        validation_start=policy.validation_start,
        test_start=policy.test_start,
        data_end_exclusive=policy.data_end_exclusive,
    )
    data["station_code"] = pd.Categorical(
        data["station"], categories=EXPECTED_STATIONS, ordered=True
    ).codes.astype("int16")
    anchors = np.arange(len(data))
    starts = anchors - (policy.lookback_hours - 1)
    valid_start = starts >= 0
    safe_starts = np.maximum(starts, 0)
    station_codes = data["station_code"].to_numpy()
    timestamps = data["timestamp"].to_numpy(dtype="datetime64[h]")
    same_station = station_codes[safe_starts] == station_codes
    elapsed = (timestamps - timestamps[safe_starts]).astype("timedelta64[h]").astype(int)
    data["valid_anchor"] = (
        valid_start
        & same_station
        & (elapsed == policy.lookback_hours - 1)
        & data["target_pm25"].notna().to_numpy()
        & data["split"].notna().to_numpy()
    )
    if data[artifacts.feature_columns].isna().any().any():
        raise ValueError("Model features contain missing values after preprocessing.")
    if not np.isfinite(data[artifacts.feature_columns].to_numpy(dtype="float32")).all():
        raise ValueError("Model features contain non-finite values.")

    split_counts = (
        data.loc[data["valid_anchor"]]
        .groupby("split", observed=True)
        .size()
        .reindex(["train", "validation", "test"], fill_value=0)
        .astype(int)
        .to_dict()
    )
    metadata = {
        "methodology_version": CONFIG.methodology_version,
        "source_fingerprint": artifacts.source_fingerprint,
        "lookback_hours": policy.lookback_hours,
        "forward_fill_limit_hours": artifacts.forward_fill_limit_hours,
        "forecast_horizon": CONFIG.forecast_horizon,
        "feature_count": len(artifacts.feature_columns),
        "features": artifacts.feature_columns,
        "split_counts": split_counts,
        "split_boundaries": {
            "train_start": policy.train_start,
            "validation_start": policy.validation_start,
            "test_start": policy.test_start,
            "data_end_exclusive": policy.data_end_exclusive,
        },
        "preprocessing_fit": {
            "start": artifacts.fit_start,
            "end_exclusive": artifacts.fit_end_exclusive,
        },
    }
    return data, metadata


def create_model_table(
    raw_data: pd.DataFrame,
    paths: ProjectPaths,
    *,
    train_start: str = CONFIG.train_start,
    validation_start: str = CONFIG.validation_start,
    test_start: str = CONFIG.test_start,
    data_end_exclusive: str = CONFIG.data_end_exclusive,
    lookback_hours: int = CONFIG.lookback_hours,
    forward_fill_limit_hours: int | None = CONFIG.forward_fill_limit_hours,
    persist: bool = True,
    source_fingerprint: str | None = None,
) -> tuple[pd.DataFrame, PreprocessingArtifacts, dict]:
    """Compatibility interface that fits then applies the deep preprocessing module."""

    policy = PreprocessingPolicy(
        train_start=train_start,
        fit_end_exclusive=validation_start,
        validation_start=validation_start,
        test_start=test_start,
        data_end_exclusive=data_end_exclusive,
        lookback_hours=lookback_hours,
        forward_fill_limit_hours=forward_fill_limit_hours,
    )
    artifacts = fit_preprocessor(raw_data, policy, source_fingerprint=source_fingerprint)
    data, metadata = transform_model_table(raw_data, artifacts, policy)
    if persist:
        joblib.dump(artifacts, paths.preprocessor_file)
        _write_json(
            paths.feature_metadata,
            {"features": artifacts.feature_columns, "count": len(artifacts.feature_columns)},
        )
        _write_json(paths.split_metadata, metadata)
    return data, artifacts, metadata


def partition_frozen_benchmark(data: pd.DataFrame, paths: ProjectPaths) -> tuple[int, int]:
    """Persist physically separate development and frozen-benchmark source tables."""

    ordered = _ordered_copy(data)
    development = ordered.loc[ordered["timestamp"] < pd.Timestamp(CONFIG.test_start)].copy()
    benchmark = ordered.loc[ordered["timestamp"] >= pd.Timestamp(CONFIG.test_start)].copy()
    if development.empty or benchmark.empty:
        raise ValueError("Chronological benchmark partition produced an empty side.")
    development.to_parquet(paths.development_parquet, index=False)
    benchmark.to_parquet(paths.frozen_benchmark_parquet, index=False)
    return len(development), len(benchmark)


def load_development_data(paths: ProjectPaths) -> pd.DataFrame:
    """Load only rows available to preprocessing, validation, and model selection."""

    if not paths.development_parquet.exists():
        raise FileNotFoundError("Development data is missing; run the acquire stage first.")
    data = pd.read_parquet(paths.development_parquet)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if bool((data["timestamp"] >= pd.Timestamp(CONFIG.test_start)).any()):
        raise RuntimeError("Development data contains frozen-benchmark timestamps.")
    return data


def load_frozen_benchmark(
    paths: ProjectPaths,
    final_selection_path: Path,
    *,
    expected_source_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Load benchmark rows only after the final model-selection record is locked."""

    if not final_selection_path.exists():
        raise PermissionError("Frozen benchmark access requires a final model-selection record.")
    selection = json.loads(final_selection_path.read_text(encoding="utf-8"))
    if not selection.get("selection_locked_before_benchmark"):
        raise PermissionError("The final model-selection record is not benchmark-locked.")
    if expected_source_fingerprint is not None:
        recorded = selection.get("source_fingerprint")
        if recorded != expected_source_fingerprint:
            raise RuntimeError("Final selection and current methodology fingerprints differ.")
    if not paths.frozen_benchmark_parquet.exists():
        raise FileNotFoundError("Frozen benchmark data is missing; run the acquire stage first.")
    data = pd.read_parquet(paths.frozen_benchmark_parquet)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if bool((data["timestamp"] < pd.Timestamp(CONFIG.test_start)).any()):
        raise RuntimeError("Frozen benchmark contains development timestamps.")
    return data


def save_model_table(data: pd.DataFrame, paths: ProjectPaths) -> None:
    """Persist the flat table; overlapping sequences remain lazy."""

    paths.processed.mkdir(parents=True, exist_ok=True)
    data.to_parquet(paths.model_table, index=False)

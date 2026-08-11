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
) -> tuple[pd.DataFrame, PreprocessingArtifacts, dict]:
    """Create the causal, train-fitted feature table used by both neural models."""

    data = raw_data.sort_values(["station", "timestamp"], kind="stable").reset_index(drop=True).copy()
    data["PM2.5_target_original"] = data["PM2.5"]
    data["PM2.5_input_raw"] = data["PM2.5"]

    for column in CONTINUOUS_INPUT_COLUMNS:
        data[f"missing__{column}"] = data[column].isna().astype("float32")

    # Forward filling is causal: each value can depend only on earlier rows from
    # the same monitoring station.
    data[list(CONTINUOUS_INPUT_COLUMNS)] = data.groupby(
        "station", sort=False, observed=True
    )[list(CONTINUOUS_INPUT_COLUMNS)].ffill(limit=forward_fill_limit_hours)

    train_row_mask = (
        (data["timestamp"] >= pd.Timestamp(train_start))
        & (data["timestamp"] < pd.Timestamp(validation_start))
    )
    if not train_row_mask.any():
        raise ValueError("The preprocessing fit period contains no rows.")
    station_medians_frame = (
        data.loc[train_row_mask]
        .groupby("station", observed=True)[list(CONTINUOUS_INPUT_COLUMNS)]
        .median()
    )
    global_medians_series = data.loc[train_row_mask, list(CONTINUOUS_INPUT_COLUMNS)].median()

    station_medians: dict[str, dict[str, float]] = {}
    present_stations = tuple(data["station"].dropna().unique())
    for station in present_stations:
        medians = station_medians_frame.loc[station].fillna(global_medians_series)
        station_medians[station] = {column: float(medians[column]) for column in CONTINUOUS_INPUT_COLUMNS}
        station_mask = data["station"].eq(station)
        for column in CONTINUOUS_INPUT_COLUMNS:
            data.loc[station_mask, column] = data.loc[station_mask, column].fillna(
                station_medians[station][column]
            )

    global_medians = {column: float(global_medians_series[column]) for column in CONTINUOUS_INPUT_COLUMNS}
    data[list(CONTINUOUS_INPUT_COLUMNS)] = data[list(CONTINUOUS_INPUT_COLUMNS)].fillna(global_medians)
    data["PM2.5_input_raw"] = data["PM2.5"]

    wind_missing = data["wd"].isna() | ~data["wd"].isin(COMPASS_DEGREES)
    wind_degrees = data["wd"].map(COMPASS_DEGREES).fillna(0.0)
    wind_radians = np.deg2rad(wind_degrees.astype(float))
    data["wind_sin"] = np.sin(wind_radians).astype("float32")
    data["wind_cos"] = np.cos(wind_radians).astype("float32")
    data["wind_missing"] = wind_missing.astype("float32")

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0).astype("float32")
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0).astype("float32")
    day_of_year = data["timestamp"].dt.dayofyear
    days_in_year = np.where(data["timestamp"].dt.is_leap_year, 366.0, 365.0)
    data["doy_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year).astype("float32")
    data["doy_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year).astype("float32")

    station_columns: list[str] = []
    for station in EXPECTED_STATIONS:
        column = f"station__{station}"
        data[column] = data["station"].eq(station).astype("float32")
        station_columns.append(column)

    scaler = StandardScaler()
    scaled_columns = [f"z__{column}" for column in CONTINUOUS_INPUT_COLUMNS]
    scaler.fit(data.loc[train_row_mask, list(CONTINUOUS_INPUT_COLUMNS)])
    data[scaled_columns] = scaler.transform(data[list(CONTINUOUS_INPUT_COLUMNS)]).astype("float32")

    missing_columns = [f"missing__{column}" for column in CONTINUOUS_INPUT_COLUMNS]
    temporal_columns = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    wind_columns = ["wind_sin", "wind_cos", "wind_missing"]
    feature_columns = scaled_columns + missing_columns + temporal_columns + wind_columns + station_columns

    data["target_pm25"] = data.groupby("station", sort=False, observed=True)[
        "PM2.5_target_original"
    ].shift(-CONFIG.forecast_horizon)
    data["target_timestamp"] = data.groupby("station", sort=False, observed=True)["timestamp"].shift(
        -CONFIG.forecast_horizon
    )
    data["split"] = assign_split(
        data["target_timestamp"],
        train_start=train_start,
        validation_start=validation_start,
        test_start=test_start,
        data_end_exclusive=data_end_exclusive,
    )

    station_code = pd.Categorical(data["station"], categories=EXPECTED_STATIONS, ordered=True).codes
    data["station_code"] = station_code.astype("int16")
    anchors = np.arange(len(data))
    starts = anchors - (lookback_hours - 1)
    valid_start = starts >= 0
    safe_starts = np.maximum(starts, 0)
    same_station = data["station_code"].to_numpy()[safe_starts] == data["station_code"].to_numpy()
    elapsed_hours = (
        data["timestamp"].to_numpy(dtype="datetime64[h]")
        - data["timestamp"].to_numpy(dtype="datetime64[h]")[safe_starts]
    ).astype("timedelta64[h]").astype(int)
    contiguous = elapsed_hours == lookback_hours - 1
    data["valid_anchor"] = (
        valid_start
        & same_station
        & contiguous
        & data["target_pm25"].notna().to_numpy()
        & data["split"].notna().to_numpy()
    )

    if data[feature_columns].isna().any().any():
        raise ValueError("Model features contain missing values after preprocessing.")
    if not np.isfinite(data[feature_columns].to_numpy(dtype="float32")).all():
        raise ValueError("Model features contain non-finite values.")

    artifacts = PreprocessingArtifacts(
        scaler=scaler,
        continuous_columns=list(CONTINUOUS_INPUT_COLUMNS),
        scaled_columns=scaled_columns,
        feature_columns=feature_columns,
        station_medians=station_medians,
        global_medians=global_medians,
        station_columns=station_columns,
        forward_fill_limit_hours=forward_fill_limit_hours,
        fit_start=train_start,
        fit_end_exclusive=validation_start,
    )
    if persist:
        joblib.dump(artifacts, paths.preprocessor_file)

    split_counts = (
        data.loc[data["valid_anchor"]]
        .groupby("split", observed=True)
        .size()
        .reindex(["train", "validation", "test"], fill_value=0)
        .astype(int)
        .to_dict()
    )
    metadata = {
        "lookback_hours": lookback_hours,
        "forward_fill_limit_hours": forward_fill_limit_hours,
        "forecast_horizon": CONFIG.forecast_horizon,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "split_counts": split_counts,
        "split_boundaries": {
            "train_start": train_start,
            "validation_start": validation_start,
            "test_start": test_start,
            "data_end_exclusive": data_end_exclusive,
        },
        "preprocessing_fit": {
            "start": train_start,
            "end_exclusive": validation_start,
        },
    }
    if persist:
        _write_json(
            paths.feature_metadata,
            {"features": feature_columns, "count": len(feature_columns)},
        )
        _write_json(paths.split_metadata, metadata)
    return data, artifacts, metadata


def save_model_table(data: pd.DataFrame, paths: ProjectPaths) -> None:
    """Persist the flat table; overlapping sequences remain lazy."""

    paths.processed.mkdir(parents=True, exist_ok=True)
    data.to_parquet(paths.model_table, index=False)

"""Central configuration and persistent local project paths.

The command-line workflow and the five notebooks use this module as their
single source of truth.  The default root is always the checked-out project;
``ODL_PROJECT_ROOT`` exists only for explicit local overrides and tests.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

PROJECT_NAME = "odl-assignment"
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_root() -> Path:
    """Return the configured project root without creating directories."""

    override = os.getenv("ODL_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return LOCAL_PROJECT_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem contract shared by every notebook."""

    root: Path

    @property
    def shared(self) -> Path:
        return self.root / "shared"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def raw_extract(self) -> Path:
        return self.raw / "PRSA2017_Data_20130301-20170228"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def environment_artifacts(self) -> Path:
        return self.root / "artifacts" / "environment"

    @property
    def histories(self) -> Path:
        return self.root / "artifacts" / "histories"

    @property
    def models(self) -> Path:
        return self.root / "artifacts" / "models"

    @property
    def preprocessing(self) -> Path:
        return self.root / "artifacts" / "preprocessing"

    @property
    def tuning(self) -> Path:
        return self.root / "artifacts" / "tuning"

    @property
    def run_state(self) -> Path:
        return self.root / "artifacts" / "run_state"

    @property
    def runtime_logs(self) -> Path:
        return self.root / "artifacts" / "runtime_logs"

    @property
    def runtime_backups(self) -> Path:
        return self.root / "artifacts" / "training_backups"

    @property
    def executed_notebooks(self) -> Path:
        return self.root / "artifacts" / "executed_notebooks"

    @property
    def assignment1_figures(self) -> Path:
        return self.root / "outputs" / "assignment1" / "figures"

    @property
    def assignment1_tables(self) -> Path:
        return self.root / "outputs" / "assignment1" / "tables"

    @property
    def assignment1_logs(self) -> Path:
        return self.root / "outputs" / "assignment1" / "logs"

    @property
    def assignment2_figures(self) -> Path:
        return self.root / "outputs" / "assignment2" / "figures"

    @property
    def assignment2_tables(self) -> Path:
        return self.root / "outputs" / "assignment2" / "tables"

    @property
    def assignment2_logs(self) -> Path:
        return self.root / "outputs" / "assignment2" / "logs"

    @property
    def source_archive(self) -> Path:
        return self.raw / "beijing_multi_site_air_quality.zip"

    @property
    def source_manifest(self) -> Path:
        return self.raw / "source_manifest.json"

    @property
    def combined_parquet(self) -> Path:
        return self.interim / "combined_hourly.parquet"

    @property
    def development_parquet(self) -> Path:
        return self.interim / "development_hourly.parquet"

    @property
    def frozen_benchmark_parquet(self) -> Path:
        return self.interim / "frozen_benchmark_hourly.parquet"

    @property
    def model_table(self) -> Path:
        return self.processed / "model_table.parquet"

    @property
    def benchmark_model_table(self) -> Path:
        return self.processed / "frozen_benchmark_model_table.parquet"

    @property
    def feature_metadata(self) -> Path:
        return self.processed / "feature_metadata.json"

    @property
    def split_metadata(self) -> Path:
        return self.processed / "split_metadata.json"

    @property
    def benchmark_split_metadata(self) -> Path:
        return self.processed / "frozen_benchmark_split_metadata.json"

    @property
    def rolling_validation_metadata(self) -> Path:
        return self.processed / "rolling_validation_metadata.json"

    @property
    def selected_preprocessing(self) -> Path:
        return self.processed / "selected_preprocessing.json"

    @property
    def preprocessor_file(self) -> Path:
        return self.preprocessing / "preprocessing.joblib"

    def ensure_directories(self) -> None:
        """Create the complete runtime directory contract."""

        for path in (
            self.raw,
            self.raw_extract,
            self.interim,
            self.processed,
            self.environment_artifacts,
            self.histories,
            self.models,
            self.preprocessing,
            self.tuning,
            self.run_state,
            self.runtime_logs,
            self.runtime_backups,
            self.executed_notebooks,
            self.assignment1_figures,
            self.assignment1_tables,
            self.assignment1_logs,
            self.assignment2_figures,
            self.assignment2_tables,
            self.assignment2_logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


EXPECTED_STATIONS = (
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
)

POLLUTANT_COLUMNS = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")
METEOROLOGICAL_COLUMNS = ("TEMP", "PRES", "DEWP", "RAIN", "WSPM")
CONTINUOUS_INPUT_COLUMNS = POLLUTANT_COLUMNS + METEOROLOGICAL_COLUMNS
RAW_COLUMNS = (
    "No",
    "year",
    "month",
    "day",
    "hour",
    *POLLUTANT_COLUMNS,
    *METEOROLOGICAL_COLUMNS[:-1],
    "wd",
    "WSPM",
    "station",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment constants that must not drift between notebooks."""

    methodology_version: str = "v2.2-focused-lstm-2026-08"
    source_url: str = (
        "https://archive.ics.uci.edu/static/public/501/"
        "beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip"
    )
    dataset_doi: str = "10.24432/C5RK5G"
    dataset_license: str = "CC BY 4.0"
    expected_total_rows: int = 420_768
    expected_rows_per_station: int = 35_064
    expected_start: str = "2013-03-01 00:00:00"
    expected_end: str = "2017-02-28 23:00:00"
    train_start: str = "2013-03-01 00:00:00"
    validation_start: str = "2016-01-01 00:00:00"
    test_start: str = "2016-08-01 00:00:00"
    data_end_exclusive: str = "2017-03-01 00:00:00"
    lookback_hours: int = 12
    forward_fill_limit_hours: int = 24
    forecast_horizon: int = 1
    seed: int = 42
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    cnn_max_epochs: int = 60
    cnn_early_stopping_patience: int = 8
    random_search_candidates: int = 12
    huber_delta: float = 10.0
    focused_max_epochs: int = 40
    focused_early_stopping_patience: int = 6
    run_mode: Literal["smoke", "full"] = "full"

    def to_dict(self) -> dict:
        return asdict(self)

    def stable_hash(self) -> str:
        """Return a short hash used to reject stale experiment artifacts."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


PATHS = ProjectPaths(resolve_project_root())
CONFIG = ExperimentConfig(
    run_mode=os.getenv("ODL_RUN_MODE", "full").strip().lower()  # type: ignore[arg-type]
)

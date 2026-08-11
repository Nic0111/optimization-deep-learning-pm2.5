"""Lazy, station-safe time-window generation for Keras."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from tensorflow import keras

    SequenceBase = keras.utils.Sequence
except ImportError:  # Enables lightweight unit tests without TensorFlow.
    class SequenceBase:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            pass


class StationWindowSequence(SequenceBase):
    """Generate batched windows without materialising all overlapping samples."""

    def __init__(
        self,
        model_table: pd.DataFrame,
        feature_columns: list[str],
        split: str,
        lookback: int = 24,
        batch_size: int = 256,
        shuffle: bool = False,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.features = model_table[feature_columns].to_numpy(dtype=np.float32, copy=True)
        self.targets = model_table["target_pm25"].to_numpy(dtype=np.float32, copy=True)
        self.station_codes = model_table["station_code"].to_numpy(dtype=np.int16, copy=True)
        self.timestamps = model_table["timestamp"].to_numpy(dtype="datetime64[h]", copy=True)
        eligible = model_table["valid_anchor"].to_numpy(dtype=bool) & model_table["split"].eq(split).to_numpy()
        self.base_anchor_indices = np.flatnonzero(eligible).astype(np.int64)
        self.anchor_indices = self.base_anchor_indices.copy()
        self.lookback = int(lookback)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self._offsets = np.arange(self.lookback - 1, -1, -1, dtype=np.int64)
        self._validate_anchors()
        if self.shuffle:
            self.on_epoch_end()

    def _validate_anchors(self) -> None:
        if not len(self.base_anchor_indices):
            raise ValueError("No valid sequence anchors were found for this split.")
        starts = self.base_anchor_indices - (self.lookback - 1)
        if (starts < 0).any():
            raise ValueError("A sequence starts before the beginning of the table.")
        if not np.array_equal(self.station_codes[starts], self.station_codes[self.base_anchor_indices]):
            raise ValueError("A sequence would cross a station boundary.")
        elapsed = (self.timestamps[self.base_anchor_indices] - self.timestamps[starts]).astype(
            "timedelta64[h]"
        ).astype(int)
        if not np.all(elapsed == self.lookback - 1):
            raise ValueError("A sequence contains a timestamp gap.")
        if not np.isfinite(self.targets[self.base_anchor_indices]).all():
            raise ValueError("A sequence has a missing or non-finite target.")

    def __len__(self) -> int:
        return int(np.ceil(len(self.anchor_indices) / self.batch_size))

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.batch_size
        stop = min(start + self.batch_size, len(self.anchor_indices))
        anchors = self.anchor_indices[start:stop]
        rows = anchors[:, None] - self._offsets[None, :]
        x_batch = self.features[rows]
        y_batch = self.targets[anchors, None]
        return x_batch.astype(np.float32, copy=False), y_batch.astype(np.float32, copy=False)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.anchor_indices)

    def ordered_anchor_indices(self) -> np.ndarray:
        """Return stable chronological anchors for targets and predictions."""

        return self.base_anchor_indices.copy()


def build_sequences(
    model_table: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    batch_size: int,
    seed: int,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, StationWindowSequence]:
    """Build train, validation, and test sequences from one feature contract."""

    unknown = set(splits) - {"train", "validation", "test"}
    if unknown:
        raise ValueError(f"Unknown requested sequence splits: {sorted(unknown)}")
    return {
        split: StationWindowSequence(
            model_table,
            feature_columns,
            split,
            lookback=lookback,
            batch_size=batch_size,
            shuffle=split == "train",
            seed=seed,
        )
        for split in splits
    }


def prediction_frame(
    model_table: pd.DataFrame,
    sequence: StationWindowSequence,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Align ordered model predictions with targets and naive baselines."""

    anchors = sequence.ordered_anchor_indices()
    values = np.asarray(predictions).reshape(-1)
    if len(values) != len(anchors):
        raise ValueError("Prediction count does not match sequence anchors.")

    frame = model_table.loc[
        anchors,
        ["station", "target_timestamp", "target_pm25", "PM2.5_input_raw"],
    ].copy()
    frame = frame.rename(columns={"target_pm25": "y_true", "PM2.5_input_raw": "persistence"})
    seasonal_rows = anchors - (sequence.lookback - 1)
    frame["seasonal_persistence"] = model_table.loc[
        seasonal_rows, "PM2.5_input_raw"
    ].to_numpy(dtype=float)
    frame["y_pred"] = values
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    return frame.reset_index(drop=True)

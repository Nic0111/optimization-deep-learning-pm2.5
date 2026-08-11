"""Basic and tunable TensorFlow/Keras forecasting models."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import CONFIG


def set_global_determinism(seed: int = 42) -> None:
    """Seed Python, NumPy, and TensorFlow and request deterministic ops."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


@keras.utils.register_keras_serializable(package="odl")
class PersistenceResidual(layers.Layer):
    """Add a learned change to the latest unscaled PM2.5 observation."""

    def __init__(
        self,
        pm25_feature_index: int,
        pm25_mean: float,
        pm25_scale: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pm25_feature_index = int(pm25_feature_index)
        self.pm25_mean = float(pm25_mean)
        self.pm25_scale = float(pm25_scale)

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        sequence, change = inputs
        standardised = sequence[:, -1, self.pm25_feature_index : self.pm25_feature_index + 1]
        persistence = standardised * self.pm25_scale + self.pm25_mean
        return persistence + change

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "pm25_feature_index": self.pm25_feature_index,
            "pm25_mean": self.pm25_mean,
            "pm25_scale": self.pm25_scale,
        }


def _compile(
    model: keras.Model,
    learning_rate: float,
    huber_delta: float = CONFIG.huber_delta,
) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=huber_delta),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def build_lstm(
    input_shape: tuple[int, int],
    units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    l2_strength: float = 0.0,
    num_layers: int = 1,
    huber_delta: float = CONFIG.huber_delta,
    formulation: str = "direct",
    pm25_feature_index: int = 0,
    pm25_mean: float = 0.0,
    pm25_scale: float = 1.0,
) -> keras.Model:
    """Build the assignment's LSTM forecasting model."""

    regularizer = keras.regularizers.l2(l2_strength) if l2_strength else None
    inputs = keras.Input(shape=input_shape, name="hourly_sequence")
    x = inputs
    for layer_index in range(num_layers):
        x = layers.LSTM(
            units,
            return_sequences=layer_index < num_layers - 1,
            kernel_regularizer=regularizer,
            name=f"lstm_{layer_index + 1}",
        )(x)
        x = layers.Dropout(dropout, name=f"lstm_dropout_{layer_index + 1}")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizer, name="dense")(x)
    if formulation == "residual":
        change = layers.Dense(
            1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="pm25_change",
        )(x)
        outputs = PersistenceResidual(
            pm25_feature_index,
            pm25_mean,
            pm25_scale,
            name="persistence_plus_change",
        )([inputs, change])
    elif formulation == "direct":
        outputs = layers.Dense(1, name="pm25_next_hour")(x)
    else:
        raise ValueError(f"Unknown forecast formulation: {formulation}")
    return _compile(
        keras.Model(inputs, outputs, name=f"lstm_{formulation}_forecaster"),
        learning_rate,
        huber_delta,
    )


def build_cnn(
    input_shape: tuple[int, int],
    filters: int = 64,
    kernel_size: int = 3,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    l2_strength: float = 0.0,
    num_blocks: int = 2,
    dilated: bool = True,
    huber_delta: float = CONFIG.huber_delta,
) -> keras.Model:
    """Build a causal 1D CNN for one-hour-ahead forecasting."""

    regularizer = keras.regularizers.l2(l2_strength) if l2_strength else None
    inputs = keras.Input(shape=input_shape, name="hourly_sequence")
    x = inputs
    for block_index in range(num_blocks):
        dilation_rate = 2**block_index if dilated else 1
        x = layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            kernel_regularizer=regularizer,
            name=f"causal_conv_{block_index + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"batch_norm_{block_index + 1}")(x)
        x = layers.Activation("relu", name=f"relu_{block_index + 1}")(x)
    # One-hour forecasting is dominated by the latest state. Retaining only the
    # final causal representation avoids diluting it through global averaging.
    x = layers.Cropping1D((input_shape[0] - 1, 0), name="latest_causal_state")(x)
    x = layers.Flatten(name="flatten_latest_state")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizer, name="dense")(x)
    outputs = layers.Dense(1, name="pm25_next_hour")(x)
    return _compile(keras.Model(inputs, outputs, name="cnn_forecaster"), learning_rate, huber_delta)


def build_tunable_lstm(hp: Any, input_shape: tuple[int, int]) -> keras.Model:
    """Keras Tuner search space for the LSTM family."""

    return build_lstm(
        input_shape=input_shape,
        units=hp.Choice("units", [32, 64, 128]),
        dense_units=hp.Choice("dense_units", [16, 32, 64]),
        dropout=hp.Float("dropout", 0.1, 0.4, step=0.1),
        learning_rate=hp.Float("learning_rate", 1e-4, 3e-3, sampling="log"),
        l2_strength=hp.Choice("l2_strength", [0.0, 1e-5, 1e-4]),
        num_layers=hp.Choice("num_layers", [1, 2]),
    )


def build_tunable_cnn(hp: Any, input_shape: tuple[int, int]) -> keras.Model:
    """Keras Tuner search space for the causal CNN family."""

    return build_cnn(
        input_shape=input_shape,
        filters=hp.Choice("filters", [32, 64, 128]),
        kernel_size=hp.Choice("kernel_size", [3, 5, 7]),
        dense_units=hp.Choice("dense_units", [16, 32, 64]),
        dropout=hp.Float("dropout", 0.1, 0.4, step=0.1),
        learning_rate=hp.Float("learning_rate", 1e-4, 3e-3, sampling="log"),
        l2_strength=hp.Choice("l2_strength", [0.0, 1e-5, 1e-4]),
        num_blocks=hp.Choice("num_blocks", [1, 2, 3]),
        dilated=True,
    )


def standard_callbacks(
    checkpoint_path: Path,
    history_log: Path,
    backup_dir: Path | None = None,
    append_log: bool = False,
) -> list[keras.callbacks.Callback]:
    """Create consistent, interruption-safe training callbacks.

    ``BackupAndRestore`` preserves the current epoch and optimizer state when a
    laptop session is interrupted.  It is deliberately separate from the best
    validation checkpoint, which remains the model used for comparison.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_log.parent.mkdir(parents=True, exist_ok=True)
    previous_best: float | None = None
    if append_log and history_log.exists():
        import pandas as pd

        existing_history = pd.read_csv(history_log)
        if "val_mae" in existing_history and existing_history["val_mae"].notna().any():
            previous_best = float(existing_history["val_mae"].min())
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=CONFIG.early_stopping_patience,
            restore_best_weights=True,
            baseline=previous_best,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            initial_value_threshold=previous_best,
        ),
        keras.callbacks.CSVLogger(history_log, append=append_log),
        keras.callbacks.TerminateOnNaN(),
    ]
    if backup_dir is not None:
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        callbacks.insert(
            0,
            keras.callbacks.BackupAndRestore(
                backup_dir=backup_dir,
                delete_checkpoint=True,
            ),
        )
    return callbacks


def save_history(history: keras.callbacks.History, csv_path: Path, json_path: Path) -> None:
    """Persist training history in both tabular and machine-readable formats."""

    import pandas as pd

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history.history)
    frame.index.name = "epoch"
    frame.to_csv(csv_path)
    json_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

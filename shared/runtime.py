"""Local WSL runtime configuration, state tracking, and GPU monitoring."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Self

from .config import CONFIG, PATHS, ExperimentConfig, ProjectPaths

RunMode = Literal["smoke", "full"]
DeviceChoice = Literal["auto", "gpu", "cpu"]
PIPELINE_STAGES = ("acquire", "prepare", "baselines", "tune", "evaluate", "render")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a temporary file so interruptions cannot corrupt state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class RuntimeOptions:
    """Options that identify one reproducible local experiment run."""

    mode: RunMode = "full"
    device: DeviceChoice = "auto"
    batch_size: int = 256
    resume: bool = True
    allow_cpu_full: bool = False
    reuse_only: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError(f"Unsupported run mode: {self.mode}")
        if self.device not in {"auto", "gpu", "cpu"}:
            raise ValueError(f"Unsupported device choice: {self.device}")
        if self.batch_size not in {64, 128, 256}:
            raise ValueError("Batch size must be one of 64, 128, or 256.")

    @property
    def artifact_suffix(self) -> str:
        return "" if self.mode == "full" else "_smoke"

    @classmethod
    def from_environment(cls) -> RuntimeOptions:
        return cls(
            mode=os.getenv("ODL_RUN_MODE", "full").strip().lower(),  # type: ignore[arg-type]
            device=os.getenv("ODL_DEVICE", "auto").strip().lower(),  # type: ignore[arg-type]
            batch_size=int(os.getenv("ODL_BATCH_SIZE", "256")),
            resume=os.getenv("ODL_RESUME", "1") != "0",
            allow_cpu_full=os.getenv("ODL_ALLOW_CPU_FULL", "0") == "1",
            reuse_only=os.getenv("ODL_REUSE_ONLY", "0") == "1",
        )

    def experiment_config(self) -> ExperimentConfig:
        return replace(CONFIG, run_mode=self.mode, batch_size=self.batch_size)

    def stable_hash(self) -> str:
        payload = {
            "experiment": self.experiment_config().to_dict(),
            "runtime": {
                "mode": self.mode,
                "device": self.device,
                "batch_size": self.batch_size,
            },
        }
        serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:16]


class RunStateStore:
    """Persistent state used for safe stage-level and component-level resume."""

    def __init__(self, paths: ProjectPaths, options: RuntimeOptions) -> None:
        self.paths = paths
        self.options = options
        self.path = paths.run_state / f"{options.mode}.json"
        self.config_hash = options.stable_hash()

    def _new_state(self) -> dict[str, Any]:
        return {
            "mode": self.options.mode,
            "config_hash": self.config_hash,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "current_stage": None,
            "stages": {},
            "components": {},
        }

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._new_state()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        recorded = state.get("config_hash")
        if recorded != self.config_hash:
            raise RuntimeError(
                "Existing run state was produced by a different configuration "
                f"({recorded} != {self.config_hash}). Archive it or use the recorded "
                "batch/device settings before resuming."
            )
        return state

    def save(self, state: dict[str, Any]) -> None:
        state["updated_at_utc"] = utc_now()
        write_json_atomic(self.path, state)

    def begin(self, key: str, component: bool = False) -> None:
        state = self.load()
        group = "components" if component else "stages"
        state[group][key] = {
            "status": "running",
            "started_at_utc": utc_now(),
        }
        if not component:
            state["current_stage"] = key
        self.save(state)

    def complete(
        self,
        key: str,
        summary: dict[str, Any] | None = None,
        outputs: list[Path] | None = None,
        component: bool = False,
    ) -> None:
        state = self.load()
        group = "components" if component else "stages"
        state[group][key] = {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "summary": summary or {},
            "outputs": [str(path) for path in outputs or []],
        }
        if not component:
            state["current_stage"] = None
        self.save(state)

    def fail(self, key: str, error: BaseException, component: bool = False) -> None:
        state = self.load()
        group = "components" if component else "stages"
        previous = state[group].get(key, {})
        state[group][key] = {
            **previous,
            "status": "failed",
            "failed_at_utc": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if not component:
            state["current_stage"] = None
        self.save(state)

    def is_complete(
        self,
        key: str,
        outputs: list[Path] | None = None,
        component: bool = False,
    ) -> bool:
        state = self.load()
        group = "components" if component else "stages"
        item = state[group].get(key, {})
        required = outputs if outputs is not None else [Path(path) for path in item.get("outputs", [])]
        return item.get("status") == "complete" and all(path.exists() for path in required)

    def summary(self, key: str) -> dict[str, Any]:
        return self.load().get("stages", {}).get(key, {}).get("summary", {})


@contextmanager
def tracked_stage(
    store: RunStateStore,
    stage: str,
    outputs: list[Path],
) -> Iterator[dict[str, Any]]:
    """Track a stage and expose a dictionary for its final summary."""

    summary: dict[str, Any] = {}
    store.begin(stage)
    try:
        yield summary
    except BaseException as error:
        store.fail(stage, error)
        raise
    else:
        store.complete(stage, summary=summary, outputs=outputs)


def git_metadata(root: Path) -> dict[str, Any]:
    if shutil.which("git") is None:
        return {"available": False}
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
        return {"available": True, "revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}


def nvidia_smi_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False}
    query = (
        "name,driver_version,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,power.draw"
    )
    try:
        output = subprocess.check_output(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).strip()
        values = [value.strip() for value in output.splitlines()[0].split(",")]
        keys = query.split(",")
        return {"available": True, **dict(zip(keys, values, strict=True))}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return {"available": False}


def _package_versions() -> dict[str, str | None]:
    packages = (
        "tensorflow",
        "keras-tuner",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def configure_tensorflow(options: RuntimeOptions) -> tuple[Any, str, list[Any]]:
    """Import and configure TensorFlow after applying the requested device policy."""

    if options.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is not installed. Run the WSL bootstrap before preflight."
        ) from error

    gpus = tf.config.list_physical_devices("GPU")
    if options.device != "cpu":
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Device initialisation may already have occurred in an interactive notebook.
                pass

    selected = "gpu" if gpus and options.device != "cpu" else "cpu"
    if options.device == "gpu" and not gpus:
        raise RuntimeError("GPU execution was requested, but TensorFlow detected no GPU.")
    if options.mode == "full" and selected == "cpu" and not options.allow_cpu_full:
        raise RuntimeError(
            "Full execution requires the local GPU. Use --allow-cpu-full only after "
            "accepting the substantially longer runtime."
        )
    return tf, selected, gpus


def probe_batch_size(tf: Any, preferred: int) -> int:
    """Choose the largest candidate that can run the assignment's largest LSTM batch."""

    candidates = [size for size in (256, 128, 64) if size <= preferred]
    last_error: BaseException | None = None
    for size in candidates:
        try:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(CONFIG.seed)
            inputs = tf.keras.Input(shape=(24, 41))
            x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
            x = tf.keras.layers.LSTM(128)(x)
            outputs = tf.keras.layers.Dense(1)(x)
            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mse")
            x_batch = tf.zeros((size, 24, 41), dtype=tf.float32)
            y_batch = tf.zeros((size, 1), dtype=tf.float32)
            model.train_on_batch(x_batch, y_batch)
            tf.keras.backend.clear_session()
            return size
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as error:
            last_error = error
            tf.keras.backend.clear_session()
    raise RuntimeError("Even batch size 64 failed the local GPU memory probe.") from last_error


def run_preflight(
    options: RuntimeOptions,
    paths: ProjectPaths = PATHS,
    probe_batch: bool = True,
) -> tuple[RuntimeOptions, dict[str, Any]]:
    """Validate the WSL/TensorFlow runtime and persist reproducibility evidence."""

    tf, selected_device, gpus = configure_tensorflow(options)
    batch_size = probe_batch_size(tf, options.batch_size) if probe_batch else options.batch_size
    resolved = replace(options, device=selected_device, batch_size=batch_size)
    uname = platform.uname()
    environment = {
        "captured_at_utc": utc_now(),
        "project_root": str(paths.root),
        "platform": platform.platform(),
        "system": uname.system,
        "release": uname.release,
        "python": platform.python_version(),
        "wsl_distribution": os.getenv("WSL_DISTRO_NAME"),
        "device": selected_device,
        "tensorflow_gpu_devices": [device.name for device in gpus],
        "batch_size": batch_size,
        "mixed_precision": False,
        "nvidia_smi": nvidia_smi_snapshot(),
        "packages": _package_versions(),
        "git": git_metadata(paths.root),
        "experiment_config": resolved.experiment_config().to_dict(),
        "config_hash": resolved.stable_hash(),
    }
    paths.ensure_directories()
    write_json_atomic(paths.environment_artifacts / f"environment_{options.mode}.json", environment)
    return resolved, environment


class GPUMonitor:
    """Sample NVIDIA utilisation to CSV while a long pipeline is running."""

    def __init__(self, path: Path, interval_seconds: int = 60) -> None:
        self.path = path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None or self._thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="odl-gpu-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        fields = [
            "timestamp_utc",
            "name",
            "driver_version",
            "memory.total",
            "memory.used",
            "utilization.gpu",
            "temperature.gpu",
            "power.draw",
        ]
        write_header = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if write_header:
                writer.writeheader()
            while not self._stop.is_set():
                sample = nvidia_smi_snapshot()
                if sample.get("available"):
                    measurement = {key: sample.get(key) for key in fields if key != "timestamp_utc"}
                    writer.writerow({"timestamp_utc": utc_now(), **measurement})
                    stream.flush()
                self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

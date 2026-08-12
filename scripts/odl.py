"""Local command-line entry point for the complete ODL experiment."""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reexec_with_wsl_gpu_libraries() -> None:
    """Expose pip-installed NVIDIA libraries before TensorFlow is imported.

    The dynamic loader reads ``LD_LIBRARY_PATH`` when the Python process starts,
    so changing it during TensorFlow preflight is too late. Re-exec exactly once
    with the locked environment's NVIDIA library directories when running in WSL.
    """

    if not os.getenv("WSL_DISTRO_NAME") or os.getenv("ODL_GPU_LIBS_CONFIGURED") == "1":
        return
    site_packages = (
        ROOT
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    libraries = sorted(str(path) for path in (site_packages / "nvidia").glob("*/lib") if path.is_dir())
    wsl_driver = Path("/usr/lib/wsl/lib")
    if wsl_driver.is_dir():
        libraries.insert(0, str(wsl_driver))
    if not libraries:
        return
    environment = os.environ.copy()
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([*libraries, *([existing] if existing else [])])
    environment["ODL_GPU_LIBS_CONFIGURED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


_reexec_with_wsl_gpu_libraries()

from shared.config import PATHS
from shared.runtime import (
    GPUMonitor,
    RunStateStore,
    RuntimeOptions,
    archive_existing_run,
    run_preflight,
    source_fingerprint,
    utc_now,
    write_json_atomic,
)
from shared.workflow import run_pipeline

MODEL_STAGES = ("acquire", "prepare", "baselines", "tune", "evaluate")
NOTEBOOKS = (
    Path("assignment1/01_data_acquisition_and_audit.ipynb"),
    Path("assignment1/02_eda_and_preprocessing.ipynb"),
    Path("assignment2/01_baselines_and_basic_models.ipynb"),
    Path("assignment2/02_hyperparameter_tuning.ipynb"),
    Path("assignment2/03_final_evaluation.ipynb"),
)


class Tee(io.TextIOBase):
    """Write training output to the terminal and a persistent log."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self.terminal = terminal
        self.log = log

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


@contextmanager
def pipeline_log(mode: str) -> Iterator[Path]:
    PATHS.runtime_logs.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = PATHS.runtime_logs / f"pipeline_{mode}_{timestamp}.log"
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        output = Tee(sys.__stdout__, stream)
        errors = Tee(sys.__stderr__, stream)
        with redirect_stdout(output), redirect_stderr(errors):
            print(f"ODL local pipeline started at {utc_now()}")
            print(f"Project root: {ROOT}")
            yield path


def _options(args: argparse.Namespace) -> RuntimeOptions:
    return RuntimeOptions(
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        resume=not getattr(args, "no_resume", False),
        allow_cpu_full=getattr(args, "allow_cpu_full", False),
        reuse_only=False,
    )


def _export_options(options: RuntimeOptions, reuse_only: bool = False) -> None:
    os.environ["ODL_PROJECT_ROOT"] = str(ROOT)
    os.environ["ODL_RUN_MODE"] = options.mode
    os.environ["ODL_DEVICE"] = options.device
    os.environ["ODL_BATCH_SIZE"] = str(options.batch_size)
    os.environ["ODL_RESUME"] = "1" if options.resume else "0"
    os.environ["ODL_ALLOW_CPU_FULL"] = "1" if options.allow_cpu_full else "0"
    os.environ["ODL_REUSE_ONLY"] = "1" if reuse_only else "0"


def _stage_slice(from_stage: str | None, through_stage: str | None) -> tuple[str, ...]:
    start = MODEL_STAGES.index(from_stage) if from_stage else 0
    stop = MODEL_STAGES.index(through_stage) + 1 if through_stage else len(MODEL_STAGES)
    if start >= stop:
        raise ValueError("--from-stage must not come after --through-stage.")
    return MODEL_STAGES[start:stop]


def render_notebooks(options: RuntimeOptions) -> list[Path]:
    """Regenerate and execute notebooks while refusing to retrain missing stages."""

    import nbformat
    from nbclient import NotebookClient

    from scripts.build_notebooks import main as build_notebooks

    store = RunStateStore(PATHS, options)
    if options.resume and store.is_complete("render", [ROOT / path for path in NOTEBOOKS]):
        return [ROOT / path for path in NOTEBOOKS]
    store.begin("render")
    executed: list[Path] = []
    try:
        build_notebooks()
        # Rendering must reuse completed stages even when the preceding training
        # command explicitly disabled resume to force a fresh experiment.
        _export_options(replace(options, resume=True), reuse_only=True)
        for relative_path in NOTEBOOKS:
            notebook_path = ROOT / relative_path
            print(f"Rendering {relative_path} from completed artifacts...", flush=True)
            notebook = nbformat.read(notebook_path, as_version=4)
            client = NotebookClient(
                notebook,
                timeout=None,
                kernel_name="odl-assignment",
                resources={"metadata": {"path": str(ROOT)}},
                allow_errors=False,
            )
            client.execute(cwd=str(ROOT))
            nbformat.write(notebook, notebook_path)
            archive_path = PATHS.executed_notebooks / options.mode / relative_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(notebook_path, archive_path)
            executed.append(notebook_path)
        store.complete(
            "render",
            summary={"notebooks": len(executed), "mode": options.mode},
            outputs=executed,
        )
        return executed
    except BaseException as error:
        store.fail("render", error)
        raise


def command_preflight(args: argparse.Namespace) -> int:
    requested = _options(args)
    resolved, environment = run_preflight(requested, probe_batch=not args.no_batch_probe)
    _export_options(resolved)
    print(json.dumps(environment, indent=2, default=str))
    return 0


def command_run(args: argparse.Namespace) -> int:
    requested = _options(args)
    if not requested.resume:
        archive_path = archive_existing_run(PATHS, requested.mode)
        if archive_path is not None:
            print(f"Archived previous compact run evidence to: {archive_path}")
    with pipeline_log(requested.mode) as log_path:
        pipeline_started = time.perf_counter()
        resolved, environment = run_preflight(requested, probe_batch=not args.no_batch_probe)
        _export_options(resolved)
        print("Resolved local runtime:")
        print(json.dumps(environment, indent=2, default=str))
        stages = _stage_slice(args.from_stage, args.through_stage)
        monitor_path = PATHS.runtime_logs / f"gpu_{resolved.mode}.csv"
        with GPUMonitor(monitor_path):
            summaries = run_pipeline(resolved, stages=stages)
            if args.render:
                render_notebooks(resolved)
        summary_path = PATHS.run_state / f"pipeline_summary_{resolved.mode}.json"
        write_json_atomic(
            summary_path,
            {
                "mode": resolved.mode,
                "device": resolved.device,
                "batch_size": resolved.batch_size,
                "elapsed_seconds": time.perf_counter() - pipeline_started,
                "completed_at_utc": utc_now(),
                "stages": summaries,
            },
        )
        print(f"Pipeline log: {log_path}")
        print(f"Pipeline summary: {summary_path}")
    return 0


def _load_state(mode: str) -> dict:
    path = PATHS.run_state / f"{mode}.json"
    if not path.exists():
        return {"mode": mode, "status": "not_started", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def command_status(args: argparse.Namespace) -> int:
    payload = {mode: _load_state(mode) for mode in ([args.mode] if args.mode else ["smoke", "full"])}
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _smoke_is_complete() -> bool:
    state = _load_state("smoke")
    required = ("acquire", "prepare", "baselines", "tune", "evaluate", "render")
    return (
        state.get("source_fingerprint") == source_fingerprint(ROOT)
        and all(
            state.get("stages", {}).get(stage, {}).get("status") == "complete"
            for stage in required
        )
    )


def command_start(args: argparse.Namespace) -> int:
    if os.name == "nt" or not os.getenv("WSL_DISTRO_NAME"):
        raise RuntimeError("Detached full training must be launched inside the Ubuntu WSL workspace.")
    if args.mode == "full":
        if not args.approved_after_smoke:
            raise RuntimeError(
                "Full training is approval-gated. Re-run with --approved-after-smoke only "
                "after the user reviews the smoke benchmark."
            )
        if not _smoke_is_complete():
            raise RuntimeError("The complete smoke pipeline and notebook render must pass first.")
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed; run the WSL bootstrap script.")
    session = f"odl-{args.mode}"
    existing = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"A persistent session named {session!r} is already running.")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--mode",
        args.mode,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--render",
    ]
    if args.no_resume:
        command.append("--no-resume")
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session, shlex.join(command)])
    print(f"Started persistent session: {session}")
    print(f"Check progress with: {sys.executable} scripts/odl.py status --mode {args.mode}")
    print(f"View logs with: {sys.executable} scripts/odl.py logs --mode {args.mode}")
    return 0


def command_logs(args: argparse.Namespace) -> int:
    candidates = sorted(PATHS.runtime_logs.glob(f"pipeline_{args.mode}_*.log"))
    if not candidates:
        raise FileNotFoundError(f"No {args.mode} pipeline log exists yet.")
    path = candidates[-1]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"Log: {path}")
    print("\n".join(lines[-args.lines :]))
    if args.follow:
        if os.name == "nt" or shutil.which("tail") is None:
            raise RuntimeError("--follow is available inside WSL where the tail command exists.")
        return subprocess.call(["tail", "-n", str(args.lines), "-f", str(path)])
    return 0


def command_render(args: argparse.Namespace) -> int:
    requested = _options(args)
    environment_path = PATHS.environment_artifacts / f"environment_{requested.mode}.json"
    if not environment_path.exists():
        raise FileNotFoundError("Run preflight and the pipeline before rendering notebooks.")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    resolved = replace(
        requested,
        device=environment["device"],
        batch_size=int(environment["batch_size"]),
    )
    render_notebooks(resolved)
    return 0


def add_runtime_arguments(parser: argparse.ArgumentParser, default_mode: str = "full") -> None:
    parser.add_argument("--mode", choices=("smoke", "full"), default=default_mode)
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--allow-cpu-full", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Validate TensorFlow and the local GPU.")
    add_runtime_arguments(preflight)
    preflight.add_argument("--no-batch-probe", action="store_true")
    preflight.set_defaults(handler=command_preflight)

    run = subparsers.add_parser("run", help="Run the resumable experiment in the foreground.")
    add_runtime_arguments(run, default_mode="smoke")
    run.add_argument("--from-stage", choices=MODEL_STAGES)
    run.add_argument("--through-stage", choices=MODEL_STAGES)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--no-batch-probe", action="store_true")
    run.add_argument("--render", action="store_true", help="Capture notebook outputs after success.")
    run.set_defaults(handler=command_run)

    start = subparsers.add_parser("start", help="Start a persistent WSL training session.")
    add_runtime_arguments(start)
    start.add_argument("--approved-after-smoke", action="store_true")
    start.add_argument("--no-resume", action="store_true")
    start.set_defaults(handler=command_start)

    status = subparsers.add_parser("status", help="Print machine-readable run state.")
    status.add_argument("--mode", choices=("smoke", "full"))
    status.set_defaults(handler=command_status)

    logs = subparsers.add_parser("logs", help="Show the latest persistent pipeline log.")
    logs.add_argument("--mode", choices=("smoke", "full"), default="full")
    logs.add_argument("--lines", type=int, default=80)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(handler=command_logs)

    render = subparsers.add_parser("render-notebooks", help="Capture outputs without retraining.")
    add_runtime_arguments(render)
    render.add_argument("--no-resume", action="store_true")
    render.set_defaults(handler=command_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted. Checkpoints and run state were preserved.", file=sys.stderr)
        return 130
    # A CLI boundary must turn operational failures into a concise non-zero exit.
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

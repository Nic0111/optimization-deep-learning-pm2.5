"""Tests for local runtime state and stale-artifact protection."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shared import workflow
from shared.config import ProjectPaths
from shared.runtime import RunStateStore, RuntimeOptions, archive_existing_run

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_options_use_separate_smoke_artifacts():
    smoke = RuntimeOptions(mode="smoke", device="gpu", batch_size=128)
    full = RuntimeOptions(mode="full", device="gpu", batch_size=128)
    assert smoke.artifact_suffix == "_smoke"
    assert full.artifact_suffix == ""
    assert smoke.stable_hash() != full.stable_hash()


def test_registered_wsl_kernel_launcher_is_executable():
    assert os.access(ROOT / "scripts" / "python_wsl.sh", os.X_OK)


def test_run_state_requires_outputs_to_exist(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    options = RuntimeOptions(mode="smoke", device="cpu", batch_size=64)
    store = RunStateStore(paths, options)
    output = tmp_path / "result.txt"
    output.write_text("complete", encoding="utf-8")
    store.complete("prepare", summary={"rows": 10}, outputs=[output])
    assert store.is_complete("prepare")
    assert store.summary("prepare") == {"rows": 10}
    output.unlink()
    assert not store.is_complete("prepare")


def test_run_state_rejects_changed_configuration(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    first = RunStateStore(paths, RuntimeOptions(mode="smoke", device="cpu", batch_size=64))
    first.begin("acquire")
    changed = RunStateStore(paths, RuntimeOptions(mode="smoke", device="cpu", batch_size=128))
    with pytest.raises(RuntimeError, match="different configuration"):
        changed.load()


def test_workflow_reuses_completed_stage_without_repeating_work(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    options = RuntimeOptions(mode="smoke", device="cpu", batch_size=64)
    calls = []

    def fake_stage(runtime_options, project_paths, _store):
        calls.append(runtime_options.mode)
        output = project_paths.interim / "fake.parquet"
        output.write_text("result", encoding="utf-8")
        return {"rows": 5}, [output]

    monkeypatch.setitem(workflow.STAGE_FUNCTIONS, "acquire", fake_stage)
    assert workflow.run_stage("acquire", options=options, paths=paths) == {"rows": 5}
    assert workflow.run_stage("acquire", options=options, paths=paths) == {"rows": 5}
    assert calls == ["smoke"]


def test_run_state_rejects_changed_methodology_source(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    source = tmp_path / "shared" / "example.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    options = RuntimeOptions(mode="smoke", device="cpu", batch_size=64)
    RunStateStore(paths, options).begin("acquire")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different methodology source"):
        RunStateStore(paths, options).load()


def test_no_resume_archive_clears_state_and_progress_only(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure_directories()
    state = paths.run_state / "smoke.json"
    state.write_text('{"status": "complete"}', encoding="utf-8")
    progress = paths.assignment2_tables / "random_search_rolling_results_smoke.csv"
    progress.write_text("candidate_id\ncandidate_00\n", encoding="utf-8")
    model = paths.models / "tuned_final_smoke.keras"
    model.write_text("model", encoding="utf-8")

    archive = archive_existing_run(paths, "smoke")
    assert archive is not None and (archive / "archive_manifest.json").exists()
    assert not state.exists()
    assert not progress.exists()
    assert model.exists()

"""Synthetic-only preparation tests for dormant real Qwen public CPU smoke."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

from clipper.books import scripting

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/book_local_cpu_smoke.py"
_SPEC = spec_from_file_location("clipper_book_local_cpu_smoke", _SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("public smoke harness file could not be loaded")
smoke = module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


@pytest.fixture(autouse=True)
def _isolate_synthetic_job_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may append a synthetic 'real execution' summary to Actions CI."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def _request() -> dict[str, object]:
    return {
        "schema": smoke.SCHEMA,
        "intent": smoke.INTENT,
        "branch": smoke.BRANCH,
        "reviewed_code_sha": "a" * 40,
        "fixture_label": smoke.LABEL,
        "fixture_sha256": smoke.FIXTURE_SHA256,
        "max_new_units": 2,
    }


def _validate(record: object, **overrides: object) -> str:
    params: dict[str, object] = {
        "ref": f"refs/heads/{smoke.BRANCH}",
        "event": "push",
        "trigger_sha": "b" * 40,
        "before_sha": "a" * 40,
        "parents": ("a" * 40,),
        "changes": (f"A\t{smoke.RUN_REQUEST}",),
        "fixture": smoke._fixture_bytes(),
    }
    params.update(overrides)
    return smoke.validate_request(record, **params)


def test_original_public_fixture_bytes_are_sha_pinned_and_two_content_units() -> None:
    source = smoke._fixture_bytes()
    assert hashlib.sha256(source).hexdigest() == smoke.FIXTURE_SHA256
    parts = smoke._paragraphs(source)
    assert len(parts) == 2
    assert all(50 <= len(text.split()) <= 220 for text in parts)
    assert smoke.LABEL.encode() in source
    assert b"Atomic Habits" not in source


def test_strict_push_gate_accepts_only_next_one_file_request() -> None:
    assert _validate(_request()) == "a" * 40


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "arbitrary-schema"),
        ("intent", "DRY_RUN"),
        ("branch", "main"),
        ("reviewed_code_sha", "c" * 40),
        ("fixture_label", "a real book"),
        ("fixture_sha256", "f" * 64),
        ("max_new_units", 0),
        ("max_new_units", 1),
        ("max_new_units", 3),
        ("max_new_units", True),
        ("max_new_units", "2"),
    ],
)
def test_request_rejects_unknown_identity_scope_and_budget(field: str, value: object) -> None:
    candidate = _request()
    candidate[field] = value
    with pytest.raises(smoke.SmokeError, match="exact pinned code/fixture/budget"):
        _validate(candidate)


@pytest.mark.parametrize(
    "override",
    [
        {"ref": "refs/heads/main"},
        {"event": "workflow_dispatch"},
        {"trigger_sha": "not-sha"},
        {"before_sha": "not-sha"},
        {"parents": ("a" * 40, "c" * 40)},
        {"parents": ("c" * 40,)},
        {"parents": ()},
        {"changes": (f"M\t{smoke.RUN_REQUEST}",)},
        {"changes": (f"A\t{smoke.RUN_REQUEST}", "A\tsrc/new.py")},
        {"changes": ("A\tprivate/source.txt",)},
        {"fixture": b"PRIVATE_INPUT"},
    ],
)
def test_gate_rejects_nonreviewed_parent_other_paths_or_invalid_fixture(
    override: dict[str, object],
) -> None:
    with pytest.raises(smoke.SmokeError):
        _validate(_request(), **override)


def test_request_json_rejects_duplicates_extra_fields_and_nonobjects() -> None:
    with pytest.raises(smoke.SmokeError, match="duplicate"):
        json.loads('{"schema":"first","schema":"second"}', object_pairs_hook=smoke._unique_pairs)
    with pytest.raises(smoke.SmokeError, match="strict schema"):
        _validate({**_request(), "fixture_path": "/home/private/book.epub"})
    with pytest.raises(smoke.SmokeError, match="JSON object"):
        _validate([_request()])


def test_public_fixture_prepare_uses_production_candidate_ingest_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_inference(*_args: object, **_kwargs: object) -> None:
        pytest.fail("dry-run fixture preparation must never load weights or call network")

    monkeypatch.setattr(scripting, "_script_pipeline", no_inference)
    monkeypatch.setattr("urllib.request.urlopen", no_inference)
    manifest = smoke.prepare(tmp_path)
    assert manifest["label"] == smoke.LABEL
    assert manifest["max_new_units"] == 2
    assert manifest["source_kind"] == "FABRICATED_IN_MEMORY_RECORDED_TRANSPORT_NO_NETWORK"
    assert manifest["publish_eligible"] is False
    assert len(manifest["work_units"]) == 2
    assert all(row["field"] == "native_text" for row in manifest["work_units"])
    assert all(row["word_count"] >= 50 for row in manifest["work_units"])
    assert manifest["analysis_ref"].startswith("private/book-candidates/analysis-refs/")
    for unit in range(2):
        stored = json.loads((tmp_path / "evidence" / f"unit-{unit:04d}-request.json").read_text())
        assert stored["model"] == scripting.SCRIPT_MODEL
        assert stored["revision"] == scripting.SCRIPT_REVISION
        assert stored["max_new_tokens"] == 960
        assert stored["seed"] == 42
        assert smoke.LABEL in stored["messages"][1]["content"]
    assert (tmp_path / "evidence" / "public-source.txt").read_bytes() == smoke._fixture_bytes()


def test_public_fixture_dry_run_cli_never_calls_model(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_args: object, **_kwargs: object) -> None:
        pytest.fail("synthetic CLI dry-run must not invoke real inference")

    monkeypatch.setattr(scripting, "_script_pipeline", no_model)
    monkeypatch.setattr("urllib.request.urlopen", no_model)
    assert smoke.main(["dry-run"]) == 0
    out = capsys.readouterr().out
    assert "PUBLIC_BOOK_CPU_SMOKE_DRY_RUN" in out
    assert '"no_inference": true' in out


def test_real_entry_refuses_gate_failure_before_any_fixture_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gate_rejected() -> None:
        raise smoke.SmokeError("no trigger file")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("failed gate must prevent all fixture and model work")

    monkeypatch.setattr(smoke, "gate", gate_rejected)
    monkeypatch.setattr(smoke, "prepare", forbidden)
    monkeypatch.setattr(scripting, "_script_pipeline", forbidden)
    with pytest.raises(smoke.SmokeError, match="no trigger file"):
        smoke._execute(tmp_path / "fresh", replay=False)
    assert not (tmp_path / "fresh").exists()


def test_bounded_reconstructable_log_evidence_ignores_unrelated_paths(tmp_path: Path) -> None:
    root = tmp_path / "public"
    (root / "evidence").mkdir(parents=True)
    original = os.urandom(1403)
    (root / "evidence" / "runtime.json").write_bytes(original)
    raw = (
        root
        / "private/book-candidates/analysis/completions"
        / smoke.BOOK_ID
        / ("a" * 64)
        / f"{'b' * 64}.txt"
    )
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'{"lessons": []}\n')
    secret = root / "private/private-book-input.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("NEVER_COLLECT_PRIVATE_INPUT")
    records = smoke.encoded_evidence(root)
    assert len(records) >= 4
    assert all(row["execution"] == "SYNTHETIC_OR_UNGATED_NO_MODEL" for row in records)
    assert all(row["label"] == smoke.LABEL for row in records)
    assert not any("private-book-input" in row["path"] for row in records)
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record["path"]), []).append(record)
    assert len(groups) == 2
    reconstructed: dict[str, bytes] = {}
    for name, parts in groups.items():
        parts.sort(key=lambda item: int(item["index"]))
        assert [part["index"] for part in parts] == list(range(parts[0]["chunk_count"]))
        data = b"".join(base64.b64decode(part["data_b64"]) for part in parts)
        assert len(data) == parts[0]["byte_count"]
        assert hashlib.sha256(data).hexdigest() == parts[0]["sha256"]
        reconstructed[name] = data
    assert reconstructed["evidence/runtime.json"] == original
    assert reconstructed[str(raw.relative_to(root))] == raw.read_bytes()


def test_evidence_bounds_reject_oversize_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "public"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    oversized = evidence / "runtime.json"
    oversized.write_bytes(b"x" * (smoke.MAX_FILE_BYTES + 1))
    with pytest.raises(smoke.SmokeError, match="individual"):
        smoke.encoded_evidence(root)
    oversized.unlink()
    target = root / "elsewhere"
    target.mkdir()
    (evidence / "reports").symlink_to(target, target_is_directory=True)
    with pytest.raises(smoke.SmokeError, match="linked ancestor"):
        smoke.encoded_evidence(root)


def test_timeout_or_oom_produces_failure_evidence_without_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    root = smoke._root()
    root.mkdir()
    smoke.finish(root, 124)
    report = json.loads((root / "evidence" / "process-exit.json").read_text())
    assert report["outcome"] == "timeout"
    assert report["exit_code"] == 124
    assert smoke.main(["collect"]) == 0
    assert "PUBLIC_BOOK_CPU_SMOKE_COLLECTION" in capsys.readouterr().out
    assert smoke.main(["finish", "--exit-code", "256"]) == 1


@pytest.mark.parametrize("memory,disk", [(1_000_000_000, 30_000_000_000), (20_000_000_000, 1000)])
def test_resource_preflight_rejects_insufficient_ram_or_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memory: int, disk: int
) -> None:
    monkeypatch.setattr(
        smoke,
        "_metric_bytes",
        lambda: {"ram_available_bytes": memory, "disk_free_bytes": disk},
    )
    with pytest.raises(smoke.SmokeError, match="insufficient"):
        smoke._require_resources(tmp_path)
    assert (tmp_path / "evidence" / "resource-preflight.json").is_file()


def test_optional_model_markers_do_not_log_prompts_when_disabled_or_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLIPPER_PUBLIC_BOOK_CPU_SMOKE", raising=False)
    scripting._public_cpu_smoke_phase("cpu_weight_loading_start")
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("CLIPPER_PUBLIC_BOOK_CPU_SMOKE", "1")
    scripting._public_cpu_smoke_phase("cpu_weight_loading_start")
    assert capsys.readouterr().out.strip() == "PUBLIC_BOOK_CPU_MODEL_PHASE cpu_weight_loading_start"


def test_workflow_github_context_rules_and_dormant_trigger_are_explicit() -> None:
    workflow_path = smoke.REPO / ".github/workflows/book-local-cpu-smoke.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    # PyYAML 1.1 treats the GitHub Actions on key as boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"push"}
    assert triggers["push"]["branches"] == [smoke.BRANCH]
    assert triggers["push"]["paths"] == [smoke.RUN_REQUEST]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    job = workflow["jobs"]["public-synthetic-only-cpu"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == 45
    assert "HF_HOME" not in job["env"]
    # GitHub context rules: jobs.<id>.env cannot use the runner context.
    # The shell step can use the default RUNNER_TEMP environment variable.
    assert not any("runner." in str(value) for value in job["env"].values())
    steps = job["steps"]
    gate_index = next(
        i for i, step in enumerate(steps) if step["name"].startswith("Enforce one-commit")
    )
    cache_index = next(i for i, step in enumerate(steps) if "ephemeral HF cache" in step["name"])
    install_index = next(i for i, step in enumerate(steps) if "CPU-only pinned" in step["name"])
    assert gate_index < cache_index < install_index
    setup = steps[cache_index]["run"]
    assert '"$RUNNER_TEMP"' in setup
    assert '"$GITHUB_ENV"' in setup
    assert "HF_HOME=%s/clipper-public-book-cpu-smoke-hf" in setup
    content = workflow_path.read_text(encoding="utf-8")
    assert "runner.temp" not in content
    assert "actions/cache" not in content
    assert "upload-artifact" not in content


def test_smoke_gate_model_replay_and_collect_share_push_before_sha() -> None:
    """The always-run collector must be able to revalidate the genuine push gate."""
    workflow_path = smoke.REPO / ".github/workflows/book-local-cpu-smoke.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["public-synthetic-only-cpu"]["steps"]
    commands = (
        "python scripts/book_local_cpu_smoke.py gate",
        "python -u scripts/book_local_cpu_smoke.py run",
        "python -u scripts/book_local_cpu_smoke.py replay",
        "python -u scripts/book_local_cpu_smoke.py collect",
    )
    for command in commands:
        matches = [step for step in steps if command in str(step.get("run", ""))]
        assert len(matches) == 1, command
        assert matches[0].get("env", {}).get("SMOKE_PUSH_BEFORE_SHA") == "${{ github.event.before }}"


@pytest.mark.parametrize("request_present", [False, True])
def test_workflow_structure_allows_absent_or_present_request_in_isolated_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_present: bool
) -> None:
    """A later request-only trigger must not break ordinary workflow-structure CI."""
    isolated_repo = tmp_path / "isolated-repo"
    relative_workflow = Path(".github/workflows/book-local-cpu-smoke.yml")
    copied_workflow = isolated_repo / relative_workflow
    copied_workflow.parent.mkdir(parents=True)
    copied_workflow.write_bytes((smoke.REPO / relative_workflow).read_bytes())

    request = isolated_repo / smoke.RUN_REQUEST
    if request_present:
        request.parent.mkdir(parents=True)
        request.write_text('{"synthetic_inert_fixture": true}\n', encoding="utf-8")

    monkeypatch.setattr(smoke, "REPO", isolated_repo)
    monkeypatch.setattr(
        smoke, "gate", lambda: pytest.fail("workflow structure must not invoke the run gate")
    )
    test_workflow_github_context_rules_and_dormant_trigger_are_explicit()
    assert request.exists() is request_present


def _sample_progress(root: Path) -> Path:
    progress = (
        root / "private/book-candidates/analysis/progress" / smoke.BOOK_ID / f"{'a' * 64}.json"
    )
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_bytes(b'{"label":"PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK"}\n')
    return progress


def test_legitimate_progress_bytes_shared_by_collector_and_replay_hash(tmp_path: Path) -> None:
    root = tmp_path / "public"
    file = _sample_progress(root)
    expected = hashlib.sha256(file.read_bytes()).hexdigest()
    assert smoke._capture_paths(root) == {str(file.relative_to(root)): expected}
    records = smoke.encoded_evidence(root)
    assert records
    assert records[0]["execution"] == "SYNTHETIC_OR_UNGATED_NO_MODEL"
    assert records[0]["sha256"] == expected


@pytest.mark.parametrize("linked_at", ["root", "root_parent", "private", "analysis", "progress"])
def test_collector_and_replay_reject_symlink_ancestors_outside_root(
    tmp_path: Path, linked_at: str
) -> None:
    root = tmp_path / "public"
    outside = tmp_path / "outside"
    external = _sample_progress(outside)
    assert external.is_file()
    if linked_at == "root":
        root.symlink_to(outside, target_is_directory=True)
    elif linked_at == "root_parent":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        root = alias / "outside"
    elif linked_at == "private":
        root.mkdir()
        (root / "private").symlink_to(outside / "private", target_is_directory=True)
    elif linked_at == "analysis":
        root.joinpath("private/book-candidates").mkdir(parents=True)
        (root / "private/book-candidates/analysis").symlink_to(
            outside / "private/book-candidates/analysis", target_is_directory=True
        )
    else:
        root.joinpath("private/book-candidates/analysis").mkdir(parents=True)
        (root / "private/book-candidates/analysis/progress").symlink_to(
            outside / "private/book-candidates/analysis/progress", target_is_directory=True
        )
    with pytest.raises(smoke.SmokeError, match="linked ancestor"):
        smoke.encoded_evidence(root)
    with pytest.raises(smoke.SmokeError, match="linked ancestor"):
        smoke._capture_paths(root)


def test_collector_and_replay_reject_matching_symlink_leaf(tmp_path: Path) -> None:
    root = tmp_path / "public"
    other = _sample_progress(tmp_path / "outside")
    leaf = other.relative_to(tmp_path / "outside")
    target = root / leaf
    target.parent.mkdir(parents=True)
    target.symlink_to(other)
    for reader in (smoke.encoded_evidence, smoke._capture_paths):
        with pytest.raises(smoke.SmokeError, match="linked ancestor"):
            reader(root)


def test_both_readers_precheck_size_without_unbounded_path_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "public"
    progress = _sample_progress(root)
    progress.write_bytes(b"X" * (smoke.MAX_FILE_BYTES + 1))

    def forbidden_read(_self: Path) -> bytes:
        pytest.fail("oversized file must be rejected without Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    for reader in (smoke.encoded_evidence, smoke._capture_paths):
        with pytest.raises(smoke.SmokeError, match="individual evidence file"):
            reader(root)


def test_evidence_total_limit_is_enforced_before_opening_next_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "public"
    file = _sample_progress(root)
    file.write_bytes(b"x" * 20)
    second = root / "evidence" / "runtime.json"
    second.parent.mkdir(parents=True)
    second.write_bytes(b"y" * 20)
    monkeypatch.setattr(smoke, "MAX_TOTAL_BYTES", 30)
    original_open = os.open
    opens: list[str] = []

    def tracked_open(
        path: str | os.PathLike[str],
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        opens.append(str(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(smoke.os, "open", tracked_open)
    for reader in (smoke.encoded_evidence, smoke._capture_paths):
        opens.clear()
        with pytest.raises(smoke.SmokeError, match="total exceeds"):
            reader(root)
        assert len(opens) == 1


def test_growing_evidence_is_rejected_before_unbounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "public"
    file = _sample_progress(root)
    original_open = os.open
    open_count = 0

    def grow_at_open(
        path: str | os.PathLike[str],
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal open_count
        if Path(path) == file:
            open_count += 1
            extra_fd = original_open(file, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(extra_fd, b"G" * (smoke.MAX_FILE_BYTES + 1))
            finally:
                os.close(extra_fd)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(smoke.os, "open", grow_at_open)
    with pytest.raises(smoke.SmokeError, match="individual evidence file"):
        smoke.encoded_evidence(root)
    assert open_count == 1


@pytest.mark.skipif(os.name == "nt", reason="FIFO check requires a POSIX runner")
def test_nonregular_fifo_is_rejected_without_opening(tmp_path: Path) -> None:
    root = tmp_path / "public"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    fifo = evidence / "runtime.json"
    os.mkfifo(fifo)
    for reader in (smoke.encoded_evidence, smoke._capture_paths):
        with pytest.raises(smoke.SmokeError, match="regular file"):
            reader(root)


def test_ambient_github_summary_never_receives_ungated_synthetic_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    summary = tmp_path / "ambient-gh-summary.md"
    summary.write_text("SENTINEL\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    root = smoke._root()
    root.mkdir()
    smoke.finish(root, 124)
    assert smoke.main(["collect"]) == 0
    assert summary.read_text(encoding="utf-8") == "SENTINEL\n"
    output = capsys.readouterr().out
    assert '"execution": "SYNTHETIC_OR_UNGATED_NO_MODEL"' in output
    assert '"outcome": "NO_VERIFIED_GATED_RUN"' in output
    assert "GATED_PUBLIC_CPU_ATTEMPT" not in output
    assert "REAL_EXECUTION" not in output


def test_public_gated_phase_classification_has_no_fake_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    root = smoke._root()
    root.mkdir()
    attested = {"trigger_sha": "a" * 40, "reviewed_code_sha": "b" * 40}
    monkeypatch.setattr(smoke, "gate", lambda: attested)
    smoke._json(root / "evidence" / "trigger.json", attested)
    assert smoke._evidence_state(root)["outcome"] == "GATED_FIXTURE_OR_PREFLIGHT_STAGE"
    phases = root / "evidence/model-phases.jsonl"
    smoke._write(phases, b'{"phase":"cpu_weight_loading_start"}\n')
    assert (
        smoke._evidence_state(root)["outcome"]
        == "MODEL_LOADING_STARTED_WITHOUT_ACCEPTED_COMPLETION"
    )
    smoke._write(
        phases,
        b'{"phase":"cpu_weight_loading_start"}\n{"phase":"real_generation_start"}\n',
    )
    assert (
        smoke._evidence_state(root)["outcome"] == "GENERATION_STARTED_WITHOUT_ACCEPTED_COMPLETION"
    )
    smoke._json(root / "evidence" / "outcome.json", {"result": {"status": "incomplete"}})
    assert smoke._evidence_state(root)["outcome"] == "INCOMPLETE_OR_INVALID_MODEL_RESULT"
    smoke._json(
        root / "evidence" / "outcome.json",
        {"result": {"status": "complete", "candidate_count": 0}},
    )
    assert smoke._evidence_state(root)["outcome"] == "VALID_EMPTY_COMPLETION_NOT_LESSON_QUALITY"
    smoke._json(
        root / "evidence" / "outcome.json",
        {"result": {"status": "complete", "candidate_count": 1}},
    )
    assert smoke._evidence_state(root)["outcome"] == "VALID_UNREVIEWED_CANDIDATES_NOT_APPROVED"
    smoke.finish(root, 124)
    assert smoke._evidence_state(root)["outcome"] == "PROCESS_TIMED_OUT_NO_ACCEPTED_COMPLETION"


def test_model_phase_logging_is_public_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    marker = tmp_path / "model-phases.jsonl"
    monkeypatch.setenv("CLIPPER_PUBLIC_BOOK_CPU_SMOKE", "1")
    monkeypatch.setenv("CLIPPER_PUBLIC_BOOK_CPU_SMOKE_PHASE_FILE", str(marker))
    scripting._public_cpu_smoke_phase("cpu_weight_loading_start")
    assert json.loads(marker.read_text()) == {"phase": "cpu_weight_loading_start"}
    assert marker.stat().st_mode & 0o777 == 0o600
    assert capsys.readouterr().out.strip() == "PUBLIC_BOOK_CPU_MODEL_PHASE cpu_weight_loading_start"
    with pytest.raises(scripting.NicheVideoError, match="phase evidence path"):
        scripting._public_cpu_smoke_phase("contains unsafe data")


def test_model_free_two_unit_dry_run_in_fresh_process() -> None:
    """Exercise the actual standalone CLI with Hub in offline mode, no model loads."""
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CLIPPER_PUBLIC_BOOK_CPU_SMOKE": "0",
    }
    env.pop("GITHUB_STEP_SUMMARY", None)
    result = subprocess.run(
        [sys.executable, str(smoke.REPO / "scripts/book_local_cpu_smoke.py"), "dry-run"],
        cwd=smoke.REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    prefix = "PUBLIC_BOOK_CPU_SMOKE_DRY_RUN "
    lines = [line for line in result.stdout.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1
    payload = json.loads(lines[0][len(prefix) :])
    assert payload == {
        "label": smoke.LABEL,
        "unit_count": 2,
        "no_inference": True,
    }
    assert "PUBLIC_BOOK_CPU_MODEL_PHASE" not in result.stdout
    assert "PUBLIC_BOOK_CPU_SMOKE_COLLECTION" not in result.stdout

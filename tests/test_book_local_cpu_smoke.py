"""Synthetic-only preparation tests for dormant real Qwen public CPU smoke."""

from __future__ import annotations

import base64
import hashlib
import json
import os
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
    with pytest.raises(smoke.SmokeError, match="linked directory"):
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
    gate_index = next(i for i, step in enumerate(steps) if "request gate" in step["name"])
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
    assert not (smoke.REPO / smoke.RUN_REQUEST).exists()



"""Offline-only contract tests for dormant public synthetic Modal launcher."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import book_modal_cpu_launcher as launcher


SHA = "a" * 40


def _manifest(tmp_path: Path, **overrides: Any) -> tuple[Path, str]:
    data = {
        "schema": "clipper-modal-book-launch-approval-v1",
        "intent": launcher.INTENT,
        "source_sha": SHA,
        "workspace": "expected-workspace",
        "environment": "qualification",
        "credential_route": "explicit-modal-profile",
        "expires_utc": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        "cost_ceiling_usd": "0.05",
        "external_control_id": "independent-ctrl-123",
        "execution_permission": True,
    }
    data.update(overrides)
    path = tmp_path / "out-of-band-not-committed.json"
    raw = json.dumps(data, sort_keys=True).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_no_import_side_effects_and_no_modal_import() -> None:
    checkout = Path(__file__).resolve().parents[1]
    probe = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "tests"))
import book_modal_cpu_launcher
assert "modal" not in sys.modules
assert "clipper" not in sys.modules
print("INERT")
"""
    observed = subprocess.run(
        [sys.executable, "-c", probe, str(checkout)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert observed.returncode == 0, observed.stderr
    assert observed.stdout.strip() == "INERT"


def test_plan_mode_never_imports_sdk_or_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(launcher, "local_source", lambda _root, _sha: None)
    assert launcher.main(["--expected-sha", SHA]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "DORMANT_PLAN_ONLY"
    assert plan["function_limits"]["cpu"] == [1, 1] or plan["function_limits"]["cpu"] == (1, 1)
    assert plan["function_limits"]["memory"] == [512, 512] or plan["function_limits"]["memory"] == (512, 512)
    assert plan["function_limits"]["include_source"] is False
    assert plan["function_limits"]["block_network"] is True
    assert plan["function_limits"]["restrict_modal_access"] is True
    assert plan["function_limits"]["max_containers"] == 1
    assert plan["function_limits"]["retries"] == 0
    assert plan["credential_or_modal_import"] is False


def test_no_approval_blocks_before_client_construction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(launcher, "local_source", lambda _root, _sha: None)
    created: list[str] = []

    def forbidden() -> Any:
        created.append("client")
        raise AssertionError("control-plane construction")

    monkeypatch.setattr(launcher, "ModalControl", forbidden)
    assert launcher.main(["--execute", "--expected-sha", SHA]) == 2
    assert created == []
    assert json.loads(capsys.readouterr().out)["status"] == "DORMANT_LAUNCH_BLOCKED"


@pytest.mark.parametrize("change", [
    {"execution_permission": False},
    {"source_sha": "b" * 40},
    {"workspace": ""},
    {"credential_route": "unknown"},
    {"cost_ceiling_usd": "0.00"},
    {"expires_utc": "2000-01-01T00:00:00+00:00"},
    {"expires_utc": "2999-01-01T00:00:00+00:00"},
])
def test_invalid_external_approval_rejected(tmp_path: Path, change: dict[str, Any]) -> None:
    path, digest = _manifest(tmp_path, **change)
    with pytest.raises(launcher.LaunchBlocked):
        launcher.load_approval(path, digest, SHA)


def test_missing_or_changed_approval_digest_fails_closed(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    with pytest.raises(launcher.LaunchBlocked):
        launcher.load_approval(path, "b" * 64, SHA)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(launcher.LaunchBlocked):
        launcher.load_approval(path, digest, SHA)


def test_workspace_mismatch_blocks_resource_call(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    events: list[str] = []

    class FakeControl:
        def identify(self, _approval: launcher.Approval) -> str:
            events.append("identity-read")
            return "foreign-workspace"

        def run_once(self, _approval: launcher.Approval) -> dict[str, object]:
            events.append("resource-created")
            raise AssertionError("workspace mismatch reached resource phase")

    with pytest.raises(launcher.LaunchBlocked, match="workspace"):
        launcher.execute(SHA, approval, FakeControl(), external_control_verified=True)
    assert events == ["identity-read"]


def test_external_control_requirement_precedes_identity(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    events: list[str] = []

    class FakeControl:
        def identify(self, _approval: launcher.Approval) -> str:
            events.append("client")
            return approval.workspace

        def run_once(self, _approval: launcher.Approval) -> dict[str, object]:
            events.append("resource")
            return {}

    with pytest.raises(launcher.LaunchBlocked, match="independent lifecycle"):
        launcher.execute(SHA, approval, FakeControl(), external_control_verified=False)
    assert events == []


def test_approved_fake_control_does_not_import_modal(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    events: list[str] = []

    class FakeControl:
        def identify(self, a: launcher.Approval) -> str:
            events.append("identity-read")
            return a.workspace

        def run_once(self, a: launcher.Approval) -> dict[str, object]:
            events.append("run-called-inert-fake")
            return {"source_sha": a.source_sha, "test_double": True}

    output = launcher.execute(SHA, approval, FakeControl(), external_control_verified=True)
    assert output["test_double"] is True
    assert events == ["identity-read", "run-called-inert-fake"]


def test_local_source_fails_on_foreign_or_dirty_source(monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = Path(launcher.__file__).resolve().parents[1]
    calls: list[tuple[str, ...]] = []

    def fake_git(_checkout: Path, *args: str) -> str:
        calls.append(args)
        if "rev-parse" in args:
            return SHA
        return " M tests/book_modal_cpu_launcher.py"

    monkeypatch.setattr(launcher, "_git", fake_git)
    with pytest.raises(launcher.LaunchBlocked, match="clean"):
        launcher.local_source(checkout, SHA)
    assert calls == [("rev-parse", "HEAD"), ("status", "--porcelain=v1", "--untracked-files=all")]


def test_child_function_uses_only_synthetic_cli_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = Path("/opt/clipper")
    fake_payload = {
        "source_sha": SHA, "status": "SYNTHETIC_CONTRACT_REHEARSAL_PASSED",
        "source_pin": "VERIFIED_CLEAN_LOCAL_GIT_HEAD",
        "publish_eligible": False,
        "tracks": {"pause": {"status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"}},
        "zero_call_evidence": {"attempts": []},
        "artifact_sha256": {"track.json": "b" * 64},
        "elapsed_seconds": 0.2, "peak_rss_kib_platform": 32000,
        "real_model_calls": 0, "network_calls": 0, "publication_calls": 0,
    }
    calls: list[dict[str, Any]] = []

    def inert_subprocess(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append({"argv": argv, "kwargs": kwargs})
        return SimpleNamespace(returncode=0, stdout=json.dumps(fake_payload))

    monkeypatch.setattr(subprocess, "run", inert_subprocess)
    output = launcher._remote_child(SHA)
    assert output["publish_eligible"] is False
    assert len(calls) == 1
    assert calls[0]["argv"][1] == str(checkout / "tests" / "book_public_rehearsal.py")
    assert calls[0]["kwargs"]["env"]["PYTHONPATH"] == "/opt/clipper/src:/opt/clipper/tests"
    assert "MODAL_TOKEN_SECRET" not in calls[0]["kwargs"]["env"]
    assert output["real_model_calls"] == 0


def _valid_child_output() -> dict[str, object]:
    return {
        "source_sha": SHA,
        "status": "SYNTHETIC_CONTRACT_REHEARSAL_PASSED",
        "source_pin": "VERIFIED_CLEAN_LOCAL_GIT_HEAD",
        "publish_eligible": False,
        "tracks": {
            "pause": {"status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"},
            "synthetic_test_review": {
                "status": "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED",
                "publish_eligible": False,
            },
        },
        "zero_call_evidence": {"attempts": []},
        "real_model_calls": 0,
        "network_calls": 0,
        "publication_calls": 0,
    }


def test_validated_result_requires_both_tracks_pin_and_empty_attempts() -> None:
    launcher._validate_child_report(_valid_child_output(), SHA)
    for mutation in (
        {"source_pin": "EXPLICIT_LOCAL_GIT_CHECK_REQUIRED_FOR_CLI"},
        {"zero_call_evidence": {"attempts": ["model"]}},
        {"tracks": {"pause": {"status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"}}},
    ):
        changed = {**_valid_child_output(), **mutation}
        with pytest.raises(launcher.LaunchBlocked):
            launcher._validate_child_report(changed, SHA)


def test_actual_run_remote_inert_sdk_stages_and_cleanup(tmp_path: Path) -> None:
    approval_file, digest = _manifest(tmp_path)
    approval = launcher.load_approval(approval_file, digest, SHA)
    events: list[str] = []
    kwargs_seen: list[dict[str, object]] = []

    class Image:
        @staticmethod
        def debian_slim(**_kw: object) -> Image:
            events.append("image")
            return Image()

        def apt_install(self, *_args: object) -> Image:
            return self

        def pip_install(self, *_args: object) -> Image:
            return self

        def run_commands(self, *_args: object) -> Image:
            events.append("clone-prepared")
            return self

        def env(self, *_args: object) -> Image:
            return self

    class Call:
        object_id = "fc-synthetic-1"

        def get(self, **_kw: object) -> dict[str, object]:
            events.append("get")
            return _valid_child_output()

        def cancel(self, **_kw: object) -> None:
            events.append("cancel")

    class Function:
        def spawn(self, *_args: object) -> Call:
            events.append("spawn")
            return Call()

    class App:
        app_id = "ap-synthetic-1"

        def __init__(self, **_kw: object) -> None:
            events.append("app")

        def function(self, **kw: object) -> object:
            kwargs_seen.append(kw)
            return lambda _fn: Function()

        def run(self, **kw: object) -> App:
            assert kw["environment_name"] == approval.environment
            events.append("app-run")
            return self

        def __enter__(self) -> App:
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("app-exit")

    controller = launcher.ModalControl()
    controller._modal = SimpleNamespace(Image=Image, App=App)
    controller._client = object()
    result = controller.run_remote(approval, lambda phase, item: events.append(phase))
    assert result["app_id"] == "ap-synthetic-1"
    assert result["call_id"] == "fc-synthetic-1"
    assert events.index("APP_ID") < events.index("SPAWN_BEGIN")
    assert events.index("CALL_ID") < events.index("GET_BEGIN")
    assert events.index("cancel") < events.index("app-exit")
    assert kwargs_seen[0]["cpu"] == (1, 1)
    assert kwargs_seen[0]["memory"] == (512, 512)
    assert kwargs_seen[0]["include_source"] is False
    assert kwargs_seen[0]["block_network"] is True


def test_actual_run_remote_call_failure_records_cancel_and_never_succeeds(
    tmp_path: Path,
) -> None:
    approval_path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(approval_path, digest, SHA)
    events: list[str] = []

    class Image:
        @staticmethod
        def debian_slim(**_kw: object) -> Image:
            return Image()

        def apt_install(self, *_args: object) -> Image:
            return self

        def pip_install(self, *_args: object) -> Image:
            return self

        def run_commands(self, *_args: object) -> Image:
            return self

        def env(self, *_args: object) -> Image:
            return self

    class Call:
        object_id = "fc-synthetic-2"

        def get(self, **_kw: object) -> None:
            raise RuntimeError("synthetic get failure")

        def cancel(self, **_kw: object) -> None:
            raise RuntimeError("synthetic cancel failure")

    class Function:
        def spawn(self, *_args: object) -> Call:
            return Call()

    class App:
        app_id = "ap-synthetic-2"

        def __init__(self, **_kw: object) -> None:
            pass

        def function(self, **_kw: object) -> object:
            return lambda _fn: Function()

        def run(self, **_kw: object) -> App:
            return self

        def __enter__(self) -> App:
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("app-exit")

    control = launcher.ModalControl()
    control._modal = SimpleNamespace(Image=Image, App=App)
    control._client = object()
    with pytest.raises(RuntimeError, match="synthetic get failure"):
        control.run_remote(approval, lambda phase, _v: events.append(phase))
    assert "APP_ID" in events
    assert "CALL_ID" in events
    assert "CALL_CANCEL_ERROR" in events
    assert "app-exit" in events


def test_stop_and_list_terminal_evidence_requires_exact_id_zero_tasks() -> None:
    commands: list[list[str]] = []

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        commands.append(argv)
        if "stop" in argv:
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"app_id": "ap-synthetic-1", "state": "stopped", "tasks": "0"}]))

    result = launcher._stop_and_observe(
        "ap-synthetic-1", "qualification", time.monotonic() + 10,
        runner=fake_cli,
    )
    assert result == (True, [])
    assert commands[0][-1] == "ap-synthetic-1"
    assert "--env" in commands[0] and "qualification" in commands[0]

    def failed_stop(argv: list[str], **_kw: object) -> SimpleNamespace:
        if "stop" in argv:
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"app_id": "ap-synthetic-1", "state": "stopped", "tasks": "0"}]))

    confirmed, errors = launcher._stop_and_observe(
        "ap-synthetic-1", "qualification", time.monotonic() + 10,
        runner=failed_stop,
    )
    assert confirmed is False
    assert "stop:nonzero" in errors


def test_build_stall_retains_unknown_cleanup_without_app_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)

    class StuckProcess:
        pid = 23456
        returncode = -9

        def wait(self, **_kw: object) -> None:
            raise subprocess.TimeoutExpired("synthetic", 1)

        def poll(self) -> None:
            return None

    monkeypatch.setattr(launcher.os, "killpg", lambda *_args: None)
    outcome = launcher._bounded_run(
        approval, path, digest, deadline_seconds=1,
        popen=lambda *_args, **_kw: StuckProcess(),
        runner=lambda *_args, **_kw: (_ for _ in ()).throw(
            AssertionError("no App ID: cannot issue stop against guessed id")
        ),
    )
    assert outcome["status"] == "BLOCKED"
    assert outcome["provider_removal"] == "UNKNOWN"
    assert "APP_ID_UNKNOWN" in outcome["cleanup_errors"]
    assert "CONTROL_PLANE_TIMEOUT" in outcome["cleanup_errors"]


def test_stop_failure_retains_exact_app_id_and_blocks(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    recorded: list[list[str]] = []

    class CompletedProcess:
        pid = 45678
        returncode = 0

        def wait(self, **_kw: object) -> int:
            return 0

        def poll(self) -> int:
            return 0

    def fake_popen(argv: list[str], **_kw: object) -> CompletedProcess:
        journal = Path(argv[argv.index("--journal") + 1])
        result_file = Path(argv[argv.index("--result-file") + 1])
        launcher._journal(journal, "APP_RUN_BEGIN")
        launcher._journal(journal, "APP_ID", "ap-synthetic-known")
        launcher._journal(journal, "CALL_ID", "fc-synthetic-known")
        result_file.write_text(json.dumps({
            "rehearsal": _valid_child_output(),
            "app_id": "ap-synthetic-known", "call_id": "fc-synthetic-known",
        }))
        return CompletedProcess()

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        recorded.append(argv)
        if "stop" in argv:
            return SimpleNamespace(returncode=1, stdout="")
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"app_id": "ap-synthetic-known", "state": "stopped", "tasks": "0"}]))

    outcome = launcher._bounded_run(
        approval, path, digest, deadline_seconds=10,
        popen=fake_popen, runner=fake_cli,
    )
    assert outcome["status"] == "BLOCKED"
    assert outcome["app_id"] == "ap-synthetic-known"
    assert outcome["provider_removal"] == "UNKNOWN"
    assert "stop:nonzero" in outcome["cleanup_errors"]
    assert len(recorded) == 2


def test_live_cli_cannot_spend_on_receipt_string_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, digest = _manifest(tmp_path)
    monkeypatch.setattr(launcher, "local_source", lambda _root, _sha: None)
    monkeypatch.setenv("MODAL_PROFILE", "approved-explicit-profile")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    created: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        created.append("controller")
        raise AssertionError("budget gate incorrectly permitted client")

    monkeypatch.setattr(launcher, "BoundedModalControl", forbidden)
    assert launcher.main([
        "--execute", "--expected-sha", SHA,
        "--approval-file", str(path),
        "--approval-sha256", digest,
        "--independent-control-receipt", "independent-ctrl-123",
    ]) == 2
    assert created == []
    assert json.loads(capsys.readouterr().out)["status"] == "DORMANT_LAUNCH_BLOCKED"


def test_internal_worker_cannot_bypass_budget_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, digest = _manifest(tmp_path)
    monkeypatch.setattr(launcher, "local_source", lambda _root, _sha: None)
    called: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        called.append("sdk")
        raise AssertionError("internal worker bypassed budget")

    monkeypatch.setattr(launcher, "_internal_worker", forbidden)
    assert launcher.main([
        "--internal-worker",
        "--expected-sha", SHA,
        "--approval-file", str(path),
        "--approval-sha256", digest,
        "--journal", str(tmp_path / "journal.jsonl"),
        "--result-file", str(tmp_path / "result.json"),
    ]) == 2
    assert called == []
    assert json.loads(capsys.readouterr().out)["status"] == "DORMANT_LAUNCH_BLOCKED"


def test_list_unknown_and_mismatched_states_never_claim_stop() -> None:
    for rows in (
        [],
        [{"app_id": "ap-other", "state": "stopped", "tasks": "0"}],
        [{"app_id": "ap-a", "state": "stopping...", "tasks": "0"}],
        [{"app_id": "ap-a", "state": "stopped", "tasks": "1"}],
        [{"app_id": "ap-a", "state": "stopped"}],
        [{"app_id": "ap-a", "state": "stopped", "tasks": "0"}] * 2,
        {"app_id": "ap-a", "state": "stopped", "tasks": "0"},
    ):
        def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
            if "stop" in argv:
                return SimpleNamespace(returncode=0, stdout="")
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows))

        confirmed, errors = launcher._stop_and_observe(
            "ap-a", "qualification", time.monotonic() + 10, runner=fake_cli
        )
        assert confirmed is False
        assert errors


def test_stop_preserves_prior_cancel_and_journal_failures(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)

    class Finished:
        returncode = 2
        pid = 9992

        def poll(self) -> int:
            return 2

        def wait(self, **_kw: object) -> int:
            return 2

    def fake_popen(argv: list[str], **_kw: object) -> Finished:
        ledger = Path(argv[argv.index("--journal") + 1])
        launcher._journal(ledger, "APP_ID", "ap-a")
        launcher._journal(ledger, "CALL_CANCEL_ERROR", "RuntimeError")
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write("not json\\n")
        return Finished()

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        if "stop" in argv:
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"app_id": "ap-a", "state": "stopped", "tasks": "0"}]),
        )

    output = launcher._bounded_run(
        approval, path, digest, deadline_seconds=10,
        popen=fake_popen, runner=fake_cli,
    )
    assert output["status"] == "BLOCKED"
    assert "CALL_CANCEL_ERROR" in output["cleanup_errors"]
    assert "INVALID_JOURNAL" in output["cleanup_errors"]
    assert "WORKER_NONZERO_OR_TERMINATED" in output["cleanup_errors"]
    assert "NO_RESULT" in output["cleanup_errors"]
    assert output["app_id"] == "ap-a"
    assert output["provider_removal"] == "STOPPED_CONFIRMED_BY_LIST"


def test_parent_retains_bounded_diagnostics_only(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    sensitive = b"inert-private-diagnostic-test-marker"

    class Finished:
        pid = 4242
        returncode = 2

        def poll(self) -> int:
            return 2

        def wait(self, **_kw: object) -> int:
            return 2

    def fake_popen(argv: list[str], **kw: object) -> Finished:
        kw["stdout"].write(sensitive)
        kw["stderr"].write(sensitive)
        return Finished()

    output = launcher._bounded_run(
        approval, path, digest, deadline_seconds=10,
        popen=fake_popen,
    )
    assert output["status"] == "BLOCKED"
    for part in ("stdout", "stderr"):
        evidence = output["worker_diagnostics"][part]
        assert evidence["size_bytes"] == len(sensitive)
        assert evidence["bounded_prefix_sha256"] == hashlib.sha256(sensitive).hexdigest()
    assert sensitive.decode() not in json.dumps(output)

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

import book_modal_cpu_launcher as launcher
import pytest

SHA = "a" * 40


def _manifest(tmp_path: Path, **overrides: Any) -> tuple[Path, str]:
    data = {
        "schema": "clipper-modal-public-synthetic-approval-v2",
        "intent": launcher.INTENT,
        "source_sha": SHA,
        "workspace": "mahmoud-bayoumi-mm",
        "environment": "qualification",
        "credential_route": "actions-existing-secret-route",
        "expires_utc": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        "included_credit_only": True,
        "paid_overage_authorized": False,
        "owner_request_ref": "2026-11-25T11:09:05+02:00",
        "credit_confirmation_ref": "2026-11-25T15:00:33+02:00",
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
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
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
    assert plan["function_limits"]["memory"] == [512, 512] or plan["function_limits"]["memory"] == (
        512,
        512,
    )
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


@pytest.mark.parametrize(
    "change",
    [
        {"execution_permission": False},
        {"source_sha": "b" * 40},
        {"workspace": ""},
        {"workspace": "wrong-workspace"},
        {"credential_route": "unknown"},
        {"included_credit_only": False},
        {"paid_overage_authorized": True},
        {"owner_request_ref": "generic"},
        {"credit_confirmation_ref": "generic"},
        {"intent": "PRIVATE_BOOK_INFERENCE"},
        {"expires_utc": "2000-01-01T00:00:00+00:00"},
        {"expires_utc": "2999-01-01T00:00:00+00:00"},
    ],
)
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
        "source_sha": SHA,
        "status": "SYNTHETIC_CONTRACT_REHEARSAL_PASSED",
        "source_pin": "VERIFIED_CLEAN_LOCAL_GIT_HEAD",
        "publish_eligible": False,
        "tracks": {"pause": {"status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"}},
        "zero_call_evidence": {"attempts": []},
        "artifact_sha256": {"track.json": "b" * 64},
        "elapsed_seconds": 0.2,
        "peak_rss_kib_platform": 32000,
        "real_model_calls": 0,
        "network_calls": 0,
        "publication_calls": 0,
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
        app_id = "ap-AAAAAAAAAAAAAAAAAAAAAA"

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
    assert result["app_id"] == "ap-AAAAAAAAAAAAAAAAAAAAAA"
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
        app_id = "ap-BBBBBBBBBBBBBBBBBBBBBB"

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
    app_id = "ap-AAAAAAAAAAAAAAAAAAAAAA"
    commands: list[list[str]] = []
    responses = iter(
        [
            [{"app_id": app_id, "state": "ephemeral", "tasks": "1"}],
            [{"app_id": app_id, "state": "stopped", "tasks": "0"}],
        ]
    )

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        commands.append(argv)
        if "stop" in argv:
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout=json.dumps(next(responses)))

    assert launcher._stop_and_observe(
        app_id, "qualification", time.monotonic() + 10, runner=fake_cli
    ) == (True, [])
    list_cmd = ["modal", "app", "list", "--json", "--env", "qualification"]
    stop_cmd = ["modal", "app", "stop", "-y", "--env", "qualification", app_id]
    assert commands == [list_cmd, stop_cmd, list_cmd]

    commands.clear()

    def failed_stop(argv: list[str], **_kw: object) -> SimpleNamespace:
        commands.append(argv)
        if "stop" in argv:
            return SimpleNamespace(returncode=1, stdout="")
        state = "ephemeral" if len(commands) == 1 else "stopped"
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"app_id": app_id, "state": state, "tasks": "0"}])
        )

    confirmed, errors = launcher._stop_and_observe(
        app_id, "qualification", time.monotonic() + 10, runner=failed_stop
    )
    assert confirmed is False and "stop:nonzero" in errors
    assert commands == [list_cmd, stop_cmd, list_cmd]


def test_build_stall_retains_unknown_cleanup_without_app_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        approval,
        path,
        digest,
        deadline_seconds=1,
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
        launcher._journal(journal, "APP_ID", "ap-CCCCCCCCCCCCCCCCCCCCCC")
        launcher._journal(journal, "CALL_ID", "fc-synthetic-known")
        result_file.write_text(
            json.dumps(
                {
                    "rehearsal": _valid_child_output(),
                    "app_id": "ap-CCCCCCCCCCCCCCCCCCCCCC",
                    "call_id": "fc-synthetic-known",
                }
            )
        )
        return CompletedProcess()

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        recorded.append(argv)
        if "stop" in argv:
            return SimpleNamespace(returncode=1, stdout="")
        state = "ephemeral" if len(recorded) == 1 else "stopped"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [{"app_id": "ap-CCCCCCCCCCCCCCCCCCCCCC", "state": state, "tasks": "0"}]
            ),
        )

    outcome = launcher._bounded_run(
        approval,
        path,
        digest,
        deadline_seconds=10,
        popen=fake_popen,
        runner=fake_cli,
    )
    assert outcome["status"] == "BLOCKED"
    assert outcome["app_id"] == "ap-CCCCCCCCCCCCCCCCCCCCCC"
    assert outcome["provider_removal"] == "UNKNOWN"
    assert "stop:nonzero" in outcome["cleanup_errors"]
    assert len(recorded) == 3


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
    assert (
        launcher.main(
            [
                "--execute",
                "--expected-sha",
                SHA,
                "--approval-file",
                str(path),
                "--approval-sha256",
                digest,
                "--independent-control-receipt",
                "independent-ctrl-123",
            ]
        )
        == 2
    )
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
    assert (
        launcher.main(
            [
                "--internal-worker",
                "--expected-sha",
                SHA,
                "--approval-file",
                str(path),
                "--approval-sha256",
                digest,
                "--journal",
                str(tmp_path / "journal.jsonl"),
                "--result-file",
                str(tmp_path / "result.json"),
            ]
        )
        == 2
    )
    assert called == []
    assert json.loads(capsys.readouterr().out)["status"] == "DORMANT_LAUNCH_BLOCKED"


def test_list_unknown_and_mismatched_states_never_claim_stop() -> None:
    for rows in (
        [],
        [{"app_id": "ap-EEEEEEEEEEEEEEEEEEEEEE", "state": "stopped", "tasks": "0"}],
        [{"app_id": "ap-DDDDDDDDDDDDDDDDDDDDDD", "state": "stopping...", "tasks": "0"}],
        [{"app_id": "ap-DDDDDDDDDDDDDDDDDDDDDD", "state": "stopped", "tasks": "1"}],
        [{"app_id": "ap-DDDDDDDDDDDDDDDDDDDDDD", "state": "stopped"}],
        [{"app_id": "ap-DDDDDDDDDDDDDDDDDDDDDD", "state": "stopped", "tasks": "0"}] * 2,
        {"app_id": "ap-DDDDDDDDDDDDDDDDDDDDDD", "state": "stopped", "tasks": "0"},
    ):

        def fake_cli(argv: list[str], rows: object = rows, **_kw: object) -> SimpleNamespace:
            if "stop" in argv:
                return SimpleNamespace(returncode=0, stdout="")
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows))

        confirmed, errors = launcher._stop_and_observe(
            "ap-DDDDDDDDDDDDDDDDDDDDDD", "qualification", time.monotonic() + 10, runner=fake_cli
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
        launcher._journal(ledger, "APP_ID", "ap-DDDDDDDDDDDDDDDDDDDDDD")
        launcher._journal(ledger, "CALL_CANCEL_ERROR", "RuntimeError")
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write("not json\\n")
        return Finished()

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        if "stop" in argv:
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [{"app_id": "ap-DDDDDDDDDDDDDDDDDDDDDD", "state": "stopped", "tasks": "0"}]
            ),
        )

    output = launcher._bounded_run(
        approval,
        path,
        digest,
        deadline_seconds=10,
        popen=fake_popen,
        runner=fake_cli,
    )
    assert output["status"] == "BLOCKED"
    assert "CALL_CANCEL_ERROR" in output["cleanup_errors"]
    assert "INVALID_JOURNAL" in output["cleanup_errors"]
    assert "WORKER_NONZERO_OR_TERMINATED" in output["cleanup_errors"]
    assert "NO_RESULT" in output["cleanup_errors"]
    assert output["app_id"] == "ap-DDDDDDDDDDDDDDDDDDDDDD"
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
        approval,
        path,
        digest,
        deadline_seconds=10,
        popen=fake_popen,
    )
    assert output["status"] == "BLOCKED"
    for part in ("stdout", "stderr"):
        evidence = output["worker_diagnostics"][part]
        assert evidence["size_bytes"] == len(sensitive)
        assert evidence["bounded_prefix_sha256"] == hashlib.sha256(sensitive).hexdigest()
    assert sensitive.decode() not in json.dumps(output)


@pytest.mark.parametrize(
    "invalid",
    [
        "ap-a",
        "ap-" + "a" * 21,
        "ap-" + "a" * 23,
        "ap-" + "a" * 10 + "-" + "a" * 11,
        "deployment-name",
    ],
)
def test_cleanup_rejects_nonexact_app_id_before_any_cli(invalid: str) -> None:
    commands: list[list[str]] = []

    def forbidden(argv: list[str], **_kw: object) -> SimpleNamespace:
        commands.append(argv)
        raise AssertionError("non-exact ID reached provider CLI")

    assert launcher._stop_and_observe(
        invalid, "qualification", time.monotonic() + 10, runner=forbidden
    ) == (False, ["INVALID_APP_ID"])
    assert commands == []


def test_journal_invalid_app_id_never_becomes_actionable(tmp_path: Path) -> None:
    ledger = tmp_path / "journal.jsonl"
    launcher._journal(ledger, "APP_RUN_BEGIN")
    launcher._journal(ledger, "APP_ID", "ap-a")
    app_id, phases = launcher._read_journal(ledger)
    assert app_id is None
    assert "INVALID_APP_ID" in phases


def test_approval_parses_same_digest_bound_bytes_during_atomic_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _manifest(tmp_path)
    original_open = Path.open
    first_read = False

    class SwappingFile:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def __enter__(self) -> SwappingFile:
            self.stream.__enter__()
            return self

        def __exit__(self, *args: object) -> Any:
            return self.stream.__exit__(*args)

        def read(self, *args: object) -> bytes:
            nonlocal first_read
            raw: bytes = self.stream.read(*args)
            if not first_read:
                first_read = True
                changed = tmp_path / "replaced.json"
                changed.write_text(json.dumps({"intent": "OTHER_OPERATION"}))
                changed.replace(path)
            return raw

    def swapping_open(self: Path, mode: str = "r", **kw: object) -> Any:
        stream = original_open(self, mode, **kw)
        if self == path and mode == "rb":
            return SwappingFile(stream)
        return stream

    monkeypatch.setattr(Path, "open", swapping_open)
    approval = launcher.load_approval(path, digest, SHA)
    assert first_read
    assert approval.workspace == "mahmoud-bayoumi-mm"
    assert json.loads(path.read_text()) == {"intent": "OTHER_OPERATION"}


def test_approval_oversize_and_nondict_fail_closed(tmp_path: Path) -> None:
    for raw in (b" " * 8193, b"[]", b"{not json"):
        path = tmp_path / "malformed.json"
        path.write_bytes(raw)
        with pytest.raises(launcher.LaunchBlocked):
            launcher.load_approval(path, hashlib.sha256(raw).hexdigest(), SHA)


def test_already_stopped_app_skips_stop_even_if_stop_would_exit_nonzero() -> None:
    app_id = "ap-" + "A" * 22
    calls: list[list[str]] = []

    def list_only(argv: list[str], **_kw: object) -> SimpleNamespace:
        calls.append(argv)
        assert argv[2] == "list"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"app_id": app_id, "state": "stopped", "tasks": "0"}]),
        )

    assert launcher._stop_and_observe(
        app_id, "qualification", time.monotonic() + 10, runner=list_only
    ) == (True, [])
    assert calls == [["modal", "app", "list", "--json", "--env", "qualification"]]


def test_keyboard_interrupt_reconciles_exact_id_and_propagates_evidence(tmp_path: Path) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    app_id = "ap-" + "A" * 22
    calls: list[list[str]] = []

    class Interrupted:
        pid = 31349
        returncode = 130

        def wait(self, **_kw: object) -> int:
            raise KeyboardInterrupt

        def poll(self) -> int:
            return 130

    def fake_popen(argv: list[str], **_kw: object) -> Interrupted:
        ledger = Path(argv[argv.index("--journal") + 1])
        launcher._journal(ledger, "APP_RUN_BEGIN")
        launcher._journal(ledger, "APP_ID", app_id)
        return Interrupted()

    def list_only(argv: list[str], **_kw: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"app_id": app_id, "state": "stopped", "tasks": "0"}]),
        )

    with pytest.raises(launcher.LaunchInterrupted) as interrupted:
        launcher._bounded_run(
            approval,
            path,
            digest,
            deadline_seconds=10,
            popen=fake_popen,
            runner=list_only,
        )
    report = interrupted.value.evidence
    assert report["status"] == "BLOCKED"
    assert report["initiator_interrupted"] is True
    assert report["app_id"] == app_id
    assert "INITIATOR_KEYBOARD_INTERRUPT" in report["cleanup_errors"]
    assert report["provider_removal"] == "STOPPED_CONFIRMED_BY_LIST"
    assert len(calls) == 1 and calls[0][2] == "list"


def test_real_adapter_identity_never_runs_outside_bounded_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    adapter = launcher.BoundedModalControl(path, digest)
    attempts: list[str] = []

    def guarded_identity(_approval: launcher.Approval) -> str:
        attempts.append("unbounded-identity")
        raise AssertionError("workspace verification escaped deadline")

    monkeypatch.setattr(adapter, "identify", guarded_identity)
    monkeypatch.setattr(
        adapter,
        "run_once",
        lambda _approval: {"status": "BLOCKED", "provider_removal": "UNKNOWN"},
    )
    result = launcher.execute(SHA, approval, adapter, external_control_verified=True)
    assert result["status"] == "BLOCKED"
    assert attempts == []


def test_identity_stage_stall_is_bounded_and_never_starts_cleanup_without_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    events: list[str] = []

    class StuckIdentity:
        pid = 34349
        returncode = -9

        def wait(self, **_kw: object) -> None:
            raise subprocess.TimeoutExpired("fake-identity", 1)

        def poll(self) -> None:
            return None

    def fake_popen(argv: list[str], **_kw: object) -> StuckIdentity:
        ledger = Path(argv[argv.index("--journal") + 1])
        launcher._journal(ledger, "WORKSPACE_VERIFY_BEGIN")
        events.append("worker-started")
        return StuckIdentity()

    monkeypatch.setattr(launcher.os, "killpg", lambda *_args: events.append("killed"))
    report = launcher._bounded_run(
        approval,
        path,
        digest,
        deadline_seconds=1,
        popen=fake_popen,
        runner=lambda *_args, **_kw: (_ for _ in ()).throw(
            AssertionError("no ID: no provider stop")
        ),
    )
    assert report["status"] == "BLOCKED"
    assert report["provider_removal"] == "UNKNOWN"
    assert "WORKSPACE_VERIFY_BEGIN" in report["observed_phases"]
    assert "APP_ID_UNKNOWN" in report["cleanup_errors"]
    assert events == ["worker-started", "killed"]


def test_internal_worker_rechecks_approval_expiry_after_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    calls: list[str] = []

    class FakeController:
        def identify(self, _approval: launcher.Approval) -> str:
            calls.append("identity")
            return approval.workspace

        def run_remote(self, *_args: object) -> None:
            calls.append("resource")
            raise AssertionError("expired approval reached resource path")

    class ExpiredDateTime:
        @staticmethod
        def now(_tz: object) -> datetime:
            return approval.expires_utc + timedelta(seconds=1)

    monkeypatch.setattr(launcher, "ModalControl", FakeController)
    monkeypatch.setattr(launcher, "datetime", ExpiredDateTime)
    ledger, result = tmp_path / "ledger.jsonl", tmp_path / "result.json"
    assert launcher._internal_worker(approval, ledger, result) == 2
    assert calls == ["identity"]
    assert not result.exists()
    assert "WORKER_ERROR" in launcher._read_journal(ledger)[1]


def test_journal_valid_app_id_then_nonobject_json_keeps_cleanup_and_interrupt(
    tmp_path: Path,
) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    app_id = "ap-" + "A" * 22
    commands: list[list[str]] = []

    class Interrupted:
        pid = 23232
        returncode = 130

        def wait(self, **_kw: object) -> int:
            raise KeyboardInterrupt

        def poll(self) -> int:
            return 130

    def fake_popen(argv: list[str], **_kw: object) -> Interrupted:
        ledger = Path(argv[argv.index("--journal") + 1])
        launcher._journal(ledger, "APP_ID", app_id)
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write("[]\nnull\n123\n")
        return Interrupted()

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        commands.append(argv)
        if "stop" in argv:
            return SimpleNamespace(returncode=0, stdout="")
        state = "ephemeral" if len(commands) == 1 else "stopped"
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"app_id": app_id, "state": state, "tasks": "0"}])
        )

    with pytest.raises(launcher.LaunchInterrupted) as interrupted:
        launcher._bounded_run(
            approval,
            path,
            digest,
            deadline_seconds=10,
            popen=fake_popen,
            runner=fake_cli,
        )
    evidence = interrupted.value.evidence
    assert evidence["app_id"] == app_id
    assert evidence["initiator_interrupted"] is True
    assert evidence["status"] == "BLOCKED"
    assert "INVALID_JOURNAL" in evidence["cleanup_errors"]
    assert "INITIATOR_KEYBOARD_INTERRUPT" in evidence["cleanup_errors"]
    assert [command[2] for command in commands] == ["list", "stop", "list"]


@pytest.mark.parametrize("initial_kind", ["nonzero", "invalid_json", "missing"])
def test_initial_inconclusive_list_still_stops_exact_id_and_retains_uncertainty(
    initial_kind: str,
) -> None:
    app_id = "ap-" + "A" * 22
    commands: list[list[str]] = []

    def fake_cli(argv: list[str], **_kw: object) -> SimpleNamespace:
        commands.append(argv)
        if argv[2] == "stop":
            return SimpleNamespace(returncode=0, stdout="")
        if len(commands) == 1:
            if initial_kind == "nonzero":
                return SimpleNamespace(returncode=1, stdout="")
            if initial_kind == "invalid_json":
                return SimpleNamespace(returncode=0, stdout="{invalid")
            return SimpleNamespace(returncode=0, stdout="[]")
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"app_id": app_id, "state": "stopped", "tasks": "0"}])
        )

    confirmed, errors = launcher._stop_and_observe(
        app_id, "qualification", time.monotonic() + 10, runner=fake_cli
    )
    assert confirmed is False
    assert errors
    assert commands == [
        ["modal", "app", "list", "--json", "--env", "qualification"],
        ["modal", "app", "stop", "-y", "--env", "qualification", app_id],
        ["modal", "app", "list", "--json", "--env", "qualification"],
    ]


def test_reached_main_scoped_action_success_blocked_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, digest = _manifest(tmp_path)
    monkeypatch.setattr(launcher, "local_source", lambda *_args: None)
    monkeypatch.setenv("MODAL_TOKEN_ID", "inert-test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "inert-test-secret")
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    seen: list[str] = []

    class FakeBounded:
        def __init__(self, _path: Path, _digest: str) -> None:
            seen.append("bounded-parent")

        def run_once(self, _approval: launcher.Approval) -> dict[str, object]:
            seen.append("worker-run")
            evidence = _valid_child_output()
            evidence["artifact_sha256"] = {"test.json": "b" * 64}
            evidence["unapproved_field"] = "marker-not-for-output"
            return {
                "status": "SYNTHETIC_REHEARSAL_CONFIRMED",
                "provider_removal": "STOPPED_CONFIRMED_BY_LIST",
                "app_id": "ap-" + "A" * 22,
                "observed_phases": [],
                "cleanup_errors": [],
                "initiator_interrupted": False,
                "result": {"rehearsal": evidence, "unapproved_field": "marker-not-for-output"},
                "unapproved_field": "marker-not-for-output",
            }

    # Keep the real scoped execute, substitute only the worker run.
    monkeypatch.setattr(launcher.BoundedModalControl, "run_once", FakeBounded.run_once)
    original_init = launcher.BoundedModalControl.__init__

    def inert_init(
        self: launcher.BoundedModalControl, approval_path: Path, digest_arg: str
    ) -> None:
        seen.append("bounded-parent")
        original_init(self, approval_path, digest_arg)

    monkeypatch.setattr(launcher.BoundedModalControl, "__init__", inert_init)
    args = [
        "--execute",
        "--expected-sha",
        SHA,
        "--approval-file",
        str(path),
        "--approval-sha256",
        digest,
        "--independent-control-receipt",
        "independent-ctrl-123",
    ]
    assert launcher.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "SYNTHETIC_REHEARSAL_CONFIRMED"
    assert "unapproved_field" not in json.dumps(output)
    assert "marker-not-for-output" not in json.dumps(output)
    assert output["rehearsal"]["source_sha"] == SHA
    assert output["rehearsal"]["tracks"]["pause"]["status"] == (
        "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    )
    assert output["rehearsal"]["tracks"]["synthetic_test_review"]["status"] == (
        "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
    )
    assert output["rehearsal"]["zero_call_evidence"] == {"attempts": []}
    assert output["rehearsal"]["real_model_calls"] == 0
    assert output["rehearsal"]["network_calls"] == 0
    assert output["rehearsal"]["publication_calls"] == 0
    expected_inventory = json.dumps(
        {"test.json": "b" * 64}, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    assert output["rehearsal"]["artifact_inventory_count"] == 1
    assert (
        output["rehearsal"]["artifact_inventory_sha256"]
        == hashlib.sha256(expected_inventory).hexdigest()
    )
    assert "artifact_sha256" not in output["rehearsal"]
    assert seen == ["bounded-parent", "worker-run"]

    def blocked(
        self: launcher.BoundedModalControl, _approval: launcher.Approval
    ) -> dict[str, object]:
        return {
            "status": "BLOCKED",
            "provider_removal": "UNKNOWN",
            "cleanup_errors": ["APP_ID_UNKNOWN"],
        }

    monkeypatch.setattr(launcher.BoundedModalControl, "run_once", blocked)
    assert launcher.main(args) == 2
    blocked_output = json.loads(capsys.readouterr().out)
    assert blocked_output["status"] == "BLOCKED"
    assert "rehearsal" not in blocked_output

    def interrupted(
        self: launcher.BoundedModalControl, _approval: launcher.Approval
    ) -> dict[str, object]:
        raise launcher.LaunchInterrupted(
            {"status": "BLOCKED", "cleanup_errors": ["INITIATOR_KEYBOARD_INTERRUPT"]}
        )

    monkeypatch.setattr(launcher.BoundedModalControl, "run_once", interrupted)
    assert launcher.main(args) == 130
    interrupted_output = json.loads(capsys.readouterr().out)
    assert interrupted_output["status"] == "BLOCKED"
    assert "rehearsal" not in interrupted_output


def test_reached_internal_worker_requires_scoped_inputs_before_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, digest = _manifest(tmp_path)
    monkeypatch.setattr(launcher, "local_source", lambda *_args: None)
    monkeypatch.setenv("MODAL_TOKEN_ID", "inert-test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "inert-test-secret")
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    nonce = "a" * 64
    monkeypatch.setenv("CLIPPER_MODAL_WORKER_NONCE", nonce)
    folder = tmp_path / "owner"
    folder.mkdir()
    journal, result = folder / "ids.jsonl", folder / "result.json"
    events: list[str] = []

    def fake_internal(approval: launcher.Approval, j: Path, r: Path) -> int:
        events.append("identity-before-resource")
        assert approval.workspace == "mahmoud-bayoumi-mm"
        assert j == journal and r == result
        return 0

    monkeypatch.setattr(launcher, "_internal_worker", fake_internal)
    args = [
        "--internal-worker",
        "--expected-sha",
        SHA,
        "--approval-file",
        str(path),
        "--approval-sha256",
        digest,
        "--independent-control-receipt",
        "independent-ctrl-123",
        "--journal",
        str(journal),
        "--result-file",
        str(result),
        "--internal-owner-nonce",
        nonce,
    ]
    assert launcher.main(args) == 0
    assert events == ["identity-before-resource"]
    for bad_args in (
        [x for x in args if x != "--internal-owner-nonce" and x != nonce],
        [
            *args[: args.index("--journal")],
            "--journal",
            str(tmp_path / "other.jsonl"),
            *args[args.index("--result-file") :],
        ],
        [*args[:-1], "b" * 64],
    ):
        assert launcher.main(bad_args) == 2
    assert events == ["identity-before-resource"]
    assert "DORMANT_LAUNCH_BLOCKED" in capsys.readouterr().out


def test_worker_identity_exact_workspace_and_unexpired_before_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _manifest(tmp_path)
    approval = launcher.load_approval(path, digest, SHA)
    calls: list[str] = []

    class FakeControl:
        def identify(self, _approval: launcher.Approval) -> str:
            calls.append("identity")
            return approval.workspace

        def run_remote(self, _approval: launcher.Approval, record: Any) -> dict[str, object]:
            calls.append("resource")
            record("RESULT_ACCEPTED", "synthetic")
            return {"rehearsal": {"status": "TEST_ONLY"}}

    monkeypatch.setattr(launcher, "ModalControl", FakeControl)
    journal, result = tmp_path / "ids.jsonl", tmp_path / "result.json"
    assert launcher._internal_worker(approval, journal, result) == 0
    assert calls == ["identity", "resource"]
    assert json.loads(result.read_text())["rehearsal"]["status"] == "TEST_ONLY"

    calls.clear()

    class ForeignControl(FakeControl):
        def identify(self, _approval: launcher.Approval) -> str:
            calls.append("identity")
            return "different-workspace"

    monkeypatch.setattr(launcher, "ModalControl", ForeignControl)
    assert launcher._internal_worker(approval, journal, result) == 2
    assert calls == ["identity"]


def test_scoped_approval_rejects_paid_and_generic_before_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(launcher, "local_source", lambda *_args: None)
    monkeypatch.setenv("MODAL_TOKEN_ID", "inert-test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "inert-test-secret")
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    created: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        created.append("SDK")
        raise AssertionError("provider reached")

    monkeypatch.setattr(launcher, "BoundedModalControl", forbidden)
    for bad in (
        {"paid_overage_authorized": True},
        {"included_credit_only": False},
        {"workspace": "foreign"},
        {"intent": "MODEL_INFERENCE"},
        {"owner_request_ref": "other"},
        {"credit_confirmation_ref": "other"},
        {"credential_route": "explicit-modal-profile"},
        {"expires_utc": "2000-01-01T00:00:00Z"},
    ):
        path, digest = _manifest(tmp_path, **bad)
        assert (
            launcher.main(
                [
                    "--execute",
                    "--expected-sha",
                    SHA,
                    "--approval-file",
                    str(path),
                    "--approval-sha256",
                    digest,
                    "--independent-control-receipt",
                    "independent-ctrl-123",
                ]
            )
            == 2
        )
    assert created == []
    assert "DORMANT_LAUNCH_BLOCKED" in capsys.readouterr().out


def test_actual_public_rehearsal_inventory_sanitized_end_to_end(tmp_path: Path) -> None:
    """Use the real 97-file fixture generator, not a fabricated flat-hash stub."""
    import book_public_rehearsal as rehearsal

    source_sha = launcher._git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD")
    report = rehearsal.rehearse(tmp_path / "public-rehearsal", source_sha)
    report["source_pin"] = "VERIFIED_CLEAN_LOCAL_GIT_HEAD"
    # Nested arbitrary metadata from test producers is not an output permission.
    report["tracks"]["pause"]["unapproved_field"] = {"marker": "never-publish-nested-marker"}
    report["unapproved_field"] = "never-publish-nested-marker"
    inventory_map = report["artifact_sha256"]
    assert isinstance(inventory_map, dict)
    assert len(inventory_map) > 32
    raw_inventory = json.dumps(
        inventory_map, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    expected_digest = hashlib.sha256(raw_inventory).hexdigest()
    result = launcher._public_result(
        {
            "status": "SYNTHETIC_REHEARSAL_CONFIRMED",
            "provider_removal": "STOPPED_CONFIRMED_BY_LIST",
            "result": {"rehearsal": report},
        }
    )
    evidence = result["rehearsal"]
    assert evidence["source_sha"] == source_sha
    assert evidence["tracks"]["pause"]["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert evidence["tracks"]["synthetic_test_review"]["status"] == (
        "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
    )
    assert evidence["zero_call_evidence"] == {"attempts": []}
    assert evidence["real_model_calls"] == 0
    assert evidence["network_calls"] == 0
    assert evidence["publication_calls"] == 0
    assert evidence["artifact_inventory_count"] == len(inventory_map)
    assert evidence["artifact_inventory_sha256"] == expected_digest
    assert "artifact_sha256" not in json.dumps(result)
    assert "never-publish-nested-marker" not in json.dumps(result)
    assert all(path not in json.dumps(result) for path in inventory_map)


@pytest.mark.parametrize(
    "bad_key",
    ["/absolute.json", "../escape.json", "a/../b.json", "a//b.json", "a\\b.json"],
)
def test_invalid_artifact_paths_cannot_publish(bad_key: str) -> None:
    report = _valid_child_output()
    report["artifact_sha256"] = {bad_key: "a" * 64}
    with pytest.raises(launcher.LaunchBlocked):
        launcher._public_result(
            {"status": "SYNTHETIC_REHEARSAL_CONFIRMED", "result": {"rehearsal": report}}
        )


def test_artifact_inventory_count_and_size_bounds() -> None:
    report = _valid_child_output()
    report["artifact_sha256"] = {f"file-{i}.json": "a" * 64 for i in range(513)}
    with pytest.raises(launcher.LaunchBlocked):
        launcher._public_result(
            {"status": "SYNTHETIC_REHEARSAL_CONFIRMED", "result": {"rehearsal": report}}
        )
    report["artifact_sha256"] = {f"folder/{i:03d}-{'x' * 100}.json": "a" * 64 for i in range(510)}
    with pytest.raises(launcher.LaunchBlocked):
        launcher._public_result(
            {"status": "SYNTHETIC_REHEARSAL_CONFIRMED", "result": {"rehearsal": report}}
        )


def test_main_publication_failure_preserves_confirmed_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, digest = _manifest(tmp_path)
    monkeypatch.setattr(launcher, "local_source", lambda *_args: None)
    monkeypatch.setenv("MODAL_TOKEN_ID", "inert-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "inert-value")
    monkeypatch.delenv("MODAL_PROFILE", raising=False)

    def fake_run(
        self: launcher.BoundedModalControl, _approval: launcher.Approval
    ) -> dict[str, object]:
        report = _valid_child_output()
        report["artifact_sha256"] = {"../invalid.json": "a" * 64}
        return {
            "status": "SYNTHETIC_REHEARSAL_CONFIRMED",
            "provider_removal": "STOPPED_CONFIRMED_BY_LIST",
            "app_id": "ap-" + "A" * 22,
            "observed_phases": ["APP_ID", "RESULT_ACCEPTED"],
            "cleanup_errors": [],
            "initiator_interrupted": False,
            "result": {"rehearsal": report},
        }

    monkeypatch.setattr(launcher.BoundedModalControl, "run_once", fake_run)
    assert (
        launcher.main(
            [
                "--execute",
                "--expected-sha",
                SHA,
                "--approval-file",
                str(path),
                "--approval-sha256",
                digest,
                "--independent-control-receipt",
                "independent-ctrl-123",
            ]
        )
        == 2
    )
    observed = json.loads(capsys.readouterr().out)
    assert observed["status"] == "BLOCKED"
    assert observed["provider_removal"] == "STOPPED_CONFIRMED_BY_LIST"
    assert observed["app_id"] == "ap-" + "A" * 22
    assert "PUBLIC_EVIDENCE_VALIDATION_FAILED" in observed["cleanup_errors"]
    assert "rehearsal" not in observed

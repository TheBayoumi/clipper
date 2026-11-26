"""Offline launcher contract tests; never authenticate or allocate a resource."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_books_private_phase import _brief

from clipper.books import private_modal, private_phase
from clipper.stage_contracts import content_fingerprint


def test_preflight_does_not_authenticate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, packet = _brief()
    forbidden = Mock(side_effect=AssertionError("no provider"))
    monkeypatch.setattr(private_modal, "identify", forbidden)
    result = private_modal.launch(
        manifest,
        packet,
        None,
        approved_manifest_sha256=content_fingerprint(manifest),
        execute=False,
        checkout=tmp_path,
        journal_root=tmp_path,
        output=tmp_path / "result.bin",
    )
    assert result["status"] == "PREFLIGHT_ONLY"
    forbidden.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_duplicate_phase_claim(tmp_path: Path) -> None:
    manifest, _ = _brief()
    journal = private_modal.Journal(tmp_path, manifest)
    assert (journal.directory / "0001.json").is_file()
    manifest["owner"] = "c" * 64
    with pytest.raises(FileExistsError):
        private_modal.Journal(tmp_path, manifest)


@pytest.mark.parametrize("tasks", ["0", "1", 0])
def test_exact_terminal_observation(monkeypatch: pytest.MonkeyPatch, tasks: object) -> None:
    app_id = "ap-" + "a" * 22
    run = Mock(
        return_value=Mock(
            stdout=json.dumps([{"app_id": app_id, "state": "stopped", "tasks": tasks}])
        )
    )
    monkeypatch.setattr(private_modal.subprocess, "run", run)
    if isinstance(tasks, int):
        with pytest.raises(private_phase.PrivatePhaseError):
            private_modal.observe(app_id)
    else:
        assert private_modal.observe(app_id) is (tasks == "0")


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "owner",
        "digest",
        "size",
        "status",
        "spawn",
        "write",
        "no_output",
        "cleanup",
        "call_journal",
        "scope_exit",
    ],
)
def test_dispatch_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    tmp_path.chmod(0o700)
    manifest, packet = _brief()
    identity = content_fingerprint(manifest)
    output = tmp_path / "result.bin"
    raw = private_phase.pack(
        {"book-candidates/hybrid_argument/" + "a" * 64 + "/raw-1.json": b"fixture"}
    )
    report = {
        "status": "BRIEF_READY_UNREVIEWED",
        "manifest_sha256": identity,
        "owner": manifest["owner"],
        "checkpoint_bytes": len(raw),
        "checkpoint_sha256": private_phase.sha(raw),
        "retention": "COMPLETE",
        "brief_identity": "a" * 64,
        "configured_execution_receipt_identity": "b" * 64,
        "publish_eligible": False,
    }
    if fault == "owner":
        report["owner"] = "c" * 64
    elif fault == "digest":
        report["checkpoint_sha256"] = "c" * 64
    elif fault == "size":
        report["checkpoint_bytes"] = 0
    elif fault == "status":
        report["status"] = "unexpected"
    elif fault == "no_output":
        report.update(
            status="PRIVATE_PHASE_FAILED_NO_RETRY",
            retention="OUTPUT_NOT_RETAINED",
            checkpoint_bytes=0,
            checkpoint_sha256=None,
            omitted_members=1,
            inventory_truncated=False,
        )
        raw = None
    client = Mock()
    app = Mock(app_id="ap-" + "a" * 22)
    function = Mock()
    call = Mock(object_id="fc-fixture")
    function.spawn.return_value = call
    call.get.return_value = (
        report,
        None if raw is None else private_modal.compress_checkpoint(raw),
    )
    if fault == "spawn":
        function.spawn.side_effect = TimeoutError("ambiguous")
    if fault == "call_journal":
        original = private_modal.Journal.record

        def record(self, event, **fields):
            if event == "CALL_ID":
                raise OSError("journal failure")
            original(self, event, **fields)

        monkeypatch.setattr(private_modal.Journal, "record", record)
    if fault == "write":
        monkeypatch.setattr(
            private_phase, "persist_checkpoint", Mock(side_effect=OSError("private failure"))
        )

    @contextmanager
    def scope(**_):
        try:
            yield
        finally:
            if fault in {"none", "cleanup", "scope_exit"}:
                assert output.read_bytes() == raw
            if fault == "scope_exit":
                raise RuntimeError("context exit failure")

    app.run.side_effect = scope
    monkeypatch.setattr(private_modal, "identify", lambda: (Mock(), client))
    monkeypatch.setattr(private_modal, "verify_checkout", lambda *_: None)
    monkeypatch.setattr(private_modal, "factory", lambda *_: (app, function))
    monkeypatch.setattr(private_modal, "cleanup", lambda *_: fault != "cleanup")
    kwargs = {
        "approved_manifest_sha256": identity,
        "execute": True,
        "checkout": tmp_path,
        "journal_root": tmp_path,
        "output": output,
    }
    result = private_modal._dispatch_qualified(manifest, packet, None, **kwargs)
    assert result["status"] == (
        "BRIEF_READY_UNREVIEWED" if fault == "none" else "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    )
    assert result["checkpoint_saved"] is (fault in {"none", "cleanup", "scope_exit"})
    assert result["app_id"] == app.app_id
    assert result["call_id"] == (None if fault == "spawn" else call.object_id)
    client.close.assert_called_once()
    if fault in {"no_output", "write"}:
        events = [json.loads(path.read_bytes()) for path in tmp_path.glob("*/*.json")]
        assert any(item["event"] == "RESPONSE_VERIFIED" for item in events)
    with pytest.raises((FileExistsError, private_phase.PrivatePhaseError)):
        private_modal._dispatch_qualified(manifest, packet, None, **kwargs)
    function.spawn.assert_called_once()


@pytest.mark.parametrize("rows", [[], [{"app_id": "wrong"}], [{"app_id": "ap-" + "a" * 22}]])
def test_cleanup_requires_exact_record(monkeypatch: pytest.MonkeyPatch, rows: list) -> None:
    monkeypatch.setattr(
        private_modal.subprocess, "run", Mock(return_value=Mock(stdout=json.dumps(rows)))
    )
    with pytest.raises(private_phase.PrivatePhaseError):
        private_modal.observe("ap-" + "a" * 22)


def test_public_execute_blocked_before_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, packet = _brief()
    forbidden = Mock(side_effect=AssertionError("no authentication or dispatch"))
    monkeypatch.setattr(private_modal, "identify", forbidden)
    monkeypatch.setattr(private_modal, "_dispatch_qualified", forbidden)
    with pytest.raises(private_phase.PrivatePhaseError, match="zero-overage"):
        private_modal.launch(
            manifest,
            packet,
            None,
            approved_manifest_sha256=content_fingerprint(manifest),
            execute=True,
            checkout=tmp_path,
            journal_root=tmp_path,
            output=tmp_path / "result.bin",
        )
    forbidden.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_cleanup_one_stop_without_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = _brief()
    journal = private_modal.Journal(tmp_path, manifest)
    app_id = "ap-" + "a" * 22
    client = Mock()
    monkeypatch.setattr(private_modal, "identify", lambda: (Mock(), client))
    run = Mock(
        return_value=Mock(
            returncode=0,
            stdout=json.dumps([{"app_id": app_id, "state": "stopping", "tasks": "1"}]),
        )
    )
    monkeypatch.setattr(private_modal.subprocess, "run", run)
    sleep = Mock()
    monkeypatch.setattr(private_modal.time, "sleep", sleep)
    assert private_modal.cleanup(app_id, journal) is False
    commands = [call.args[0] for call in run.call_args_list]
    assert all(command[:3] == [sys.executable, "-m", "modal"] for command in commands)
    assert len([command for command in commands if command[4] == "stop"]) == 1
    assert sleep.call_count == 6
    client.close.assert_called_once()


@pytest.mark.parametrize("terminal", ["stopped", "running", "absent"])
def test_stop_timeout_reobserves_exact_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    manifest, _ = _brief()
    journal = private_modal.Journal(tmp_path, manifest)
    app_id = "ap-" + "a" * 22
    monkeypatch.setattr(private_modal, "identify", lambda: (Mock(), Mock()))
    clock = [0.0]
    commands = []

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    def run(command, **kwargs):
        commands.append(command)
        timeout = kwargs["timeout"]
        assert 0 < timeout <= 8.0
        assert clock[0] + timeout <= 45.0
        if command[4] == "stop":
            clock[0] += timeout
            raise subprocess.TimeoutExpired(command, timeout)
        clock[0] += min(1.0, timeout)
        if len(commands) == 1 or terminal == "running":
            rows = [{"app_id": app_id, "state": "running", "tasks": "1"}]
        elif terminal == "absent":
            rows = []
        else:
            rows = [{"app_id": app_id, "state": "stopped", "tasks": "0"}]
        return Mock(stdout=json.dumps(rows))

    monkeypatch.setattr(private_modal.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(private_modal.time, "sleep", sleep)
    monkeypatch.setattr(private_modal.subprocess, "run", run)
    assert private_modal.cleanup(app_id, journal) is (terminal == "stopped")
    assert clock[0] <= 45.0
    stops = [command for command in commands if command[4] == "stop"]
    assert len(stops) == 1
    assert stops[0][-1] == app_id
    events = [json.loads(path.read_bytes()) for path in journal.directory.glob("*.json")]
    assert any(event["event"] == "EXACT_STOP_TIMEOUT_AMBIGUOUS" for event in events)
    assert any(event["event"] == "STOPPED_ZERO_TASKS" for event in events) is (
        terminal == "stopped"
    )

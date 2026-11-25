"""Offline-only contract tests for dormant public synthetic Modal launcher."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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

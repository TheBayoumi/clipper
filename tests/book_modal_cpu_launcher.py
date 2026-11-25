"""Dormant, explicitly authorized, model-free Modal CPU rehearsal launcher.

Default CLI only verifies local source and prints a plan: no Modal import.
Live execution is separately scoped; import and plan remain inert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

SOURCE_URL = "https://github.com/TheBayoumi/clipper.git"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
APP_ID = re.compile(r"ap-[A-Za-z0-9]{22}\Z")
CALL_ID = re.compile(r"fc-[A-Za-z0-9_-]{3,128}\Z")
INTENT = "PUBLIC_SYNTHETIC_MODEL_FREE_BOOK_REHEARSAL"
LIMITS = {
    "cpu": (1, 1),
    "memory": (512, 512),
    "gpu": None,
    "block_network": True,
    "restrict_modal_access": True,
    "single_use_containers": True,
    "max_containers": 1,
    "min_containers": 0,
    "buffer_containers": 0,
    "retries": 0,
    "startup_timeout": 60,
    "timeout": 30,
    "include_source": False,
    "scaledown_window": 2,
}


class LaunchBlocked(RuntimeError):
    """No resources may be allocated for this request."""


class SyntheticChildError(RuntimeError):
    """Fixed-category child failure with bounded exit-code evidence."""

    def __init__(self, category: str, returncode: int | None = None) -> None:
        super().__init__(category)
        self.returncode = returncode


class LaunchInterrupted(KeyboardInterrupt):
    """Original user interrupt retained with bounded reconciliation evidence."""

    def __init__(self, evidence: dict[str, object]) -> None:
        super().__init__("bounded worker interrupted; reconciliation attempted")
        self.evidence = evidence


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise LaunchBlocked(reason)


def _approval_object(raw: bytes) -> dict[str, Any]:
    """Decode the exact already-digested approval bytes; never reopen the path."""
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise LaunchBlocked("approval JSON malformed") from exc
    _require(isinstance(data, dict), "approval must be a JSON object")
    return data


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


def local_source(checkout: Path, sha: str) -> None:
    _require(bool(HEX40.fullmatch(sha)), "explicit full source SHA required")
    _require(
        checkout.resolve() == Path(__file__).resolve().parents[1], "launcher must use own checkout"
    )
    _require(_git(checkout, "rev-parse", "HEAD") == sha, "source checkout SHA differs")
    _require(
        not _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"),
        "source checkout is not clean",
    )
    _require(
        (checkout / "tests" / "book_public_rehearsal.py").is_file(), "rehearsal entrypoint missing"
    )


@dataclass(frozen=True)
class Approval:
    source_sha: str
    workspace: str
    environment: str
    credential_route: str
    expires_utc: datetime
    included_credit_only: bool
    paid_overage_authorized: bool
    owner_request_ref: str
    credit_confirmation_ref: str
    external_control_id: str


def load_approval(path: Path, digest: str, sha: str, *, now: datetime | None = None) -> Approval:
    _require(bool(HEX64.fullmatch(digest)), "out-of-band approval file SHA-256 required")
    _require(path.is_file() and not path.is_symlink(), "approval must be a regular file")
    _require(0 < path.stat().st_size <= 8192, "missing or oversized approval")
    with path.open("rb") as stream:
        raw = stream.read(8193)
    _require(0 < len(raw) <= 8192, "missing or oversized approval")
    _require(hashlib.sha256(raw).hexdigest() == digest, "approval digest mismatch")
    data = _approval_object(raw)
    expected = {
        "schema",
        "intent",
        "source_sha",
        "workspace",
        "environment",
        "credential_route",
        "expires_utc",
        "included_credit_only",
        "paid_overage_authorized",
        "owner_request_ref",
        "credit_confirmation_ref",
        "external_control_id",
        "execution_permission",
    }
    _require(set(data) == expected, "approval fields mismatch")
    _require(
        data["schema"] == "clipper-modal-public-synthetic-approval-v2", "approval schema mismatch"
    )
    _require(data["intent"] == INTENT, "wrong operation authorized")
    _require(data["execution_permission"] is True, "no explicit permission")
    _require(
        isinstance(data["source_sha"], str) and data["source_sha"] == sha, "approval SHA mismatch"
    )
    for key in ("workspace", "environment", "credential_route", "external_control_id"):
        value = data[key]
        _require(
            isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_.:/-]{3,128}", value)),
            "invalid or missing approval " + key,
        )
    _require(data["workspace"] == "mahmoud-bayoumi-mm", "workspace scope mismatch")
    _require(
        data["credential_route"] == "actions-existing-secret-route",
        "Actions-only credential route required",
    )
    _require(data["included_credit_only"] is True, "included-credit-only scope required")
    _require(data["paid_overage_authorized"] is False, "paid overage prohibited")
    _require(
        data["owner_request_ref"] == "2026-11-25T11:09:05+02:00", "owner request reference mismatch"
    )
    _require(
        data["credit_confirmation_ref"] == "2026-11-25T15:00:33+02:00", "credit reference mismatch"
    )
    try:
        expiry = datetime.fromisoformat(data["expires_utc"].replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise LaunchBlocked("invalid UTC approval expiry") from exc
    offset = expiry.utcoffset()
    _require(offset is not None and offset.total_seconds() == 0, "approval expiry must be UTC")
    current = now if now is not None else datetime.now(UTC)
    _require(current < expiry, "approval expired")
    _require((expiry - current).total_seconds() <= 3600, "approval exceeds one-hour validity")
    return Approval(
        sha,
        data["workspace"],
        data["environment"],
        data["credential_route"],
        expiry,
        True,
        False,
        data["owner_request_ref"],
        data["credit_confirmation_ref"],
        data["external_control_id"],
    )


def _remote_child(sha: str) -> dict[str, object]:
    """Only executed in a separately approved restricted CPU Function."""
    from pathlib import Path as ChildPath

    checkout = ChildPath("/opt/clipper")
    # A new directory is created *within* the temporary container root;
    # the rehearsal CLI demands a path that does not already exist.
    with tempfile.TemporaryDirectory(prefix="clipper-") as fresh:
        return _run_synthetic_child(checkout, ChildPath(fresh) / "artifacts", sha)


def _run_synthetic_child(checkout: Path, output: Path, sha: str) -> dict[str, object]:
    import json as child_json
    import os as child_os
    import subprocess as child_subprocess

    env = {
        "PATH": child_os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(checkout / "src") + ":" + str(checkout / "tests"),
        "HOME": str(output.parent),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = child_subprocess.run(
        [
            "python",
            str(checkout / "tests" / "book_public_rehearsal.py"),
            "--fresh-root",
            str(output),
            "--expected-sha",
            sha,
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=25,
    )
    if proc.returncode != 0:
        raise SyntheticChildError("SYNTHETIC_CHILD_NONZERO", proc.returncode)
    if len(proc.stdout) > 220_000:
        raise SyntheticChildError("SYNTHETIC_CHILD_STDOUT_OVERSIZE")

    payload = child_json.loads(proc.stdout)
    if not isinstance(payload, dict) or payload.get("source_sha") != sha:
        raise RuntimeError("unexpected synthetic result identity")
    if (
        payload.get("status") != "SYNTHETIC_CONTRACT_REHEARSAL_PASSED"
        or payload.get("publish_eligible") is not False
        or payload.get("source_pin") != "VERIFIED_CLEAN_LOCAL_GIT_HEAD"
    ):
        raise RuntimeError("synthetic result not accepted")
    return _compact_rehearsal(payload, sha)


class Control(Protocol):
    def identify(self, approval: Approval) -> str: ...
    def run_once(self, approval: Approval) -> dict[str, object]: ...


class ModalControl:
    """Imported/constructed only after ALL local approval/identity prerequisites."""

    def __init__(self) -> None:
        self._modal: Any = None
        self._client: Any = None

    def identify(self, approval: Approval) -> str:
        # Does not create App/Image/Function. Credentials must already be explicitly selected.
        from importlib.metadata import version

        import modal

        _require(version("modal") == "1.5.5", "Modal SDK must be exactly 1.5.5")
        client = modal.Client.from_env()
        workspace = modal.Workspace.from_context(client=client)
        workspace.hydrate(client=client)
        self._modal, self._client = modal, client
        return str(workspace.name)

    def run_once(self, approval: Approval) -> dict[str, object]:
        raise LaunchBlocked("control-plane work must run in bounded child; not directly")

    def run_remote(self, approval: Approval, record: Any) -> dict[str, object]:
        """Only internal bounded child may construct image and App."""
        modal, client = self._modal, self._client
        _require(modal is not None and client is not None, "identity not established")
        sha = approval.source_sha
        record("BUILD_BEGIN", "")
        clone = (
            "set -eu; export GIT_TERMINAL_PROMPT=0; "
            "git -c credential.helper= clone --no-checkout " + SOURCE_URL + " /opt/clipper; "
            "git -C /opt/clipper checkout --detach " + sha + "; "
            'test "$(git -C /opt/clipper rev-parse HEAD)" = ' + sha + "; "
            'test -z "$(git -C /opt/clipper status --porcelain=v1 --untracked-files=all)"; '
            "test -f /opt/clipper/tests/book_public_rehearsal.py"
        )
        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git")
            .pip_install("PyYAML==6.0.3")
            .run_commands(clone)
            .env({"PYTHONPATH": "/opt/clipper/tests"})
        )
        # The public clone supplies the function module at /opt/clipper/tests.
        # Neither App nor Function mounts the ambient host source.
        app = modal.App(include_source=False)
        function = app.function(image=image, **LIMITS)(_remote_child)
        record("APP_RUN_BEGIN", "")
        # Context-entry can create an App before its ID is returned: the
        # parent retains BUILD/APP_RUN_BEGIN with unknown resource identity.
        with app.run(client=client, detach=False, environment_name=approval.environment):
            app_id = getattr(app, "app_id", None)
            _require(
                isinstance(app_id, str) and bool(APP_ID.fullmatch(app_id)),
                "exact provider App ID missing",
            )
            record("APP_ID", app_id)
            call: Any = None
            try:
                record("SPAWN_BEGIN", app_id)
                call = function.spawn(sha)
                call_id = getattr(call, "object_id", None)
                _require(isinstance(call_id, str) and bool(call_id), "FunctionCall id missing")
                record("CALL_ID", call_id)
                record("GET_BEGIN", call_id)
                output = call.get(timeout=30)
                _validate_compact_report(output, sha)
                record("RESULT_ACCEPTED", call_id)
                return {
                    "source_sha": sha,
                    "app_id": app_id,
                    "call_id": call_id,
                    "rehearsal": output,
                }
            finally:
                # Cancellation runs BEFORE potentially blocking __exit__.
                if call is not None:
                    try:
                        call.cancel(terminate_containers=True)
                        record("CALL_CANCEL_ATTEMPTED", str(getattr(call, "object_id", "")))
                    except Exception as exc:
                        record("CALL_CANCEL_ERROR", type(exc).__name__)


def _validate_child_report(payload: object, sha: str) -> None:
    _require(isinstance(payload, dict), "synthetic result not a dictionary")
    assert isinstance(payload, dict)
    _require(
        payload.get("source_sha") == sha
        and payload.get("source_pin") == "VERIFIED_CLEAN_LOCAL_GIT_HEAD",
        "synthetic source pin mismatch",
    )
    _require(
        payload.get("status") == "SYNTHETIC_CONTRACT_REHEARSAL_PASSED"
        and payload.get("publish_eligible") is False,
        "synthetic status rejected",
    )
    tracks = payload.get("tracks")
    _require(isinstance(tracks, dict), "synthetic tracks missing")
    assert isinstance(tracks, dict)
    _require(
        isinstance(tracks.get("pause"), dict)
        and tracks["pause"].get("status") == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
        "human review gate missing",
    )
    follow = tracks.get("synthetic_test_review")
    _require(
        isinstance(follow, dict)
        and follow.get("status") == "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
        and follow.get("publish_eligible") is False,
        "synthetic test continuation invalid",
    )
    proof = payload.get("zero_call_evidence")
    _require(isinstance(proof, dict) and proof.get("attempts") == [], "guard attempts not empty")
    _require(
        all(
            type(payload.get(key)) is int and payload[key] == 0
            for key in ("real_model_calls", "network_calls", "publication_calls")
        ),
        "synthetic zero-call evidence invalid",
    )


COMPACT_SCHEMA = "clipper-public-book-modal-compact-v1"
COMPACT_KEYS = frozenset(
    {
        "schema",
        "source_sha",
        "source_pin",
        "status",
        "publish_eligible",
        "tracks",
        "zero_call_evidence",
        "real_model_calls",
        "network_calls",
        "publication_calls",
        "artifact_inventory_sha256",
        "artifact_inventory_count",
    }
)
ERROR_PHASES = frozenset(
    {
        "WORKSPACE_VERIFY_BEGIN",
        "BUILD_BEGIN",
        "APP_RUN_BEGIN",
        "APP_ID",
        "SPAWN_BEGIN",
        "CALL_ID",
        "GET_BEGIN",
        "RESULT_ACCEPTED",
    }
)
SAFE_WORKER_CLASS = re.compile(
    r"(?:LaunchBlocked|[A-Za-z_][A-Za-z0-9_]{0,63}(?:Error|Exception))\Z"
)
JOURNAL_PHASES = ERROR_PHASES | frozenset(
    {
        "CALL_CANCEL_ATTEMPTED",
        "CALL_CANCEL_ERROR",
        "WORKER_ERROR",
        "WORKER_ERROR_PHASE",
        "WORKER_ERROR_HTTP_STATUS",
        "CHILD_EXIT_CODE",
    }
)


def _compact_rehearsal(payload: object, sha: str) -> dict[str, object]:
    """Validate full synthetic fixture in-container; return only bounded, fixed evidence."""
    _validate_child_report(payload, sha)
    assert isinstance(payload, dict)
    hashes = payload.get("artifact_sha256")
    _require(
        isinstance(hashes, dict) and 0 < len(hashes) <= 512,
        "synthetic artifact inventory count invalid",
    )
    for path, digest in hashes.items():
        _require(
            isinstance(path, str)
            and bool(path)
            and len(path.encode("utf-8")) <= 1024
            and not path.startswith("/")
            and "\\" not in path
            and ":" not in path
            and all(part not in {"", ".", ".."} for part in path.split("/"))
            and isinstance(digest, str)
            and bool(HEX64.fullmatch(digest)),
            "invalid relative POSIX artifact inventory",
        )
    inventory = json.dumps(hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    _require(len(inventory) <= 65536, "synthetic artifact inventory exceeds 64 KiB")
    compact: dict[str, object] = {
        "schema": COMPACT_SCHEMA,
        "source_sha": sha,
        "source_pin": "VERIFIED_CLEAN_LOCAL_GIT_HEAD",
        "status": "SYNTHETIC_CONTRACT_REHEARSAL_PASSED",
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
        "artifact_inventory_sha256": hashlib.sha256(inventory).hexdigest(),
        "artifact_inventory_count": len(hashes),
    }
    _validate_compact_report(compact, sha)
    return compact


def _validate_compact_report(payload: object, sha: str) -> None:
    """Strict compact wire contract applied at remote, worker-parent and public edges."""
    _require(
        isinstance(payload, dict) and set(payload) == COMPACT_KEYS, "invalid compact report keys"
    )
    assert isinstance(payload, dict)
    _require(
        isinstance(sha, str)
        and bool(HEX40.fullmatch(sha))
        and payload["schema"] == COMPACT_SCHEMA
        and type(payload["source_sha"]) is str
        and payload["source_sha"] == sha
        and payload["source_pin"] == "VERIFIED_CLEAN_LOCAL_GIT_HEAD"
        and payload["status"] == "SYNTHETIC_CONTRACT_REHEARSAL_PASSED"
        and payload["publish_eligible"] is False,
        "invalid compact identity/status",
    )
    tracks = payload["tracks"]
    _require(
        isinstance(tracks, dict) and set(tracks) == {"pause", "synthetic_test_review"},
        "invalid compact track keys",
    )
    assert isinstance(tracks, dict)
    _require(
        type(tracks["pause"]) is dict
        and tracks["pause"] == {"status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"}
        and type(tracks["synthetic_test_review"]) is dict
        and tracks["synthetic_test_review"].get("publish_eligible") is False
        and tracks["synthetic_test_review"]
        == {
            "status": "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED",
            "publish_eligible": False,
        },
        "invalid compact track evidence",
    )
    _require(
        type(payload["zero_call_evidence"]) is dict
        and payload["zero_call_evidence"] == {"attempts": []}
        and all(
            type(payload[key]) is int and payload[key] == 0
            for key in ("real_model_calls", "network_calls", "publication_calls")
        ),
        "invalid compact zero-call evidence",
    )
    _require(
        type(payload["artifact_inventory_sha256"]) is str
        and bool(HEX64.fullmatch(payload["artifact_inventory_sha256"]))
        and type(payload["artifact_inventory_count"]) is int
        and 1 <= payload["artifact_inventory_count"] <= 512,
        "invalid compact inventory evidence",
    )


def _journal(path: Path, phase: str, item: str = "") -> None:
    """Durable minimal IDs/phase evidence; never tokens or source text."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": phase, "id": item}, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_journal(path: Path) -> tuple[str | None, list[str]]:
    app_id: str | None = None
    phases: list[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                phases.append("MALFORMED_RECORD")
                continue
            if not isinstance(entry, dict):
                phases.append("MALFORMED_RECORD")
                continue
            phase, identifier = entry.get("phase"), entry.get("id")
            if (
                not isinstance(phase, str)
                or phase not in JOURNAL_PHASES
                or not isinstance(identifier, str)
                or len(identifier) > 128
            ):
                phases.append("MALFORMED_RECORD")
                continue
            phases.append(phase)
            if phase == "APP_ID":
                if APP_ID.fullmatch(identifier):
                    app_id = identifier
                else:
                    phases.append("INVALID_APP_ID")
                    app_id = None
    return app_id, phases


def _read_worker_error(path: Path) -> dict[str, object] | None:
    """Extract only typed, allowlisted worker failure metadata from bounded journal."""
    if not path.is_file() or path.stat().st_size > 65536:
        return None
    found: dict[str, object] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines()[:512]:
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                continue
            key, value = entry.get("phase"), entry["id"]
            if key == "WORKER_ERROR" and SAFE_WORKER_CLASS.fullmatch(value):
                found["class"] = value
            elif key == "WORKER_ERROR_PHASE" and value in ERROR_PHASES:
                found["phase"] = value
            elif key == "WORKER_ERROR_HTTP_STATUS" and value.isascii() and value.isdecimal():
                number = int(value)
                if 100 <= number <= 599:
                    found["http_status"] = number
            elif key == "CHILD_EXIT_CODE" and re.fullmatch(r"-?[0-9]{1,3}", value):
                number = int(value)
                if -255 <= number <= 255:
                    found["child_exit_code"] = number
    except (OSError, UnicodeError):
        return None
    return found if "class" in found else None


def _read_accepted_call_id(path: Path, app_id: str | None) -> str | None:
    """Require one matching journal call ID and its accepted-result phase."""
    if app_id is None or not APP_ID.fullmatch(app_id):
        return None
    if not path.is_file() or path.stat().st_size > 65536:
        return None
    try:
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, ValueError):
        return None
    if len(entries) > 512 or not all(type(entry) is dict for entry in entries):
        return None
    expected = None
    for phase in ("CALL_ID", "GET_BEGIN", "RESULT_ACCEPTED"):
        ids = [entry.get("id") for entry in entries if entry.get("phase") == phase]
        if len(ids) != 1 or type(ids[0]) is not str or not CALL_ID.fullmatch(ids[0]):
            return None
        if expected is not None and expected != ids[0]:
            return None
        expected = ids[0]
    app_ids = [entry.get("id") for entry in entries if entry.get("phase") == "APP_ID"]
    return expected if app_ids == [app_id] else None


def _stop_and_observe(
    app_id: str,
    environment: str,
    deadline: float,
    runner: Any = subprocess.run,
    *,
    warnings: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """One exact-ID stop at most; bounded terminal reobservation shares the deadline."""
    if not APP_ID.fullmatch(app_id):
        return False, ["INVALID_APP_ID"]

    local_warnings: list[str] = []
    list_command = ["modal", "app", "list", "--json", "--env", environment]

    transient_observation = False

    def observe() -> tuple[bool | None, str | None]:
        nonlocal transient_observation
        transient_observation = False
        remaining = min(8.0, deadline - time.monotonic())
        if remaining <= 0:
            return None, "list:deadline"
        try:
            response = runner(
                list_command, check=False, text=True, capture_output=True, timeout=remaining
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            transient_observation = True
            return None, "list:" + type(exc).__name__
        except subprocess.SubprocessError as exc:
            return None, "list:" + type(exc).__name__
        if response.returncode != 0:
            transient_observation = True
            return None, "list:nonzero"
        try:
            rows = json.loads(response.stdout)
        except (TypeError, ValueError):
            return None, "list:invalid_json"
        if not isinstance(rows, list):
            return None, "list:invalid_schema"
        matches = [item for item in rows if isinstance(item, dict) and item.get("app_id") == app_id]
        if len(matches) != 1:
            return None, "list:exact_id_not_found_or_duplicate"
        row = matches[0]
        state, tasks = row.get("state"), row.get("tasks")
        if type(state) is not str or type(tasks) is not str or len(state) > 40:
            return None, "list:invalid_row_schema"
        if not tasks.isascii() or not tasks.isdecimal() or len(tasks) > 16:
            return None, "list:invalid_row_schema"
        if state == "stopped" and tasks == "0":
            return True, None
        return False, None

    terminal, issue = observe()
    if issue:
        if not transient_observation:
            return False, [issue]
        local_warnings.append(issue)
    if terminal:
        return True, []

    remaining = min(8.0, deadline - time.monotonic())
    if remaining <= 0:
        return False, [*local_warnings, "stop:deadline"]
    try:
        response = runner(
            ["modal", "app", "stop", "-y", "--env", environment, app_id],
            check=False,
            text=True,
            capture_output=True,
            timeout=remaining,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        local_warnings.append("stop:" + type(exc).__name__)
    else:
        if response.returncode != 0:
            local_warnings.append("stop:nonzero")

    # Maximum three post-stop reads with short positive sleeps; no new budget.
    last_issue = "list:terminal_or_tasks_unconfirmed"
    for attempt in range(3):
        if attempt:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.2, remaining))
        terminal, issue = observe()
        if terminal is True:
            if warnings is not None:
                warnings.extend(local_warnings)
            return True, []
        if issue:
            last_issue = issue
            # An invalid or ambiguous exact-ID record cannot be healed by
            # observing a different snapshot later.
            if issue in {
                "list:invalid_json",
                "list:invalid_schema",
                "list:invalid_row_schema",
                "list:exact_id_not_found_or_duplicate",
            }:
                break
        else:
            last_issue = "list:terminal_or_tasks_unconfirmed"
    return False, [*local_warnings, last_issue]



def _bounded_run(
    approval: Approval,
    manifest_path: Path,
    digest: str,
    *,
    deadline_seconds: float = 120,
    popen: Any = subprocess.Popen,
    runner: Any = subprocess.run,
) -> dict[str, object]:
    """One bounded control-plane process, plus bounded exact-ID stop/list."""
    _require(0 < deadline_seconds <= 180, "unbounded control-plane deadline")
    deadline = time.monotonic() + deadline_seconds
    with tempfile.TemporaryDirectory(prefix="clipper-modal-owner-") as directory:
        folder = Path(directory)
        ledger, result_file = folder / "ids.jsonl", folder / "result.json"
        stdout_file, stderr_file = folder / "worker.stdout", folder / "worker.stderr"
        worker_nonce = secrets.token_hex(32)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--internal-worker",
            "--expected-sha",
            approval.source_sha,
            "--approval-file",
            str(manifest_path),
            "--approval-sha256",
            digest,
            "--journal",
            str(ledger),
            "--result-file",
            str(result_file),
            "--independent-control-receipt",
            approval.external_control_id,
            "--internal-owner-nonce",
            worker_nonce,
        ]
        # Capture only bounded diagnostic digests/sizes; never expose raw
        # stdout/stderr, which might contain provider or environment details.
        with stdout_file.open("wb") as stdout_sink, stderr_file.open("wb") as stderr_sink:
            proc = popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                start_new_session=True,
                stdout=stdout_sink,
                stderr=stderr_sink,
                env={**os.environ, "CLIPPER_MODAL_WORKER_NONCE": worker_nonce},
            )
        timed_out = False
        interrupted = False
        process_cleanup_errors: list[str] = []
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic() - 20))
        except subprocess.TimeoutExpired:
            timed_out = True
        except KeyboardInterrupt:
            interrupted = True
            process_cleanup_errors.append("INITIATOR_KEYBOARD_INTERRUPT")
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError) as exc:
                    process_cleanup_errors.append("KILL_GROUP:" + type(exc).__name__)
                try:
                    proc.wait(timeout=min(2.0, max(0.1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    timed_out = True
                except KeyboardInterrupt:
                    interrupted = True
                    process_cleanup_errors.append("INTERRUPT_DURING_WORKER_REAP")
        app_id, phases = _read_journal(ledger)
        worker_error = _read_worker_error(ledger)
        confirmed = False
        cleanup_warnings: list[str] = []
        cleanup_errors: list[str] = list(process_cleanup_errors)
        if "CALL_CANCEL_ERROR" in phases:
            cleanup_errors.append("CALL_CANCEL_ERROR")
        if "MALFORMED_RECORD" in phases:
            cleanup_errors.append("INVALID_JOURNAL")
        if "INVALID_APP_ID" in phases:
            cleanup_errors.append("INVALID_APP_ID")
        if app_id is not None and "INVALID_APP_ID" not in phases:
            try:
                confirmed, stop_errors = _stop_and_observe(
                    app_id, approval.environment, deadline, runner=runner,
                    warnings=cleanup_warnings,
                )
                cleanup_errors.extend(stop_errors)
            except KeyboardInterrupt:
                interrupted = True
                cleanup_errors.append("INTERRUPT_DURING_STOP_OBSERVATION")
        else:
            cleanup_errors.append("APP_ID_UNKNOWN")
        if timed_out:
            cleanup_errors.append("CONTROL_PLANE_TIMEOUT")
        if proc.returncode != 0:
            cleanup_errors.append("WORKER_NONZERO_OR_TERMINATED")
        diagnostics: dict[str, dict[str, object]] = {}
        for label, path in (("stdout", stdout_file), ("stderr", stderr_file)):
            if path.exists():
                with path.open("rb") as stream:
                    prefix = stream.read(4096)
                diagnostics[label] = {
                    "size_bytes": path.stat().st_size,
                    "bounded_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                    "prefix_bytes_hashed": len(prefix),
                    "truncated": path.stat().st_size > len(prefix),
                }
        output: dict[str, object] | None = None
        accepted_call_id = _read_accepted_call_id(ledger, app_id)
        if result_file.is_file() and result_file.stat().st_size <= 220_000:
            try:
                parsed = json.loads(result_file.read_text())
                if type(parsed) is not dict or set(parsed) != {
                    "source_sha", "app_id", "call_id", "rehearsal"
                }:
                    raise LaunchBlocked("unexpected result envelope")
                if (
                    parsed["source_sha"] != approval.source_sha
                    or parsed["app_id"] != app_id
                    or type(parsed["app_id"]) is not str
                    or not APP_ID.fullmatch(parsed["app_id"])
                    or type(parsed["call_id"]) is not str
                    or parsed["call_id"] != accepted_call_id
                    or not CALL_ID.fullmatch(parsed["call_id"])
                ):
                    raise LaunchBlocked("result identity does not match journal")
                _validate_compact_report(parsed["rehearsal"], approval.source_sha)
                if (
                    proc.returncode != 0 or timed_out or interrupted
                    or "INVALID_JOURNAL" in cleanup_errors
                    or "INVALID_APP_ID" in cleanup_errors
                ):
                    raise LaunchBlocked("worker incomplete or journal unsafe")
                output = parsed
            except (ValueError, TypeError, UnicodeError, OSError, LaunchBlocked):
                cleanup_errors.append("RESULT_UNVERIFIED")
        else:
            cleanup_errors.append("NO_RESULT")
        execution_status = "ACCEPTED" if output is not None else "UNVERIFIED"
        lifecycle_status = "TERMINAL_CONFIRMED" if confirmed else "UNKNOWN"
        result: dict[str, object] = {
            "status": "SYNTHETIC_REHEARSAL_CONFIRMED"
            if (confirmed and not cleanup_errors and output is not None)
            else "BLOCKED",
            "execution_status": execution_status,
            "lifecycle_status": lifecycle_status,
            "provider_removal": "STOPPED_CONFIRMED_BY_LIST" if confirmed else "UNKNOWN",
            "app_id": app_id,
            "call_id": accepted_call_id if output is not None else None,
            "observed_phases": phases,
            "cleanup_errors": cleanup_errors,
            "cleanup_warnings": cleanup_warnings,
            "worker_diagnostics": diagnostics,
            "result": output,
            "initiator_interrupted": interrupted,
            "worker_error": worker_error,
        }
        if interrupted:
            result["status"] = "BLOCKED"
            result["result"] = None
            result["execution_status"] = "UNVERIFIED"
            result["call_id"] = None
            raise LaunchInterrupted(result)
        return result


def _internal_worker(approval: Approval, journal: Path, result_file: Path) -> int:
    controller = ModalControl()
    phase = "WORKSPACE_VERIFY_BEGIN"

    def record(item: str, value: str = "") -> None:
        nonlocal phase
        if item in ERROR_PHASES:
            phase = item
        _journal(journal, item, value)

    try:
        record("WORKSPACE_VERIFY_BEGIN")
        _require(controller.identify(approval) == approval.workspace, "workspace mismatch")
        _require(
            datetime.now(UTC) < approval.expires_utc,
            "approval expired during workspace identity lookup",
        )
        result = controller.run_remote(approval, record)
        _validate_compact_report(result.get("rehearsal"), approval.source_sha)
        result_file.write_text(json.dumps(result, sort_keys=True))
        with result_file.open("rb") as f:
            os.fsync(f.fileno())
        return 0
    except Exception as exc:
        name = type(exc).__name__
        record("WORKER_ERROR", name if SAFE_WORKER_CLASS.fullmatch(name) else "OtherException")
        record("WORKER_ERROR_PHASE", phase)
        code = next(
            (
                value
                for value in (
                    getattr(exc, "status_code", None),
                    getattr(exc, "http_status", None),
                    getattr(exc, "status", None),
                )
                if type(value) is int and 100 <= value <= 599
            ),
            None,
        )
        if code is not None:
            record("WORKER_ERROR_HTTP_STATUS", str(code))
        child_code = getattr(exc, "returncode", None)
        if type(child_code) is int and -255 <= child_code <= 255:
            record("CHILD_EXIT_CODE", str(child_code))
        return 2


class BoundedModalControl:
    """Client identity before resources; image/build runs in a bounded subprocess."""

    def __init__(self, manifest_path: Path, digest: str) -> None:
        self.manifest_path = manifest_path
        self.digest = digest

    def identify(self, approval: Approval) -> str:
        # The real SDK workspace lookup must be inside the bounded worker.
        raise LaunchBlocked("workspace identity can only be read inside bounded worker")

    def run_once(self, approval: Approval) -> dict[str, object]:
        return _bounded_run(approval, self.manifest_path, self.digest)


def execute(
    sha: str, approval: Approval, control: Control, *, external_control_verified: bool
) -> dict[str, object]:
    _require(external_control_verified, "independent lifecycle control not verified")
    _require(approval.source_sha == sha, "approval source mismatch")
    _require(datetime.now(UTC) < approval.expires_utc, "approval expired before execution")
    if isinstance(control, BoundedModalControl):
        # Real workspace hydration happens inside the one-shot worker's deadline
        # before image/App construction. Never hydrate in the parent.
        return control.run_once(approval)
    observed = control.identify(approval)
    _require(observed == approval.workspace, "client-bound workspace identity mismatch")
    _require(datetime.now(UTC) < approval.expires_utc, "approval expired before resources")
    return control.run_once(approval)


def _require_credential_route() -> None:
    """Actions-provided token pair only, without token values in output."""
    _require(
        bool(os.environ.get("MODAL_TOKEN_ID"))
        and bool(os.environ.get("MODAL_TOKEN_SECRET"))
        and not os.environ.get("MODAL_PROFILE"),
        "existing Actions token pair required without profile ambiguity",
    )


def _safe_worker_error(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    _require(
        isinstance(value, dict)
        and bool(value)
        and set(value) <= {"class", "phase", "http_status", "child_exit_code"}
        and "class" in value,
        "invalid worker error metadata",
    )
    assert isinstance(value, dict)
    _require(
        type(value["class"]) is str and bool(SAFE_WORKER_CLASS.fullmatch(value["class"])),
        "invalid worker class",
    )
    if "phase" in value:
        _require(
            type(value["phase"]) is str and value["phase"] in ERROR_PHASES,
            "invalid worker error phase",
        )
    if "http_status" in value:
        _require(
            type(value["http_status"]) is int and 100 <= value["http_status"] <= 599,
            "invalid worker HTTP status",
        )
    if "child_exit_code" in value:
        _require(
            type(value["child_exit_code"]) is int and -255 <= value["child_exit_code"] <= 255,
            "invalid child exit code",
        )
    return dict(value)


def _public_result(result: dict[str, object], expected_sha: str) -> dict[str, object]:
    """Publish only verified execution evidence, separately from lifecycle status."""
    _require(type(result) is dict, "invalid parent result")
    _require(result.get("status") in {"BLOCKED", "SYNTHETIC_REHEARSAL_CONFIRMED"},
             "invalid overall status")
    _require(isinstance(expected_sha, str) and bool(HEX40.fullmatch(expected_sha)),
             "trusted expected source SHA required")
    safe: dict[str, object] = {"status": result["status"]}
    for key in ("execution_status", "lifecycle_status", "provider_removal"):
        value = result.get(key)
        allowed = {
            "execution_status": {"ACCEPTED", "UNVERIFIED"},
            "lifecycle_status": {"TERMINAL_CONFIRMED", "UNKNOWN"},
            "provider_removal": {"STOPPED_CONFIRMED_BY_LIST", "UNKNOWN"},
        }[key]
        if value is not None:
            _require(type(value) is str and value in allowed, "unsafe outcome taxonomy")
            safe[key] = value
    app_id = result.get("app_id")
    if app_id is not None:
        _require(type(app_id) is str and bool(APP_ID.fullmatch(app_id)), "unsafe App ID")
        safe["app_id"] = app_id
    for key in ("observed_phases", "cleanup_errors", "cleanup_warnings"):
        value = result.get(key)
        if value is not None:
            _require(type(value) is list and len(value) <= 128, "unsafe diagnostic sequence")
            accepted = (
                JOURNAL_PHASES | {"MALFORMED_RECORD", "INVALID_APP_ID"}
                if key == "observed_phases" else {
                    "CALL_CANCEL_ERROR", "INVALID_JOURNAL", "INVALID_APP_ID", "APP_ID_UNKNOWN",
                    "CONTROL_PLANE_TIMEOUT", "WORKER_NONZERO_OR_TERMINATED",
                    "RESULT_UNVERIFIED", "NO_RESULT", "INITIATOR_KEYBOARD_INTERRUPT",
                    "INTERRUPT_DURING_WORKER_REAP", "INTERRUPT_DURING_STOP_OBSERVATION",
                    "PUBLIC_EVIDENCE_VALIDATION_FAILED", "list:deadline",
                    "list:nonzero", "list:invalid_json", "list:invalid_schema",
                    "list:invalid_row_schema", "list:exact_id_not_found_or_duplicate",
                    "list:terminal_or_tasks_unconfirmed", "stop:deadline", "stop:nonzero",
                }
            )
            allowed_prefix = ("KILL_GROUP:", "stop:", "list:")
            for entry in value:
                _require(type(entry) is str and len(entry) <= 96, "unsafe diagnostic")
                _require(entry in accepted or (
                    key != "observed_phases" and entry.startswith(allowed_prefix)
                    and re.fullmatch(r"(?:stop|list|KILL_GROUP):[A-Za-z][A-Za-z0-9_]{0,63}", entry)
                ), "unknown diagnostic")
            safe[key] = value
    if "initiator_interrupted" in result:
        _require(type(result["initiator_interrupted"]) is bool, "invalid interrupt flag")
        safe["initiator_interrupted"] = result["initiator_interrupted"]
    if "worker_error" in result:
        safe["worker_error"] = _safe_worker_error(result["worker_error"])
    envelope = result.get("result")
    if envelope is not None:
        _require(
            type(envelope) is dict
            and set(envelope) == {"source_sha", "app_id", "call_id", "rehearsal"},
            "invalid execution envelope",
        )
        call_id = envelope["call_id"]
        _require(
            envelope["source_sha"] == expected_sha
            and type(call_id) is str and bool(CALL_ID.fullmatch(call_id))
            and type(app_id) is str and envelope["app_id"] == app_id
            and result.get("call_id") == call_id
            and result.get("execution_status") == "ACCEPTED"
            and result.get("initiator_interrupted") is not True,
            "unverified execution identity",
        )
        _validate_compact_report(envelope["rehearsal"], expected_sha)
        safe["call_id"] = call_id
        safe["rehearsal"] = envelope["rehearsal"]
    else:
        _require(result.get("execution_status") != "ACCEPTED",
                 "accepted execution has no validated envelope")
        _require(result.get("call_id") is None, "unvalidated call ID")
    if result["status"] == "SYNTHETIC_REHEARSAL_CONFIRMED":
        _require(
            result.get("execution_status") == "ACCEPTED"
            and result.get("lifecycle_status") == "TERMINAL_CONFIRMED"
            and result.get("provider_removal") == "STOPPED_CONFIRMED_BY_LIST"
            and not result.get("cleanup_errors")
            and envelope is not None,
            "overall success without full terminal execution",
        )
    return safe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--independent-control-receipt")
    # Internal-only child is only spawned by the bounded parent after approval.
    parser.add_argument("--internal-worker", action="store_true", default=False)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--internal-owner-nonce")
    args = parser.parse_args(argv)
    checkout = Path(__file__).resolve().parents[1]
    try:
        local_source(checkout, args.expected_sha)
        if args.internal_worker:
            _require(
                not args.execute
                and args.approval_file is not None
                and args.approval_sha256 is not None
                and args.journal is not None
                and args.result_file is not None,
                "worker approval and output paths required",
            )
            _require(
                isinstance(args.internal_owner_nonce, str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", args.internal_owner_nonce))
                and secrets.compare_digest(
                    args.internal_owner_nonce,
                    os.environ.get("CLIPPER_MODAL_WORKER_NONCE", ""),
                ),
                "bounded owner capability missing",
            )
            _require(
                args.journal.parent == args.result_file.parent
                and args.journal.name == "ids.jsonl"
                and args.result_file.name == "result.json"
                and args.journal.parent.is_dir()
                and not args.journal.is_symlink()
                and not args.result_file.is_symlink(),
                "invalid worker output paths",
            )
            approval = load_approval(args.approval_file, args.approval_sha256, args.expected_sha)
            _require(
                args.independent_control_receipt == approval.external_control_id,
                "lifecycle receipt mismatch",
            )
            _require_credential_route()
            return _internal_worker(approval, args.journal, args.result_file)
        if not args.execute:
            _require(
                not any(
                    (
                        args.approval_file,
                        args.approval_sha256,
                        args.independent_control_receipt,
                        args.journal,
                        args.result_file,
                        args.internal_owner_nonce,
                    )
                ),
                "plan mode accepts no execution/credential fields",
            )
            print(
                json.dumps(
                    {
                        "status": "DORMANT_PLAN_ONLY",
                        "source_sha": args.expected_sha,
                        "intent": INTENT,
                        "function_limits": LIMITS,
                        "credential_or_modal_import": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        _require(
            args.approval_file is not None and args.approval_sha256 is not None,
            "separate owner approval required",
        )
        approval = load_approval(args.approval_file, args.approval_sha256, args.expected_sha)
        _require(
            args.independent_control_receipt == approval.external_control_id,
            "independent control authority receipt missing",
        )
        _require(
            args.journal is None and args.result_file is None and args.internal_owner_nonce is None,
            "owner-only worker flags prohibited",
        )
        _require_credential_route()
        result = execute(
            args.expected_sha,
            approval,
            BoundedModalControl(args.approval_file, args.approval_sha256),
            external_control_verified=True,
        )
        try:
            visible = _public_result(result, args.expected_sha)
        except (LaunchBlocked, ValueError, TypeError):
            # Preserve verified exact-ID cleanup evidence if publication validation fails.
            prior = result.get("cleanup_errors")
            errors = list(prior) if isinstance(prior, list) else []
            errors.append("PUBLIC_EVIDENCE_VALIDATION_FAILED")
            # Build an independently safe fallback; never revalidate the
            # rejected envelope or raw diagnostics after publication fails.
            blocked: dict[str, object] = {
                "status": "BLOCKED",
                "execution_status": "UNVERIFIED",
                "cleanup_errors": ["PUBLIC_EVIDENCE_VALIDATION_FAILED"],
            }
            if result.get("provider_removal") in {"UNKNOWN", "STOPPED_CONFIRMED_BY_LIST"}:
                blocked["provider_removal"] = result["provider_removal"]
            candidate_app = result.get("app_id")
            if type(candidate_app) is str and APP_ID.fullmatch(candidate_app):
                blocked["app_id"] = candidate_app
            print(json.dumps(_public_result(blocked, args.expected_sha), sort_keys=True))
            return 2
        print(json.dumps(visible, sort_keys=True))
        return 0 if result.get("status") == "SYNTHETIC_REHEARSAL_CONFIRMED" else 2
    except LaunchInterrupted as interrupted:
        print(json.dumps(_public_result(interrupted.evidence, args.expected_sha), sort_keys=True))
        return 130
    except (LaunchBlocked, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        print(json.dumps({"status": "DORMANT_LAUNCH_BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

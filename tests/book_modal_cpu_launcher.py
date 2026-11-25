"""Dormant, explicitly authorized, model-free Modal CPU rehearsal launcher.

Default CLI only verifies local source and prints a plan: no Modal import.
Live execute is separately authorized, currently not approved. No workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise LaunchBlocked(reason)


def _json(path: Path, limit: int = 8192) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "approval must be a regular local file")
    _require(path.stat().st_size <= limit, "approval file exceeds limit")
    raw = path.read_bytes()
    data = json.loads(raw)
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
    _require(checkout.resolve() == Path(__file__).resolve().parents[1], "launcher must use own checkout")
    _require(_git(checkout, "rev-parse", "HEAD") == sha, "source checkout SHA differs")
    _require(not _git(checkout, "status", "--porcelain=v1", "--untracked-files=all"),
             "source checkout is not clean")
    _require((checkout / "tests" / "book_public_rehearsal.py").is_file(),
             "rehearsal entrypoint missing")


@dataclass(frozen=True)
class Approval:
    source_sha: str
    workspace: str
    environment: str
    credential_route: str
    expires_utc: datetime
    cost_ceiling_usd: str
    external_control_id: str


def load_approval(path: Path, digest: str, sha: str, *, now: datetime | None = None) -> Approval:
    _require(bool(HEX64.fullmatch(digest)), "out-of-band approval file SHA-256 required")
    raw = path.read_bytes() if path.is_file() and not path.is_symlink() else b""
    _require(0 < len(raw) <= 8192, "missing or invalid external approval")
    _require(hashlib.sha256(raw).hexdigest() == digest, "approval digest mismatch")
    data = _json(path)
    expected = {
        "schema", "intent", "source_sha", "workspace", "environment",
        "credential_route", "expires_utc", "cost_ceiling_usd",
        "external_control_id", "execution_permission",
    }
    _require(set(data) == expected, "approval fields mismatch")
    _require(data["schema"] == "clipper-modal-book-launch-approval-v1", "approval schema mismatch")
    _require(data["intent"] == INTENT, "wrong operation authorized")
    _require(data["execution_permission"] is True, "no explicit permission")
    _require(isinstance(data["source_sha"], str) and data["source_sha"] == sha,
             "approval SHA mismatch")
    for key in ("workspace", "environment", "credential_route", "external_control_id"):
        value = data[key]
        _require(isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_.:/-]{3,128}", value)),
                 "invalid or missing approval " + key)
    _require(
        data["credential_route"] in {"explicit-modal-profile", "actions-existing-secret-route"},
        "unapproved credential route",
    )
    _require(isinstance(data["cost_ceiling_usd"], str) and bool(
        re.fullmatch(r"(?:0|[1-9][0-9]{0,3})\.[0-9]{2}", data["cost_ceiling_usd"])
    ) and float(data["cost_ceiling_usd"]) > 0, "explicit positive cost cap required")
    try:
        expiry = datetime.fromisoformat(data["expires_utc"].replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise LaunchBlocked("invalid UTC approval expiry") from exc
    _require(expiry.tzinfo is not None and expiry.utcoffset().total_seconds() == 0,
             "approval expiry must be UTC")
    current = now if now is not None else datetime.now(UTC)
    _require(current < expiry, "approval expired")
    _require((expiry-current).total_seconds() <= 3600, "approval exceeds one-hour validity")
    return Approval(sha, data["workspace"], data["environment"],
                    data["credential_route"], expiry,
                    data["cost_ceiling_usd"], data["external_control_id"])


def _remote_child(sha: str) -> dict[str, object]:
    """Only executed in a separately approved restricted CPU Function."""
    import os as child_os
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

    env = {"PATH": child_os.environ.get("PATH", "/usr/bin:/bin"),
           "PYTHONPATH": str(checkout / "src") + ":" + str(checkout / "tests"),
           "HOME": str(output.parent),
           "PYTHONDONTWRITEBYTECODE": "1"}
    proc = child_subprocess.run(
        ["python", str(checkout / "tests" / "book_public_rehearsal.py"),
         "--fresh-root", str(output), "--expected-sha", sha],
        cwd=checkout, env=env, capture_output=True, text=True,
        check=False, timeout=25,
    )
    if proc.returncode != 0 or len(proc.stdout) > 220_000:
        raise RuntimeError("synthetic child failed; full output withheld")
    payload = child_json.loads(proc.stdout)
    if not isinstance(payload, dict) or payload.get("source_sha") != sha:
        raise RuntimeError("unexpected synthetic result identity")
    if (payload.get("status") != "SYNTHETIC_CONTRACT_REHEARSAL_PASSED"
            or payload.get("publish_eligible") is not False
            or payload.get("source_pin") != "VERIFIED_CLEAN_LOCAL_GIT_HEAD"):
        raise RuntimeError("synthetic result not accepted")
    return {
        "source_sha": sha,
        "status": payload["status"],
        "source_pin": payload["source_pin"],
        "publish_eligible": False,
        "tracks": payload["tracks"],
        "zero_call_evidence": payload["zero_call_evidence"],
        "artifact_sha256": payload["artifact_sha256"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "peak_rss_kib_platform": payload["peak_rss_kib_platform"],
        "real_model_calls": payload.get("real_model_calls"),
        "network_calls": payload.get("network_calls"),
        "publication_calls": payload.get("publication_calls"),
    }


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
        import modal

        from importlib.metadata import version

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
            "git -c credential.helper= clone --no-checkout "
            + SOURCE_URL + " /opt/clipper; "
            "git -C /opt/clipper checkout --detach " + sha + "; "
            "test \"$(git -C /opt/clipper rev-parse HEAD)\" = " + sha + "; "
            "test -z \"$(git -C /opt/clipper status --porcelain=v1 --untracked-files=all)\"; "
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
            _require(isinstance(app_id, str) and bool(re.fullmatch(r"ap-[A-Za-z0-9-]+", app_id)),
                     "App id missing")
            record("APP_ID", app_id)
            call: Any = None
            try:
                record("SPAWN_BEGIN", app_id)
                call = function.spawn(sha)
                call_id = getattr(call, "object_id", None)
                _require(isinstance(call_id, str) and bool(call_id),
                         "FunctionCall id missing")
                record("CALL_ID", call_id)
                record("GET_BEGIN", call_id)
                output = call.get(timeout=30)
                _validate_child_report(output, sha)
                record("RESULT_ACCEPTED", call_id)
                return {"source_sha": sha, "app_id": app_id, "call_id": call_id,
                        "rehearsal": output}
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
    _require(payload.get("source_sha") == sha
             and payload.get("source_pin") == "VERIFIED_CLEAN_LOCAL_GIT_HEAD",
             "synthetic source pin mismatch")
    _require(payload.get("status") == "SYNTHETIC_CONTRACT_REHEARSAL_PASSED"
             and payload.get("publish_eligible") is False, "synthetic status rejected")
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
    _require(isinstance(proof, dict) and proof.get("attempts") == [],
             "guard attempts not empty")
    _require(payload.get("real_model_calls") == 0
             and payload.get("network_calls") == 0
             and payload.get("publication_calls") == 0,
             "synthetic zero-call evidence invalid")


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
            phase, identifier = entry.get("phase"), entry.get("id")
            if not isinstance(phase, str) or not isinstance(identifier, str):
                phases.append("MALFORMED_RECORD")
                continue
            phases.append(phase)
            if phase == "APP_ID" and re.fullmatch(r"ap-[A-Za-z0-9-]+", identifier):
                app_id = identifier
    return app_id, phases


def _stop_and_observe(
    app_id: str,
    environment: str,
    deadline: float,
    runner: Any = subprocess.run,
) -> tuple[bool, list[str]]:
    """Stop by exact ID, then require exact stopped row from supported app list.

    Modal 1.5.5 has app stop/list, NOT app info. The current public CLI
    formats list --json as [{app_id, state, tasks, ...}]. Unverified or
    version-divergent schemas are never interpreted as terminal success.
    """
    errors: list[str] = []
    commands = (
        ("stop", ["modal", "app", "stop", "-y", "--env", environment, app_id]),
        ("list", ["modal", "app", "list", "--json", "--limit", "0", "--env", environment]),
    )
    stopped = False
    for kind, command in commands:
        remaining = min(8.0, deadline - time.monotonic())
        if remaining <= 0:
            errors.append(kind + ":deadline")
            return False, errors
        try:
            response = runner(
                command, check=False, text=True, capture_output=True, timeout=remaining
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(kind + ":" + type(exc).__name__)
            continue
        if response.returncode != 0:
            errors.append(kind + ":nonzero")
            continue
        if kind == "stop":
            stopped = True
            continue
        try:
            rows = json.loads(response.stdout)
        except (ValueError, TypeError):
            errors.append("list:invalid_json")
            continue
        if not isinstance(rows, list):
            errors.append("list:invalid_schema")
            continue
        matches = [
            item for item in rows
            if isinstance(item, dict) and item.get("app_id") == app_id
        ]
        if len(matches) != 1:
            errors.append("list:exact_id_not_found_or_duplicate")
            continue
        row = matches[0]
        if (row.get("state") == "stopped" and row.get("tasks") == "0"
                and stopped and not errors):
            return True, []
        errors.append("list:terminal_or_tasks_unconfirmed")
    return False, errors


def _bounded_run(approval: Approval, manifest_path: Path, digest: str,
                 *, deadline_seconds: float = 120,
                 popen: Any = subprocess.Popen,
                 runner: Any = subprocess.run) -> dict[str, object]:
    """One bounded control-plane process, plus bounded exact-ID stop/info."""
    _require(0 < deadline_seconds <= 180, "unbounded control-plane deadline")
    deadline = time.monotonic() + deadline_seconds
    with tempfile.TemporaryDirectory(prefix="clipper-modal-owner-") as directory:
        folder = Path(directory)
        ledger, result_file = folder / "ids.jsonl", folder / "result.json"
        stdout_file, stderr_file = folder / "worker.stdout", folder / "worker.stderr"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--internal-worker",
            "--expected-sha", approval.source_sha,
            "--approval-file", str(manifest_path), "--approval-sha256", digest,
            "--journal", str(ledger), "--result-file", str(result_file),
        ]
        # Capture only bounded diagnostic digests/sizes; never expose raw
        # stdout/stderr, which might contain provider or environment details.
        with stdout_file.open("wb") as stdout_sink, stderr_file.open("wb") as stderr_sink:
            proc = popen(
                command, cwd=Path(__file__).resolve().parents[1],
                start_new_session=True, stdout=stdout_sink, stderr=stderr_sink
            )
        timed_out = False
        process_cleanup_errors: list[str] = []
        try:
            proc.wait(timeout=max(0.1, deadline-time.monotonic()-20))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError) as exc:
                    process_cleanup_errors.append("KILL_GROUP:" + type(exc).__name__)
                try:
                    proc.wait(timeout=min(2.0, max(0.1, deadline-time.monotonic())))
                except subprocess.TimeoutExpired:
                    timed_out = True
        app_id, phases = _read_journal(ledger)
        confirmed = False
        cleanup_errors: list[str] = list(process_cleanup_errors)
        if "CALL_CANCEL_ERROR" in phases:
            cleanup_errors.append("CALL_CANCEL_ERROR")
        if "MALFORMED_RECORD" in phases:
            cleanup_errors.append("INVALID_JOURNAL")
        if app_id is not None:
            confirmed, stop_errors = _stop_and_observe(
                app_id, approval.environment, deadline, runner=runner
            )
            cleanup_errors.extend(stop_errors)
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
        if result_file.is_file() and result_file.stat().st_size <= 220_000:
            try:
                parsed = json.loads(result_file.read_text())
                if isinstance(parsed, dict):
                    output = parsed
                    _validate_child_report(output.get("rehearsal"), approval.source_sha)
            except (ValueError, LaunchBlocked):
                cleanup_errors.append("RESULT_UNVERIFIED")
        else:
            cleanup_errors.append("NO_RESULT")
        return {
            "status": "SYNTHETIC_REHEARSAL_CONFIRMED" if (
                confirmed and not cleanup_errors and output is not None
            ) else "BLOCKED",
            "provider_removal": "STOPPED_CONFIRMED_BY_LIST" if confirmed else "UNKNOWN",
            "app_id": app_id,
            "observed_phases": phases,
            "cleanup_errors": cleanup_errors,
            "worker_diagnostics": diagnostics,
            "result": output if confirmed and not cleanup_errors else None,
        }


def _internal_worker(approval: Approval, journal: Path, result_file: Path) -> int:
    controller = ModalControl()
    try:
        _require(controller.identify(approval) == approval.workspace, "workspace mismatch")
        result = controller.run_remote(
            approval, lambda phase, value: _journal(journal, phase, value)
        )
        result_file.write_text(json.dumps(result, sort_keys=True))
        with result_file.open("rb") as f:
            os.fsync(f.fileno())
        return 0
    except Exception as exc:
        _journal(journal, "WORKER_ERROR", type(exc).__name__)
        return 2


class BoundedModalControl:
    """Client identity before resources; image/build runs in a bounded subprocess."""

    def __init__(self, manifest_path: Path, digest: str) -> None:
        self.manifest_path = manifest_path
        self.digest = digest
        self._client = ModalControl()

    def identify(self, approval: Approval) -> str:
        return self._client.identify(approval)

    def run_once(self, approval: Approval) -> dict[str, object]:
        return _bounded_run(approval, self.manifest_path, self.digest)


def execute(sha: str, approval: Approval, control: Control,
            *, external_control_verified: bool) -> dict[str, object]:
    _require(external_control_verified, "independent lifecycle control not verified")
    _require(approval.source_sha == sha, "approval source mismatch")
    _require(datetime.now(UTC) < approval.expires_utc, "approval expired before execution")
    observed = control.identify(approval)
    _require(observed == approval.workspace, "client-bound workspace identity mismatch")
    _require(datetime.now(UTC) < approval.expires_utc, "approval expired before resources")
    return control.run_once(approval)


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
    args = parser.parse_args(argv)
    checkout = Path(__file__).resolve().parents[1]
    try:
        local_source(checkout, args.expected_sha)
        if args.internal_worker:
            # The worker is an implementation detail, not a second user
            # entrypoint. The current workspace cannot enforce the approved
            # dollar cap. Do not allow a direct worker invocation to bypass
            # the outer --execute blocker by supplying its arguments.
            raise LaunchBlocked("internal worker disabled pending hard budget gate")
        if not args.execute:
            _require(not any((args.approval_file, args.approval_sha256,
                              args.independent_control_receipt,
                              args.journal, args.result_file)),
                     "plan mode accepts no execution/credential fields")
            print(json.dumps({"status": "DORMANT_PLAN_ONLY",
                              "source_sha": args.expected_sha,
                              "intent": INTENT, "function_limits": LIMITS,
                              "credential_or_modal_import": False}, sort_keys=True))
            return 0
        _require(args.approval_file is not None and args.approval_sha256 is not None,
                 "separate owner approval required")
        approval = load_approval(args.approval_file, args.approval_sha256, args.expected_sha)
        _require(
            args.independent_control_receipt == approval.external_control_id,
            "independent control authority receipt missing",
        )
        # Fail closed on implicit/ambiguous credential sources before importing Modal.
        profile = os.environ.get("MODAL_PROFILE", "")
        token_id = os.environ.get("MODAL_TOKEN_ID", "")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
        if approval.credential_route == "explicit-modal-profile":
            _require(bool(profile) and not token_id and not token_secret,
                     "explicit profile required without environment token overrides")
        else:
            _require(bool(token_id) and bool(token_secret) and not profile,
                     "existing Actions token pair required without profile ambiguity")
        # A parsed dollar amount and a matching receipt string are not
        # independently enforceable spend authority. The bounded subprocess
        # and cleanup controls are implemented/testable offline, but a live
        # budget guard is NOT supplied by the current Starter workspace.
        # Reject BEFORE constructing any SDK/client or remote resource.
        raise LaunchBlocked("independent enforceable account budget cap missing")
        # No live execution branch is reachable until a separately reviewed
        # authenticated budget control replaces the explicit blocker above.
    except (LaunchBlocked, OSError, ValueError, json.JSONDecodeError,
            subprocess.SubprocessError):
        print(json.dumps({"status": "DORMANT_LAUNCH_BLOCKED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

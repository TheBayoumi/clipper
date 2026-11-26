"""Explicit private Modal Function launcher; import and preflight allocate nothing."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from clipper.stage_contracts import content_fingerprint

from . import acquisition, evaluation, pipeline_plan, private_phase

APP = re.compile(r"ap-[A-Za-z0-9]{22}")
CALL = re.compile(r"fc-[A-Za-z0-9_-]{3,128}")
SOURCE = "https://github.com/TheBayoumi/clipper.git"
STARTUP_TIMEOUT = 120
RESULT_TRANSPORT_MARGIN = 180


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise private_phase.PrivatePhaseError(reason)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def verify_checkout(root: Path, commit: str) -> None:
    require(root.resolve() == Path(__file__).resolve().parents[3], "own checkout required")
    require(git(root, "rev-parse", "HEAD") == commit, "checkout commit mismatch")
    require(
        not git(root, "status", "--porcelain=v1", "--untracked-files=all"),
        "checkout not clean",
    )


def identify() -> tuple[Any, Any]:
    """Explicitly injected environment credentials only; never a profile fallback."""
    require(
        bool(os.environ.get("MODAL_TOKEN_ID"))
        and bool(os.environ.get("MODAL_TOKEN_SECRET"))
        and not os.environ.get("MODAL_PROFILE"),
        "private SDK credentials required",
    )
    require(os.environ.get("MODAL_ENVIRONMENT") == "main", "environment must be main")
    require(version("modal") == "1.5.5", "unsupported Modal SDK")
    import modal

    client = None
    try:
        client = modal.Client.from_env()
        workspace = modal.Workspace.from_context(client=client)
        workspace.hydrate(client=client)
        require(str(workspace.name) == private_phase.WORKSPACE, "workspace mismatch")
        return modal, client
    except Exception:
        if client is not None:
            close_client(client)
        raise private_phase.PrivatePhaseError("private SDK identity unavailable") from None


def close_client(client: Any) -> bool:
    try:
        client.close()
    except Exception:
        return False
    return True


def _remote_phase(
    manifest: dict[str, Any], packet: bytes, checkpoint: bytes | None
) -> tuple[dict[str, object], bytes | None]:
    """Reviewed image; workspace scope comes from the authenticated caller guard."""
    import subprocess
    from pathlib import Path

    from clipper.books.private_phase import execute_private_phase

    checkout = Path("/opt/clipper")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "diff", "--exit-code", "HEAD"],
        capture_output=True,
        timeout=10,
    ).returncode
    if head != manifest["commit"] or dirty:
        raise RuntimeError("remote source commit mismatch")
    return execute_private_phase(
        manifest,
        packet,
        checkpoint,
        verified_commit=head,
        verified_workspace="mahmoud-bayoumi-mm",
    )


def factory(modal: Any, manifest: dict[str, Any]) -> tuple[Any, Any]:
    """Pinned public source and dependencies; no local file upload or secret mount."""
    sha = manifest["commit"]
    require(private_phase.HEX40.fullmatch(sha) is not None, "invalid source commit")
    clone = (
        "set -eu; export GIT_TERMINAL_PROMPT=0; "
        f"git -c credential.helper= clone --no-checkout {SOURCE} /opt/clipper; "
        f"git -C /opt/clipper checkout --detach {sha}; "
        f'test "$(git -C /opt/clipper rev-parse HEAD)" = {sha}; '
        'test -z "$(git -C /opt/clipper status --porcelain=v1 --untracked-files=all)"'
    )
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .run_commands(clone)
        .pip_install(
            "PyYAML==6.0.3",
            "numpy==2.2.6",
            "opencv-python-headless==4.12.0.88",
            "gdown==6.1.0",
            "yt-dlp==2026.7.4",
            "transformers==4.57.3",
            "huggingface_hub==0.36.2",
            "lm-format-enforcer==0.11.3",
        )
        .run_commands(
            "python -m pip install --no-cache-dir "
            "--index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu"
        )
        .env({"PYTHONPATH": "/opt/clipper/src", "HF_HOME": "/opt/hf"})
        .run_commands(
            "python -c 'from clipper.books.private_modal import _remote_phase; "
            "from clipper.books import constrained, private_phase; "
            "from clipper.books.scripting import _pinned_qwen_snapshot; "
            "_pinned_qwen_snapshot()'"
        )
        .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    )
    app = modal.App(include_source=False)
    function = app.function(
        image=image,
        include_source=False,
        cpu=(4, 4),
        memory=(16384, 16384),
        gpu=None,
        block_network=True,
        restrict_modal_access=True,
        single_use_containers=True,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        retries=0,
        startup_timeout=STARTUP_TIMEOUT,
        timeout=manifest["timeout_seconds"],
        scaledown_window=2,
    )(_remote_phase)
    return app, function


class Journal:
    """Exclusive per-phase owner claim and fsynced immutable event records."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.identity = content_fingerprint(manifest)
        self.sequence = 0
        # Exclude owner/config changes from dispatch uniqueness for this source phase.
        key = content_fingerprint(
            {
                "workspace": manifest["workspace"],
                "bundle": manifest["bundle_sha256"],
                "phase": manifest["phase"],
            }
        )
        self.directory = root / key
        self.directory.mkdir(mode=0o700, exist_ok=False)
        self.record("CLAIMED")
        self.sync(root)

    @staticmethod
    def sync(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def record(self, event: str, **fields: object) -> None:
        self.sequence += 1
        payload = {
            "event": event,
            "sequence": self.sequence,
            "manifest_sha256": self.identity,
            "owner": self.manifest["owner"],
            **fields,
        }
        acquisition._private_write(
            self.directory / f"{self.sequence:04d}.json",
            evaluation._json_bytes(payload),
            private_root=self.root,
        )
        self.sync(self.directory)


def observe(app_id: str) -> bool:
    """Exact positive terminal evidence; no absent-ID success."""
    require(APP.fullmatch(app_id) is not None, "invalid App ID")
    proc = subprocess.run(
        [sys.executable, "-m", "modal", "app", "list", "--json", "--env", "main"],
        capture_output=True,
        text=True,
        check=True,
        timeout=8,
    )
    rows = json.loads(proc.stdout)
    require(isinstance(rows, list), "invalid app inventory")
    matches = [row for row in rows if isinstance(row, dict) and row.get("app_id") == app_id]
    require(len(matches) == 1, "exact app record absent or ambiguous")
    row = matches[0]
    require(
        type(row.get("state")) is str
        and type(row.get("tasks")) is str
        and row["tasks"].isascii()
        and row["tasks"].isdecimal(),
        "invalid app state evidence",
    )
    return bool(row["state"] == "stopped" and row["tasks"] == "0")


def cleanup(app_id: str, journal: Journal) -> bool:
    try:
        # Recheck credentials/workspace before using the independently invoked CLI.
        _, observer = identify()
        if not close_client(observer):
            raise private_phase.PrivatePhaseError("observer client close failed")
        journal.record("OBSERVE_BEGIN", app_id=app_id)
        if observe(app_id):
            journal.record("STOPPED_ZERO_TASKS", app_id=app_id)
            return True
        journal.record("EXACT_STOP_BEGIN", app_id=app_id)
        proc = subprocess.run(
            [sys.executable, "-m", "modal", "app", "stop", "-y", "--env", "main", app_id],
            capture_output=True,
            check=False,
            timeout=8,
        )
        journal.record("EXACT_STOP_RETURNED", app_id=app_id, returncode=proc.returncode)
        for delay in (0.5, 1.0, 2.0):
            time.sleep(delay)
            if observe(app_id):
                journal.record("STOPPED_ZERO_TASKS", app_id=app_id)
                return True
    except Exception:
        journal.record("CLEANUP_OBSERVATION_FAILED", app_id=app_id)
    journal.record("CLEANUP_UNVERIFIED", app_id=app_id)
    return False


def private_directory(path: Path) -> bool:
    return (
        path.is_dir()
        and path.resolve() == path.absolute()
        and path.stat().st_uid == os.getuid()
        and path.stat().st_mode & 0o077 == 0
    )


def response(report: object, raw: object, run: dict[str, Any], identity: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise private_phase.PrivatePhaseError("invalid phase envelope")
    expected = (
        "BRIEF_READY_UNREVIEWED" if run["phase"] == "brief" else "CHAPTER_SCRIPT_READY_UNREVIEWED"
    )
    status = report.get("status")
    require(
        status in (expected, "PRIVATE_PHASE_FAILED_NO_RETRY")
        and report.get("manifest_sha256") == identity
        and report.get("owner") == run["owner"]
        and report.get("publish_eligible") is False,
        "private response identity mismatch",
    )
    retention = report.get("retention")
    if raw is None:
        require(
            status == "PRIVATE_PHASE_FAILED_NO_RETRY"
            and retention == "OUTPUT_NOT_RETAINED"
            and report.get("checkpoint_sha256") is None
            and report.get("checkpoint_bytes") == 0,
            "missing checkpoint evidence",
        )
    else:
        if not isinstance(raw, bytes):
            raise private_phase.PrivatePhaseError("invalid checkpoint type")
        require(
            type(report.get("checkpoint_bytes")) is int
            and report["checkpoint_bytes"] == len(raw)
            and report.get("checkpoint_sha256") == private_phase.sha(raw)
            and retention == ("COMPLETE" if status == expected else "PARTIAL_NONRESUMABLE"),
            "private checkpoint response mismatch",
        )
        files = private_phase.unpack(raw)
        require(
            all(private_phase.NATIVE_PATH.fullmatch(name) for name in files),
            "response checkpoint inventory",
        )
        require(
            ("book-candidates/failed-phase.json" in files) == (status != expected),
            "checkpoint retention marker mismatch",
        )
    safe: dict[str, Any] = {
        key: report[key]
        for key in (
            "status",
            "manifest_sha256",
            "owner",
            "retention",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "publish_eligible",
        )
    }
    for key in ("brief_identity", "configured_execution_receipt_identity", "final_result_identity"):
        if pipeline_plan.is_digest(report.get(key)):
            safe[key] = report[key]
    if status == expected:
        require(
            "brief_identity" in safe
            and "configured_execution_receipt_identity" in safe
            and (run["phase"] != "script" or "final_result_identity" in safe),
            "missing completed artifact identity",
        )
    else:
        require(
            type(report.get("omitted_members")) is int
            and report["omitted_members"] >= 0
            and type(report.get("inventory_truncated")) is bool,
            "missing partial inventory evidence",
        )
        safe["omitted_members"] = report["omitted_members"]
        safe["inventory_truncated"] = report["inventory_truncated"]
    return safe


def _dispatch_qualified(
    manifest: object,
    packet: bytes,
    checkpoint: bytes | None,
    *,
    approved_manifest_sha256: str,
    execute: bool,
    checkout: Path,
    journal_root: Path,
    output: Path,
) -> dict[str, object]:
    run, _, _ = private_phase.preflight(manifest, packet, checkpoint)
    identity = content_fingerprint(run)
    require(identity == approved_manifest_sha256, "reviewed manifest digest mismatch")
    require(type(execute) is bool, "explicit execution boolean required")
    if not execute:
        return {"status": "PREFLIGHT_ONLY", "manifest_sha256": identity}
    verify_checkout(checkout, run["commit"])
    require(
        private_directory(journal_root),
        "existing private journal root required",
    )
    require(
        private_directory(output.parent),
        "private output parent required",
    )
    require(not output.exists() and not output.is_symlink(), "checkpoint output already exists")
    journal = Journal(journal_root, run)
    client = None
    app_id: str | None = None
    call_id: str | None = None
    saved = False
    result: dict[str, Any] = {}
    cleaned = False
    scope_exited = False
    try:
        journal.record("IDENTIFY_BEGIN")
        modal, client = identify()
        journal.record("WORKSPACE_VERIFIED")
        journal.record("BUILD_BEGIN")
        app, function = factory(modal, run)
        journal.record("APP_RUN_BEGIN")
        with app.run(client=client, detach=False, environment_name="main"):
            candidate = app.app_id
            require(
                isinstance(candidate, str) and APP.fullmatch(candidate) is not None,
                "exact App ID unavailable",
            )
            app_id = candidate
            journal.record("APP_ID", app_id=app_id)
            journal.record("SPAWN_BEGIN", app_id=app_id)
            call = function.spawn(run, packet, checkpoint)
            candidate = call.object_id
            require(
                isinstance(candidate, str) and CALL.fullmatch(candidate) is not None,
                "exact Call ID unavailable",
            )
            call_id = candidate
            journal.record("CALL_ID", app_id=app_id, call_id=call_id)
            journal.record("GET_BEGIN", app_id=app_id, call_id=call_id)
            report, raw = call.get(
                timeout=run["timeout_seconds"] + STARTUP_TIMEOUT + RESULT_TRANSPORT_MARGIN
            )
            result = response(report, raw, run, identity)
            journal.record("RESPONSE_VERIFIED", app_id=app_id, call_id=call_id, response=result)
            if raw is not None:
                journal.record("CHECKPOINT_WRITE_BEGIN", app_id=app_id, call_id=call_id)
                private_phase.persist_checkpoint(output, raw, result["checkpoint_sha256"])
                saved = True
                journal.record(
                    "CHECKPOINT_VERIFIED",
                    app_id=app_id,
                    call_id=call_id,
                    checkpoint_sha256=result["checkpoint_sha256"],
                )
            journal.record("APP_SCOPE_EXIT_BEGIN", app_id=app_id, call_id=call_id)
        journal.record("APP_SCOPE_EXIT_RETURNED", app_id=app_id)
        scope_exited = True
    except Exception:
        journal.record("DISPATCH_FAILED_NO_RETRY", app_id=app_id, call_id=call_id)
    finally:
        try:
            if app_id is not None:
                cleaned = cleanup(app_id, journal)
            else:
                journal.record("RESOURCE_IDENTITY_UNVERIFIED")
        finally:
            if client is not None:
                closed = close_client(client)
                journal.record("CLIENT_CLOSED" if closed else "CLIENT_CLOSE_FAILED")
                cleaned = cleaned and closed
    expected = (
        "BRIEF_READY_UNREVIEWED" if run["phase"] == "brief" else "CHAPTER_SCRIPT_READY_UNREVIEWED"
    )
    return {
        "status": (
            expected
            if saved and cleaned and scope_exited and result.get("status") == expected
            else "PRIVATE_PHASE_BLOCKED_NO_RETRY"
        ),
        "app_id": app_id,
        "call_id": call_id,
        "checkpoint_saved": saved,
        "retention": result.get("retention") if saved else "OUTPUT_NOT_RETAINED",
        "cleanup_verified": cleaned,
        "manifest_sha256": identity,
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "publish_eligible": False,
    }


def launch(
    manifest: object,
    packet: bytes,
    checkpoint: bytes | None,
    *,
    approved_manifest_sha256: str,
    execute: bool,
    checkout: Path,
    journal_root: Path,
    output: Path,
) -> dict[str, object]:
    """Public boundary remains preflight-only until merchant/owner qualification."""
    run, _, _ = private_phase.preflight(manifest, packet, checkpoint)
    identity = content_fingerprint(run)
    require(identity == approved_manifest_sha256, "reviewed manifest digest mismatch")
    require(type(execute) is bool, "explicit execution boolean required")
    if execute:
        raise private_phase.PrivatePhaseError(
            "live phase blocked pending merchant zero-overage and owner resource approval"
        )
    return {"status": "PREFLIGHT_ONLY", "manifest_sha256": identity}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "packet", "checkout", "journal-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--approved-manifest-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = launch(
            pipeline_plan.strict_json(pipeline_plan._read(args.manifest, 16384)),
            evaluation._read(args.packet, private_phase.MAX_PACKET),
            evaluation._read(args.checkpoint, private_phase.MAX_PACKET)
            if args.checkpoint
            else None,
            approved_manifest_sha256=args.approved_manifest_sha256,
            execute=args.execute,
            checkout=args.checkout,
            journal_root=args.journal_root,
            output=args.output,
        )
    except Exception:
        print(json.dumps({"status": "PRIVATE_PHASE_BLOCKED_NO_RETRY"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 2 if result["status"] == "PRIVATE_PHASE_BLOCKED_NO_RETRY" else 0


if __name__ == "__main__":
    raise SystemExit(main())

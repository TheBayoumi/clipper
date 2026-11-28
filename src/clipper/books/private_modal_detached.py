"""Detached private Modal phase: start once, then poll in short separate invocations.

The long CPU phase must not depend on the client process that started it. `start`
spawns the reviewed Function under a detached App, journals the exact App ID and
FunctionCall ID in a private attach record, and exits without stopping the App.
`poll` is a separate short invocation: it re-authenticates, re-attaches by ID and
either reports a pending run or verifies the result, persists the checkpoint and
runs the same exact-App cleanup as the synchronous launcher. Network stays blocked
in the container. The packet goes in and the compressed checkpoint comes out through a
dedicated private Volume named from the manifest digest. The Volume is deleted, and its
deletion verified, on every terminal path.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clipper.stage_contracts import content_fingerprint

from . import acquisition, evaluation, pipeline_plan, private_modal, private_phase

# Probe-decided: True keeps the container from using Modal resources beyond its mounted
# Volume. If the data-free probe shows the Volume commit needs more, this becomes False.
RESTRICT_ACCESS = True
CALL = re.compile(r"fc-[A-Za-z0-9]{10,64}")
ATTACH = "ATTACH.json"
DEADLINE_MARGIN_SECONDS = 900
MAX_ATTACH_BYTES = 32768
require = private_modal.require


def _finished(directory: Path) -> bool:
    for path in directory.glob("[0-9][0-9][0-9][0-9].json"):
        if json.loads(path.read_bytes()).get("event") == "DETACHED_FINISHED":
            return True
    return False


def _volume_exists(modal: Any, client: Any, name: str) -> bool:
    listed = modal.Volume.objects.list(environment_name="main", client=client)
    return any(getattr(item, "name", None) == name for item in listed)


def delete_volume(modal: Any, client: Any, name: str, journal: private_modal.Journal) -> bool:
    """Delete the run's own Volume and verify by exact-name inventory."""
    try:
        journal.record("VOLUME_DELETE_BEGIN", volume=name)
        modal.Volume.objects.delete(
            name, allow_missing=True, environment_name="main", client=client
        )
        if _volume_exists(modal, client, name):
            journal.record("VOLUME_DELETE_UNVERIFIED", volume=name)
            return False
        journal.record("VOLUME_DELETE_VERIFIED", volume=name)
        return True
    except Exception:
        journal.record("VOLUME_DELETE_FAILED", volume=name)
        return False


def _read_volume_file(volume: Any, path: str) -> bytes:
    chunks = []
    total = 0
    for chunk in volume.read_file(path):
        total += len(chunk)
        require(total <= private_modal.VOLUME_CHECKPOINT_SIZE_LIMIT, "volume file too large")
        chunks.append(chunk)
    return b"".join(chunks)


def start(
    manifest: object,
    packet: bytes,
    checkpoint: bytes | None,
    *,
    approved_manifest_sha256: str,
    execute: bool,
    checkout: Path,
    journal_root: Path,
    now: Callable[[], float] = time.time,
) -> dict[str, object]:
    run, _, _ = private_phase.preflight(manifest, packet, checkpoint)
    profile = private_phase.packet_profile(packet)
    identity = content_fingerprint(run)
    require(identity == approved_manifest_sha256, "reviewed manifest digest mismatch")
    require(type(execute) is bool, "explicit execution boolean required")
    if not execute:
        return {"status": "PREFLIGHT_ONLY", "manifest_sha256": identity}
    private_modal.verify_checkout(checkout, run["commit"])
    require(private_modal.private_directory(journal_root), "existing private journal root required")
    journal = private_modal.Journal(journal_root, run)
    client = None
    app_id: str | None = None
    spawned = False
    call_id: str | None = None
    modal: Any = None
    name = private_modal.volume_name(run)
    volume_made = False
    try:
        journal.record("IDENTIFY_BEGIN")
        modal, client = private_modal.identify()
        journal.record("WORKSPACE_VERIFIED")
        journal.record("VOLUME_CREATE_BEGIN", volume=name)
        modal.Volume.objects.create(
            name, allow_existing=False, environment_name="main", client=client
        )
        volume_made = True
        volume = modal.Volume.from_name(name, environment_name="main", client=client)
        with volume.batch_upload() as upload:
            upload.put_file(io.BytesIO(packet), "/" + private_modal.VOLUME_PACKET)
            if checkpoint is not None:
                upload.put_file(io.BytesIO(checkpoint), "/prior.bin")
        journal.record("VOLUME_PACKET_UPLOADED", volume=name)
        app, function = private_modal.factory(
            modal, run, restrict_access=RESTRICT_ACCESS, volume=volume, profile=profile
        )
        journal.record("APP_RUN_BEGIN", detached=True)
        with app.run(client=client, detach=True, environment_name="main"):
            candidate = app.app_id
            require(
                isinstance(candidate, str) and private_modal.APP.fullmatch(candidate) is not None,
                "exact App ID unavailable",
            )
            app_id = candidate
            journal.record("APP_ID", app_id=app_id)
            journal.record("SPAWN_BEGIN", app_id=app_id)
            call = function.spawn(run, name)
            candidate_call = call.object_id
            require(
                isinstance(candidate_call, str) and CALL.fullmatch(candidate_call) is not None,
                "exact FunctionCall ID unavailable",
            )
            call_id = candidate_call
            spawned = True
            record = {
                "app_id": app_id,
                "call_id": call_id,
                "volume": name,
                "manifest_sha256": identity,
                "owner": run["owner"],
                "run": run,
                "started_at": now(),
                "timeout_seconds": run["timeout_seconds"],
            }
            acquisition._private_write(
                journal.directory / ATTACH,
                evaluation._json_bytes(record),
                private_root=journal_root,
            )
            journal.record("SPAWNED", app_id=app_id, call_id=call_id)
        journal.record("DETACHED_RUNNING", app_id=app_id)
    except Exception:
        journal.record("START_FAILED_NO_RETRY", app_id=app_id)
        spawned = False
        if app_id is not None:
            private_modal.cleanup(app_id, journal)
        if volume_made:
            delete_volume(modal, client, name, journal)
    finally:
        if client is not None:
            closed = private_modal.close_client(client)
            journal.record("CLIENT_CLOSED" if closed else "CLIENT_CLOSE_FAILED")
    return {
        "status": "DETACHED_RUNNING" if spawned else "PRIVATE_PHASE_BLOCKED_NO_RETRY",
        "app_id": app_id,
        "call_id": call_id if spawned else None,
        "manifest_sha256": identity,
        "publish_eligible": False,
    }


def _load_attach(journal: private_modal.Journal, approved: str) -> dict[str, Any]:
    raw = evaluation._read(journal.directory / ATTACH, MAX_ATTACH_BYTES)
    record = pipeline_plan.strict_json(raw)
    run = record.get("run")
    require(
        isinstance(run, dict)
        and content_fingerprint(run) == approved == record.get("manifest_sha256")
        and record.get("owner") == run.get("owner")
        and isinstance(record.get("app_id"), str)
        and private_modal.APP.fullmatch(record["app_id"]) is not None
        and record.get("volume") == private_modal.volume_name(run)
        and isinstance(record.get("call_id"), str)
        and CALL.fullmatch(record["call_id"]) is not None
        and isinstance(record.get("started_at"), int | float)
        and isinstance(record.get("timeout_seconds"), int)
        and 0 < record["timeout_seconds"] <= 86400,
        "attach record does not match the reviewed manifest",
    )
    return record


def poll(
    manifest: object,
    *,
    approved_manifest_sha256: str,
    journal_root: Path,
    output: Path,
    now: Callable[[], float] = time.time,
) -> dict[str, object]:
    run_manifest = private_phase.validate_manifest(manifest)
    require(private_modal.private_directory(journal_root), "existing private journal root required")
    require(
        private_modal.private_directory(output.parent),
        "private output parent required",
    )
    journal = private_modal.Journal.reopen(journal_root, run_manifest)
    if _finished(journal.directory):
        return {"status": "ALREADY_FINISHED", "manifest_sha256": approved_manifest_sha256}
    record = _load_attach(journal, approved_manifest_sha256)
    run: dict[str, Any] = record["run"]
    app_id: str = record["app_id"]
    name: str = record["volume"]
    volume_gone = False
    identity = approved_manifest_sha256
    expected = (
        "BRIEF_READY_UNREVIEWED" if run["phase"] == "brief" else "CHAPTER_SCRIPT_READY_UNREVIEWED"
    )
    client = None
    modal: Any = None
    saved = False
    result: dict[str, Any] = {}
    cleaned = False
    pending = False
    try:
        modal, client = private_modal.identify()
        journal.record("POLL_BEGIN", app_id=app_id)
        call = modal.FunctionCall.from_id(record["call_id"], client=client)
        deadline = record["started_at"] + record["timeout_seconds"] + DEADLINE_MARGIN_SECONDS
        try:
            report, _unused = call.get(timeout=0)
        except TimeoutError:
            if now() <= deadline:
                journal.record("POLL_PENDING", app_id=app_id)
                pending = True
            else:
                journal.record("DEADLINE_EXCEEDED_NO_RETRY", app_id=app_id)
        else:
            require(not output.exists() and not output.is_symlink(), "checkpoint output exists")
            raw = None
            if isinstance(report, dict) and report.get("checkpoint_bytes"):
                try:
                    volume = modal.Volume.from_name(name, environment_name="main", client=client)
                    blob = _read_volume_file(volume, private_modal.VOLUME_CHECKPOINT)
                except Exception:
                    # A transport failure is not a verdict on the checkpoint: keep the
                    # Volume and the App so a later poll can read it again, until the deadline.
                    if now() > deadline:
                        raise
                    journal.record("VOLUME_READ_RETRYABLE", app_id=app_id)
                    pending = True
                else:
                    raw = private_modal.decompress_checkpoint(blob)
            if not pending:
                result = private_modal.response(report, raw, run, identity)
                journal.record("RESPONSE_VERIFIED", app_id=app_id, response=result)
            if raw is not None:
                journal.record("CHECKPOINT_WRITE_BEGIN", app_id=app_id)
                private_phase.persist_checkpoint(output, raw, result["checkpoint_sha256"])
                saved = True
                journal.record(
                    "CHECKPOINT_VERIFIED",
                    app_id=app_id,
                    checkpoint_sha256=result["checkpoint_sha256"],
                )
    except Exception:
        journal.record("POLL_FAILED_NO_RETRY", app_id=app_id)
    finally:
        try:
            if pending:
                journal.record("POLL_RELEASED_APP_KEPT", app_id=app_id)
            else:
                cleaned = private_modal.cleanup(app_id, journal)
                if client is not None:
                    volume_gone = delete_volume(modal, client, name, journal)
                cleaned = cleaned and volume_gone
        finally:
            if client is not None:
                closed = private_modal.close_client(client)
                journal.record("CLIENT_CLOSED" if closed else "CLIENT_CLOSE_FAILED")
                cleaned = cleaned and closed
    if pending:
        return {
            "status": "DETACHED_PENDING",
            "app_id": app_id,
            "manifest_sha256": identity,
            "publish_eligible": False,
        }
    if cleaned:
        journal.record("DETACHED_FINISHED", app_id=app_id)
    return {
        "status": (
            expected
            if saved and cleaned and result.get("status") == expected
            else "PRIVATE_PHASE_BLOCKED_NO_RETRY"
        ),
        "app_id": app_id,
        "checkpoint_saved": saved,
        "retention": result.get("retention") if saved else "OUTPUT_NOT_RETAINED",
        "cleanup_verified": cleaned,
        "manifest_sha256": identity,
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "publish_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "poll"):
        item = sub.add_parser(name)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--journal-root", type=Path, required=True)
        item.add_argument("--approved-manifest-sha256", required=True)
    start_parser = sub.choices["start"]
    start_parser.add_argument("--packet", type=Path, required=True)
    start_parser.add_argument("--checkout", type=Path, required=True)
    start_parser.add_argument("--checkpoint", type=Path)
    start_parser.add_argument("--execute", action="store_true")
    sub.choices["poll"].add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = pipeline_plan.strict_json(pipeline_plan._read(args.manifest, 16384))
        if args.command == "start":
            result = start(
                manifest,
                evaluation._read(args.packet, private_phase.MAX_PACKET),
                evaluation._read(args.checkpoint, private_phase.MAX_PACKET)
                if args.checkpoint
                else None,
                approved_manifest_sha256=args.approved_manifest_sha256,
                execute=args.execute,
                checkout=args.checkout,
                journal_root=args.journal_root,
            )
        else:
            result = poll(
                manifest,
                approved_manifest_sha256=args.approved_manifest_sha256,
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

"""Explicit two-chapter private Codespace driver; no public book content or rendering.

All private paths are outside the worktree. A caller must use the single
Codespace deadline wrapper; this module independently checks the shared clock.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from clipper.stage_contracts import content_fingerprint

from . import (
    acquisition,
    chapter,
    chapter_hybrid,
    chapter_lifecycle,
    evaluation,
    narration,
    scripting,
    selected_scope,
    whole_book,
)
from .orchestrator import BookOrchestrator

BATCH_SCHEMA = "clipper-two-private-chapter-codespace-v2"
DEADLINE_SECONDS = 7200
CLEANUP_RESERVE_SECONDS = 300
# Set only after a fresh REST + signature check. A caller's environment is not
# a source of authority for longer runtimes.
_VERIFIED_CLOCK: ContextVar[tuple[datetime, datetime] | None] = ContextVar(
    "_private_chapter_signed_clock", default=None
)


def _execution_cutoff(created: datetime) -> datetime:
    verified = _VERIFIED_CLOCK.get()
    if verified is not None and verified[0] == created:
        return verified[1]
    return created + timedelta(seconds=DEADLINE_SECONDS - CLEANUP_RESERVE_SECONDS)


DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class PrivateHostStop(ValueError):
    """Sanitized private-host phase refusal; source text is never emitted."""


class PhaseClaimContended(PrivateHostStop):
    """A concurrent caller lost an immutable claim; observation only."""


def _created_at_timestamp(value: str) -> datetime:
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise PrivateHostStop("verified Codespace creation timestamp required") from None
    if date.tzinfo is None:
        raise PrivateHostStop("Codespace timestamp must include UTC offset")
    return date.astimezone(UTC)


def _created_at(value: str, now: datetime | None = None) -> datetime:
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise PrivateHostStop("verified Codespace creation timestamp required") from None
    if date.tzinfo is None:
        raise PrivateHostStop("Codespace timestamp must include UTC offset")
    date = date.astimezone(UTC)
    current = now or datetime.now(UTC)
    if date > current or current >= _execution_cutoff(date):
        raise PrivateHostStop("one-host shared deadline expired or invalid")
    return date


def _check_deadline(created: datetime) -> None:
    if datetime.now(UTC) >= _execution_cutoff(created):
        raise PrivateHostStop("one-host shared execution budget exhausted")


def _private_path(value: str, root: Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise PrivateHostStop("private host path must be absolute and not a symlink")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise PrivateHostStop("private host input must be inside private artifact root")
    if must_exist and not resolved.is_file():
        raise PrivateHostStop("required private input does not exist")
    return resolved


def _config(value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "chapter_1", "chapter_7", "plan"}
        or value["schema"] != BATCH_SCHEMA
    ):
        raise PrivateHostStop("two-chapter private host manifest schema mismatch")
    bundles: set[str] = set()
    for number in (1, 7):
        block = value[f"chapter_{number}"]
        if not isinstance(block, dict) or set(block) != {
            "bundle_path",
            "bundle_sha256",
            "scope_sha256",
            "review_file",
        }:
            raise PrivateHostStop("chapter bundle/review fields missing")
        if (
            not isinstance(block["bundle_path"], str)
            or not block["bundle_path"]
            or block["bundle_path"] in bundles
            or not isinstance(block["bundle_sha256"], str)
            or DIGEST.fullmatch(block["bundle_sha256"]) is None
        ):
            raise PrivateHostStop("two independent immutable chapter bundles required")
        bundles.add(block["bundle_path"])
        if block["scope_sha256"] is not None and (
            not isinstance(block["scope_sha256"], str)
            or DIGEST.fullmatch(block["scope_sha256"]) is None
        ):
            raise PrivateHostStop("chapter scope digest malformed")
        if block["review_file"] is not None and not isinstance(block["review_file"], str):
            raise PrivateHostStop("chapter review must be an explicit private file")
    chapter.ChapterPlan.from_manifest(value["plan"])
    return value


def _section_manifest(
    runner: BookOrchestrator, verified: selected_scope.SelectedEvidence
) -> tuple[Path, str, dict[str, object]]:
    """Reissue only the original exporter-attested scoped manifest, never a full parent."""
    manifest = selected_scope._section_manifest(verified.ref, verified.book, verified.work)
    raw = evaluation._json_bytes(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != verified.section_manifest_sha256:
        raise PrivateHostStop("selected chapter's parent-bound manifest changed")
    root = runner.root / "book-candidates"
    path = root / "section-scopes" / f"{digest}.json"
    acquisition._private_write(path, raw, private_root=root)
    return (
        path,
        digest,
        {
            "work_units": len(verified.work.units),
            "skipped_units": len(verified.work.skipped),
            "words": sum(len(unit.text.split()) for unit in verified.work.units),
            "source_bytes": sum(len(unit.text.encode("utf-8")) for unit in verified.work.units),
            "excluded_parent_units": verified.work.excluded_units,
            "workplan_sha256": verified.work.fingerprint,
            "skipped_markers": list(verified.work.skipped),
        },
    )


Prepared = tuple[
    int,
    dict[str, Any],
    Path,
    selected_scope.SelectedEvidence,
    Path,
    str,
    dict[str, object],
]


def _prepare(runner: BookOrchestrator, inputs: dict[str, Any]) -> list[Prepared]:
    """Validate both bundles, independent chapter identities, and disjoint source BEFORE CPU."""
    prepared: list[Prepared] = []
    sections: set[str] = set()
    parent: tuple[str, str, str] | None = None
    for number in (1, 7):
        block = inputs[f"chapter_{number}"]
        bundle = _private_path(block["bundle_path"], runner.root, must_exist=True)
        verified = selected_scope.import_chapter(runner.root, bundle)
        identity = (
            verified.book.fingerprint,
            verified.ref.raw_sha256,
            verified.ref.ingest_stage_key,
        )
        if (
            verified.chapter_number != number
            or verified.bundle_sha256 != block["bundle_sha256"]
            or (parent is not None and parent != identity)
            or sections.intersection(verified.work.section_ids)
        ):
            raise PrivateHostStop(
                "selected chapter number, parent edition or independent scope mismatch"
            )
        parent = identity
        sections.update(verified.work.section_ids)
        section_path, scope_sha, counts = _section_manifest(runner, verified)
        if block["scope_sha256"] is not None and block["scope_sha256"] != scope_sha:
            raise PrivateHostStop("selected chapter scope identity mismatch")
        if (
            cast(int, counts["work_units"]) + cast(int, counts["skipped_units"])
            > selected_scope.MAX_CHAPTER_ENTRIES
        ):
            raise PrivateHostStop("selected full chapter exceeds original unit/gap budget")
        prepared.append((number, block, bundle, verified, section_path, scope_sha, counts))
    return prepared


def _inspection(
    runner: BookOrchestrator,
    prepared: list[Prepared],
) -> dict[str, object]:
    root = runner.root / "book-candidates"
    chapters: dict[str, dict[str, object]] = {}
    for number, _block, _bundle, verified, section, digest, counts in prepared:
        chapters[str(number)] = {
            "bundle_sha256": verified.bundle_sha256,
            "scope_sha256": digest,
            "parent_book_fingerprint": verified.book.fingerprint,
            "parent_ingest_stage_key": verified.ref.ingest_stage_key,
            "work_units": counts["work_units"],
            "skipped_units": counts["skipped_units"],
            "words": counts["words"],
            "source_bytes": counts["source_bytes"],
            "selected_sections": list(verified.work.section_ids),
            "linked_section_ids": list(verified.linked_section_ids),
            "selected_reading_blocks": [
                {
                    "anchor": block.anchor,
                    "section_id": block.section_id,
                    "source_path": block.source_path,
                    "origin": block.origin,
                    "asset_sha256": block.asset_sha256,
                    "unrecovered": block.origin == "unrecovered_image" or not block.text.strip(),
                }
                for block in verified.book.selected_reading_blocks
            ],
            "gaps": list(verified.work.skipped),
            "section_manifest_relative_path": str(section.relative_to(root)),
            "linkage_completeness": "UNVERIFIED_REQUIRES_SOURCE_REVIEW",
            "parent_original_bytes_reverified_on_host": False,
        }
    report = {
        "schema": "clipper-private-selected-chapter-inspection-v2",
        "chapter_count": 2,
        "chapters": chapters,
        "parent_full_book_complete": False,
        "parent_original_bytes_reverified_on_host": False,
        "source_boundary": "HOST_VERIFIED_SELECTED_BUNDLES_EXPORTER_ATTESTED_PARENT",
        "source_linkage_review": "REQUIRED",
        "semantic_approval": "NOT_ATTESTED",
        "publish_eligible": False,
    }
    digest = hashlib.sha256(evaluation._json_bytes(report)).hexdigest()
    evaluation._store(root, root / "private-host" / f"scope-inspection-{digest}.json", report)
    return {
        "status": "SCOPE_INSPECTION_REQUIRES_HUMAN_SELECTION_LOCK",
        "inspection_sha256": digest,
        "chapters": {
            key: {
                "scope_sha256": item["scope_sha256"],
                "bundle_sha256": item["bundle_sha256"],
                "work_units": item["work_units"],
                "skipped_units": item["skipped_units"],
                "words": item["words"],
                "source_bytes": item["source_bytes"],
            }
            for key, item in chapters.items()
        },
        "publish_eligible": False,
    }


def _review_packet(
    runner: BookOrchestrator, chapter_number: int, summary: dict[str, object]
) -> None:
    root = runner.root / "book-candidates"
    evaluations = root / "chapter-evaluations"
    source_paths = list(evaluations.glob("*/source-ledger.json"))
    matching: list[dict[str, Any]] = []
    for path in source_paths:
        raw = evaluation._read(path)
        source: object = whole_book.load_strict_reply(raw)
        if (
            isinstance(source, dict)
            and source.get("section_manifest_sha256") == summary["section_scope_sha256"]
        ):
            matching.append(source)
    if len(matching) != 1:
        raise PrivateHostStop("private source ledger for reviewed brief is ambiguous")
    briefs: list[dict[str, Any]] = []
    for path in (root / "brief").glob("*/validated.json"):
        record: object = whole_book.load_strict_reply(evaluation._read(path))
        if (
            isinstance(record, dict)
            and record.get("artifact_identity") == summary["brief_identity"]
        ):
            briefs.append(record)
    if len(briefs) != 1:
        raise PrivateHostStop("private verified brief cannot be uniquely exported")
    brief = briefs[0]["payload"]
    source = matching[0]
    packet = {
        "schema": "clipper-chapter-private-human-review-packet-v1",
        "chapter": chapter_number,
        "brief_identity": summary["brief_identity"],
        "brief": brief,
        "source_ledger": source,
        "review_template": {
            "schema": chapter.REVIEW_SCHEMA,
            "brief_sha256": summary["brief_identity"],
            "reviewer": "",
            "decisions": [
                {
                    "beat_id": row["beat_id"],
                    "decision": "REVIEW_REQUIRED",
                    "reason": "",
                    "qualifier_quote": "",
                }
                for row in brief["beats"]
            ],
            "unit_decisions": [
                {
                    "source_id": row["source_id"],
                    "decision": row["decision"],
                    "review": "REVIEW_REQUIRED",
                    "reason": "",
                    "qualifier_quote": "",
                }
                for row in brief["coverage"]
            ],
        },
        "publish_eligible": False,
    }
    evaluation._store(
        root, root / "private-host" / f"chapter-{chapter_number}-review-packet.json", packet
    )


def _release_models() -> None:
    scripting._script_pipeline.cache_clear()
    narration._kokoro_pipeline.cache_clear()
    gc.collect()


PHASE_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"NOT_STARTED"},
    "NOT_STARTED": {"MINING_IN_FLIGHT", "HYBRID_BRIEF_IN_FLIGHT", "STOPPED"},
    "HYBRID_BRIEF_IN_FLIGHT": {"AWAITING_HUMAN_REVIEW", "STOPPED"},
    "MINING_IN_FLIGHT": {"SCOPED_MINING_INCOMPLETE", "BRIEF_IN_FLIGHT", "STOPPED"},
    "BRIEF_IN_FLIGHT": {"AWAITING_HUMAN_REVIEW", "STOPPED"},
    "AWAITING_HUMAN_REVIEW": {"RESUME_IN_FLIGHT", "STOPPED"},
    "RESUME_IN_FLIGHT": {"ADAPTATION_IN_FLIGHT", "PREPRODUCTION_READY", "STOPPED"},
    "ADAPTATION_IN_FLIGHT": {"ADAPTATION_VERIFIED", "STOPPED"},
    "ADAPTATION_VERIFIED": {"SCRIPT_IN_FLIGHT", "STOPPED"},
    "SCRIPT_IN_FLIGHT": {"SCRIPT_VERIFIED", "STOPPED"},
    "SCRIPT_VERIFIED": {"NARRATION_IN_FLIGHT", "STOPPED"},
    "NARRATION_IN_FLIGHT": {"NARRATION_VERIFIED", "STOPPED"},
    "NARRATION_VERIFIED": {"DIRECTION_IN_FLIGHT", "STOPPED"},
    "DIRECTION_IN_FLIGHT": {"DIRECTION_VERIFIED", "STOPPED"},
    "DIRECTION_VERIFIED": {"PREPRODUCTION_READY", "STOPPED"},
}


def _phase_head(root: Path, number: int) -> Path:
    return root / "book-candidates/private-host" / f"chapter-{number}-phase-head.json"


@contextmanager
def _phase_lock(root: Path, number: int) -> Iterator[None]:
    """Exclusive process-level claim around read/append/committed-head updates."""
    import fcntl
    import stat

    folder = root / "book-candidates/private-host"
    acquisition._private_directory(folder, private_root=root / "book-candidates")
    path = folder / f"chapter-{number}-phase.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise PrivateHostStop("private phase lock is not a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _commit_phase_head(root: Path, number: int, record: dict[str, Any], raw: bytes) -> None:
    """Atomically replace fsynced journal head; unmatched event/head fails closed."""
    path = _phase_head(root, number)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PrivateHostStop("private chapter journal head redirected")
    head = {
        "schema": "clipper-private-chapter-committed-head-v1",
        "sequence": record["sequence"],
        "event_sha256": hashlib.sha256(raw).hexdigest(),
        "chapter": number,
        "bundle_sha256": record["bundle_sha256"],
        "scope_sha256": record["scope_sha256"],
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(evaluation._json_bytes(head))
            handle.flush()
            os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        folder_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(folder_fd)
        finally:
            os.close(folder_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _phase_records(root: Path, number: int, bundle: str, scope: str) -> list[dict[str, Any]]:
    """Verify every exact, append-only private transition before trusting resume."""
    folder = root / "book-candidates/private-host" / f"chapter-{number}-phase-events"
    head_path = _phase_head(root, number)
    if not folder.exists():
        if head_path.exists() or head_path.is_symlink():
            raise PrivateHostStop("private chapter journal events missing from committed head")
        return []
    if folder.is_symlink() or not folder.is_dir():
        raise PrivateHostStop("private chapter phase history is redirected")
    if not any(folder.iterdir()):
        raise PrivateHostStop("existing private phase history cannot be empty")
    records: list[dict[str, Any]] = []
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise PrivateHostStop("private chapter phase history has invalid entry")
        raw = evaluation._read(path, 16384)
        record: object = whole_book.load_strict_reply(raw)
        index = len(records) + 1
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "schema",
                "sequence",
                "chapter",
                "bundle_sha256",
                "scope_sha256",
                "phase",
                "artifacts",
                "publish_eligible",
            }
            or record["schema"] != "clipper-private-chapter-phase-event-v1"
            or type(record["sequence"]) is not int
            or record["sequence"] != index
            or record["chapter"] != number
            or record["bundle_sha256"] != bundle
            or record["scope_sha256"] != scope
            or record["publish_eligible"] is not False
            or not isinstance(record["phase"], str)
            or not isinstance(record["artifacts"], dict)
            or path.name != f"{index:05d}-{hashlib.sha256(raw).hexdigest()}.json"
            or raw != evaluation._json_bytes(record)
        ):
            raise PrivateHostStop("private chapter phase history changed or is incomplete")
        records.append(record)
    if head_path.is_symlink() or not head_path.is_file():
        raise PrivateHostStop("private chapter committed journal head missing")
    head_raw = evaluation._read(head_path, 16384)
    head = whole_book.load_strict_reply(head_raw)
    last_raw = evaluation._json_bytes(records[-1])
    expected = {
        "schema": "clipper-private-chapter-committed-head-v1",
        "sequence": len(records),
        "event_sha256": hashlib.sha256(last_raw).hexdigest(),
        "chapter": number,
        "bundle_sha256": bundle,
        "scope_sha256": scope,
    }
    if head != expected or head_raw != evaluation._json_bytes(expected):
        raise PrivateHostStop("private chapter phase tail differs from committed head")
    return records


def _phase(
    runner: BookOrchestrator,
    number: int,
    bundle: str,
    scope: str,
    state: str,
    artifacts: dict[str, object] | None = None,
) -> dict[str, Any]:
    root = runner.root
    with _phase_lock(root, number):
        return _append_phase_locked(runner, number, bundle, scope, state, artifacts)


def _append_phase_locked(
    runner: BookOrchestrator,
    number: int,
    bundle: str,
    scope: str,
    state: str,
    artifacts: dict[str, object] | None,
) -> dict[str, Any]:
    root = runner.root
    records = _phase_records(root, number, bundle, scope)
    previous = str(records[-1]["phase"]) if records else None
    if state not in PHASE_TRANSITIONS.get(previous, set()):
        raise PhaseClaimContended("duplicate or superseded chapter phase claim")
    # Terminal/follow-on transitions are reserved for the caller which won
    # MINING_IN_FLIGHT or RESUME_IN_FLIGHT. No losing caller may STOP a winner.
    if state not in {
        "NOT_STARTED",
        "MINING_IN_FLIGHT",
        "HYBRID_BRIEF_IN_FLIGHT",
        "RESUME_IN_FLIGHT",
    }:
        active = next(
            (
                row["artifacts"].get("claim_owner")
                for row in reversed(records)
                if row["phase"]
                in {
                    "MINING_IN_FLIGHT",
                    "HYBRID_BRIEF_IN_FLIGHT",
                    "RESUME_IN_FLIGHT",
                }
            ),
            None,
        )
        if active is not None and (artifacts or {}).get("claim_owner") != active:
            raise PhaseClaimContended("private chapter transition owner mismatch")
    record: dict[str, Any] = {
        "schema": "clipper-private-chapter-phase-event-v1",
        "sequence": len(records) + 1,
        "chapter": number,
        "bundle_sha256": bundle,
        "scope_sha256": scope,
        "phase": state,
        "artifacts": artifacts or {},
        "publish_eligible": False,
    }
    raw = evaluation._json_bytes(record)
    base = root / "book-candidates"
    path = (
        base
        / "private-host"
        / f"chapter-{number}-phase-events"
        / f"{len(records) + 1:05d}-{hashlib.sha256(raw).hexdigest()}.json"
    )
    acquisition._private_write(path, raw, private_root=base)
    _commit_phase_head(root, number, record, raw)
    return record


def _claim_reviewed_resume(
    runner: BookOrchestrator,
    number: int,
    digest: str,
    scope_sha: str,
    brief_identity: str | None,
    progress_path: Path,
    claim_owner: str,
) -> None:
    """Bind the immutable review identity before any post-review CPU producer."""
    if brief_identity is None:
        raise PrivateHostStop("resume claim lacks verified journaled brief identity")
    _phase(
        runner,
        number,
        digest,
        scope_sha,
        "RESUME_IN_FLIGHT",
        {
            "brief_identity": brief_identity,
            "progress_path": str(progress_path),
            "claim_owner": claim_owner,
        },
    )


def _record_verified_stage(
    runner: BookOrchestrator,
    number: int,
    bundle: str,
    scope: str,
    claim_owner: str,
    stage: str,
    identity: str | None,
) -> None:
    key = {
        "ADAPTATION_VERIFIED": "adaptation_identity",
        "SCRIPT_VERIFIED": "script_identity",
        "NARRATION_VERIFIED": "narration_report_identity",
        "DIRECTION_VERIFIED": "direction_identity",
    }.get(stage)
    if key is None and not stage.endswith("_IN_FLIGHT"):
        raise PrivateHostStop("unsupported private chapter subphase")
    if key is not None and (identity is None or DIGEST.fullmatch(identity) is None):
        raise PrivateHostStop("private chapter stage has no verified artifact identity")
    artifacts: dict[str, object] = {"claim_owner": claim_owner}
    if key is not None:
        artifacts[key] = identity
    _phase(runner, number, bundle, scope, stage, artifacts)


def _outcome_from_phase(record: dict[str, Any]) -> dict[str, object]:
    state = str(record["phase"])
    artifacts = record["artifacts"]
    status = {
        "NOT_STARTED": "NOT_STARTED",
        "MINING_IN_FLIGHT": "INTERRUPTED_DURING_MINING",
        "HYBRID_BRIEF_IN_FLIGHT": "INTERRUPTED_DURING_HYBRID_RECONCILIATION",
        "SCOPED_MINING_INCOMPLETE": "SCOPED_MINING_INCOMPLETE",
        "BRIEF_IN_FLIGHT": "INTERRUPTED_DURING_CHAPTER_BRIEF",
        "AWAITING_HUMAN_REVIEW": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
        "RESUME_IN_FLIGHT": "INTERRUPTED_DURING_CHAPTER_RESUME",
        "ADAPTATION_IN_FLIGHT": "INTERRUPTED_DURING_ADAPTATION",
        "ADAPTATION_VERIFIED": "ADAPTATION_VERIFIED_CHECKPOINT",
        "SCRIPT_IN_FLIGHT": "INTERRUPTED_DURING_CHAPTER_SCRIPT",
        "SCRIPT_VERIFIED": "SCRIPT_VERIFIED_CHECKPOINT",
        "NARRATION_IN_FLIGHT": "INTERRUPTED_DURING_NARRATION",
        "NARRATION_VERIFIED": "NARRATION_VERIFIED_CHECKPOINT",
        "DIRECTION_IN_FLIGHT": "INTERRUPTED_DURING_DIRECTION",
        "DIRECTION_VERIFIED": "DIRECTION_VERIFIED_CHECKPOINT",
        "PREPRODUCTION_READY": "CHAPTER_PREPRODUCTION_READY_UNREVIEWED",
        "STOPPED": "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE",
    }.get(state, "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE")
    return {
        "status": status,
        "bundle_sha256": record["bundle_sha256"],
        "last_verified_artifacts": artifacts,
        "publish_eligible": False,
    }


def execute(
    *,
    mode: str,
    artifact_root: Path,
    manifest_path: Path,
    created_at: str,
) -> dict[str, object]:
    """Verify REST identity AND live external signer before even inspecting sources."""
    # Clear any previous invocation's grant before touching the new input.
    _VERIFIED_CLOCK.set(None)
    codespace_name = os.environ.get("CODESPACE_NAME")
    if not codespace_name:
        raise PrivateHostStop("single approved Codespace environment required")
    root = artifact_root.resolve()
    worktree = Path(__file__).resolve().parents[3]
    if (
        not artifact_root.is_absolute()
        or root.is_relative_to(worktree)
        or artifact_root.is_symlink()
        or root == Path("/")
    ):
        raise PrivateHostStop("private artifacts must live outside repository checkout")
    # A caller's time/environment assertion is not authoritative. REST checks
    # exact machine, numeric instance ID, creation, repository and approved SHA.
    # The signed 45-second heartbeat is produced by a separate owner machine.
    context = chapter_lifecycle.verify_lifecycle(
        root / "book-candidates", caller_created_at=created_at, codespace_name=codespace_name
    )
    supplied_created = _created_at_timestamp(created_at)
    if context.created_at != supplied_created:
        raise PrivateHostStop("authoritative Codespace creation time mismatch")
    execution_deadline = context.execution_deadline or (
        context.created_at + timedelta(seconds=DEADLINE_SECONDS - CLEANUP_RESERVE_SECONDS)
    )
    _VERIFIED_CLOCK.set((context.created_at, execution_deadline))
    created = _created_at(created_at)
    path = _private_path(str(manifest_path), root, must_exist=True)
    inputs = _config(whole_book.load_strict_reply(evaluation._read(path, 32768)))
    plan = chapter.ChapterPlan.from_manifest(inputs["plan"])
    runner = BookOrchestrator(root)
    _check_deadline(created)
    prepared = _prepare(runner, inputs)
    if mode == "inspect":
        return _inspection(runner, prepared)
    if mode in {"hybrid-brief", "hybrid-resume"}:
        return _execute_hybrid(runner, prepared, plan, mode, root, created_at)
    if mode not in {"brief", "resume"}:
        raise PrivateHostStop("unsupported private chapter host phase")
    if not all(
        (plan.brief_cpu, plan.adaptation_cpu, plan.script_cpu, plan.direction_cpu, plan.kokoro_cpu)
    ):
        raise PrivateHostStop("real chapter phases require explicit full CPU permissions")
    if any(
        len(verified.work.units) + len(verified.work.skipped) > chapter.MAX_SOURCE_UNITS
        for _, _, _, verified, _, _, _ in prepared
    ):
        raise PrivateHostStop("legacy mined-chapter route exceeds original 64-unit safety cap")
    if any(block["scope_sha256"] != scope_sha for _, block, _, _, _, scope_sha, _ in prepared):
        raise PrivateHostStop("both original selected scopes must be explicitly operator-locked")

    # BOTH chapters have durable NOT_STARTED checkpoints before the first
    # irreversible call. A terminated process leaves an IN_FLIGHT marker.
    for number, _block, _bundle, verified, _section, scope_sha, _counts in prepared:
        if not _phase_records(root, number, verified.bundle_sha256, scope_sha):
            # A concurrent initialization may already own this immutable phase.
            with suppress(PhaseClaimContended):
                _phase(runner, number, verified.bundle_sha256, scope_sha, "NOT_STARTED")

    outcomes: dict[str, object] = {}
    for number, block, bundle, verified, section_path, scope_sha, counts in prepared:
        digest = verified.bundle_sha256
        records = _phase_records(root, number, digest, scope_sha)
        last = records[-1]
        state = str(last["phase"])
        if mode == "brief" and state != "NOT_STARTED":
            outcomes[str(number)] = _outcome_from_phase(last)
            continue
        if mode == "resume" and state != "AWAITING_HUMAN_REVIEW":
            outcomes[str(number)] = _outcome_from_phase(last)
            continue
        private_root = root / "book-candidates"
        saved = private_root / "private-host" / f"chapter-{number}-brief-record.json"
        journaled_brief_identity: str | None = None
        claim_owner = uuid4().hex
        try:
            # The independent heartbeat must remain current between chapter
            # phases; a previous invocation never extends the creation clock.
            chapter_lifecycle.verify_lifecycle(
                private_root, caller_created_at=created_at, codespace_name=codespace_name
            )
            _check_deadline(created)
            if mode == "brief":
                _phase(
                    runner,
                    number,
                    digest,
                    scope_sha,
                    "MINING_IN_FLIGHT",
                    {"claim_owner": claim_owner},
                )
                mined = runner.mine_candidate(
                    bundle,
                    local_cpu=True,
                    section_ids=verified.work.section_ids,
                    section_manifest_sha256=scope_sha,
                    selected_scope_path=bundle,
                    max_new_units=cast(int, counts["work_units"]),
                )
                if mined["status"] != "complete":
                    last = _phase(
                        runner,
                        number,
                        digest,
                        scope_sha,
                        "SCOPED_MINING_INCOMPLETE",
                        {
                            "progress_path": mined.get("progress_path"),
                            "units": mined.get("total_work_units"),
                            "claim_owner": claim_owner,
                        },
                    )
                    outcomes[str(number)] = _outcome_from_phase(last)
                    continue
                progress_path = Path(mined["progress_path"])
            else:
                record: object = whole_book.load_strict_reply(evaluation._read(saved, 16384))
                if (
                    not isinstance(record, dict)
                    or record.get("scope_sha256") != scope_sha
                    or record.get("bundle_sha256") != digest
                    or record.get("chapter") != number
                    or record.get("phase") != "AWAITING_HUMAN_REVIEW"
                    or record.get("plan_fingerprint") != plan.fingerprint
                    or record.get("brief_identity") != last["artifacts"].get("brief_identity")
                    or not isinstance(record.get("progress_path"), str)
                ):
                    raise PrivateHostStop("chapter has no immutable prior private brief phase")
                if (
                    not isinstance(record["brief_identity"], str)
                    or DIGEST.fullmatch(record["brief_identity"]) is None
                ):
                    raise PrivateHostStop("journaled chapter brief identity invalid")
                journaled_brief_identity = record["brief_identity"]
                progress_path = _private_path(record["progress_path"], root, must_exist=True)
                if hashlib.sha256(evaluation._read(progress_path, 20_000_000)).hexdigest() != (
                    record.get("progress_sha256")
                ):
                    raise PrivateHostStop("chapter scoped mining ledger changed")
            review_path = None
            if mode == "resume":
                if block["review_file"] is None:
                    outcomes[str(number)] = {
                        **_outcome_from_phase(last),
                        "review_status": "REVIEW_REQUIRED",
                    }
                    continue
                try:
                    review_path = _private_path(block["review_file"], root, must_exist=True)
                except (PrivateHostStop, OSError):
                    outcomes[str(number)] = {
                        **_outcome_from_phase(last),
                        "review_status": "REVIEW_REQUIRED",
                    }
                    continue
            else:
                _phase(
                    runner,
                    number,
                    digest,
                    scope_sha,
                    "BRIEF_IN_FLIGHT",
                    {"progress_path": str(progress_path), "claim_owner": claim_owner},
                )
            output = chapter.evaluate_chapter(
                runner,
                bundle,
                progress_path,
                section_path,
                selected_scope_path=bundle,
                review_file=review_path,
                plan=plan,
                brief_completion=scripting._local_chapter_completion,
                adaptation_completion=scripting._local_chapter_completion,
                script_completion=scripting._local_chapter_completion,
                direction_completion=scripting._local_chapter_completion,
                narration_producer=narration._kokoro_wav,
                expected_brief_identity=journaled_brief_identity,
                on_stage_progress=(
                    partial(_record_verified_stage, runner, number, digest, scope_sha, claim_owner)
                    if mode == "resume"
                    else None
                ),
                on_resume_verified=(
                    partial(
                        _claim_reviewed_resume,
                        runner,
                        number,
                        digest,
                        scope_sha,
                        journaled_brief_identity,
                        progress_path,
                        claim_owner,
                    )
                    if mode == "resume"
                    else None
                ),
            )
            if mode == "brief":
                if output["status"] != "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW":
                    raise PrivateHostStop("chapter did not stop at mandatory human checkpoint")
                evidence = {
                    "chapter": number,
                    "phase": "AWAITING_HUMAN_REVIEW",
                    "scope_sha256": scope_sha,
                    "bundle_sha256": digest,
                    "progress_path": str(progress_path),
                    "progress_sha256": hashlib.sha256(
                        evaluation._read(progress_path, 20_000_000)
                    ).hexdigest(),
                    "brief_identity": output["brief_identity"],
                    "plan_fingerprint": plan.fingerprint,
                }
                evaluation._store(private_root, saved, evidence)
                _review_packet(runner, number, output)
                last = _phase(
                    runner,
                    number,
                    digest,
                    scope_sha,
                    "AWAITING_HUMAN_REVIEW",
                    {
                        "brief_identity": output["brief_identity"],
                        "progress_path": str(progress_path),
                        "progress_sha256": evidence["progress_sha256"],
                        "claim_owner": claim_owner,
                    },
                )
            else:
                if output["status"] != "CHAPTER_PREPRODUCTION_READY_UNREVIEWED":
                    raise PrivateHostStop("chapter resume did not produce real complete audio")
                output_manifest_sha = _save_final_manifest(
                    root, number, digest, scope_sha, str(output["final_result_identity"])
                )
                last = _phase(
                    runner,
                    number,
                    digest,
                    scope_sha,
                    "PREPRODUCTION_READY",
                    {
                        "final_result_identity": output.get("final_result_identity"),
                        "output_manifest_sha256": output_manifest_sha,
                        "audio_provenance": output.get("audio_provenance"),
                        "claim_owner": claim_owner,
                    },
                )
            outcomes[str(number)] = _outcome_from_phase(last)
        except PhaseClaimContended:
            # The caller that lost a MINING/RESUME claim can only observe.
            current = _phase_records(root, number, digest, scope_sha)[-1]
            outcomes[str(number)] = _outcome_from_phase(current)
        except chapter.ChapterReviewPending:
            # No RESUME_IN_FLIGHT claim was written; a corrected human decision
            # file can be submitted on the same immutable brief checkpoint.
            outcomes[str(number)] = {
                **_outcome_from_phase(last),
                "review_status": "REVIEW_REQUIRED",
            }
        except (ValueError, OSError, TypeError, RuntimeError):
            # An integrity/preflight refusal before the resume claim must not
            # consume a previously reviewable chapter.
            current = _phase_records(root, number, digest, scope_sha)[-1]
            if mode == "resume" and current["phase"] == "AWAITING_HUMAN_REVIEW":
                outcomes[str(number)] = {
                    **_outcome_from_phase(current),
                    "review_status": "RESUME_PREFLIGHT_BLOCKED",
                }
            else:
                try:
                    stopped = _phase(
                        runner,
                        number,
                        digest,
                        scope_sha,
                        "STOPPED",
                        {"claim_owner": claim_owner},
                    )
                except PhaseClaimContended:
                    stopped = _phase_records(root, number, digest, scope_sha)[-1]
                outcomes[str(number)] = _outcome_from_phase(stopped)
        finally:
            _release_models()
    return {
        "status": "PRIVATE_TWO_CHAPTER_CHECKPOINT",
        "chapters": outcomes,
        "publish_eligible": False,
    }


def _hybrid_producer_sha256(
    ledger: dict[str, Any],
    scope: str,
    bundle: str,
    plan: chapter.ChapterPlan,
) -> str:
    """Recompute the same producer/runtime binding as the CPU stage consumer."""
    chapter_hybrid.validate_ledger(ledger)
    identity = chapter_hybrid.producer_identity(
        ledger_sha256=ledger["ledger_sha256"],
        chapter_scope_sha256=scope,
        selected_bundle_sha256=bundle,
        plan_fingerprint=plan.fingerprint,
        model_producer=whole_book.hybrid_chapter_producer_material(),
    )
    return chapter_hybrid.digest(identity)


def _claim_hybrid_resume(
    runner: BookOrchestrator,
    number: int,
    bundle: str,
    scope: str,
    brief_identity: str,
    owner: str,
) -> None:
    """Reserve one source-driven post-review continuation without a mining ledger."""
    _phase(
        runner,
        number,
        bundle,
        scope,
        "RESUME_IN_FLIGHT",
        {
            "brief_identity": brief_identity,
            "chapter_mode": "DIRECT_SOURCE_HYBRID",
            "claim_owner": owner,
        },
    )


def _hybrid_review_packet(
    runner: BookOrchestrator, number: int, summary: dict[str, object]
) -> None:
    """Bind original source, final reconciled beats, and human decision template."""
    base = runner.root / "book-candidates"
    key = summary["chapter_scope_identity"]
    if not isinstance(key, str) or DIGEST.fullmatch(key) is None:
        raise PrivateHostStop("hybrid review chapter scope identity missing")
    ledger_path = base / "chapter-evaluations" / key / "source-ledger.json"
    raw = evaluation._read(ledger_path, chapter_hybrid.MAX_SOURCE_BYTES + 1024)
    if hashlib.sha256(raw).hexdigest() != summary["source_ledger_sha256"]:
        raise PrivateHostStop("hybrid original source ledger changed before review")
    original: object = whole_book.load_strict_reply(raw)
    if not isinstance(original, dict):
        raise PrivateHostStop("hybrid original source ledger malformed")
    chapter_hybrid.validate_ledger(original)
    candidates = list((base / "hybrid_reconcile").glob("*/validated.json"))
    if len(candidates) > 128:
        raise PrivateHostStop("hybrid reconciliation inventory exceeds finite bound")
    matching = [
        path
        for path in candidates
        if isinstance(
            (obj := whole_book.load_strict_reply(evaluation._read(path, 250_000))),
            dict,
        )
        and obj.get("artifact_identity") == summary["brief_identity"]
    ]
    if len(matching) != 1:
        raise PrivateHostStop("final reviewed hybrid brief absent or ambiguous")
    envelope = whole_book.load_strict_reply(evaluation._read(matching[0], 250_000))
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise PrivateHostStop("hybrid reconciliation final payload missing")
    brief = envelope["payload"]
    if brief.get("ledger_sha256") != original["ledger_sha256"]:
        raise PrivateHostStop("hybrid final chapter brief detached from source ledger")
    packet = chapter_hybrid.human_review_packet(brief, original)
    packet.update(
        {
            "chapter": number,
            "brief_identity": summary["brief_identity"],
            "source_ledger_sha256": summary["source_ledger_sha256"],
            "producer_identity_sha256": summary["producer_identity_sha256"],
            "review_template": {
                "schema": chapter_hybrid.REVIEW_SCHEMA,
                "brief_sha256": brief["brief_sha256"],
                "brief_artifact_identity": summary["brief_identity"],
                "reviewer": "",
                "unit_decisions": [
                    {"id": item["id"], "review": "REVIEW_REQUIRED", "reason": ""}
                    for item in original["entries"]
                ],
                "beat_decisions": [
                    {"beat_id": beat["beat_id"], "review": "REVIEW_REQUIRED", "reason": ""}
                    for beat in brief["beats"]
                ],
            },
        }
    )
    evaluation._store(base, base / "private-host" / f"chapter-{number}-review-packet.json", packet)


def _execute_hybrid(
    runner: BookOrchestrator,
    prepared: list[Prepared],
    plan: chapter.ChapterPlan,
    mode: str,
    root: Path,
    created_at: str,
) -> dict[str, object]:
    """Separate host journal for source-driven chapters, never invoke lesson mining."""
    if mode not in {"hybrid-brief", "hybrid-resume"} or not all(
        (plan.brief_cpu, plan.adaptation_cpu, plan.script_cpu, plan.direction_cpu, plan.kokoro_cpu)
    ):
        raise PrivateHostStop("hybrid private CPU phases require explicitly authorized plan")
    if any(block["scope_sha256"] != scope for _, block, _, _, _, scope, _ in prepared):
        raise PrivateHostStop("hybrid requires operator-locked original selected scopes")
    for number, _block, _bundle, verified, _section, scope, _counts in prepared:
        if not _phase_records(root, number, verified.bundle_sha256, scope):
            with suppress(PhaseClaimContended):
                _phase(runner, number, verified.bundle_sha256, scope, "NOT_STARTED")
    outcomes: dict[str, object] = {}
    for number, block, bundle, verified, section, scope, _counts in prepared:
        bundle_sha = verified.bundle_sha256
        prior = _phase_records(root, number, bundle_sha, scope)[-1]
        state = str(prior["phase"])
        if (mode == "hybrid-brief" and state != "NOT_STARTED") or (
            mode == "hybrid-resume" and state != "AWAITING_HUMAN_REVIEW"
        ):
            outcomes[str(number)] = _outcome_from_phase(prior)
            continue
        private = root / "book-candidates"
        saved = private / "private-host" / f"chapter-{number}-brief-record.json"
        owner = uuid4().hex
        identity: str | None = None
        try:
            refreshed = chapter_lifecycle.verify_lifecycle(
                private, caller_created_at=created_at, codespace_name=os.environ["CODESPACE_NAME"]
            )
            if refreshed.created_at != _created_at_timestamp(created_at):
                raise PrivateHostStop("rest creation time mismatch")
            _VERIFIED_CLOCK.set(
                (
                    refreshed.created_at,
                    refreshed.execution_deadline
                    or refreshed.created_at
                    + timedelta(seconds=DEADLINE_SECONDS - CLEANUP_RESERVE_SECONDS),
                )
            )
            _check_deadline(_created_at(created_at))
            review = None
            if mode == "hybrid-brief":
                _phase(
                    runner,
                    number,
                    bundle_sha,
                    scope,
                    "HYBRID_BRIEF_IN_FLIGHT",
                    {"claim_owner": owner, "chapter_mode": "DIRECT_SOURCE_HYBRID"},
                )
            else:
                record: object = whole_book.load_strict_reply(evaluation._read(saved, 16384))
                if (
                    not isinstance(record, dict)
                    or set(record)
                    != {
                        "chapter_mode",
                        "chapter",
                        "phase",
                        "scope_sha256",
                        "bundle_sha256",
                        "brief_identity",
                        "source_ledger_sha256",
                        "argument_identity",
                        "audit_identities",
                        "plan_fingerprint",
                        "plan",
                        "chapter_scope_identity",
                        "producer_identity_sha256",
                    }
                    or record["chapter_mode"] != "DIRECT_SOURCE_HYBRID"
                    or record["chapter"] != number
                    or record["phase"] != "AWAITING_HUMAN_REVIEW"
                    or record["scope_sha256"] != scope
                    or record["bundle_sha256"] != bundle_sha
                    or not isinstance(record["chapter_scope_identity"], str)
                    or DIGEST.fullmatch(record["chapter_scope_identity"]) is None
                    or record["plan_fingerprint"] != plan.fingerprint
                    or chapter.ChapterPlan.from_manifest(record["plan"]).fingerprint
                    != plan.fingerprint
                    or record["brief_identity"] != prior["artifacts"].get("brief_identity")
                    or record["source_ledger_sha256"]
                    != prior["artifacts"].get("source_ledger_sha256")
                    or record["argument_identity"] != prior["artifacts"].get("argument_identity")
                    or record["audit_identities"] != prior["artifacts"].get("audit_identities")
                    or record["producer_identity_sha256"]
                    != prior["artifacts"].get("producer_identity_sha256")
                    or not isinstance(record["producer_identity_sha256"], str)
                    or DIGEST.fullmatch(record["producer_identity_sha256"]) is None
                    or not isinstance(record["brief_identity"], str)
                    or DIGEST.fullmatch(record["brief_identity"]) is None
                    or not isinstance(record["audit_identities"], list)
                    or not 1 <= len(record["audit_identities"]) <= chapter_hybrid.MAX_AUDITS
                    or any(
                        not isinstance(value, str) or DIGEST.fullmatch(value) is None
                        for value in record["audit_identities"]
                    )
                ):
                    raise PrivateHostStop("hybrid resume requires exact immutable brief journal")
                ledger_path = (
                    private
                    / "chapter-evaluations"
                    / record["chapter_scope_identity"]
                    / "source-ledger.json"
                )
                ledger_bytes = evaluation._read(ledger_path, chapter_hybrid.MAX_SOURCE_BYTES + 1024)
                ledger_obj = whole_book.load_strict_reply(ledger_bytes)
                if (
                    not isinstance(ledger_obj, dict)
                    or hashlib.sha256(ledger_bytes).hexdigest() != record["source_ledger_sha256"]
                    or _hybrid_producer_sha256(ledger_obj, scope, bundle_sha, plan)
                    != record["producer_identity_sha256"]
                ):
                    raise PrivateHostStop("hybrid resume producer/runtime identity changed")
                identity = record["brief_identity"]
                if block["review_file"] is None:
                    outcomes[str(number)] = {
                        **_outcome_from_phase(prior),
                        "review_status": "REVIEW_REQUIRED",
                    }
                    continue
                review = _private_path(block["review_file"], root, must_exist=True)
            output = chapter.evaluate_hybrid_chapter(
                runner,
                bundle,
                section,
                review_file=review,
                plan=plan,
                argument_completion=scripting._local_chapter_completion,
                audit_completion=scripting._local_chapter_completion,
                reconcile_completion=scripting._local_chapter_completion,
                adaptation_completion=scripting._local_chapter_completion,
                script_completion=scripting._local_chapter_completion,
                direction_completion=scripting._local_chapter_completion,
                narration_producer=narration._kokoro_wav,
                expected_brief_identity=identity,
                on_stage_progress=(
                    partial(
                        _record_verified_stage,
                        runner,
                        number,
                        bundle_sha,
                        scope,
                        owner,
                    )
                    if mode == "hybrid-resume"
                    else None
                ),
                on_resume_verified=(
                    partial(
                        _claim_hybrid_resume,
                        runner,
                        number,
                        bundle_sha,
                        scope,
                        identity,
                        owner,
                    )
                    if mode == "hybrid-resume" and identity is not None
                    else None
                ),
            )
            if mode == "hybrid-brief":
                if output["status"] != "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW":
                    raise PrivateHostStop("hybrid brief skipped mandatory semantic human gate")
                evidence = {
                    "chapter_mode": "DIRECT_SOURCE_HYBRID",
                    "chapter": number,
                    "phase": "AWAITING_HUMAN_REVIEW",
                    "chapter_scope_identity": output["chapter_scope_identity"],
                    "scope_sha256": scope,
                    "bundle_sha256": bundle_sha,
                    "brief_identity": output["brief_identity"],
                    "source_ledger_sha256": output["source_ledger_sha256"],
                    "argument_identity": output["argument_identity"],
                    "audit_identities": output["audit_identities"],
                    "producer_identity_sha256": output["producer_identity_sha256"],
                    "plan_fingerprint": plan.fingerprint,
                    "plan": {
                        "schema": chapter.PLAN_SCHEMA,
                        **{key: getattr(plan, key) for key in plan.__dataclass_fields__},
                    },
                }
                evaluation._store(private, saved, evidence)
                _hybrid_review_packet(runner, number, output)
                final = _phase(
                    runner,
                    number,
                    bundle_sha,
                    scope,
                    "AWAITING_HUMAN_REVIEW",
                    {
                        **{
                            name: evidence[name]
                            for name in (
                                "brief_identity",
                                "source_ledger_sha256",
                                "argument_identity",
                                "audit_identities",
                                "producer_identity_sha256",
                                "chapter_mode",
                                "chapter_scope_identity",
                            )
                        },
                        "claim_owner": owner,
                    },
                )
            else:
                if output["status"] != "CHAPTER_PREPRODUCTION_READY_UNREVIEWED":
                    raise PrivateHostStop("hybrid reviewed chapter did not finish CPU audio")
                manifest = _save_hybrid_manifest(
                    root, number, bundle_sha, scope, str(output["final_result_identity"])
                )
                final = _phase(
                    runner,
                    number,
                    bundle_sha,
                    scope,
                    "PREPRODUCTION_READY",
                    {
                        "final_result_identity": output["final_result_identity"],
                        "output_manifest_sha256": manifest,
                        "audio_provenance": output["audio_provenance"],
                        "claim_owner": owner,
                    },
                )
            outcomes[str(number)] = _outcome_from_phase(final)
        except PhaseClaimContended:
            current = _phase_records(root, number, bundle_sha, scope)[-1]
            outcomes[str(number)] = _outcome_from_phase(current)
        except chapter.ChapterReviewPending:
            outcomes[str(number)] = {
                **_outcome_from_phase(prior),
                "review_status": "REVIEW_REQUIRED",
            }
        except (ValueError, OSError, RuntimeError, TypeError, KeyError):
            current = _phase_records(root, number, bundle_sha, scope)[-1]
            if mode == "hybrid-resume" and current["phase"] == "AWAITING_HUMAN_REVIEW":
                outcomes[str(number)] = {
                    **_outcome_from_phase(current),
                    "review_status": "RESUME_PREFLIGHT_BLOCKED",
                }
            else:
                try:
                    stopped = _phase(
                        runner,
                        number,
                        bundle_sha,
                        scope,
                        "STOPPED",
                        {"claim_owner": owner},
                    )
                except PhaseClaimContended:
                    stopped = _phase_records(root, number, bundle_sha, scope)[-1]
                outcomes[str(number)] = _outcome_from_phase(stopped)
        finally:
            _release_models()
    return {
        "status": "PRIVATE_TWO_CHAPTER_HYBRID_CHECKPOINT",
        "chapters": outcomes,
        "source_mode": "DIRECT_SOURCE_HYBRID",
        "publish_eligible": False,
    }


def _read_recovery_artifact(path: Path, budget: int) -> bytes:
    """Preserve bounded reads, normalize absent/unsafe artifacts at the host boundary."""
    try:
        return evaluation._read(path, budget)
    except (evaluation.EvaluationStop, OSError):
        raise PrivateHostStop("private recovery artifact absent, unsafe, or oversized") from None


def _parse_recovery_artifact(raw: bytes) -> Any:
    """Never leak a lower-layer parser failure or private text in recovery errors."""
    try:
        return whole_book.load_strict_reply(raw)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise PrivateHostStop("private recovery artifact invalid") from None


def _private_output_entry(
    root: Path, path: Path, number: int, kind: str, limit: int
) -> dict[str, object]:
    """Integrity snapshot for an allowlisted private file, never prose in status."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        raise PrivateHostStop("recovery artifact escaped private root") from None
    if (
        not rel.parts
        or rel.parts[0] != "book-candidates"
        or any(part in {"", ".", ".."} for part in rel.parts)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PrivateHostStop("recovery artifact is missing or redirected")
    raw = _read_recovery_artifact(path, limit)
    return {
        "chapter": number,
        "kind": kind,
        "relative_path": rel.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _stage_output_entries(
    root: Path,
    number: int,
    stage: str,
    identity: str,
    *,
    validate: Callable[[object], dict[str, Any]] | None = None,
) -> list[dict[str, object]]:
    """Select one exact validated chapter stage; retain bounded replay evidence."""
    base = root / "book-candidates" / stage
    if base.is_symlink() or not base.is_dir():
        raise PrivateHostStop("reviewed chapter stage directory missing")
    matched: list[Path] = []
    directories = list(base.iterdir())
    if len(directories) > 128:
        raise PrivateHostStop("reviewed chapter stage inventory exceeds finite limit")
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir():
            raise PrivateHostStop("reviewed chapter stage directory redirected")
        candidate = directory / "validated.json"
        if not candidate.exists():
            continue
        record = _parse_recovery_artifact(_read_recovery_artifact(candidate, 250_000))
        if isinstance(record, dict) and record.get("artifact_identity") == identity:
            if (
                not {"payload", "provenance", "token_receipt", "attempt", "owner"}.issubset(record)
                or type(record["attempt"]) is not int
                or not 1 <= record["attempt"] <= 2
                or not isinstance(record["owner"], str)
                or re.fullmatch(r"[0-9a-f]{32}", record["owner"]) is None
                or record.get("contract") != stage
                or record.get("request_key") != directory.name
                or record.get("validator_contract") != chapter.TEXT_ARTIFACT_CONTRACT
                or not isinstance(record.get("raw_sha256"), str)
                or DIGEST.fullmatch(record["raw_sha256"]) is None
                or not isinstance(record.get("payload_sha256"), str)
                or DIGEST.fullmatch(record["payload_sha256"]) is None
                or hashlib.sha256(evaluation._json_bytes(record["payload"])).hexdigest()
                != record["payload_sha256"]
                or chapter._text_artifact_identity(
                    stage,
                    record["request_key"],
                    record["provenance"],
                    record["raw_sha256"],
                    record["payload_sha256"],
                    record["token_receipt"],
                    record.get("prior_rejection_sha256"),
                )
                != identity
            ):
                raise PrivateHostStop("reviewed chapter stage identity changed")
            raw_path = directory / f"raw-{record['attempt']}.json"
            raw_record = _parse_recovery_artifact(_read_recovery_artifact(raw_path, 110_000))
            if (
                not isinstance(raw_record, dict)
                or not isinstance(raw_record.get("raw"), str)
                or hashlib.sha256(raw_record["raw"].encode("utf-8")).hexdigest()
                != record["raw_sha256"]
                or raw_record.get("owner") != record.get("owner")
            ):
                raise PrivateHostStop("reviewed chapter raw evidence changed")
            matched.append(directory)
    if len(matched) != 1:
        raise PrivateHostStop("reviewed chapter stage identity absent or ambiguous")
    selected = matched[0]
    # Verify full winning claim/raw/validated identity and all preceding
    # retry records. Offline recovery uses an original-validator rejection
    # receipt bound at rejection time: a shape-only replay cannot determine
    # whether a syntactically valid reply failed SOURCE-bound validation.
    # Normal producer/cache replay still independently runs the real validator.
    shapes: dict[str, set[str]] = {
        "brief": {"argument", "beats", "coverage"},
        "adaptation": {
            "throughline",
            "narrative_form",
            "choice_reason",
            "beats",
            "characters",
            "locations",
            "props",
        },
        "chapter_script": {"title", "scenes"},
        "chapter_direction": {"shots"},
    }
    if validate is None and stage not in shapes:
        raise PrivateHostStop("unsupported reviewed chapter text stage")
    selected_record = _parse_recovery_artifact(
        _read_recovery_artifact(selected / "validated.json", 250_000)
    )
    if not isinstance(selected_record, dict):
        raise PrivateHostStop("reviewed chapter validated replay envelope malformed")
    request_key = selected.name
    provenance = selected_record.get("provenance")
    if provenance not in {"ACTUAL_CPU_QWEN", "INJECTED_TEST_PRODUCER"}:
        raise PrivateHostStop("reviewed chapter stage provenance invalid")
    try:
        _, replay_identity = chapter._verified_text_artifact(
            root / "book-candidates",
            selected,
            stage,
            request_key,
            provenance,
            validate
            if validate is not None
            else lambda raw: chapter._shape(raw, shapes[stage], "chapter recovery shape invalid"),
            2,
            8192,
            recovery_evidence_only=True,
        )
    except chapter.ChapterStop:
        raise PrivateHostStop("reviewed chapter replay history cannot be verified") from None
    if replay_identity != identity:
        raise PrivateHostStop("reviewed chapter replay identity changed")
    files = list(selected.iterdir())
    if len(files) > 24:
        raise PrivateHostStop("reviewed chapter replay file inventory exceeds finite limit")
    allowed = re.compile(r"(?:validated|started-[12]|raw-[12]|failed-[12])\.json\Z")
    # Only the exact winning claim and every preceding rejected attempt may
    # appear as trusted stage evidence. Future/extra failed attempts cannot
    # become recovered source or a retroactive new retry.
    winning_attempt = selected_record["attempt"]
    expected_history = {"validated.json"}
    for index in range(1, winning_attempt + 1):
        expected_history.update({f"started-{index}.json", f"raw-{index}.json"})
        if index < winning_attempt:
            expected_history.add(f"failed-{index}.json")
    if {file.name for file in files if allowed.fullmatch(file.name)} != expected_history:
        raise PrivateHostStop("reviewed chapter replay file history mismatch")
    collected = [
        _private_output_entry(root, file, number, "text", 250_000)
        for file in files
        if allowed.fullmatch(file.name)
    ]
    if not any(
        row["relative_path"] == (selected / "validated.json").relative_to(root).as_posix()
        for row in collected
    ):
        raise PrivateHostStop("reviewed chapter validated replay artifact absent")
    return collected


def _narration_output_entries(root: Path, number: int, key: str) -> list[dict[str, object]]:
    path = root / "book-candidates" / f"narration-report-{key}.json"
    report = _parse_recovery_artifact(_read_recovery_artifact(path, 300_000))
    if (
        not isinstance(report, dict)
        or content_fingerprint(report) != key
        or report.get("schema") != "clipper-chapter-narration-v2"
        or not isinstance(report.get("records"), list)
        or len(report["records"]) > 80
        or report.get("provenance") not in {"ACTUAL_CPU_KOKORO", "INJECTED_TEST_WAV"}
    ):
        raise PrivateHostStop("chapter narration report identity changed")
    results = [_private_output_entry(root, path, number, "narration_report", 300_000)]
    expected_prefix = f"chapter-audio-{report['provenance'].lower()}"
    for record in report["records"]:
        if not isinstance(record, dict):
            raise PrivateHostStop("chapter WAV inventory malformed")
        rel_text = record.get("wav_relative_path")
        if (
            not isinstance(rel_text, str)
            or not isinstance(record.get("sha256"), str)
            or DIGEST.fullmatch(record["sha256"]) is None
        ):
            raise PrivateHostStop("chapter WAV inventory malformed")
        rel = Path(rel_text)
        if (
            rel.is_absolute()
            or not rel.parts
            or rel.parts[0] != expected_prefix
            or any(part in {"", ".", ".."} for part in rel.parts)
            or rel.suffix.lower() != ".wav"
        ):
            raise PrivateHostStop("chapter WAV path escaped attested audio scope")
        audio = _private_output_entry(
            root, root / "book-candidates" / rel, number, "wav", narration.MAX_WAV_BYTES
        )
        if audio["sha256"] != record["sha256"]:
            raise PrivateHostStop("chapter WAV hash changed after report")
        results.append(audio)
    return results


def _chapter_output_manifest(
    root: Path,
    number: int,
    bundle: str,
    scope: str,
    *,
    final_result_identity: str | None = None,
) -> dict[str, object]:
    """Scoped, private, bounded inventory from committed stage identities."""
    records = _phase_records(root, number, bundle, scope)
    if any(record["artifacts"].get("chapter_mode") == "DIRECT_SOURCE_HYBRID" for record in records):
        return _hybrid_output_manifest(
            root, number, bundle, scope, final_result_identity=final_result_identity
        )
    identities: dict[str, str] = {}
    for item in records:
        for key in (
            "brief_identity",
            "adaptation_identity",
            "script_identity",
            "narration_report_identity",
            "direction_identity",
            "final_result_identity",
        ):
            value = item["artifacts"].get(key)
            if isinstance(value, str) and DIGEST.fullmatch(value):
                if key in identities and identities[key] != value:
                    raise PrivateHostStop("chapter recovery identity changed within journal")
                identities[key] = value
    if final_result_identity is not None:
        if DIGEST.fullmatch(final_result_identity) is None:
            raise PrivateHostStop("chapter final result identity invalid")
        identities["final_result_identity"] = final_result_identity
    entries: list[dict[str, object]] = []
    for key, stage in (
        ("brief_identity", "brief"),
        ("adaptation_identity", "adaptation"),
        ("script_identity", "chapter_script"),
        ("direction_identity", "chapter_direction"),
    ):
        if key in identities:
            entries.extend(_stage_output_entries(root, number, stage, identities[key]))
    if "narration_report_identity" in identities:
        entries.extend(
            _narration_output_entries(root, number, identities["narration_report_identity"])
        )
    if "brief_identity" in identities:
        folder = root / "book-candidates/private-host"
        for name in (f"chapter-{number}-brief-record.json", f"chapter-{number}-review-packet.json"):
            path = folder / name
            if path.is_file():
                obj = _parse_recovery_artifact(_read_recovery_artifact(path, 250_000))
                if (
                    not isinstance(obj, dict)
                    or obj.get("brief_identity") != identities["brief_identity"]
                ):
                    raise PrivateHostStop("private chapter review record changed")
                entries.append(_private_output_entry(root, path, number, "review", 250_000))
    if "final_result_identity" in identities:
        result_id = identities["final_result_identity"]
        eval_root = root / "book-candidates/chapter-evaluations"
        matches = list(eval_root.glob(f"*/chapter-result-{result_id}.json"))
        if len(matches) != 1:
            raise PrivateHostStop("chapter final result missing or ambiguous")
        result_file = matches[0]
        result = _parse_recovery_artifact(_read_recovery_artifact(result_file, 500_000))
        if (
            not isinstance(result, dict)
            or content_fingerprint(result) != result_id
            or result.get("selected_source_bundle_sha256") != bundle
            or result.get("source_boundary")
            != "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT"
            or result.get("parent_original_reverified_on_host") is not False
            or result.get("publish_eligible") is not False
            or result.get("brief_key") != identities.get("brief_identity")
            or result.get("adaptation_key") != identities.get("adaptation_identity")
            or result.get("script_key") != identities.get("script_identity")
            or result.get("direction_key") != identities.get("direction_identity")
            or result.get("narration_report_key") != identities.get("narration_report_identity")
        ):
            raise PrivateHostStop("private chapter final result differs from reviewed stages")
        entries.append(_private_output_entry(root, result_file, number, "final_result", 500_000))
        source = result_file.parent / "source-ledger.json"
        source_entry = _private_output_entry(root, source, number, "source_ledger", 300_000)
        if source_entry["sha256"] != result["source_ledger_sha256"]:
            raise PrivateHostStop("private chapter source ledger identity changed")
        entries.append(source_entry)
        review_id = result.get("human_review_sha256")
        if not isinstance(review_id, str) or DIGEST.fullmatch(review_id) is None:
            raise PrivateHostStop("private chapter human review identity missing")
        entries.append(
            _private_output_entry(
                root,
                result_file.parent / f"human-review-{review_id}.json",
                number,
                "human_review",
                250_000,
            )
        )
    unique: dict[str, dict[str, object]] = {}
    for item in entries:
        name = str(item["relative_path"])
        if name in unique and unique[name] != item:
            raise PrivateHostStop("private chapter output path conflicted")
        unique[name] = item
    if len(unique) > 256:
        raise PrivateHostStop("private chapter output count exceeds finite allowance")
    return {
        "schema": "clipper-chapter-private-output-manifest-v1",
        "chapter": number,
        "bundle_sha256": bundle,
        "scope_sha256": scope,
        "verified_artifact_identities": identities,
        "files": [unique[key] for key in sorted(unique)],
        "publish_eligible": False,
    }


def _hybrid_output_manifest(
    root: Path,
    number: int,
    bundle: str,
    scope: str,
    *,
    final_result_identity: str | None = None,
) -> dict[str, object]:
    """Revalidate source-grounded raw/cache chains before private recovery."""
    records = _phase_records(root, number, bundle, scope)
    reference = next(
        (
            row["artifacts"]
            for row in records
            if row["phase"] == "AWAITING_HUMAN_REVIEW"
            and row["artifacts"].get("chapter_mode") == "DIRECT_SOURCE_HYBRID"
        ),
        None,
    )
    if reference is None:
        # A claimed but interrupted model request has no attested response.
        # Never invent a recoverable stage identity from loose files.
        return {
            "schema": "clipper-chapter-private-output-manifest-v1",
            "chapter": number,
            "bundle_sha256": bundle,
            "scope_sha256": scope,
            "verified_artifact_identities": {},
            "files": [],
            "source_mode": "DIRECT_SOURCE_HYBRID",
            "inflight_inventory": "UNATTESTED_FAIL_CLOSED",
            "publish_eligible": False,
        }
    base = root / "book-candidates"
    brief_id = reference.get("brief_identity")
    argument_id = reference.get("argument_identity")
    audit_ids = reference.get("audit_identities")
    key = reference.get("chapter_scope_identity")
    source_sha = reference.get("source_ledger_sha256")
    producer_sha = reference.get("producer_identity_sha256")
    if (
        not all(
            isinstance(v, str) and DIGEST.fullmatch(v)
            for v in (brief_id, argument_id, key, source_sha, producer_sha)
        )
        or not isinstance(audit_ids, list)
        or not 1 <= len(audit_ids) <= chapter_hybrid.MAX_AUDITS
        or any(not isinstance(v, str) or DIGEST.fullmatch(v) is None for v in audit_ids)
    ):
        raise PrivateHostStop("hybrid reviewed source-stage identities invalid")
    saved_path = base / "private-host" / f"chapter-{number}-brief-record.json"
    saved = _parse_recovery_artifact(_read_recovery_artifact(saved_path, 16384))
    if (
        not isinstance(saved, dict)
        or saved.get("chapter_mode") != "DIRECT_SOURCE_HYBRID"
        or saved.get("chapter_scope_identity") != key
        or saved.get("brief_identity") != brief_id
        or saved.get("argument_identity") != argument_id
        or saved.get("audit_identities") != audit_ids
        or saved.get("source_ledger_sha256") != source_sha
        or saved.get("producer_identity_sha256") != producer_sha
        or saved.get("bundle_sha256") != bundle
        or saved.get("scope_sha256") != scope
        or saved.get("chapter") != number
    ):
        raise PrivateHostStop("hybrid recovery source brief journal changed")
    try:
        plan = chapter.ChapterPlan.from_manifest(saved["plan"])
    except (ValueError, TypeError, KeyError):
        raise PrivateHostStop("hybrid recovery frozen CPU plan invalid") from None
    if saved.get("plan_fingerprint") != plan.fingerprint:
        raise PrivateHostStop("hybrid recovery CPU plan identity changed")
    ledger_path = base / "chapter-evaluations" / key / "source-ledger.json"
    ledger_bytes = _read_recovery_artifact(ledger_path, chapter_hybrid.MAX_SOURCE_BYTES + 1024)
    ledger_obj = _parse_recovery_artifact(ledger_bytes)
    if (
        not isinstance(ledger_obj, dict)
        or hashlib.sha256(ledger_bytes).hexdigest() != source_sha
        or ledger_obj.get("identity", {}).get("chapter") != number
        or ledger_obj.get("identity", {}).get("section_manifest_sha256") != scope
        or ledger_obj.get("identity", {}).get("selected_bundle_sha256") != bundle
    ):
        raise PrivateHostStop("hybrid recoverable source differs from selected original")
    if _hybrid_producer_sha256(ledger_obj, scope, bundle, plan) != producer_sha:
        raise PrivateHostStop("hybrid recovered producer/runtime identity changed")
    files: list[dict[str, object]] = [
        _private_output_entry(
            root, ledger_path, number, "source_ledger", chapter_hybrid.MAX_SOURCE_BYTES + 1024
        ),
        _private_output_entry(root, saved_path, number, "review_record", 16384),
    ]

    def verified_stage(
        stage: str, identity: str, validator: Callable[[object], dict[str, Any]]
    ) -> dict[str, Any]:
        evidence = _stage_output_entries(root, number, stage, identity, validate=validator)
        files.extend(evidence)
        records_for_stage = [
            item for item in evidence if str(item["relative_path"]).endswith("/validated.json")
        ]
        if len(records_for_stage) != 1:
            raise PrivateHostStop("hybrid stage validated source record not unique")
        path = root / str(records_for_stage[0]["relative_path"])
        envelope = _parse_recovery_artifact(_read_recovery_artifact(path, 250_000))
        if (
            not isinstance(envelope, dict)
            or envelope.get("artifact_identity") != identity
            or not isinstance(envelope.get("payload"), dict)
        ):
            raise PrivateHostStop("hybrid stage payload changed after raw replay")
        return cast(dict[str, Any], envelope["payload"])

    argument = verified_stage(
        "hybrid_argument",
        argument_id,
        lambda raw: chapter_hybrid.validate_argument(raw, ledger_obj),
    )
    windows = chapter_hybrid.audit_windows(ledger_obj)
    if len(windows) != len(audit_ids):
        raise PrivateHostStop("hybrid recovered local source coverage incomplete")
    audits: list[dict[str, Any]] = []
    for window, identity in zip(windows, audit_ids, strict=True):

        def validate_local(raw: object, scoped: dict[str, object] = window) -> dict[str, Any]:
            return chapter_hybrid.validate_audit(raw, ledger_obj, scoped)

        audits.append(verified_stage("hybrid_audit", identity, validate_local))

    def validate_reconcile(raw: object) -> dict[str, Any]:
        result = chapter_hybrid.reconcile(raw, ledger_obj, argument, audits)
        if len(result["beats"]) > plan.max_beats:
            raise chapter.ChapterStop("hybrid recovered beat limit differs from frozen plan")
        return result

    brief = verified_stage("hybrid_reconcile", brief_id, validate_reconcile)
    packet_path = base / "private-host" / f"chapter-{number}-review-packet.json"
    packet = _parse_recovery_artifact(_read_recovery_artifact(packet_path, 1_000_000))
    if (
        not isinstance(packet, dict)
        or packet.get("brief_identity") != brief_id
        or packet.get("brief_sha256") != brief["brief_sha256"]
        or packet.get("source_ledger_sha256") != source_sha
        or packet.get("producer_identity_sha256") != producer_sha
    ):
        raise PrivateHostStop("hybrid source review packet changed after human gate")
    files.append(_private_output_entry(root, packet_path, number, "review", 1_000_000))
    stage_identities: dict[str, object] = {
        "brief_identity": brief_id,
        "argument_identity": argument_id,
        "audit_identities": audit_ids,
        "producer_identity_sha256": producer_sha,
    }
    derived: dict[str, str] = {}
    for row in records:
        for field in (
            "adaptation_identity",
            "script_identity",
            "narration_report_identity",
            "direction_identity",
            "final_result_identity",
        ):
            value = row["artifacts"].get(field)
            if isinstance(value, str) and DIGEST.fullmatch(value):
                if field in derived and derived[field] != value:
                    raise PrivateHostStop("hybrid stage journal identities changed")
                derived[field] = value
    stage_identities.update(derived)
    treatment: dict[str, Any] | None = None
    script: dict[str, Any] | None = None
    if "adaptation_identity" in derived:
        treatment = verified_stage(
            "adaptation",
            derived["adaptation_identity"],
            lambda raw: chapter_hybrid.validate_adaptation(raw, brief, plan),
        )
    if "script_identity" in derived:
        if treatment is None:
            raise PrivateHostStop("hybrid script recovered without adaptation")
        script = verified_stage(
            "chapter_script",
            derived["script_identity"],
            lambda raw: chapter_hybrid.validate_script(raw, treatment, brief, plan),
        )
    if "narration_report_identity" in derived:
        files.extend(_narration_output_entries(root, number, derived["narration_report_identity"]))
    if final_result_identity is not None:
        if DIGEST.fullmatch(final_result_identity) is None:
            raise PrivateHostStop("hybrid final identity invalid")
        if (
            "final_result_identity" in derived
            and derived["final_result_identity"] != final_result_identity
        ):
            raise PrivateHostStop("hybrid final identity changed after committed phase")
        derived["final_result_identity"] = final_result_identity
        stage_identities["final_result_identity"] = final_result_identity
    if "final_result_identity" in derived:
        final_path = (
            base
            / "chapter-evaluations"
            / key
            / f"chapter-result-{derived['final_result_identity']}.json"
        )
        record = _parse_recovery_artifact(_read_recovery_artifact(final_path, 500_000))
        if (
            not isinstance(record, dict)
            or content_fingerprint(record) != derived["final_result_identity"]
            or record.get("source_mining_provenance") != "DIRECT_SOURCE_NO_MINING"
            or record.get("chapter_scope_identity") != key
            or record.get("source_ledger_sha256") != source_sha
            or record.get("producer_identity_sha256") != producer_sha
            or record.get("selected_source_bundle_sha256") != bundle
            or record.get("parent_original_reverified_on_host") is not False
            or record.get("publish_eligible") is not False
            or record.get("argument_key") != argument_id
            or record.get("audit_keys") != audit_ids
            or record.get("brief_key") != brief_id
            or any(
                record.get(record_field) != derived.get(identity_field)
                for record_field, identity_field in (
                    ("adaptation_key", "adaptation_identity"),
                    ("script_key", "script_identity"),
                    ("direction_key", "direction_identity"),
                    ("narration_report_key", "narration_report_identity"),
                )
            )
        ):
            raise PrivateHostStop("hybrid final output differs from complete source journal")
        if treatment is None or script is None:
            raise PrivateHostStop("hybrid final creative stages absent")
        if "direction_identity" in derived:
            verified_stage(
                "chapter_direction",
                derived["direction_identity"],
                lambda raw: chapter._direction_validate(
                    raw, script, treatment, record["timings"], plan
                ),
            )
        review_id = record.get("human_review_sha256")
        if not isinstance(review_id, str) or DIGEST.fullmatch(review_id) is None:
            raise PrivateHostStop("hybrid final human review digest missing")
        review_path = final_path.parent / f"human-review-{review_id}.json"
        review_obj = _parse_recovery_artifact(_read_recovery_artifact(review_path, 250_000))
        if (
            not isinstance(review_obj, dict)
            or review_obj.get("review_sha256") != review_id
            or review_obj.get("schema") != chapter_hybrid.REVIEW_SCHEMA
            or not isinstance(review_obj.get("review"), dict)
            or review_obj["review"].get("brief_artifact_identity") != brief_id
        ):
            raise PrivateHostStop("hybrid final human approval record changed")
        review = {
            field: value
            for field, value in review_obj["review"].items()
            if field != "brief_artifact_identity"
        }
        chapter_hybrid.validate_human_review(review, brief, ledger_obj)
        files.extend(
            (
                _private_output_entry(root, final_path, number, "final_result", 500_000),
                _private_output_entry(root, review_path, number, "human_review", 250_000),
            )
        )
    if len(files) > 256:
        raise PrivateHostStop("hybrid recovery files exceed finite allowance")
    unique: dict[str, dict[str, object]] = {}
    for file in files:
        relative = str(file["relative_path"])
        if relative in unique and unique[relative] != file:
            raise PrivateHostStop("hybrid recovery filename collides")
        unique[relative] = file
    return {
        "schema": "clipper-chapter-private-output-manifest-v1",
        "chapter": number,
        "bundle_sha256": bundle,
        "scope_sha256": scope,
        "verified_artifact_identities": stage_identities,
        "files": [unique[key] for key in sorted(unique)],
        "source_mode": "DIRECT_SOURCE_HYBRID",
        "publish_eligible": False,
    }


def _save_hybrid_manifest(
    root: Path, number: int, bundle: str, scope: str, final_identity: str
) -> str:
    manifest = _hybrid_output_manifest(
        root, number, bundle, scope, final_result_identity=final_identity
    )
    raw = evaluation._json_bytes(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    evaluation._store(
        root / "book-candidates",
        root / "book-candidates/private-host" / f"chapter-{number}-output-manifest.json",
        manifest,
    )
    return digest


def _save_final_manifest(
    root: Path, number: int, bundle: str, scope: str, final_result_identity: str
) -> str:
    manifest = _chapter_output_manifest(
        root, number, bundle, scope, final_result_identity=final_result_identity
    )
    raw = evaluation._json_bytes(manifest)
    digest = hashlib.sha256(raw).hexdigest()
    evaluation._store(
        root / "book-candidates",
        root / "book-candidates/private-host" / f"chapter-{number}-output-manifest.json",
        manifest,
    )
    return digest


def private_recovery_archive(artifact_root: Path) -> dict[str, object]:
    """Package only checksum-verified scoped outputs into a private owner archive."""
    status = recovery_report(artifact_root, tolerate_corrupt=True)
    root = artifact_root.resolve()
    manifests: list[dict[str, object]] = []
    files: dict[str, dict[str, object]] = {}
    chapters = cast(dict[str, dict[str, object]], status["chapters"])
    for number in (1, 7):
        info = chapters[str(number)]
        state = info["recovery_state"]
        if state in {"NEVER_STARTED", "INCOMPLETE"}:
            manifests.append(
                {
                    "chapter": number,
                    "inventory_status": "NOT_STARTED" if state == "NEVER_STARTED" else "INCOMPLETE",
                    "files": [],
                }
            )
            continue
        try:
            manifest = _chapter_output_manifest(
                root, number, str(info["bundle_sha256"]), str(info["scope_sha256"])
            )
            if state == "READY":
                saved_path = (
                    root / "book-candidates/private-host" / f"chapter-{number}-output-manifest.json"
                )
                raw = _read_recovery_artifact(saved_path, 150_000)
                if raw != evaluation._json_bytes(manifest):
                    raise PrivateHostStop(
                        "chapter output inventory differs from final committed manifest"
                    )
                if hashlib.sha256(raw).hexdigest() != info["output_manifest_sha256"]:
                    raise PrivateHostStop("chapter output inventory digest changed")
            scoped = cast(list[dict[str, object]], manifest["files"])
            # Check ALL collisions before mutating the accepted file inventory.
            for entry in scoped:
                key = str(entry["relative_path"])
                previous = files.get(key)
                if previous is not None and any(
                    previous[field] != entry[field] for field in ("kind", "sha256", "bytes")
                ):
                    raise PrivateHostStop("private shared output hash or type conflict")
        except (PrivateHostStop, evaluation.EvaluationStop, ValueError, OSError, KeyError):
            info["retrieval_status"] = "INCOMPLETE"
            manifests.append(
                {
                    "chapter": number,
                    "inventory_status": "INCOMPLETE",
                    "files": [],
                }
            )
            continue
        info["retrieval_status"] = "INVENTORY_VERIFIED_NOT_RETRIEVED"
        manifests.append({**manifest, "inventory_status": "VERIFIED"})
        for entry in scoped:
            key = str(entry["relative_path"])
            if key in files:
                owned = cast(list[int], files[key]["chapters"])
                if number not in owned:
                    owned.append(number)
            else:
                files[key] = {
                    "kind": entry["kind"],
                    "relative_path": key,
                    "sha256": entry["sha256"],
                    "bytes": entry["bytes"],
                    "chapters": [number],
                }
    if len(files) > 512:
        raise PrivateHostStop("shared chapter private recovery file limit exceeded")
    listing = {
        "schema": "clipper-chapter-private-recovery-archive-manifest-v1",
        "chapters": manifests,
        "files": [files[key] for key in sorted(files)],
        "runtime_status_sha256": hashlib.sha256(evaluation._json_bytes(status)).hexdigest(),
    }
    target = root / "book-candidates/private-host/chapter-recovery.tar"
    if target.is_symlink():
        raise PrivateHostStop("private chapter recovery archive is redirected")
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for entry in cast(list[dict[str, object]], listing["files"]):
                path = root / str(entry["relative_path"])
                raw = _read_recovery_artifact(path, narration.MAX_WAV_BYTES)
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise PrivateHostStop("private chapter output changed during archive creation")
                member = tarfile.TarInfo(name=str(entry["relative_path"]))
                member.size = len(raw)
                member.mode = 0o600
                archive.addfile(member, BytesIO(raw))
            data = evaluation._json_bytes(listing)
            member = tarfile.TarInfo(name="private-recovery-manifest.json")
            member.size = len(data)
            member.mode = 0o600
            archive.addfile(member, BytesIO(data))
            safe = evaluation._json_bytes(status)
            member = tarfile.TarInfo(name="safe-recovery-status.json")
            member.size = len(safe)
            member.mode = 0o600
            archive.addfile(member, BytesIO(safe))
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    total = 0
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            if total > 800_000_000:
                raise PrivateHostStop("private recovery archive exceeds bounded size")
            digest.update(block)
    return {
        "schema": "clipper-private-chapter-recovery-created-v1",
        "archive_sha256": digest.hexdigest(),
        "manifest_sha256": hashlib.sha256(evaluation._json_bytes(listing)).hexdigest(),
        "file_count": len(files),
        "files_retrieved": False,
        "host_cleanup_verified": False,
    }


def recovery_report(artifact_root: Path, *, tolerate_corrupt: bool = False) -> dict[str, object]:
    """Read-only, bounded, model-free owner recovery manifest with no source prose.

    This is checkpoint evidence, not proof that remote files were recovered.
    Any corrupt history fails closed instead of guessing a resumable phase.
    """
    root = artifact_root.resolve()
    worktree = Path(__file__).resolve().parents[3]
    if (
        not artifact_root.is_absolute()
        or artifact_root.is_symlink()
        or root == Path("/")
        or root.is_relative_to(worktree)
    ):
        raise PrivateHostStop("private recovery report needs a separate absolute artifact root")
    chapters: dict[str, dict[str, object]] = {}
    for number in (1, 7):
        try:
            head_path = _phase_head(root, number)
            folder = root / "book-candidates/private-host" / f"chapter-{number}-phase-events"
            if not head_path.exists() and not head_path.is_symlink() and not folder.exists():
                chapters[str(number)] = {
                    "recovery_state": "NEVER_STARTED",
                    "last_subphase": "NOT_STARTED",
                    "runtime_journal_status": "NOT_STARTED",
                    "retrieval_status": "NOT_ATTEMPTED",
                    "last_verified_artifact_identities": {},
                }
                continue
            head_raw = _read_recovery_artifact(head_path, 16384)
            head = _parse_recovery_artifact(head_raw)
            if (
                not isinstance(head, dict)
                or head.get("chapter") != number
                or not isinstance(head.get("bundle_sha256"), str)
                or DIGEST.fullmatch(head["bundle_sha256"]) is None
                or not isinstance(head.get("scope_sha256"), str)
                or DIGEST.fullmatch(head["scope_sha256"]) is None
            ):
                raise PrivateHostStop("private recovery journal head cannot be trusted")
            records = _phase_records(root, number, head["bundle_sha256"], head["scope_sha256"])
            if not records:
                raise PrivateHostStop("private recovery journal empty")
            phase = str(records[-1]["phase"])
            identities: dict[str, str] = {}
            for record in reversed(records):
                artifacts = record["artifacts"]
                for key in (
                    "brief_identity",
                    "adaptation_identity",
                    "script_identity",
                    "narration_report_identity",
                    "direction_identity",
                    "final_result_identity",
                    "output_manifest_sha256",
                    "progress_sha256",
                ):
                    value = artifacts.get(key)
                    if key not in identities and isinstance(value, str) and DIGEST.fullmatch(value):
                        identities[key] = value
            last_subphase = (
                str(records[-2]["phase"]) if phase == "STOPPED" and len(records) > 1 else phase
            )
            category = (
                "READY"
                if phase == "PREPRODUCTION_READY"
                else "NEVER_STARTED"
                if phase == "NOT_STARTED"
                else "IN_FLIGHT"
                if phase.endswith("_IN_FLIGHT")
                else "AWAITING_REVIEW"
                if phase == "AWAITING_HUMAN_REVIEW"
                else "INTERRUPTED"
            )
            chapters[str(number)] = {
                "recovery_state": category,
                "last_subphase": last_subphase,
                "runtime_journal_status": phase,
                "retrieval_status": "NOT_ATTEMPTED",
                "bundle_sha256": head["bundle_sha256"],
                "scope_sha256": head["scope_sha256"],
                "last_verified_artifact_identities": identities,
                "committed_phase_event_sha256": head["event_sha256"],
                **(
                    {"output_manifest_sha256": identities["output_manifest_sha256"]}
                    if phase == "PREPRODUCTION_READY" and "output_manifest_sha256" in identities
                    else {}
                ),
            }
        except (PrivateHostStop, evaluation.EvaluationStop, ValueError, OSError):
            if not tolerate_corrupt:
                raise
            # Keep the other chapter's scope independently recoverable.
            # No unverified stage identity may be surfaced as recovered.
            chapters[str(number)] = {
                "recovery_state": "INCOMPLETE",
                "last_subphase": "UNVERIFIED",
                "runtime_journal_status": "UNVERIFIED",
                "retrieval_status": "INCOMPLETE",
                "last_verified_artifact_identities": {},
            }
    return {
        "schema": "clipper-private-chapter-safe-recovery-v1",
        "status": "PRIVATE_CHAPTER_RECOVERY_CHECKPOINT_ONLY",
        "chapters": chapters,
        "files_retrieved": False,
        "publish_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private two-chapter Codespace checkpoint")
    parser.add_argument(
        "mode",
        choices=(
            "inspect",
            "brief",
            "resume",
            "hybrid-brief",
            "hybrid-resume",
            "status",
            "recover",
        ),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--codespace-created-at")
    args = parser.parse_args(argv)
    try:
        if args.mode == "status":
            print(json.dumps(recovery_report(args.artifact_root), sort_keys=True))
            return 0
        if args.mode == "recover":
            print(json.dumps(private_recovery_archive(args.artifact_root), sort_keys=True))
            return 0
        if args.manifest is None or args.codespace_created_at is None:
            raise PrivateHostStop("private host manifest and creation timestamp required")
        result = execute(
            mode=args.mode,
            artifact_root=args.artifact_root,
            manifest_path=args.manifest,
            created_at=args.codespace_created_at,
        )
        print(json.dumps(result, sort_keys=True))
        return (
            0
            if args.mode == "inspect"
            or all(
                item["status"]
                in {"AWAITING_HUMAN_CHAPTER_BEAT_REVIEW", "CHAPTER_PREPRODUCTION_READY_UNREVIEWED"}
                for item in cast(dict[str, dict[str, Any]], result["chapters"]).values()
            )
            else 2
        )
    except Exception:
        print(json.dumps({"status": "PRIVATE_CHAPTER_HOST_FAILED_CLOSED"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from . import acquisition, chapter, evaluation, narration, scripting, whole_book, selected_scope
from .orchestrator import BookOrchestrator

BATCH_SCHEMA = "clipper-two-private-chapter-codespace-v2"
DEADLINE_SECONDS = 7200
CLEANUP_RESERVE_SECONDS = 300
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class PrivateHostStop(ValueError):
    """Sanitized private-host phase refusal; source text is never emitted."""


def _created_at(value: str, now: datetime | None = None) -> datetime:
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise PrivateHostStop("verified Codespace creation timestamp required") from None
    if date.tzinfo is None:
        raise PrivateHostStop("Codespace timestamp must include UTC offset")
    date = date.astimezone(UTC)
    current = now or datetime.now(UTC)
    if date > current or current >= date + timedelta(
        seconds=DEADLINE_SECONDS - CLEANUP_RESERVE_SECONDS
    ):
        raise PrivateHostStop("one-host shared deadline expired or invalid")
    return date


def _check_deadline(created: datetime) -> None:
    if datetime.now(UTC) >= created + timedelta(seconds=DEADLINE_SECONDS - CLEANUP_RESERVE_SECONDS):
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


def _prepare(
    runner: BookOrchestrator, inputs: dict[str, Any]
) -> list[tuple[int, dict[str, Any], Path, selected_scope.SelectedEvidence, Path, str, dict[str, object]]]:
    """Validate both bundles, independent chapter identities, and disjoint source BEFORE CPU."""
    prepared: list[
        tuple[int, dict[str, Any], Path, selected_scope.SelectedEvidence, Path, str, dict[str, object]]
    ] = []
    sections: set[str] = set()
    parent: tuple[str, str, str] | None = None
    for number in (1, 7):
        block = inputs[f"chapter_{number}"]
        bundle = _private_path(block["bundle_path"], runner.root, must_exist=True)
        verified = selected_scope.import_chapter(runner.root, bundle)
        identity = (verified.book.fingerprint, verified.ref.raw_sha256, verified.ref.ingest_stage_key)
        if (
            verified.chapter_number != number
            or verified.bundle_sha256 != block["bundle_sha256"]
            or (parent is not None and parent != identity)
            or sections.intersection(verified.work.section_ids)
        ):
            raise PrivateHostStop("selected chapter number, parent edition or independent scope mismatch")
        parent = identity
        sections.update(verified.work.section_ids)
        section_path, scope_sha, counts = _section_manifest(runner, verified)
        if (
            block["scope_sha256"] is not None
            and block["scope_sha256"] != scope_sha
        ):
            raise PrivateHostStop("selected chapter scope identity mismatch")
        if (
            cast(int, counts["work_units"]) + cast(int, counts["skipped_units"])
            > chapter.MAX_SOURCE_UNITS
        ):
            raise PrivateHostStop("selected full chapter exceeds original unit/gap budget")
        prepared.append((number, block, bundle, verified, section_path, scope_sha, counts))
    return prepared


def _inspection(
    runner: BookOrchestrator,
    prepared: list[
        tuple[int, dict[str, Any], Path, selected_scope.SelectedEvidence, Path, str, dict[str, object]]
    ],
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


def execute(
    *,
    mode: str,
    artifact_root: Path,
    manifest_path: Path,
    created_at: str,
) -> dict[str, object]:
    """Only inspection is model-free. Runtime must be under the Codespace hard timeout."""
    created = _created_at(created_at)
    if not os.environ.get("CODESPACE_NAME"):
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
    path = _private_path(str(manifest_path), root, must_exist=True)
    inputs = _config(whole_book.load_strict_reply(evaluation._read(path, 32768)))
    plan = chapter.ChapterPlan.from_manifest(inputs["plan"])
    runner = BookOrchestrator(root)
    _check_deadline(created)
    prepared = _prepare(runner, inputs)
    if mode == "inspect":
        return _inspection(runner, prepared)
    if mode not in {"brief", "resume"}:
        raise PrivateHostStop("unsupported private chapter host phase")
    if not all(
        [
            plan.brief_cpu,
            plan.adaptation_cpu,
            plan.script_cpu,
            plan.direction_cpu,
            plan.kokoro_cpu,
        ]
    ):
        raise PrivateHostStop("real chapter phases require explicit full CPU permissions")
    # Lock BOTH scopes before making even the first irreversible mining call.
    # A mismatched chapter 7 must not waste chapter 1's shared host allowance.
    if any(block["scope_sha256"] != scope_sha for _, block, _, _, _, scope_sha, _ in prepared):
        raise PrivateHostStop("both original selected scopes must be explicitly operator-locked")
    outcomes: dict[str, object] = {}
    for number, block, bundle, verified, section_path, scope_sha, counts in prepared:
        _check_deadline(created)
        private_root = root / "book-candidates"
        state = private_root / "private-host" / f"chapter-{number}-brief-record.json"
        try:
            if mode == "brief":
                mined = runner.mine_candidate(
                    bundle,
                    local_cpu=True,
                    section_ids=verified.work.section_ids,
                    section_manifest_sha256=scope_sha,
                    selected_scope_path=bundle,
                    max_new_units=cast(int, counts["work_units"]),
                )
                if mined["status"] != "complete":
                    outcomes[str(number)] = {
                        "status": "SCOPED_MINING_INCOMPLETE",
                        "units": mined["total_work_units"],
                    }
                    continue
                progress_path = Path(mined["progress_path"])
            else:
                record: object = whole_book.load_strict_reply(evaluation._read(state, 16384))
                if (
                    not isinstance(record, dict)
                    or record.get("scope_sha256") != scope_sha
                    or record.get("bundle_sha256") != verified.bundle_sha256
                    or record.get("chapter") != number
                    or record.get("phase") != "AWAITING_HUMAN_REVIEW"
                    or not isinstance(record.get("progress_path"), str)
                ):
                    raise PrivateHostStop("chapter has no immutable prior private brief phase")
                progress_path = _private_path(record["progress_path"], root, must_exist=True)
                if hashlib.sha256(evaluation._read(progress_path, 20_000_000)).hexdigest() != (
                    record.get("progress_sha256")
                ):
                    raise PrivateHostStop("chapter scoped mining ledger changed")
            review_path = None
            if mode == "resume":
                if block["review_file"] is None:
                    raise PrivateHostStop("independent human review file required per chapter")
                review_path = _private_path(block["review_file"], root, must_exist=True)
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
            )
            if mode == "brief":
                if output["status"] != "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW":
                    raise PrivateHostStop("chapter did not stop at mandatory human checkpoint")
                evidence = {
                    "chapter": number,
                    "phase": "AWAITING_HUMAN_REVIEW",
                    "scope_sha256": scope_sha,
                    "bundle_sha256": verified.bundle_sha256,
                    "progress_path": str(progress_path),
                    "progress_sha256": hashlib.sha256(
                        evaluation._read(progress_path, 20_000_000)
                    ).hexdigest(),
                    "brief_identity": output["brief_identity"],
                }
                evaluation._store(private_root, state, evidence)
                _review_packet(runner, number, output)
            outcomes[str(number)] = {
                "status": output["status"],
                "bundle_sha256": verified.bundle_sha256,
                "brief_identity": output.get("brief_identity"),
                "final_result_identity": output.get("final_result_identity"),
                "audio_provenance": output.get("audio_provenance"),
                "publish_eligible": False,
            }
        except (ValueError, OSError, TypeError, RuntimeError):
            outcomes[str(number)] = {
                "status": "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE",
                "publish_eligible": False,
            }
        finally:
            _release_models()
    return {
        "status": "PRIVATE_TWO_CHAPTER_CHECKPOINT",
        "chapters": outcomes,
        "publish_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private two-chapter Codespace checkpoint")
    parser.add_argument("mode", choices=("inspect", "brief", "resume"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--codespace-created-at", required=True)
    args = parser.parse_args(argv)
    try:
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

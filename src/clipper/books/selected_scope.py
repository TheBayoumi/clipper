"""Native, strict, private selected-chapter evidence; never reconstruct BookText.

Exporter verifies the original parent on the retained-source machine. Importer
verifies ONLY immutable selected scope and its internal parent-bound identities.
No claim of host-level re-verification of absent original full-book bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from clipper.stage_contracts import content_fingerprint

from . import acquisition, faithful, mining, whole_book
from .contracts import BookSource, BookText, ReadingBlock

SCHEMA = "clipper-private-selected-chapter-evidence-v1"
MAX_BUNDLE_BYTES = 1_500_000
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class SelectedSourceStop(ValueError):
    """Sanitized selected-scope verification failure."""


@dataclass(frozen=True, slots=True)
class SelectedRef:
    """Parent identity references, NOT an ordinary locally reverified ingest ref."""

    book_id: str
    candidate_fingerprint: str
    acquisition_fingerprint: str
    source_fingerprint: str
    raw_sha256: str
    ingest_stage_key: str
    extracted_sha256: str
    dossier_sha256: str


@dataclass(frozen=True, slots=True)
class SelectedSource:
    """Minimal source view for the existing mining/chapter validators."""

    source: BookSource
    edition_identity: str | None
    content_sha256: str
    source_fingerprint: str
    fingerprint: str
    parser_version: str
    parser_config_sha256: str
    recovery_status: str
    selected_reading_blocks: tuple[ReadingBlock, ...]
    selected_plan: whole_book.WorkPlan
    excluded_pages: tuple[()] = ()
    parent_original_reverified_on_host: bool = False
    source_boundary: str = "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT"


@dataclass(frozen=True, slots=True)
class SelectedEvidence:
    ref: SelectedRef
    book: SelectedSource
    work: whole_book.WorkPlan
    chapter_number: int
    bundle_sha256: str
    section_manifest_sha256: str
    linked_section_ids: tuple[str, ...]


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _section_manifest(ref: SelectedRef, book: SelectedSource, work: whole_book.WorkPlan) -> dict:
    return {
        "schema": "clipper-book-section-evaluation-scope-v1",
        "book_fingerprint": book.fingerprint,
        "source_fingerprint": book.source_fingerprint,
        "ingest_stage_key": ref.ingest_stage_key,
        "parent_plan_fingerprint": work.parent_plan_fingerprint,
        "section_ids": list(work.section_ids),
        "parent_total_units": work.parent_total_units,
        "included_units": len(work.units),
        "excluded_units": work.excluded_units,
    }


def _work_from_blocks(
    blocks: tuple[ReadingBlock, ...], expected: whole_book.WorkPlan
) -> None:
    """Rebuild every text chunk/gap from selected blocks, ignoring absent parent text."""
    if not expected.units or not blocks:
        raise SelectedSourceStop("selected chapter has no original work evidence")
    fields = {
        "source": "native_text",
        "reviewed_image": "original_image_text",
        "original_research_unreviewed": "original_research_text",
        "normalized_research": "normalized_research_text",
        "machine_ocr": "machine_ocr_text",
        "unrecovered_image": "unrecovered_image_text",
    }
    units: list[whole_book.WorkUnit] = []
    gaps: list[dict[str, Any]] = []
    seen_blocks: set[tuple[str, str, str]] = set()
    positions = iter(expected.units)
    selected = set(expected.section_ids)
    for block in blocks:
        key = (block.section_id, block.anchor, block.source_path)
        if (
            block.section_id not in selected
            or key in seen_blocks
            or block.origin not in fields
        ):
            raise SelectedSourceStop("selected reading block is duplicated or out of scope")
        seen_blocks.add(key)
        field = fields[block.origin]
        if block.origin == "reviewed_image" and block.text_provenance == "image_normalized":
            field = "normalized_image_text"
        if block.text_provenance == "machine_ocr":
            field = "machine_ocr_text"
        if not block.text.strip() or block.origin == "unrecovered_image":
            gaps.append(
                {
                    "anchor": block.anchor,
                    "section_id": block.section_id,
                    "source_path": block.source_path,
                    "asset_sha256": block.asset_sha256,
                    "origin": block.origin,
                    "review_status": block.review_status,
                    "research_link_status": block.research_link_status,
                    "reason": (
                        "unrecovered_image" if block.origin == "unrecovered_image" else "empty_text"
                    ),
                }
            )
            continue
        words = block.text.split()
        for offset in range(0, len(words), mining.CHUNK_WORDS):
            original = next(positions, None)
            if original is None:
                raise SelectedSourceStop("selected bundle drops or invents original work units")
            derived = whole_book.WorkUnit(
                ordinal=original.ordinal,
                anchor=block.anchor,
                section_id=block.section_id,
                source_path=block.source_path,
                word_offset=offset,
                evidence_field=field,
                origin=block.origin,
                review_status=block.review_status,
                asset_sha256=block.asset_sha256,
                research_link_status=block.research_link_status,
                text=" ".join(words[offset : offset + mining.CHUNK_WORDS]),
            )
            if derived != original:
                raise SelectedSourceStop("selected original source unit does not match reading block")
            units.append(derived)
    if (
        next(positions, None) is not None
        or tuple(units) != expected.units
        or tuple(gaps) != expected.skipped
        or any(a.ordinal >= b.ordinal for a, b in zip(units, units[1:]))
        or any(u.ordinal < 0 or u.ordinal >= expected.parent_total_units for u in units)
        or expected.parent_total_units != len(units) + expected.excluded_units
        or expected.parent_plan_fingerprint is None
        or not set(expected.section_ids).issubset({b.section_id for b in blocks})
    ):
        raise SelectedSourceStop("selected bundle ordered source/gap/parent accounting differs")


def export_chapter(
    runner: Any,
    candidate_ref: Path,
    *,
    chapter_number: int,
    chapter_section_ids: tuple[str, ...],
    linked_section_ids: tuple[str, ...] = (),
) -> tuple[Path, str]:
    """Verify ORIGINAL parent locally, then export ONLY selected canonical evidence."""
    if type(chapter_number) is not int or chapter_number < 1:
        raise SelectedSourceStop("chapter number must be positive integer")
    if (
        not chapter_section_ids
        or set(chapter_section_ids) & set(linked_section_ids)
        or len(set((*chapter_section_ids, *linked_section_ids)))
        != len(chapter_section_ids) + len(linked_section_ids)
    ):
        raise SelectedSourceStop("chapter/linked section selection overlaps or is empty")
    ref, book = runner.load_candidate_for_analysis(candidate_ref)
    order = tuple(
        sid
        for sid in dict.fromkeys(block.section_id for block in faithful.reading_blocks(book))
        if sid in set((*chapter_section_ids, *linked_section_ids))
    )
    if set(order) != set((*chapter_section_ids, *linked_section_ids)):
        raise SelectedSourceStop("original linked/chapter section selection is incomplete")
    work = whole_book.plan_for(book, section_ids=order)
    if not work.section_ids or len(work.units) + len(work.skipped) > 64:
        raise SelectedSourceStop("selected chapter exceeds finite original-unit budget")
    blocks = tuple(
        block for block in faithful.reading_blocks(book) if block.section_id in set(order)
    )
    _work_from_blocks(blocks, work)
    parent = SelectedRef(
        book_id=ref.book_id,
        candidate_fingerprint=ref.candidate_fingerprint,
        acquisition_fingerprint=ref.acquisition_fingerprint,
        source_fingerprint=ref.source_fingerprint,
        raw_sha256=ref.raw_sha256,
        ingest_stage_key=ref.ingest_stage_key,
        extracted_sha256=ref.extracted_sha256,
        dossier_sha256=ref.dossier_sha256,
    )
    selected = SelectedSource(
        source=book.source,
        edition_identity=book.edition_identity,
        content_sha256=book.content_sha256,
        source_fingerprint=book.source_fingerprint,
        fingerprint=book.fingerprint,
        parser_version=book.parser_version,
        parser_config_sha256=book.parser_config_sha256,
        recovery_status=book.recovery_status,
        selected_reading_blocks=blocks,
        selected_plan=work,
    )
    section = _section_manifest(parent, selected, work)
    evidence = {
        "schema": SCHEMA,
        "chapter_number": chapter_number,
        "verification": "PARENT_VERIFIED_AT_EXPORTER_SELECTED_BUNDLE_VERIFIED_ON_HOST",
        "parent_original_reverified_on_host": False,
        "parent_full_book_complete": False,
        "review_status": "SELECTED_SCOPE_SOURCE_UNREVIEWED",
        "publish_eligible": False,
        "parent_ref": asdict(parent),
        "source": asdict(book.source),
        "source_identity": {
            "edition_identity": book.edition_identity,
            "parent_book_fingerprint": book.fingerprint,
            "parent_content_sha256": book.content_sha256,
            "source_fingerprint": book.source_fingerprint,
            "parser_version": book.parser_version,
            "parser_config_sha256": book.parser_config_sha256,
            "recovery_status": book.recovery_status,
        },
        "chapter_section_ids": list(chapter_section_ids),
        "linked_section_ids": list(linked_section_ids),
        "reading_blocks": [asdict(block) for block in blocks],
        "work_plan": {
            "book_fingerprint": work.book_fingerprint,
            "units": [asdict(unit) for unit in work.units],
            "skipped": list(work.skipped),
            "section_ids": list(work.section_ids),
            "parent_plan_fingerprint": work.parent_plan_fingerprint,
            "parent_total_units": work.parent_total_units,
            "excluded_units": work.excluded_units,
            "fingerprint": work.fingerprint,
        },
        "section_manifest": section,
        "section_manifest_sha256": hashlib.sha256(_json(section)).hexdigest(),
        "linked_notes_figures_completeness": "UNVERIFIED_REQUIRES_SOURCE_REVIEW",
        "parent_excluded_page_count": len(book.excluded_pages),
    }
    data = _json(evidence)
    if len(data) > MAX_BUNDLE_BYTES:
        raise SelectedSourceStop("selected chapter evidence exceeds bounded private bundle")
    sha = hashlib.sha256(data).hexdigest()
    path = runner.root / "book-candidates" / "selected-scopes" / f"{sha}.json"
    acquisition._private_write(path, data, private_root=runner.root / "book-candidates")
    return path, sha


def import_chapter(private_root: Path, path: Path) -> SelectedEvidence:
    """Check canonical immutable chapter only; NEVER access absent parent raw bytes."""
    try:
        base = private_root / "book-candidates"
        if (
            path.is_symlink()
            or path.resolve(strict=True).parent != (base / "selected-scopes").resolve()
            or DIGEST.fullmatch(path.stem) is None
            or path.suffix != ".json"
        ):
            raise SelectedSourceStop("selected bundle path is not content-addressed private evidence")
        raw = acquisition._read_bounded_file(path, MAX_BUNDLE_BYTES + 1)
        if len(raw) > MAX_BUNDLE_BYTES or hashlib.sha256(raw).hexdigest() != path.stem:
            raise SelectedSourceStop("selected chapter bundle length/hash mismatch")
        data: object = whole_book.load_strict_reply(raw)
        if not isinstance(data, dict) or set(data) != {
            "schema", "chapter_number", "verification",
            "parent_original_reverified_on_host", "parent_full_book_complete",
            "review_status", "publish_eligible", "parent_ref", "source",
            "source_identity", "chapter_section_ids", "linked_section_ids",
            "reading_blocks", "work_plan", "section_manifest",
            "section_manifest_sha256", "linked_notes_figures_completeness",
            "parent_excluded_page_count",
        } or raw != _json(data):
            raise SelectedSourceStop("selected chapter bundle is noncanonical or wrong schema")
        if (
            data["schema"] != SCHEMA
            or data["verification"] != "PARENT_VERIFIED_AT_EXPORTER_SELECTED_BUNDLE_VERIFIED_ON_HOST"
            or data["parent_original_reverified_on_host"] is not False
            or data["parent_full_book_complete"] is not False
            or data["publish_eligible"] is not False
            or data["review_status"] != "SELECTED_SCOPE_SOURCE_UNREVIEWED"
            or data["linked_notes_figures_completeness"] != "UNVERIFIED_REQUIRES_SOURCE_REVIEW"
            or type(data["chapter_number"]) is not int
            or data["chapter_number"] < 1
            or type(data["parent_excluded_page_count"]) is not int
            or data["parent_excluded_page_count"] < 0
        ):
            raise SelectedSourceStop("selected source boundary falsely claims approval/completeness")
        reference = data["parent_ref"]
        if not isinstance(reference, dict) or set(reference) != set(SelectedRef.__dataclass_fields__):
            raise SelectedSourceStop("selected original parent references malformed")
        ref = SelectedRef(**reference)
        if not isinstance(ref.book_id, str) or not ref.book_id or any(
            type(getattr(ref, k)) is not str or DIGEST.fullmatch(getattr(ref, k)) is None
            for k in SelectedRef.__dataclass_fields__ if k != "book_id"
        ):
            raise SelectedSourceStop("selected original parent identity digest malformed")
        from .contracts import BookSource

        if not isinstance(data["source"], dict) or set(data["source"]) != set(
            BookSource.__dataclass_fields__
        ):
            raise SelectedSourceStop("selected parent source metadata malformed")
        source = BookSource(**data["source"])
        identity = data["source_identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "edition_identity", "parent_book_fingerprint", "parent_content_sha256",
            "source_fingerprint", "parser_version", "parser_config_sha256", "recovery_status",
        }:
            raise SelectedSourceStop("selected parent source identity malformed")
        if any(
            type(identity[k]) is not str or DIGEST.fullmatch(identity[k]) is None
            for k in ("parent_book_fingerprint", "parent_content_sha256", "source_fingerprint",
                      "parser_config_sha256")
        ) or (
            identity["parent_content_sha256"] != ref.raw_sha256
            or source.content_sha256 != ref.raw_sha256
            or source.fingerprint != ref.source_fingerprint
            or identity["source_fingerprint"] != ref.source_fingerprint
            or source.authorization_status != "unverified"
        ):
            raise SelectedSourceStop("selected original source parent hashes disagree")
        if not isinstance(data["reading_blocks"], list) or not 1 <= len(data["reading_blocks"]) <= 100:
            raise SelectedSourceStop("selected original reading blocks absent/oversized")
        allowed = set(ReadingBlock.__dataclass_fields__)
        if any(not isinstance(block, dict) or set(block) != allowed for block in data["reading_blocks"]):
            raise SelectedSourceStop("selected reading block shape invalid")
        blocks = tuple(ReadingBlock(**block) for block in data["reading_blocks"])
        if not isinstance(data["work_plan"], dict) or set(data["work_plan"]) != {
            "book_fingerprint", "units", "skipped", "section_ids", "parent_plan_fingerprint",
            "parent_total_units", "excluded_units", "fingerprint",
        }:
            raise SelectedSourceStop("selected original work plan shape invalid")
        record = data["work_plan"]
        if (
            not isinstance(record["units"], list)
            or not 1 <= len(record["units"]) <= whole_book.MAX_PLAN_UNITS
            or not isinstance(record["skipped"], list)
            or len(record["units"]) + len(record["skipped"]) > 64
            or any(not isinstance(u, dict) or set(u) != set(whole_book.WorkUnit.__dataclass_fields__) for u in record["units"])
            or type(record["parent_total_units"]) is not int
            or type(record["excluded_units"]) is not int
            or type(record["book_fingerprint"]) is not str
        ):
            raise SelectedSourceStop("selected original work-unit bound/shape mismatch")
        work = whole_book.WorkPlan(
            book_fingerprint=record["book_fingerprint"],
            units=tuple(whole_book.WorkUnit(**u) for u in record["units"]),
            skipped=tuple(record["skipped"]),
            section_ids=tuple(record["section_ids"]),
            parent_plan_fingerprint=record["parent_plan_fingerprint"],
            parent_total_units=record["parent_total_units"],
            excluded_units=record["excluded_units"],
        )
        ids = data["chapter_section_ids"]
        linked = data["linked_section_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or not isinstance(linked, list)
            or any(not isinstance(x, str) or not x for x in [*ids, *linked])
            or len(set([*ids, *linked])) != len(ids) + len(linked)
            or set([*ids, *linked]) != set(work.section_ids)
            or work.fingerprint != record["fingerprint"]
            or work.book_fingerprint != identity["parent_book_fingerprint"]
            or type(work.parent_total_units) is not int
            or work.parent_total_units <= 0
            or type(work.excluded_units) is not int
            or work.excluded_units < 0
        ):
            raise SelectedSourceStop("selected chapter original ordered scope identity mismatch")
        # All reading blocks and linked notes remain in their original parent order.
        projected = tuple(dict.fromkeys(block.section_id for block in blocks))
        if projected != work.section_ids:
            raise SelectedSourceStop("selected chapter source block section order changed")
        _work_from_blocks(blocks, work)
        book = SelectedSource(
            source=source, edition_identity=identity["edition_identity"],
            content_sha256=ref.raw_sha256, source_fingerprint=ref.source_fingerprint,
            fingerprint=identity["parent_book_fingerprint"],
            parser_version=identity["parser_version"],
            parser_config_sha256=identity["parser_config_sha256"],
            recovery_status=identity["recovery_status"],
            selected_reading_blocks=blocks, selected_plan=work,
        )
        manifest = _section_manifest(ref, book, work)
        if (
            data["section_manifest"] != manifest
            or data["section_manifest_sha256"] != hashlib.sha256(_json(manifest)).hexdigest()
        ):
            raise SelectedSourceStop("selected source scope manifest/parent hashes diverge")
        return SelectedEvidence(
            ref, book, work, data["chapter_number"], path.stem,
            data["section_manifest_sha256"], tuple(linked),
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError, OverflowError):
        raise SelectedSourceStop("selected chapter evidence cannot be verified") from None


def reading_blocks(book: BookText | SelectedSource) -> tuple[ReadingBlock, ...]:
    if isinstance(book, SelectedSource):
        return book.selected_reading_blocks
    return tuple(faithful.reading_blocks(book))


def main(argv: list[str] | None = None) -> int:
    """Exporter runs on retained-source owner machine only, never in Codespace."""
    parser = argparse.ArgumentParser(description="Export original verified selected chapter privately")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-ref", type=Path, required=True)
    parser.add_argument("--chapter-number", type=int, required=True)
    parser.add_argument("--chapter-sections", nargs="+", required=True)
    parser.add_argument("--linked-sections", nargs="*", default=[])
    args = parser.parse_args(argv)
    try:
        from .orchestrator import BookOrchestrator

        path, digest = export_chapter(
            BookOrchestrator(args.artifact_root),
            args.candidate_ref,
            chapter_number=args.chapter_number,
            chapter_section_ids=tuple(args.chapter_sections),
            linked_section_ids=tuple(args.linked_sections),
        )
        print(json.dumps({"status": "SELECTED_SCOPE_EXPORTED", "sha256": digest, "path": str(path)}))
        return 0
    except Exception:
        print(json.dumps({"status": "SELECTED_SCOPE_EXPORT_FAILED_CLOSED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

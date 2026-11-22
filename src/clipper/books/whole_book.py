"""Bounded, provenance-labelled, offline whole-book candidate analysis.

The legacy curated selector and one-shot mine_book remain separate. This module
never loads weights or selects approved editorial facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from clipper.stage_contracts import content_fingerprint

from . import faithful, mining, scripting, selection
from .contracts import BookText

REF_SCHEMA = "clipper-book-candidate-analysis-ref-v1"
PLAN_SCHEMA = "clipper-book-evidence-work-plan-v1"
RESULT_SCHEMA = "clipper-book-whole-analysis-v1"
WINDOW_UNITS = 32
MAX_PLAN_UNITS = 20_000
MAX_RECORDED_REPLY_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class CandidateAnalysisRef:
    schema: str
    book_id: str
    candidate_fingerprint: str
    candidate: dict[str, Any]
    acquisition_fingerprint: str
    source_fingerprint: str
    raw_sha256: str
    ingest_stage_key: str
    extracted_sha256: str
    dossier_sha256: str
    book_fingerprint: str
    parser_version: str
    parser_config_sha256: str
    recovery_status: str


@dataclass(frozen=True, slots=True)
class WorkUnit:
    ordinal: int
    anchor: str
    section_id: str
    source_path: str
    word_offset: int
    evidence_field: str
    origin: str
    review_status: str
    asset_sha256: str | None
    research_link_status: str
    text: str

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkPlan:
    book_fingerprint: str
    units: tuple[WorkUnit, ...]
    skipped: tuple[dict[str, Any], ...]

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(
            {
                "schema": PLAN_SCHEMA,
                "book": self.book_fingerprint,
                "units": [unit.fingerprint for unit in self.units],
                "skipped": self.skipped,
                "window_units": WINDOW_UNITS,
            }
        )


def plan_for(book: BookText) -> WorkPlan:
    """Visit every ordered source/image block; never silently discard an image gap."""
    if not book.blocks:
        raise mining.BookMiningError("candidate analysis requires faithful ordered reading blocks")
    if book.source.authorization_status != "unverified":
        raise mining.BookMiningError("analysis cannot upgrade source authorization")
    if not selection._source_url(book.source.source_uri):
        raise mining.BookMiningError("candidate analysis requires the pinned HTTPS citation URI")
    fields = {
        "source": "native_text",
        "reviewed_image": "original_image_text",
        "original_research_unreviewed": "original_research_text",
        "normalized_research": "normalized_research_text",
        "machine_ocr": "machine_ocr_text",
        "unrecovered_image": "unrecovered_image_text",
    }
    units: list[WorkUnit] = []
    skipped: list[dict[str, Any]] = []
    for block in faithful.reading_blocks(book):
        field = fields[block.origin]
        if block.origin == "reviewed_image" and block.text_provenance == "image_normalized":
            field = "normalized_image_text"
        if block.text_provenance == "machine_ocr":
            field = "machine_ocr_text"
        if not block.text.strip() or block.origin == "unrecovered_image":
            skipped.append(
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
            if len(units) >= MAX_PLAN_UNITS:
                raise mining.BookMiningError(
                    "whole-book evidence work plan exceeds bounded unit limit"
                )
            units.append(
                WorkUnit(
                    ordinal=len(units),
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
            )
    if not units and not skipped:
        raise mining.BookMiningError("faithful projection contains no accountable evidence")
    return WorkPlan(book.fingerprint, tuple(units), tuple(skipped))


def request_for(book: BookText, unit: WorkUnit) -> dict[str, object]:
    """Public standalone request, deriving its own verified content identity."""
    return _request_for_identity(book, unit, book.fingerprint)


def _request_for_identity(
    book: BookText, unit: WorkUnit, book_fingerprint: str
) -> dict[str, object]:
    """Private batching helper: identity comes from plan_for(book), not caller input."""
    evidence = {
        "book_title": book.source.title,
        "book_author": book.source.author,
        "edition": book.edition_identity,
        "source_sha256": book.content_sha256,
        "book_fingerprint": book_fingerprint,
        "anchor": unit.anchor,
        "section_id": unit.section_id,
        "source_path": unit.source_path,
        "asset_sha256": unit.asset_sha256,
        "origin": unit.origin,
        "review_status": unit.review_status,
        "research_link_status": unit.research_link_status,
        "evidence_field": unit.evidence_field,
        "text_sha256": unit.text_sha256,
        "word_offset": unit.word_offset,
        "text": unit.text,
    }
    return {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 960,
        "messages": [
            {
                "role": "system",
                "content": (
                    'Return JSON only: {"lessons":[{"id":str,"topic":str,"claim":str,'
                    '"hook_angle":str,"visual_action":str,"evidence":str,'
                    '"evidence_field":str}]}. Return zero to three unreviewed candidates.'
                    " Quote an exact substring only from the supplied text field. "
                    "Repeat its evidence_field verbatim. Image/OCR/research wording is "
                    "not authenticated original book prose. Never assert approval."
                ),
            },
            {"role": "user", "content": json.dumps(evidence, sort_keys=True, ensure_ascii=False)},
        ],
    }


def validate_reply(payload: object, unit: WorkUnit) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or set(payload) != {"lessons"}:
        raise mining.BookMiningError("analysis response requires only a lessons array")
    items = payload["lessons"]
    if not isinstance(items, list) or len(items) > 3:
        raise mining.BookMiningError(
            "analysis response allows zero to three lessons per evidence unit"
        )
    seen: set[str] = set()
    validated: list[dict[str, str]] = []
    expected = {"id", "topic", "claim", "hook_angle", "visual_action", "evidence", "evidence_field"}
    for item in items:
        if not isinstance(item, dict) or set(item) != expected:
            raise mining.BookMiningError("analysis lesson fields are incomplete or unexpected")
        if any(
            not isinstance(item[key], str) or not item[key].strip() or len(item[key]) > 400
            for key in expected
        ):
            raise mining.BookMiningError("analysis lesson fields require bounded nonempty text")
        if not selection.IDENTIFIER.fullmatch(item["id"]) or item["id"] in seen:
            raise mining.BookMiningError("invalid or duplicate candidate id within evidence unit")
        seen.add(item["id"])
        if item["evidence_field"] != unit.evidence_field or item["evidence"] not in unit.text:
            raise mining.BookMiningError("evidence quote does not match its exact provenance field")
        validated.append(item)
    return validated


def candidate_for(book: BookText, unit: WorkUnit, item: dict[str, str]) -> dict[str, Any]:
    local_key = content_fingerprint({"unit": unit.fingerprint, "candidate_id": item["id"]})
    return {
        "id": f"{item['id']}-{local_key}",
        "proposed_id": item["id"],
        "topic": item["topic"],
        "claim": item["claim"],
        "hook_angle": item["hook_angle"],
        "visual_action": item["visual_action"],
        "claim_source_url": book.source.source_uri,
        "review_status": "MINED_UNREVIEWED",
        "publish_eligible": False,
        "evidence_anchor": {
            "anchor": unit.anchor,
            "section_id": unit.section_id,
            "source_path": unit.source_path,
            "asset_sha256": unit.asset_sha256,
            "word_offset": unit.word_offset,
            "field": unit.evidence_field,
            "origin": unit.origin,
            "review_status": unit.review_status,
            "research_link_status": unit.research_link_status,
            "source_sha256": book.content_sha256,
            "text_sha256": unit.text_sha256,
            "quote": item["evidence"],
        },
    }


def recorded_answer(replies: dict[str, object], request: dict[str, object]) -> object | None:
    """Absent answer means unresolved work; malformed answer remains a failure."""
    return replies.get(content_fingerprint(request))

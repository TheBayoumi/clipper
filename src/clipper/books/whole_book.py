"""Bounded, provenance-labelled, offline whole-book candidate analysis.

The legacy curated selector and one-shot mine_book remain separate. This module
never loads weights or selects approved editorial facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Protocol, cast

from clipper.stage_contracts import content_fingerprint

from . import constrained, faithful, mining, model_profile, scripting, selection
from .contracts import BookSource, BookText

if TYPE_CHECKING:
    from .selected_scope import SelectedSource


class SourceMetadata(Protocol):
    """Narrow source identity shared by retained and selected work-unit validators."""

    @property
    def source(self) -> BookSource: ...

    @property
    def edition_identity(self) -> str | None: ...

    @property
    def content_sha256(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...


REF_SCHEMA = "clipper-book-candidate-analysis-ref-v1"
PLAN_SCHEMA = "clipper-book-evidence-work-plan-v1"
RESULT_SCHEMA = "clipper-book-whole-analysis-v1"
WINDOW_UNITS = 32
MAX_PLAN_UNITS = 20_000
MAX_RECORDED_REPLY_BYTES = 16_384
LOCAL_COMPLETION_SCHEMA = "clipper-book-local-completion-v1"
LOCAL_PRODUCER_CONTRACT = "clipper-book-cpu-qwen-evidence-support-public-callback-v8"


def _local_generation() -> dict[str, object]:
    """Generation identity of the CPU producer, derived from the active model profile."""
    profile = model_profile.active()
    return {
        "model": profile.model,
        "revision": profile.revision,
        "seed": 42,
        "max_new_tokens": 960,
        **profile.sampling,
        **profile.template_kwargs,
        "dtype": "auto",
        "device": "cpu",
        "cpu_threads_max": 4,
    }


def local_producer_material() -> dict[str, object]:
    """Resolve installed runtime versions without importing torch or loading weights."""
    runtime: dict[str, str] = {}
    for package in ("torch", "transformers", constrained.DECODER_PACKAGE):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "not-installed"
    return {
        "contract": LOCAL_PRODUCER_CONTRACT,
        "schema": LOCAL_COMPLETION_SCHEMA,
        "generation": {**_local_generation(), **scripting.QWEN_GENERATION_SETTINGS},
        "runtime": runtime,
        "request_schema": PLAN_SCHEMA,
        "reply_max_bytes": MAX_RECORDED_REPLY_BYTES,
        "validator": "exact-quote-evidence-field-and-0-to-3-lessons-v1",
        "decoder": {
            "package": constrained.DECODER_PACKAGE,
            "version": constrained.DECODER_VERSION,
            "grammar": constrained.GRAMMAR_CONTRACT,
            "span_policy": constrained.SPAN_POLICY,
            "pool_schema": constrained.POOL_SCHEMA,
            "tokenizer_model": scripting.SCRIPT_MODEL,
            "tokenizer_revision": scripting.SCRIPT_REVISION,
            "tokenizer_json_sha256": constrained.PINNED_TOKENIZER_SHA256,
            "literal_spelling": "per-codepoint-direct-or-json-escaped-v1",
            "prompt_context_preflight": "pinned-tokenizer-full-chat-plus-960-before-weights-v1",
        },
    }


def hybrid_chapter_producer_material() -> dict[str, object]:
    """Bind the verified public tokenizer observation only to chapter work.

    The legacy miner reserves its DAG attempt before the provider may warm
    the cache. This chapter-only material must never enter the legacy key.
    """
    return {
        **local_producer_material(),
        "effective_tokenizer_cache": scripting._cached_chapter_tokenizer_material(),
    }


class LocalReplyFailure(RuntimeError):
    """A sanitized producer/response failure; never includes private content."""

    def __init__(self, category: str, raw_sha256: str | None = None) -> None:
        self.category = category
        self.raw_sha256 = raw_sha256
        super().__init__(f"local candidate completion {category}")


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
    section_ids: tuple[str, ...] = ()
    parent_plan_fingerprint: str | None = None
    parent_total_units: int = 0
    excluded_units: int = 0

    @property
    def section_scope(self) -> dict[str, object] | None:
        if not self.section_ids:
            return None
        return {
            "schema": "clipper-book-section-scope-v1",
            "section_ids": list(self.section_ids),
            "parent_plan_fingerprint": self.parent_plan_fingerprint,
            "parent_total_units": self.parent_total_units,
            "included_units": len(self.units),
            "excluded_units": self.excluded_units,
            "parent_full_book_complete": False,
        }

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(
            {
                "schema": PLAN_SCHEMA,
                "book": self.book_fingerprint,
                "units": [unit.fingerprint for unit in self.units],
                "skipped": self.skipped,
                "window_units": WINDOW_UNITS,
                **({"section_scope": self.section_scope} if self.section_ids else {}),
            }
        )


def plan_for(
    book: BookText | SelectedSource, *, section_ids: tuple[str, ...] | None = None
) -> WorkPlan:
    """Visit every ordered source/image block; never silently discard an image gap."""
    from .selected_scope import SelectedSource

    if isinstance(book, SelectedSource):
        # Preserve original parent ordinals; do not reassemble truncated BookText.
        if section_ids != book.selected_plan.section_ids:
            raise mining.BookMiningError("selected chapter cannot broaden or reorder source scope")
        return book.selected_plan
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
    parent = WorkPlan(book.fingerprint, tuple(units), tuple(skipped))
    if section_ids is None:
        return parent
    available = {block.section_id for block in faithful.reading_blocks(book)}
    if (
        not isinstance(section_ids, tuple)
        or not 1 <= len(section_ids) <= 16
        or any(
            not isinstance(value, str) or not value or value not in available
            for value in section_ids
        )
        or len(set(section_ids)) != len(section_ids)
    ):
        raise mining.BookMiningError("section scope must contain exact canonical section IDs")
    # Preserve parent ordinal, source path, offsets and underlying BookText identity.
    selected = set(section_ids)
    scoped_units = tuple(unit for unit in parent.units if unit.section_id in selected)
    if not scoped_units:
        raise mining.BookMiningError("section scope contains no faithful source work units")
    order = tuple(
        section_id
        for section_id in dict.fromkeys(block.section_id for block in faithful.reading_blocks(book))
        if section_id in selected
    )
    if order != section_ids:
        raise mining.BookMiningError("section IDs must follow canonical source reading order")
    return WorkPlan(
        parent.book_fingerprint,
        scoped_units,
        tuple(item for item in parent.skipped if item["section_id"] in selected),
        section_ids,
        parent.fingerprint,
        len(parent.units),
        len(parent.units) - len(scoped_units),
    )


def request_for(book: BookText, unit: WorkUnit) -> dict[str, object]:
    """Public standalone request, deriving its own verified content identity."""
    return _request_for_identity(book, unit, book.fingerprint)


def _request_for_identity(
    book: SourceMetadata, unit: WorkUnit, book_fingerprint: str
) -> dict[str, object]:
    """Private batching helper: identity comes from plan_for(book), not caller input."""
    pool = constrained.span_pool(unit.text, unit.evidence_field)
    spans = cast(list[dict[str, object]], pool["spans"])
    evidence = {
        "source_evidence_choices": [
            {
                "id": span["id"],
                "evidence": span["text"],
                "start": span["start"],
                "end": span["end"],
                "boundary": span["boundary"],
            }
            for span in spans
        ],
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
        "constraints": pool,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return exactly one valid JSON object with only the top-level "
                    'key "lessons"; no prose, Markdown, or code fences. '
                    '"lessons" must contain 0 to 3 candidate objects; use an empty '
                    "array only when no supported candidate exists, never as "
                    "useful-lesson success. Each candidate must contain exactly "
                    'the keys "id", "topic", "claim", "hook_angle", '
                    '"visual_action", "evidence", and "evidence_field" with no others. '
                    "Every value must be a nonempty string of at most 400 characters. "
                    'Choose each "id" and "evidence" as one exact paired entry '
                    'from "source_evidence_choices" in the user message, not '
                    "independently from different entries. Do not change either "
                    "field, even when multiple occurrences quote the same text. "
                    "IDs are safe lowercase slugs matching "
                    "^[a-z0-9]+(?:-[a-z0-9]+)*$ and must not repeat in one reply. "
                    "The full source text remains available for context and uncertainty, "
                    "but the exact selected evidence entry ALONE must directly "
                    "support the proposed claim. Do not use unrelated nearby "
                    "sentences to justify a stronger claim. Preserve limiting "
                    "words, exceptions, doubts, and counterexamples from surrounding "
                    "context; never generalize beyond the observed example. "
                    "Prefer a narrow useful mechanism or testable action over "
                    "trivia or a generic topic. Abstain with empty lessons rather "
                    "than invent facts, smooth over contradictions, or force three. "
                    "The bounded choice menu identifies eligible occurrences. "
                    'Write a specific claim directly supported by its "evidence", '
                    "not a generic topic or invented fact. Keep the topic, "
                    "hook angle, and visual action grounded in that claim. "
                    'The "evidence" must reproduce the selected entry exactly '
                    "(maximum 400 characters), without paraphrase or ellipses. "
                    "Do not substitute a different substring from source text. "
                    'The "evidence_field" must exactly repeat the supplied '
                    '"evidence_field" value. All candidates remain unreviewed '
                    "and not approved or publishable. Image/OCR/research wording "
                    "is not authenticated original book prose."
                ),
            },
            {"role": "user", "content": json.dumps(evidence, sort_keys=True, ensure_ascii=False)},
        ],
    }


def load_strict_reply(raw: str | bytes) -> object:
    """Reject repeated JSON keys, including nested candidate objects, without repair."""

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, entry in pairs:
            if key in value:
                raise mining.BookMiningError("duplicate JSON object key in local book reply")
            value[key] = entry
        return value

    return json.loads(raw, object_pairs_hook=unique)


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


def candidate_for(
    book: SourceMetadata,
    unit: WorkUnit,
    item: dict[str, str],
    *,
    local_span: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Preserve recorded provenance; attach local verified occurrence only."""
    local_key = content_fingerprint({"unit": unit.fingerprint, "candidate_id": item["id"]})
    if local_span is not None:
        start, end = cast(int, local_span["start"]), cast(int, local_span["end"])
        if (
            start < 0
            or end > len(unit.text)
            or start >= end
            or unit.text[start:end] != item["evidence"]
            or local_span["id"] != item["id"]
            or hashlib.sha256(unit.text[start:end].encode("utf-8")).hexdigest()
            != local_span["sha256"]
        ):
            raise mining.BookMiningError("selected local evidence occurrence is inconsistent")
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
            **(
                {
                    "selected_occurrence": {
                        "unit_start": local_span["start"],
                        "unit_end": local_span["end"],
                        "quote_sha256": local_span["sha256"],
                        "boundary": local_span["boundary"],
                        "id": item["id"],
                    }
                }
                if local_span is not None
                else {}
            ),
        },
    }


def recorded_answer(replies: dict[str, object], request: dict[str, object]) -> object | None:
    """Absent answer means unresolved work; malformed answer remains a failure."""
    return replies.get(content_fingerprint(request))


def __getattr__(name: str) -> dict[str, object]:
    """Historical name for the CPU producer's generation identity under the active profile."""
    if name == "LOCAL_GENERATION":
        return _local_generation()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

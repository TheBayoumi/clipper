"""Private, source-pinned chapter preproduction; never short Episode or visual rendering."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from typing import TYPE_CHECKING, Any, cast

from clipper.stage_contracts import content_fingerprint

from . import acquisition, evaluation, faithful, narration, scripting, whole_book

if TYPE_CHECKING:
    from .orchestrator import BookOrchestrator

PLAN_SCHEMA = "clipper-chapter-preproduction-run-v2"
BRIEF_SCHEMA = "clipper-chapter-brief-v2"
REVIEW_SCHEMA = "clipper-chapter-human-beat-review-v2"
ADAPT_SCHEMA = "clipper-chapter-adaptation-v1"
SCRIPT_SCHEMA = "clipper-chapter-script-graph-v1"
DIRECTION_SCHEMA = "clipper-chapter-continuity-shots-v2"
MAX_RAW_BYTES = 100_000
MAX_CONTEXT_TOKENS = 40_960
CHAT_TOKEN_RESERVE = 512
MAX_SOURCE_BYTES = 25_000
ENTITY_ID = re.compile(r"[a-z][a-z0-9-]{1,60}\Z")
TEXT_STAGES = ("brief", "adaptation", "chapter_script", "chapter_direction")


class ChapterStop(ValueError):
    """Fail closed without exposing source or model output in public errors."""


@dataclass(frozen=True)
class ChapterPlan:
    brief_cpu: bool = False
    adaptation_cpu: bool = False
    script_cpu: bool = False
    direction_cpu: bool = False
    kokoro_cpu: bool = False
    max_scenes: int = 8
    max_words: int = 1000
    max_beats: int = 20
    max_shots_per_scene: int = 3
    max_new_tokens: int = 4096
    max_attempts_per_stage: int = 2
    max_runtime_seconds: int = 900
    voice: str = "af_heart"

    @classmethod
    def from_manifest(cls, raw: object) -> ChapterPlan:
        defaults = cls()
        expected = {"schema", *defaults.__dataclass_fields__}
        if not isinstance(raw, dict) or set(raw) != expected or raw["schema"] != PLAN_SCHEMA:
            raise ChapterStop("invalid chapter plan schema")
        flags = ("brief_cpu", "adaptation_cpu", "script_cpu", "direction_cpu", "kokoro_cpu")
        if any(type(raw[key]) is not bool for key in flags):
            raise ChapterStop("chapter permissions must be explicit booleans")
        ranges = {
            "max_scenes": (2, 20),
            "max_words": (130, 2400),
            "max_beats": (2, 24),
            "max_shots_per_scene": (1, 4),
            "max_new_tokens": (1024, 8192),
            "max_attempts_per_stage": (1, 2),
            "max_runtime_seconds": (60, 900),
        }
        if any(
            type(raw[name]) is not int or not low <= raw[name] <= high
            for name, (low, high) in ranges.items()
        ):
            raise ChapterStop("chapter plan exceeds explicit finite limits")
        if raw["voice"] not in scripting.VOICES:
            raise ChapterStop("chapter plan voice is unsupported")
        return cls(**{key: raw[key] for key in defaults.__dataclass_fields__})

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(
            {
                "contract": PLAN_SCHEMA,
                **{key: getattr(self, key) for key in self.__dataclass_fields__},
                "model": scripting.SCRIPT_MODEL,
                "revision": scripting.SCRIPT_REVISION,
            }
        )


def _shape(obj: object, fields: set[str], code: str) -> dict[str, Any]:
    if not isinstance(obj, dict) or set(obj) != fields:
        raise ChapterStop(code)
    return cast(dict[str, Any], obj)


def _words(value: object, code: str, maximum: int = 1500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ChapterStop(code)
    return value


def _items(value: object, maximum: int, code: str) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ChapterStop(code)
    return value


def _identity(value: object, code: str) -> str:
    if not isinstance(value, str) or ENTITY_ID.fullmatch(value) is None:
        raise ChapterStop(code)
    return value


def _unique(values: list[str], code: str) -> None:
    if len(set(values)) != len(values):
        raise ChapterStop(code)


def _full_source_scope(
    book: Any,
    scoped: whole_book.WorkPlan,
    ledger: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Retain every original canonical work unit, including zero-candidate units.

    Skipped/unrecovered reading blocks are interleaved as explicit gap markers,
    not treated as source-authenticated passages or silently eliminated.
    """
    by_unit: dict[tuple[str, int, str], list[str]] = {}
    for candidate in candidates:
        anchor = candidate["evidence_anchor"]
        key = (anchor["anchor"], anchor["word_offset"], anchor["text_sha256"])
        by_unit.setdefault(key, []).append(candidate["id"])
    units: list[dict[str, Any]] = []
    for work, record in zip(scoped.units, ledger["progress"], strict=True):
        source_id = f"unit-{work.fingerprint}"
        ids = by_unit.pop((work.anchor, work.word_offset, work.text_sha256), [])
        units.append(
            {
                "source_id": source_id,
                "kind": "CANONICAL_WORK_UNIT",
                "original_parent_ordinal": work.ordinal,
                "unit_fingerprint": work.fingerprint,
                "anchor": work.anchor,
                "section_id": work.section_id,
                "source_path": work.source_path,
                "unit_word_offset": work.word_offset,
                "evidence_field": work.evidence_field,
                "origin": work.origin,
                "review_status": work.review_status,
                "research_link_status": work.research_link_status,
                "asset_sha256": work.asset_sha256,
                "text_sha256": work.text_sha256,
                "text": work.text,
                "candidate_ids": ids,
                "mining_status": record["status"],
                "mined_candidate_count": record["candidate_count"],
            }
        )
    if by_unit:
        raise ChapterStop("candidate annotation is not bound to an included source unit")
    gaps: list[dict[str, Any]] = []
    for offset, skipped in enumerate(scoped.skipped):
        gaps.append(
            {
                "source_id": "gap-"
                + content_fingerprint(
                    {
                        "parent_plan": scoped.parent_plan_fingerprint,
                        "skip_position": offset,
                        "metadata": skipped,
                    }
                ),
                "kind": "UNRECOVERED_OR_EMPTY_SOURCE_MARKER",
                **skipped,
                "text": "",
                "candidate_ids": [],
                "mining_status": "SKIPPED_UNRESOLVED",
            }
        )
    # Use the actual faithful block traversal, not a guessed ordinal for gaps.
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for unit in units:
        key = (unit["section_id"], unit["anchor"], unit["source_path"])
        grouped.setdefault(key, []).append(unit["source_id"])
    skipped_grouped: dict[tuple[str, str, str], list[str]] = {}
    for gap in gaps:
        key = (gap["section_id"], gap["anchor"], gap["source_path"])
        skipped_grouped.setdefault(key, []).append(gap["source_id"])
    reading_order: list[str] = []
    for block in faithful.reading_blocks(book):
        if block.section_id not in scoped.section_ids:
            continue
        key = (block.section_id, block.anchor, block.source_path)
        if grouped.get(key):
            reading_order.extend(grouped.pop(key))
        elif skipped_grouped.get(key):
            reading_order.append(skipped_grouped[key].pop(0))
        else:
            raise ChapterStop("chapter faithful reading order has an unaccounted source block")
    if (
        grouped
        or any(skipped_grouped.values())
        or len(reading_order) != len(units) + len(gaps)
        or len(set(reading_order)) != len(reading_order)
    ):
        raise ChapterStop("chapter section has missing/duplicated faithful reading units")
    return units, gaps, reading_order


def _brief_validate(
    raw: object,
    units: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    plan: ChapterPlan,
) -> dict[str, Any]:
    obj = _shape(raw, {"argument", "beats", "coverage"}, "chapter brief schema")
    _words(obj["argument"], "chapter argument missing")
    source_by_id = {item["source_id"]: item for item in [*units, *gaps]}
    coverage = _items(obj["coverage"], len(source_by_id), "chapter unit coverage budget")
    decisions: dict[str, str] = {}
    for entry in coverage:
        item = _shape(
            entry, {"source_id", "decision", "reason"}, "chapter source unit coverage schema"
        )
        source_id = item["source_id"]
        if (
            source_id not in source_by_id
            or source_id in decisions
            or item["decision"] not in {"keep", "compress", "omit"}
        ):
            raise ChapterStop("chapter source unit coverage identity mismatch")
        if source_id.startswith("gap-") and item["decision"] != "omit":
            raise ChapterStop("unrecovered chapter source cannot be silently retained as text")
        _words(item["reason"], "chapter unit retention/omission needs a reason", 500)
        decisions[source_id] = item["decision"]
    if set(decisions) != set(source_by_id):
        raise ChapterStop("chapter brief silently omitted original source units or gaps")
    beats = _items(obj["beats"], plan.max_beats, "chapter beat limit")
    if len(beats) < 2:
        raise ChapterStop("chapter requires multiple source-bound beats")
    unit_index = {unit["source_id"]: unit for unit in units}
    included: set[str] = set()
    seen_spans: set[tuple[str, int, int]] = set()
    for index, entry in enumerate(beats, 1):
        beat = _shape(
            entry,
            {
                "beat_id",
                "unit_id",
                "span_start",
                "span_end",
                "quote",
                "candidate_id",
                "meaning",
                "qualification",
                "essential",
                "source_role",
            },
            "chapter source-span beat schema",
        )
        unit_id = beat["unit_id"]
        unit = unit_index.get(unit_id)
        if (
            beat["beat_id"] != f"beat-{index:02d}"
            or unit is None
            or decisions.get(unit_id) not in {"keep", "compress"}
        ):
            raise ChapterStop("chapter beat does not refer to a retained original source unit")
        start, stop = beat["span_start"], beat["span_end"]
        if (
            type(start) is not int
            or type(stop) is not int
            or not 0 <= start < stop <= len(unit["text"])
            or not 20 <= stop - start <= 700
            or beat["quote"] != unit["text"][start:stop]
            or (unit_id, start, stop) in seen_spans
        ):
            raise ChapterStop("chapter beat quote is not an exact canonical source span")
        candidate_id = beat["candidate_id"]
        if candidate_id is not None and candidate_id not in unit["candidate_ids"]:
            raise ChapterStop("chapter beat candidate annotation not bound to original unit")
        if type(beat["essential"]) is not bool or beat["source_role"] not in {
            "BOOK_ASSERTION",
            "AUTHOR_EXAMPLE",
        }:
            raise ChapterStop("chapter beat source role or essential flag invalid")
        _words(beat["meaning"], "chapter beat meaning missing", 500)
        if not isinstance(beat["qualification"], str) or len(beat["qualification"]) > 500:
            raise ChapterStop("chapter beat qualification malformed")
        included.add(unit_id)
        seen_spans.add((unit_id, start, stop))
    if included != {sid for sid, decision in decisions.items() if decision != "omit"}:
        raise ChapterStop("chapter retained a unit without an explicit source-span beat")
    return obj


def _review_validate(
    raw: object,
    brief: dict[str, Any],
    brief_identity: str,
    units: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    doc = _shape(
        raw,
        {"schema", "brief_sha256", "reviewer", "decisions", "unit_decisions"},
        "chapter human review schema",
    )
    if doc["schema"] != REVIEW_SCHEMA or doc["brief_sha256"] != brief_identity:
        raise ChapterStop("chapter human review does not match generated brief")
    _words(doc["reviewer"], "chapter reviewer identifier required", 80)
    sources = {entry["source_id"]: entry for entry in [*units, *gaps]}
    covered = {entry["source_id"]: entry["decision"] for entry in brief["coverage"]}
    checked: set[str] = set()
    for entry in _items(doc["unit_decisions"], len(covered), "human source coverage limit"):
        decision = _shape(
            entry,
            {"source_id", "review", "decision", "reason", "qualifier_quote"},
            "human original-source unit review schema",
        )
        sid = decision["source_id"]
        if (
            sid not in covered
            or sid in checked
            or decision["decision"] != covered[sid]
            or decision["review"] not in {"accept_coverage", "reject_coverage"}
        ):
            raise ChapterStop("human source coverage decision identity mismatch")
        _words(decision["reason"], "human source coverage explanation required", 600)
        quote = decision["qualifier_quote"]
        if (
            not isinstance(quote, str)
            or len(quote) > 400
            or (quote and quote not in sources[sid]["text"])
        ):
            raise ChapterStop("human source coverage qualifier is not original text")
        if decision["review"] != "accept_coverage":
            raise ChapterStop("human rejected chapter unit coverage; revised brief required")
        checked.add(sid)
    if checked != set(covered):
        raise ChapterStop("human review did not assess all original and skipped source units")
    beats = {entry["beat_id"]: entry for entry in brief["beats"]}
    decisions: dict[str, dict[str, Any]] = {}
    for item in _items(doc["decisions"], len(beats), "chapter beat review size mismatch"):
        choice = _shape(
            item,
            {"beat_id", "decision", "reason", "qualifier_quote"},
            "chapter human beat review schema",
        )
        bid = choice["beat_id"]
        if (
            bid not in beats
            or bid in decisions
            or choice["decision"] not in {"support_attested", "reject"}
        ):
            raise ChapterStop("chapter human review has duplicate or unknown beat decisions")
        _words(choice["reason"], "human beat support reason required", 600)
        qualifier = choice["qualifier_quote"]
        context = sources[beats[bid]["unit_id"]]["text"]
        if (
            not isinstance(qualifier, str)
            or len(qualifier) > 400
            or (qualifier and qualifier not in context)
        ):
            raise ChapterStop("human beat qualifier must quote original source unit")
        decisions[bid] = choice
    if set(decisions) != set(beats):
        raise ChapterStop("chapter requires a human decision for every sourced beat")
    return doc, decisions


def _adapt_validate(
    raw: object, brief: dict[str, Any], decisions: dict[str, dict[str, Any]], plan: ChapterPlan
) -> dict[str, Any]:
    obj = _shape(
        raw,
        {
            "throughline",
            "narrative_form",
            "choice_reason",
            "beats",
            "characters",
            "locations",
            "props",
        },
        "chapter adaptation schema",
    )
    for key in ("throughline", "choice_reason"):
        _words(obj[key], "chapter throughline/creative rationale missing")
    if obj["narrative_form"] not in {
        "DRAMATIZED_ILLUSTRATION",
        "DOCUMENTARY_OBSERVATION",
        "CONCEPTUAL_SEQUENCE",
    }:
        raise ChapterStop("chapter narrative form must be explicit")
    source_beats = {x["beat_id"]: x for x in brief["beats"]}
    coverage: dict[str, str] = {}
    for entry in _items(obj["beats"], plan.max_beats, "chapter adaptation beat limit"):
        beat = _shape(
            entry,
            {"beat_id", "decision", "reason", "qualification", "illustration"},
            "chapter adaptation coverage schema",
        )
        bid = beat["beat_id"]
        if (
            bid not in source_beats
            or bid in coverage
            or beat["decision"] not in {"keep", "compress", "omit"}
        ):
            raise ChapterStop("chapter adaptation has unknown/duplicate beat")
        _words(beat["reason"], "chapter adaptation needs a retention/omission reason", 500)
        if beat["illustration"] not in {"BOOK_GROUNDED", "INVENTED_ILLUSTRATION"}:
            raise ChapterStop("chapter adaptation fiction label missing")
        qualifier = decisions[bid]["qualifier_quote"]
        original_qualification = source_beats[bid]["qualification"]
        if (
            not isinstance(beat["qualification"], str)
            or (qualifier and qualifier not in beat["qualification"])
            or (original_qualification and original_qualification not in beat["qualification"])
            or len(beat["qualification"]) > 600
        ):
            raise ChapterStop("chapter adaptation lost reviewer-attested qualification")
        if decisions[bid]["decision"] == "reject" and beat["decision"] != "omit":
            raise ChapterStop("human-rejected chapter beat cannot become a scene")
        if source_beats[bid]["essential"] and (
            decisions[bid]["decision"] == "support_attested" and beat["decision"] == "omit"
        ):
            raise ChapterStop("essential source beat cannot be omitted without new review")
        coverage[bid] = beat["decision"]
    if (
        set(coverage) != set(source_beats)
        or sum(decision in {"keep", "compress"} for decision in coverage.values()) < 2
    ):
        raise ChapterStop("chapter adaptation silently omitted its source throughline")
    identities: set[str] = set()
    for key, ceiling in (("characters", 12), ("locations", 12), ("props", 16)):
        entries = _items(obj[key], ceiling, "chapter continuity registry size exceeded")
        for entry in entries:
            person = _shape(entry, {"id", "name", "initial_state"}, "chapter entity schema")
            entity_id = _identity(person["id"], "chapter entity identifier invalid")
            if entity_id in identities:
                raise ChapterStop("chapter entity identities must be globally unique")
            identities.add(entity_id)
            _words(person["name"], "chapter entity name missing", 120)
            _words(person["initial_state"], "chapter entity initial state missing", 220)
    if not obj["locations"] or not (obj["characters"] or obj["props"]):
        raise ChapterStop("chapter direction requires a location and a visible actor/prop")
    return obj


def _script_validate(
    raw: object, adaptation: dict[str, Any], decisions: dict[str, dict[str, Any]], plan: ChapterPlan
) -> dict[str, Any]:
    obj = _shape(raw, {"title", "scenes"}, "chapter script schema")
    _words(obj["title"], "chapter title missing", 140)
    scenes = _items(obj["scenes"], plan.max_scenes, "chapter scene budget")
    if len(scenes) < 2:
        raise ChapterStop("chapter needs a connected multi-scene argument")
    allowed = {x["beat_id"] for x in adaptation["beats"] if x["decision"] in {"keep", "compress"}}
    characters = {x["id"] for x in adaptation["characters"]}
    locations = {x["id"] for x in adaptation["locations"]}
    observed: set[str] = set()
    words = 0
    for i, entry in enumerate(scenes):
        scene = _shape(
            entry,
            {
                "scene_id",
                "previous_id",
                "next_id",
                "purpose",
                "action",
                "environment_id",
                "narration",
                "dialogue",
                "beat_ids",
                "illustration_label",
                "transition",
                "consequence",
            },
            "chapter scene schema",
        )
        if (
            scene["scene_id"] != f"scene-{i + 1:02d}"
            or scene["previous_id"] != (f"scene-{i:02d}" if i else None)
            or scene["next_id"] != (f"scene-{i + 2:02d}" if i + 1 < len(scenes) else None)
            or scene["environment_id"] not in locations
        ):
            raise ChapterStop("chapter scene graph has broken links or locations")
        for key in ("purpose", "action", "narration", "transition", "consequence"):
            _words(scene[key], "chapter scene needs causal narrative content", 3200)
        if scene["illustration_label"] not in {"SOURCE_ACCOUNT", "INVENTED_ILLUSTRATION"}:
            raise ChapterStop("chapter illustration factual/fiction label missing")
        beat_ids = _items(scene["beat_ids"], len(allowed), "chapter scene source coverage")
        if (
            not beat_ids
            or len(set(beat_ids)) != len(beat_ids)
            or any(bid not in allowed for bid in beat_ids)
        ):
            raise ChapterStop("chapter scene cites omitted/unknown source beats")
        observed.update(beat_ids)
        for bid in beat_ids:
            qualifier = decisions[bid]["qualifier_quote"]
            if qualifier and qualifier not in scene["narration"]:
                raise ChapterStop("chapter script lost exact human-attested qualification")
        dialogue = _items(scene["dialogue"], 2, "chapter dialogue budget")
        if dialogue and scene["illustration_label"] != "INVENTED_ILLUSTRATION":
            raise ChapterStop("invented chapter dialogue cannot be framed as author fact")
        for spoken in dialogue:
            utterance = _shape(
                spoken, {"speaker_id", "text", "fiction_label"}, "chapter dialogue schema"
            )
            if utterance["speaker_id"] not in characters or (
                utterance["fiction_label"] != "INVENTED_DIALOGUE"
            ):
                raise ChapterStop("chapter dialogue has unknown voice/fiction authority")
            _words(utterance["text"], "chapter dialogue must be substantive", 800)
        words += len(scene["narration"].split())
        words += sum(len(x["text"].split()) for x in dialogue)
    if observed != allowed or words < 130 or words > plan.max_words:
        raise ChapterStop("chapter script misses source beats or violates chapter word budget")
    return obj


def _timings(
    script: dict[str, Any], plan: ChapterPlan, audio: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    start = 0.0
    for scene in script["scenes"]:
        if audio is None:
            word_count = len(scene["narration"].split()) + sum(
                len(v["text"].split()) for v in scene["dialogue"]
            )
            duration = max(1.0, word_count * 60.0 / 140.0)
        else:
            duration = sum(
                x["duration_seconds"] for x in audio if x["scene_id"] == scene["scene_id"]
            )
            if duration <= 0:
                raise ChapterStop("chapter audio track missing an utterance duration")
        end = round(start + duration, 4)
        output.append(
            {
                "scene_id": scene["scene_id"],
                "start_seconds": round(start, 4),
                "end_seconds": end,
                "duration_seconds": round(duration, 4),
                "basis": "MEASURED_WAV" if audio is not None else "NO_AUDIO_ESTIMATE",
            }
        )
        start = end
    if start > plan.max_runtime_seconds:
        raise ChapterStop("chapter exceeds finite narrative runtime allowance")
    return output


def _direction_validate(
    raw: object,
    script: dict[str, Any],
    adaptation: dict[str, Any],
    timings: list[dict[str, Any]],
    plan: ChapterPlan,
) -> dict[str, Any]:
    obj = _shape(raw, {"shots"}, "chapter shot schema")
    shots = _items(
        obj["shots"], plan.max_scenes * plan.max_shots_per_scene, "chapter shot budget exceeded"
    )
    characters = {x["id"]: x["initial_state"] for x in adaptation["characters"]}
    props = {x["id"]: x["initial_state"] for x in adaptation["props"]}
    locations = {x["id"] for x in adaptation["locations"]}
    current = {**characters, **props}
    if not shots:
        raise ChapterStop("chapter direction returned no shots")
    index = 0
    previous_end = 0.0
    for i, scene in enumerate(script["scenes"]):
        scene_timing = timings[i]
        scene_shots = [
            shot
            for shot in shots
            if isinstance(shot, dict) and shot.get("scene_id") == scene["scene_id"]
        ]
        if not 1 <= len(scene_shots) <= plan.max_shots_per_scene:
            raise ChapterStop("chapter scene missing bounded visual action shots")
        changed = False
        for shot in scene_shots:
            index += 1
            shot = _shape(
                shot,
                {
                    "shot_id",
                    "scene_id",
                    "purpose",
                    "action",
                    "environment_id",
                    "camera",
                    "transition",
                    "entity_ids",
                    "state_before",
                    "state_after",
                    "start_seconds",
                    "end_seconds",
                },
                "chapter shot fields malformed",
            )
            if (
                shot["shot_id"] != f"shot-{index:03d}"
                or shot["scene_id"] != scene["scene_id"]
                or shot["environment_id"] not in locations
                or shot["environment_id"] != scene["environment_id"]
            ):
                raise ChapterStop("chapter shot has orphan ID or unknown location")
            for key in ("purpose", "action", "camera", "transition"):
                _words(shot[key], "chapter shot lacks concrete action/camera", 400)
            entities = _items(shot["entity_ids"], 16, "chapter shot entity budget")
            if (
                not entities
                or len(set(entities)) != len(entities)
                or any(entity not in current for entity in entities)
            ):
                raise ChapterStop("chapter shot refers to unknown or duplicated entity")
            before = shot["state_before"]
            after = shot["state_after"]
            if (
                not isinstance(before, dict)
                or not isinstance(after, dict)
                or (set(before) != set(entities) or set(after) != set(entities))
            ):
                raise ChapterStop("chapter continuity state ledger keys mismatch")
            for entity in entities:
                if before[entity] != current[entity]:
                    raise ChapterStop("chapter shot continuity prior state impossible")
                _words(after[entity], "chapter shot post-action state missing", 220)
                changed |= after[entity] != current[entity]
                current[entity] = after[entity]
            start, end = shot["start_seconds"], shot["end_seconds"]
            if (
                type(start) not in (int, float)
                or type(end) not in (int, float)
                or abs(start - previous_end) > 0.02
                or not start < end
                or end > scene_timing["end_seconds"] + 0.02
                or start < scene_timing["start_seconds"] - 0.02
            ):
                raise ChapterStop("chapter shot timing gap, overlap or scene boundary error")
            previous_end = end
        if not changed or abs(previous_end - scene_timing["end_seconds"]) > 0.02:
            raise ChapterStop("chapter scene has no state-changing action or incomplete timing")
    if index != len(shots) or abs(previous_end - timings[-1]["end_seconds"]) > 0.02:
        raise ChapterStop("chapter direction includes orphan shots or incomplete timeline")
    canonical = [
        shot
        for scene in script["scenes"]
        for shot in shots
        if shot["scene_id"] == scene["scene_id"]
    ]
    if any(a is not b for a, b in zip(shots, canonical, strict=True)):
        raise ChapterStop("chapter shot output is not in canonical source scene order")
    return obj


def _request(stage: str, payload: dict[str, Any], plan: ChapterPlan) -> dict[str, object]:
    descriptions = {
        "brief": "Produce a source-bound argument with candidate coverage and grounded beats. "
        "Keep uncertainty; OMIT means explicit loss, not approval.",
        "adaptation": "Create a connected causal adaptation, preserving essential beats "
        "and qualifiers. Label invented illustration as fiction, never author fact.",
        "chapter_script": "Write ordered linked action scenes with substantive narration. "
        "Do NOT use short-form Episode, presenter slides or sentence compaction.",
        "chapter_direction": "Generate contiguous timed shots and explicit entity state "
        "transitions that enact the scene mechanisms; no video assets.",
    }
    contract_shapes: dict[str, object] = {
        "brief": {
            "argument": "one narrow defensible chapter argument",
            "beats": [
                {
                    "beat_id": "beat-01",
                    "unit_id": "<exact canonical unit-... source_id>",
                    "span_start": 0,
                    "span_end": 70,
                    "quote": "<exact original unit text substring>",
                    "candidate_id": "<optional mined candidate ID or null>",
                    "meaning": "<narrow source-grounded argument/example/contrast>",
                    "qualification": "<limitations or empty if no stated qualifier>",
                    "essential": True,
                    "source_role": "BOOK_ASSERTION or AUTHOR_EXAMPLE",
                }
            ],
            "coverage": [
                {
                    "source_id": "<every scoped original source unit or gap ID>",
                    "decision": "keep or compress or omit",
                    "reason": "<source-grounded reason, including zero-candidate units>",
                }
            ],
        },
        "adaptation": {
            "throughline": "<one causal story through all supported beats>",
            "narrative_form": "DRAMATIZED_ILLUSTRATION or DOCUMENTARY_OBSERVATION "
            "or CONCEPTUAL_SEQUENCE",
            "choice_reason": "<why this form honestly serves the chapter>",
            "beats": [
                {
                    "beat_id": "<each source beat>",
                    "decision": "keep or compress or omit",
                    "reason": "<specific omission/retention reason>",
                    "qualification": "<full attested qualifier, if present>",
                    "illustration": "BOOK_GROUNDED or INVENTED_ILLUSTRATION",
                }
            ],
            "characters": [
                {"id": "worker-one", "name": "Worker", "initial_state": "standing at the bench"}
            ],
            "locations": [
                {"id": "workshop", "name": "Workshop", "initial_state": "well-lit workbench"}
            ],
            "props": [{"id": "checklist", "name": "Paper checklist", "initial_state": "unmarked"}],
        },
        "chapter_script": {
            "title": "<chapter film title>",
            "scenes": [
                {
                    "scene_id": "scene-01",
                    "previous_id": None,
                    "next_id": "scene-02",
                    "purpose": "<specific dramatic purpose>",
                    "action": "<visible physical action with consequence>",
                    "environment_id": "<registered location id>",
                    "narration": "<complete original narration, not a short clip>",
                    "dialogue": [
                        {
                            "speaker_id": "<registered character ID>",
                            "text": "<invented dialogue>",
                            "fiction_label": "INVENTED_DIALOGUE",
                        }
                    ],
                    "beat_ids": ["beat-01"],
                    "illustration_label": "INVENTED_ILLUSTRATION",
                    "transition": "<causal link from previous/to next>",
                    "consequence": "<what changed due to this action>",
                }
            ],
        },
        "chapter_direction": {
            "shots": [
                {
                    "shot_id": "shot-001",
                    "scene_id": "scene-01",
                    "purpose": "<how this shot explains the mechanism>",
                    "action": "<concrete visible action>",
                    "environment_id": "<registered scene location>",
                    "camera": "<framing and motion>",
                    "transition": "<reason for transition>",
                    "entity_ids": ["checklist"],
                    "state_before": {"checklist": "unmarked"},
                    "state_after": {"checklist": "marked with missing tool"},
                    "start_seconds": 0.0,
                    "end_seconds": 6.0,
                }
            ],
        },
    }
    # Every stage must fit the complete prompt plus its full output reservation.
    # UTF-8 bytes upper-bound byte-level BPE token counts; no tokenizer/model
    # initialization, network or truncation occurs during this conservative test.
    system_message = (
        descriptions[stage]
        + " Return one strict JSON object only; obey the response_contract schema. "
        "Examples describe structure, not book facts to copy. "
        + (
            "Cover every original source unit and gap with explicit reasons; "
            "tie each sourced beat to exact original unit offsets and quote bytes. "
            if stage == "brief"
            else ""
        )
        + "Never convert invented illustrations into author history. "
        "Human semantic review and permission checks remain mandatory."
    )
    user_message = json.dumps(
        {"input": payload, "response_contract": contract_shapes[stage]},
        ensure_ascii=False,
        sort_keys=True,
    )
    bound = (
        len(system_message.encode("utf-8")) + len(user_message.encode("utf-8")) + CHAT_TOKEN_RESERVE
    )
    if bound + plan.max_new_tokens > MAX_CONTEXT_TOKENS:
        raise ChapterStop(
            "complete chapter source+context+reserved output exceeds conservative "
            "pinned model context; freeze a narrower exact canonical section"
        )
    return {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": plan.max_new_tokens,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    }



def _claim(root: Path, marker: Path, request_key: str, attempt: int) -> str:
    """Exclusive local-filesystem reservation; a partial write stays unresolved.

    O_EXCL chooses a single process/thread on a filesystem honoring exclusive
    creation. This is not a distributed lease or an execution timeout.
    """
    acquisition._private_directory(marker.parent, private_root=root)
    owner = uuid4().hex
    record = evaluation._json_bytes(
        {
            "request_key": request_key,
            "attempt": attempt,
            "owner": owner,
            "status": "STARTED_UNRESOLVED",
        }
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError:
        raise ChapterStop("chapter attempt has prior unresolved reservation") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(record)
        handle.flush()
        os.fsync(handle.fileno())
    return owner


def _verified_owner(root: Path, marker: Path, key: str, attempt: int, owner: str) -> None:
    try:
        started = whole_book.load_strict_reply(evaluation._read(marker, 2048))
    except (ValueError, TypeError, OSError):
        raise ChapterStop("chapter cached attempt has ambiguous reservation") from None
    if started != {
        "request_key": key,
        "attempt": attempt,
        "owner": owner,
        "status": "STARTED_UNRESOLVED",
    }:
        raise ChapterStop("chapter cached attempt owner mismatch")


def _stage(
    root: Path,
    stage: str,
    payload: dict[str, Any],
    plan: ChapterPlan,
    validate: Callable[[object], dict[str, Any]],
    completion: Callable[[dict[str, object]], str] | None,
) -> tuple[dict[str, Any], str, str, int]:
    """Reserve each irreversible provider attempt before calling any producer."""
    if completion is None:
        raise ChapterStop("actual chapter CPU route blocked pending a verified bounded private host")
    request = _request(stage, payload, plan)
    provenance = "INJECTED_TEST_PRODUCER"
    stage_key = content_fingerprint(
        {
            "contract": stage,
            "stage_version": {
                "brief": BRIEF_SCHEMA,
                "adaptation": ADAPT_SCHEMA,
                "chapter_script": SCRIPT_SCHEMA,
                "chapter_direction": DIRECTION_SCHEMA,
            }[stage],
            "request": request,
            "stage_limits": {
                "max_scenes": plan.max_scenes,
                "max_words": plan.max_words,
                "max_beats": plan.max_beats,
                "max_shots_per_scene": plan.max_shots_per_scene,
                "max_attempts_per_stage": plan.max_attempts_per_stage,
                "max_runtime_seconds": plan.max_runtime_seconds,
                "max_new_tokens": plan.max_new_tokens,
                "voice": plan.voice,
            },
            "provenance": provenance,
        }
    )
    directory = root / stage / stage_key
    complete = directory / "validated.json"
    if complete.is_file():
        saved = _shape(
            whole_book.load_strict_reply(evaluation._read(complete)),
            {"request_key", "provenance", "contract", "payload", "owner", "attempt"},
            "chapter stage cache schema/owner mismatch; legacy execution requires reconciliation",
        )
        if (
            saved["request_key"] != stage_key
            or saved["provenance"] != provenance
            or saved["contract"] != stage
            or not isinstance(saved["owner"], str)
            or type(saved["attempt"]) is not int
            or not 1 <= saved["attempt"] <= plan.max_attempts_per_stage
        ):
            raise ChapterStop("chapter stage cache identity changed")
        _verified_owner(
            root,
            directory / f"started-{saved['attempt']}.json",
            stage_key,
            saved["attempt"],
            saved["owner"],
        )
        return validate(saved["payload"]), stage_key, provenance, 0
    if directory.is_dir() and any(directory.glob("started-*.json")):
        raise ChapterStop("chapter stage has prior unresolved/failed attempts")
    for index in range(1, plan.max_attempts_per_stage + 1):
        marker = directory / f"started-{index}.json"
        owner = _claim(root, marker, stage_key, index)
        try:
            response = evaluation._capture_private_generator(
                root, directory, lambda: completion(request)
            )
        except Exception:
            evaluation._store(
                root, directory / f"failed-{index}.json",
                {"attempt": index, "owner": owner, "status": "PRODUCER_FAILURE"},
            )
            raise ChapterStop("private chapter producer failed; no automatic retry") from None
        if not isinstance(response, str) or len(response.encode("utf-8")) > MAX_RAW_BYTES:
            evaluation._store(
                root, directory / f"failed-{index}.json",
                {"attempt": index, "owner": owner, "status": "NON_TEXT_OR_OVERSIZE"},
            )
            raise ChapterStop("chapter producer returned invalid bounded raw output")
        evaluation._store(
            root, directory / f"raw-{index}.json",
            {
                "attempt": index,
                "owner": owner,
                "raw": response,
                "sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            },
        )
        try:
            result = validate(whole_book.load_strict_reply(response))
        except (ValueError, TypeError, KeyError, RecursionError):
            evaluation._store(
                root, directory / f"failed-{index}.json",
                {"attempt": index, "owner": owner, "status": "SCHEMA_OR_SOURCE_REJECTED"},
            )
            if index == plan.max_attempts_per_stage:
                raise ChapterStop("chapter stage exhausted all recorded attempts") from None
            continue
        evaluation._store(
            root,
            complete,
            {
                "request_key": stage_key,
                "provenance": provenance,
                "contract": stage,
                "owner": owner,
                "attempt": index,
                "payload": result,
            },
        )
        return result, stage_key, provenance, index
    raise ChapterStop("chapter stage exhausted bounded attempts")


def _narrate(
    root: Path,
    script: dict[str, Any],
    plan: ChapterPlan,
    producer: Callable[[str, str], bytes] | None,
) -> tuple[list[dict[str, Any]], str, str]:
    """One exclusive audio owner; record reproducible sample-accurate utterance offsets."""
    from .scripting import NicheConfig, Scene

    if producer is None:
        raise ChapterStop("actual chapter Kokoro route blocked pending a verified bounded private host")
    config = NicheConfig(
        niche="private_chapter_preproduction",
        audience="private",
        brand="CLIPPER",
        voice=plan.voice,
        facts=(),
        topic="private chapter",
        format="general",
    )
    provenance = "INJECTED_TEST_WAV"
    audio_root = root / f"chapter-audio-{provenance.lower()}"
    records: list[dict[str, Any]] = []
    position = 0
    for scene in script["scenes"]:
        utterances = [("narration", scene["narration"], "narrator")]
        utterances += [
            ("invented_dialogue", line["text"], line["speaker_id"]) for line in scene["dialogue"]
        ]
        for index, (kind, text, speaker_id) in enumerate(utterances):
            item = Scene(headline=scene["scene_id"], narration=text)
            path = narration.tts_cache_path(item, config, audio_root)
            request = narration.tts_request(item, config)
            marker_key = content_fingerprint({"request": request, "provenance": provenance})
            marker = audio_root / "attempts" / f"{marker_key}.json"
            complete = audio_root / "attempts" / f"completed-{marker_key}.json"
            if complete.is_file():
                try:
                    saved = _shape(
                        whole_book.load_strict_reply(evaluation._read(complete, 2048)),
                        {"request_key", "owner", "attempt", "status", "sha256", "bytes"},
                        "chapter audio completion ownership schema",
                    )
                except (ValueError, OSError, TypeError):
                    raise ChapterStop("chapter audio completion record invalid") from None
                if (
                    not path.is_file()
                    or saved["request_key"] != marker_key
                    or saved["attempt"] != 1
                    or saved["status"] != "VALIDATED_WAV"
                    or not isinstance(saved["owner"], str)
                ):
                    raise ChapterStop("chapter audio has incomplete or conflicting publication")
                _verified_owner(root, marker, marker_key, 1, saved["owner"])
                data = evaluation._read(path, narration.MAX_WAV_BYTES)
                if len(data) != saved["bytes"] or hashlib.sha256(data).hexdigest() != saved["sha256"]:
                    raise ChapterStop("chapter audio cache changed after owner publication")
                channels, sample_rate, frames = narration.validated_wav(data, cached=True)
            else:
                if marker.exists() or path.exists():
                    raise ChapterStop("prior Kokoro utterance is unresolved; no retry")
                _claim(root, marker, marker_key, 1)
                try:

                    def make_utterance(utterance: Scene = item) -> Path:
                        return narration.synthesize_scene(
                            utterance, config, audio_root, producer=producer
                        )

                    wave_path = evaluation._capture_private_generator(root, audio_root, make_utterance)
                    data = evaluation._read(wave_path, narration.MAX_WAV_BYTES)
                    channels, sample_rate, frames = narration.validated_wav(data, cached=True)
                    owner = whole_book.load_strict_reply(evaluation._read(marker, 2048))["owner"]
                    evaluation._store(
                        root,
                        complete,
                        {
                            "request_key": marker_key,
                            "attempt": 1,
                            "owner": owner,
                            "status": "VALIDATED_WAV",
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data),
                        },
                    )
                except Exception:
                    raise ChapterStop("chapter narration failed; private attempt retained") from None
            start_sample = position
            position += frames
            records.append(
                {
                    "scene_id": scene["scene_id"],
                    "utterance": index,
                    "kind": kind,
                    "speaker_id": speaker_id,
                    "tts_request_sha256": content_fingerprint(request),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "wav_relative_path": str(path.relative_to(root)),
                    "voice": plan.voice,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "samples": frames,
                    "start_sample": start_sample,
                    "end_sample": position,
                    "start_seconds": round(start_sample / sample_rate, 6),
                    "end_seconds": round(position / sample_rate, 6),
                    "duration_seconds": round(frames / sample_rate, 6),
                    "provenance": provenance,
                }
            )
    report = {
        "schema": "clipper-chapter-narration-v2",
        "records": records,
        "voice_model": scripting.VOICE_MODEL,
        "voice_revision_label": scripting.VOICE_REVISION,
        "resolved_model_commit": None,
        "resolved_commit_verified": False,
        "provenance": provenance,
        "publish_eligible": False,
    }
    report_key = content_fingerprint(report)
    evaluation._store(root, root / f"narration-report-{report_key}.json", report)
    return records, provenance, report_key


def evaluate_chapter(
    runner: BookOrchestrator,
    ref_path: Path,
    progress_path: Path,
    section_manifest: Path,
    *,
    review_file: Path | None = None,
    plan: ChapterPlan | None = None,
    brief_completion: Callable[[dict[str, object]], str] | None = None,
    adaptation_completion: Callable[[dict[str, object]], str] | None = None,
    script_completion: Callable[[dict[str, object]], str] | None = None,
    direction_completion: Callable[[dict[str, object]], str] | None = None,
    narration_producer: Callable[[str, str], bytes] | None = None,
) -> dict[str, object]:
    """Finite, source-bound chapter preproduction; no legacy catalog, short or render."""
    limits = plan or ChapterPlan()
    scoped, manifest_sha = evaluation._section_scope(runner, ref_path, section_manifest)
    ledger, candidates, book = evaluation._mined_export(
        runner,
        ref_path,
        progress_path,
        section_plan=scoped,
        section_manifest_sha256=manifest_sha,
    )
    if len(scoped.units) + len(scoped.skipped) > limits.max_beats:
        raise ChapterStop(
            "full chapter scope has more original units/gaps than finite unit coverage "
            "budget; freeze a narrower canonical section, do not omit source text"
        )
    units, gaps, reading_order = _full_source_scope(book, scoped, ledger, candidates)
    root = runner.root / "book-candidates"
    chapter_key = content_fingerprint(
        {
            "contract": PLAN_SCHEMA,
            "plan_fingerprint": scoped.fingerprint,
            "progress_sha256": evaluation._sha(evaluation._read(progress_path, 20_000_000)),
            "section_manifest_sha256": manifest_sha,
            "run_plan": limits.fingerprint,
        }
    )
    base = root / "chapter-evaluations" / chapter_key
    candidate_to_unit = {
        candidate_id: unit["source_id"] for unit in units for candidate_id in unit["candidate_ids"]
    }
    evidence = [
        {
            "candidate_id": item["id"],
            "unit_id": candidate_to_unit[item["id"]],
            "proposed_id": item["proposed_id"],
            "claim": item["claim"],
            "quote": item["evidence_anchor"]["quote"],
            "anchor": item["evidence_anchor"]["anchor"],
            "section_id": item["_evaluation_context"]["section_id"],
            "source_path": item["_evaluation_context"]["source_path"],
            "unit_word_offset": item["_evaluation_context"]["unit_word_offset"],
            "source_context": item["_evaluation_context"]["source_context"],
            "evidence_field": item["_evaluation_context"]["evidence_field"],
            "review_status": item["review_status"],
        }
        for item in candidates
    ]
    source = {
        "schema": "clipper-chapter-source-ledger-v2",
        "parent_book_fingerprint": book.fingerprint,
        "source_fingerprint": book.source_fingerprint,
        "source_sha256": book.content_sha256,
        "edition_identity": book.edition_identity,
        "scope": scoped.section_scope,
        "section_manifest_sha256": manifest_sha,
        "recovery_status": book.recovery_status,
        "authorization_status": book.source.authorization_status,
        "warnings": list(book.warnings),
        "excluded_source_pages": [
            {"page_number": x.page_number, "reason": x.reason} for x in book.excluded_pages
        ],
        "scope_skipped": list(scoped.skipped),
        "zero_candidate_units": ledger["counts"]["zero_candidate"],
        "progress_sha256": evaluation._sha(evaluation._read(progress_path, 20_000_000)),
        "source_review": "MINED_UNREVIEWED",
        "parent_full_book_complete": False,
        "publish_eligible": False,
        "units": units,
        "gaps": gaps,
        "reading_order": reading_order,
        "evidence": evidence,
    }
    if len(evaluation._json_bytes(source)) > MAX_SOURCE_BYTES:
        raise ChapterStop(
            "complete original chapter source exceeds private source preflight budget; "
            "freeze a smaller canonical section, never truncate to candidate quotes"
        )
    evaluation._store(root, base / "source-ledger.json", source)
    summary: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "status": "AWAITING_EXPLICIT_CPU_CHAPTER_BRIEF",
        "section_scope_sha256": manifest_sha,
        "parent_full_book_complete": False,
        "included_units": len(scoped.units),
        "excluded_units": scoped.excluded_units,
        "candidate_count": len(candidates),
        "source_review": "MINED_UNREVIEWED",
        "recovery_status": book.recovery_status,
        "authorization_status": book.source.authorization_status,
        "publish_eligible": False,
        "visual_calls": 0,
        "max_text_attempts": limits.max_attempts_per_stage * len(TEXT_STAGES),
        "max_kokoro_utterances": limits.max_scenes * 3 if limits.kokoro_cpu else 0,
        "actual_audio": False,
        "automation_claim": "NOT_YET_GENERATED",
    }
    if not limits.brief_cpu:
        return summary
    brief, brief_key, brief_origin, brief_attempts = _stage(
        root,
        "brief",
        {"schema": BRIEF_SCHEMA, "source": source},
        limits,
        lambda x: _brief_validate(x, units, gaps, limits),
        brief_completion,
    )
    summary.update(
        {
            "brief_identity": brief_key,
            "brief_provenance": brief_origin,
            "brief_attempts": brief_attempts,
            "status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
            "automation_claim": "INJECTED_TEST" if brief_completion else "ACTUAL_CPU",
        }
    )
    if review_file is None:
        return summary
    review_bytes = evaluation._read(review_file)
    reviewer, decisions = _review_validate(
        whole_book.load_strict_reply(review_bytes), brief, brief_key, units, gaps
    )
    supported = {
        bid for bid, decision in decisions.items() if decision["decision"] == "support_attested"
    }
    review_sha = hashlib.sha256(review_bytes).hexdigest()
    evaluation._store(
        root,
        base / f"human-review-{review_sha}.json",
        {
            "schema": REVIEW_SCHEMA,
            "review": reviewer,
            "semantic_intervention": "HUMAN_ATTESTED_NOT_INDEPENDENTLY_PROVEN",
            "review_sha256": review_sha,
            "publish_eligible": False,
        },
    )
    summary["human_review_sha256"] = review_sha
    summary["human_attested_beat_count"] = len(supported)
    if len(supported) < 2:
        return {**summary, "status": "ABSTAINED_INSUFFICIENT_ATTESTED_CHAPTER_BEATS"}
    if not limits.adaptation_cpu:
        return {**summary, "status": "AWAITING_EXPLICIT_CPU_CHAPTER_ADAPTATION"}
    treatment, adapt_key, adapt_origin, adapt_attempts = _stage(
        root,
        "adaptation",
        {
            "schema": ADAPT_SCHEMA,
            "brief": brief,
            "source": source,
            "human_review": reviewer,
        },
        limits,
        lambda x: _adapt_validate(x, brief, decisions, limits),
        adaptation_completion,
    )
    summary.update({"adaptation_identity": adapt_key, "adaptation_attempts": adapt_attempts})
    if not limits.script_cpu:
        return {**summary, "status": "AWAITING_EXPLICIT_CPU_CHAPTER_SCRIPT"}
    script, script_key, script_origin, script_attempts = _stage(
        root,
        "chapter_script",
        {
            "schema": SCRIPT_SCHEMA,
            "brief": brief,
            "adaptation": treatment,
            "source": source,
            "human_review": reviewer,
        },
        limits,
        lambda x: _script_validate(x, treatment, decisions, limits),
        script_completion,
    )
    summary.update(
        {
            "script_identity": script_key,
            "script_scene_count": len(script["scenes"]),
            "script_words": sum(
                len(s["narration"].split()) + sum(len(v["text"].split()) for v in s["dialogue"])
                for s in script["scenes"]
            ),
            "script_attempts": script_attempts,
        }
    )
    audio: list[dict[str, Any]] | None = None
    audio_origin = "NO_AUDIO"
    narration_report_key: str | None = None
    if limits.kokoro_cpu:
        audio, audio_origin, narration_report_key = _narrate(root, script, limits, narration_producer)
    times = _timings(script, limits, audio)
    summary.update(
        {
            "actual_audio": audio_origin == "ACTUAL_CPU_KOKORO",
            "audio_provenance": audio_origin,
            "narration_report_identity": narration_report_key,
            "duration_seconds": times[-1]["end_seconds"],
            "timing_basis": times[0]["basis"],
        }
    )
    if not limits.direction_cpu:
        return {**summary, "status": "AWAITING_EXPLICIT_CPU_CHAPTER_DIRECTION"}
    shots, direction_key, direction_origin, direction_attempts = _stage(
        root,
        "chapter_direction",
        {
            "schema": DIRECTION_SCHEMA,
            "scenes": script["scenes"],
            "adaptation": treatment,
            "timings": times,
            "source": source,
            "human_review": reviewer,
        },
        limits,
        lambda x: _direction_validate(x, script, treatment, times, limits),
        direction_completion,
    )
    synthetic = (
        any(
            v == "INJECTED_TEST_PRODUCER"
            for v in (brief_origin, adapt_origin, script_origin, direction_origin)
        )
        or audio_origin == "INJECTED_TEST_WAV"
    )
    final_status = (
        "TEST_DOUBLE_CHAPTER_TEXT_READY_UNREVIEWED"
        if synthetic
        else "CHAPTER_PREPRODUCTION_READY_UNREVIEWED"
        if audio is not None
        else "CHAPTER_TEXT_PLAN_READY_NO_AUDIO_UNREVIEWED"
    )
    evaluation._store(
        root,
        base / f"chapter-result-{direction_key}.json",
        {
            "schema": "clipper-chapter-private-preproduction-result-v1",
            "brief_key": brief_key,
            "human_review_sha256": review_sha,
            "adaptation_key": adapt_key,
            "script_key": script_key,
            "direction_key": direction_key,
            "shot_count": len(shots["shots"]),
            "scene_count": len(script["scenes"]),
            "timings": times,
            "audio_provenance": audio_origin,
            "text_provenances": [brief_origin, adapt_origin, script_origin, direction_origin],
            "story_review": "PENDING_INDEPENDENT_REVIEW",
            "visual_identity_enforced": False,
            "visual_calls": 0,
            "source_authenticity": book.source.authorization_status,
            "source_recovery": book.recovery_status,
            "publish_eligible": False,
        },
    )
    return {
        **summary,
        "status": final_status,
        "direction_identity": direction_key,
        "direction_attempts": direction_attempts,
        "shot_count": len(shots["shots"]),
        "automation_claim": "INJECTED_TEST" if synthetic else "ACTUAL_CPU",
        "visual_calls": 0,
    }

"""Private, source-pinned chapter preproduction; never short Episode or visual rendering."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from clipper.stage_contracts import content_fingerprint

from . import evaluation, narration, scripting, whole_book

if TYPE_CHECKING:
    from .orchestrator import BookOrchestrator

PLAN_SCHEMA = "clipper-chapter-preproduction-run-v1"
BRIEF_SCHEMA = "clipper-chapter-brief-v1"
REVIEW_SCHEMA = "clipper-chapter-human-beat-review-v1"
ADAPT_SCHEMA = "clipper-chapter-adaptation-v1"
SCRIPT_SCHEMA = "clipper-chapter-script-graph-v1"
DIRECTION_SCHEMA = "clipper-chapter-continuity-shots-v1"
MAX_RAW_BYTES = 100_000
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


def _brief_validate(
    raw: object, evidence: list[dict[str, Any]], plan: ChapterPlan
) -> dict[str, Any]:
    obj = _shape(raw, {"argument", "beats", "coverage"}, "chapter brief schema")
    _words(obj["argument"], "chapter argument missing")
    source_ids = {x["candidate_id"] for x in evidence}
    coverage = _items(obj["coverage"], plan.max_beats, "chapter coverage budget")
    seen: dict[str, str] = {}
    for value in coverage:
        c = _shape(value, {"candidate_id", "decision", "reason"}, "brief coverage schema")
        cid = c["candidate_id"]
        if cid not in source_ids or cid in seen or c["decision"] not in {"include", "omit"}:
            raise ChapterStop("chapter candidate coverage identity mismatch")
        _words(c["reason"], "chapter omission/retention reason missing", 500)
        seen[cid] = c["decision"]
    if set(seen) != source_ids:
        raise ChapterStop("chapter brief silently dropped source candidates")
    beats = _items(obj["beats"], plan.max_beats, "chapter brief beat budget")
    if len(beats) < 2:
        raise ChapterStop("chapter requires multiple substantive source beats")
    used: set[str] = set()
    for index, entry in enumerate(beats, 1):
        beat = _shape(
            entry,
            {"beat_id", "candidate_id", "meaning", "qualification", "essential", "source_role"},
            "chapter beat schema",
        )
        cid = beat["candidate_id"]
        if beat["beat_id"] != f"beat-{index:02d}" or seen.get(cid) != "include" or cid in used:
            raise ChapterStop("chapter beat is not an included canonical source candidate")
        if type(beat["essential"]) is not bool or beat["source_role"] not in {
            "BOOK_ASSERTION",
            "AUTHOR_EXAMPLE",
        }:
            raise ChapterStop("chapter source role or essential marker invalid")
        _words(beat["meaning"], "chapter beat meaning missing", 500)
        if not isinstance(beat["qualification"], str) or len(beat["qualification"]) > 500:
            raise ChapterStop("chapter qualification malformed")
        used.add(cid)
    if {cid for cid, decision in seen.items() if decision == "include"} != used:
        raise ChapterStop("chapter coverage includes an unrepresented source beat")
    return obj


def _review_validate(
    raw: object, brief: dict[str, Any], brief_identity: str, evidence: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    doc = _shape(raw, {"schema", "brief_sha256", "reviewer", "decisions"}, "chapter review schema")
    if doc["schema"] != REVIEW_SCHEMA or doc["brief_sha256"] != brief_identity:
        raise ChapterStop("chapter human review does not match generated brief")
    _words(doc["reviewer"], "chapter human reviewer identifier required", 80)
    sources = {entry["candidate_id"]: entry for entry in evidence}
    beats = {entry["beat_id"]: entry for entry in brief["beats"]}
    decisions: dict[str, dict[str, Any]] = {}
    for item in _items(doc["decisions"], len(beats), "chapter review size mismatch"):
        choice = _shape(
            item,
            {"beat_id", "decision", "reason", "qualifier_quote"},
            "chapter human review decision schema",
        )
        bid = choice["beat_id"]
        if (
            bid not in beats
            or bid in decisions
            or choice["decision"] not in {"support_attested", "reject"}
        ):
            raise ChapterStop("chapter human review has invalid/duplicate beat decisions")
        _words(choice["reason"], "chapter human reason required", 600)
        qualifier = choice["qualifier_quote"]
        context = sources[beats[bid]["candidate_id"]]["source_context"]
        if (
            not isinstance(qualifier, str)
            or len(qualifier) > 400
            or (qualifier and qualifier not in context)
        ):
            raise ChapterStop("chapter qualifier must quote pinned original context")
        decisions[bid] = choice
    if set(decisions) != set(beats):
        raise ChapterStop("chapter requires a human decision for every beat")
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
    plan: ChapterPlan
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
                    "candidate_id": "<exact evidence candidate_id>",
                    "meaning": "<narrow sourced mechanism or example>",
                    "qualification": "<limitations or empty if no stated qualifier>",
                    "essential": True,
                    "source_role": "BOOK_ASSERTION or AUTHOR_EXAMPLE",
                }
            ],
            "coverage": [
                {
                    "candidate_id": "<every candidate exactly once>",
                    "decision": "include or omit",
                    "reason": "<why substantively retain or omit>",
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
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if len(text) > 52_000:
        raise ChapterStop("private chapter request exceeds bounded text context")
    return {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": plan.max_new_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    descriptions[stage]
                    + " Return one strict JSON object only; obey the field-by-field "
                    "response_contract shown alongside the input. The example response "
                    "contract is structural, NOT content to copy. Scale lists to cover "
                    "all required actual source IDs, scenes, entities and timings. "
                    "Mark invented actions and dialogue explicitly. Do not convert "
                    "unverified source or invented illustrations into historical facts. "
                    "Human semantic review remains mandatory."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"input": payload, "response_contract": contract_shapes[stage]},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
    }


def _stage(
    root: Path,
    stage: str,
    payload: dict[str, Any],
    plan: ChapterPlan,
    validate: Callable[[object], dict[str, Any]],
    completion: Callable[[dict[str, object]], str] | None,
) -> tuple[dict[str, Any], str, str, int]:
    """Raw-preserving, immutable, content-bound stage with a pre-call reservation."""
    request = _request(stage, payload, plan)
    provenance = "INJECTED_TEST_PRODUCER" if completion is not None else "ACTUAL_CPU_QWEN"
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
            {"request_key", "provenance", "contract", "payload"},
            "chapter stage cache schema mismatch",
        )
        if (
            saved["request_key"] != stage_key
            or saved["provenance"] != provenance
            or (saved["contract"] != stage)
        ):
            raise ChapterStop("chapter stage cache identity changed")
        return validate(saved["payload"]), stage_key, provenance, 0
    if directory.is_dir() and any(directory.glob("started-*.json")):
        raise ChapterStop("chapter stage has prior unresolved/failed attempts")
    producer = completion or scripting._local_script_completion
    for index in range(1, plan.max_attempts_per_stage + 1):
        evaluation._store(
            root,
            directory / f"started-{index}.json",
            {
                "request_key": stage_key,
                "attempt": index,
                "status": "STARTED_UNRESOLVED",
            },
        )
        try:
            response = evaluation._capture_private_generator(
                root, directory, lambda: producer(request)
            )
        except Exception:
            evaluation._store(
                root,
                directory / f"failed-{index}.json",
                {
                    "attempt": index,
                    "status": "PRODUCER_FAILURE",
                },
            )
            raise ChapterStop("private chapter producer failed; no automatic retry") from None
        if not isinstance(response, str) or len(response.encode("utf-8")) > MAX_RAW_BYTES:
            evaluation._store(
                root,
                directory / f"failed-{index}.json",
                {
                    "attempt": index,
                    "status": "NON_TEXT_OR_OVERSIZE",
                },
            )
            raise ChapterStop("chapter producer returned invalid bounded raw output")
        evaluation._store(
            root,
            directory / f"raw-{index}.json",
            {
                "attempt": index,
                "raw": response,
                "sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            },
        )
        try:
            result = validate(whole_book.load_strict_reply(response))
        except (ValueError, TypeError, KeyError, RecursionError):
            evaluation._store(
                root,
                directory / f"failed-{index}.json",
                {
                    "attempt": index,
                    "status": "SCHEMA_OR_SOURCE_REJECTED",
                },
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
) -> tuple[list[dict[str, Any]], str]:
    """Kokoro only when explicitly opted into; one immutable WAV per utterance."""
    from .scripting import NicheConfig, Scene

    config = NicheConfig(
        niche="private_chapter_preproduction",
        audience="private",
        brand="CLIPPER",
        voice=plan.voice,
        facts=(),
        topic="private chapter",
        format="general",
    )
    provenance = "INJECTED_TEST_WAV" if producer is not None else "ACTUAL_CPU_KOKORO"
    audio_root = root / f"chapter-audio-{provenance.lower()}"
    records: list[dict[str, Any]] = []
    for scene in script["scenes"]:
        utterances = [("narration", scene["narration"])]
        utterances += [("invented_dialogue", line["text"]) for line in scene["dialogue"]]
        for index, (kind, text) in enumerate(utterances):
            item = Scene(headline=scene["scene_id"], narration=text)
            path = narration.tts_cache_path(item, config, audio_root)
            marker_key = content_fingerprint(
                {
                    "request": narration.tts_request(item, config),
                    "provenance": provenance,
                }
            )
            marker = audio_root / "attempts" / f"{marker_key}.json"
            if not path.is_file():
                if marker.exists():
                    raise ChapterStop("prior Kokoro utterance is unresolved; no retry")
                evaluation._store(
                    root, marker, {"status": "STARTED_UNRESOLVED", "request_sha256": marker_key}
                )
            try:
                def make_utterance(utterance: Scene = item) -> Path:
                    return narration.synthesize_scene(
                        utterance, config, audio_root, producer=producer
                    )

                wave_path = evaluation._capture_private_generator(
                    root, audio_root, make_utterance
                )
                data = evaluation._read(wave_path, 40_000_000)
                channels, sample_rate, frames = narration.validated_wav(data, cached=True)
            except Exception:
                raise ChapterStop("chapter narration failed; private attempt retained") from None
            records.append(
                {
                    "scene_id": scene["scene_id"],
                    "utterance": index,
                    "kind": kind,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "voice": plan.voice,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "samples": frames,
                    "duration_seconds": round(frames / sample_rate, 4),
                    "provenance": provenance,
                }
            )
    report_key = content_fingerprint(
        {
            "script": script,
            "records": records,
            "voice": plan.voice,
            "model": scripting.VOICE_MODEL,
            "voice_revision_label": scripting.VOICE_REVISION,
        }
    )
    evaluation._store(
        root,
        root / f"narration-report-{report_key}.json",
        {
            "schema": "clipper-chapter-narration-v1",
            "records": records,
            "voice_model": scripting.VOICE_MODEL,
            "voice_revision_label": scripting.VOICE_REVISION,
            "resolved_model_commit": None,
            "resolved_commit_verified": False,
            "provenance": provenance,
            "publish_eligible": False,
        },
    )
    return records, provenance


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
    if len(candidates) < 2 or len(candidates) > limits.max_beats:
        raise ChapterStop("bounded chapter lacks sufficient eligible source candidates")
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
    evidence = [
        {
            "candidate_id": item["id"],
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
        "schema": "clipper-chapter-source-ledger-v1",
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
        "evidence": evidence,
    }
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
        lambda x: _brief_validate(x, evidence, limits),
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
        whole_book.load_strict_reply(review_bytes), brief, brief_key, evidence
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
    if limits.kokoro_cpu:
        audio, audio_origin = _narrate(root, script, limits, narration_producer)
    times = _timings(script, limits, audio)
    summary.update(
        {
            "actual_audio": audio_origin == "ACTUAL_CPU_KOKORO",
            "audio_provenance": audio_origin,
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

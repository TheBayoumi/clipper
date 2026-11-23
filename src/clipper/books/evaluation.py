"""Private, opt-in book-to-previsual evaluation; never a publication path.

Only a canonical candidate analysis ref or its immutable mined ledger is accepted.
This deliberately does not use the legacy editorial catalog/plan/run selector.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from clipper.stage_contracts import content_fingerprint

from . import acquisition, constrained, narration, scripting, selected_scope, whole_book
from .contracts import BookText

if TYPE_CHECKING:
    from .orchestrator import BookOrchestrator

EVALUATION_SCHEMA = "clipper-book-private-previsual-evaluation-v1"
REVIEW_SCHEMA = "clipper-book-human-source-support-review-v1"
MANIFEST_SCHEMA = "clipper-book-private-cpu-evaluation-manifest-v1"
DIRECTION_SCHEMA = "clipper-book-generated-text-shots-v1"
SECTION_MANIFEST_SCHEMA = "clipper-book-section-evaluation-scope-v1"
MAX_PRIVATE_JSON = 2_000_000
MAX_DIRECTION_ATTEMPTS = 2
MAX_DIRECTION_BYTES = 16_384
SHOT_FIELDS = frozenset(
    {
        "scene_id",
        "purpose",
        "subject",
        "action",
        "environment",
        "mechanism",
        "continuity",
        "evidence_id",
    }
)


class EvaluationStop(ValueError):
    """Sanitized fail-stop; private source/model content is never an error string."""


@dataclass(frozen=True)
class EvaluationOptions:
    """Explicit later-run CPU permissions; default authorizes no model at all."""

    mine_cpu: bool = False
    script_cpu: bool = False
    direction_cpu: bool = False
    kokoro_cpu: bool = False
    max_new_units: int = 1
    pillar: str = "self_improvement"
    voice: str = "af_heart"

    @classmethod
    def from_manifest(cls, raw: object) -> EvaluationOptions:
        if (
            not isinstance(raw, dict)
            or set(raw)
            != {
                "schema",
                "mine_cpu",
                "script_cpu",
                "direction_cpu",
                "kokoro_cpu",
                "max_new_units",
                "pillar",
                "voice",
            }
            or raw["schema"] != MANIFEST_SCHEMA
        ):
            raise EvaluationStop("invalid private evaluation run manifest")
        if any(
            type(raw[field]) is not bool
            for field in ("mine_cpu", "script_cpu", "direction_cpu", "kokoro_cpu")
        ):
            raise EvaluationStop("CPU permissions must be explicit booleans")
        if type(raw["max_new_units"]) is not int or not 1 <= raw["max_new_units"] <= 8:
            raise EvaluationStop("private CPU mining budget must be between 1 and 8")
        if raw["pillar"] not in ("self_improvement", "psychology", "business"):
            raise EvaluationStop("evaluation requires an explicit supported pillar")
        if raw["voice"] not in scripting.VOICES:
            raise EvaluationStop("evaluation requires a supported voice")
        return cls(**{k: v for k, v in raw.items() if k != "schema"})


def _read(path: Path, budget: int = MAX_PRIVATE_JSON) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise EvaluationStop("private evaluation input is absent or linked")
    data = acquisition._read_bounded_file(path, budget + 1)
    if len(data) > budget:
        raise EvaluationStop("private evaluation input exceeds byte limit")
    return data


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _store(root: Path, path: Path, value: object) -> None:
    acquisition._private_write(path, _json_bytes(value), private_root=root)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _section_scope(
    runner: BookOrchestrator,
    ref_path: Path,
    section_manifest: Path,
    *,
    selected_scope_path: Path | None = None,
) -> tuple[whole_book.WorkPlan, str]:
    """Verify original source-bound section IDs without inventing parent BookText."""
    ref: whole_book.CandidateAnalysisRef | selected_scope.SelectedRef
    book: BookText | selected_scope.SelectedSource
    if selected_scope_path is not None:
        if ref_path != selected_scope_path:
            raise EvaluationStop("selected source cannot substitute ordinary candidate")
        verified = selected_scope.import_chapter(runner.root, selected_scope_path)
        ref, book = verified.ref, verified.book
    else:
        ref, book = runner.load_candidate_for_analysis(ref_path)
    raw = _read(section_manifest, 16_384)
    try:
        manifest: object = whole_book.load_strict_reply(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise EvaluationStop("invalid private section scope JSON") from None
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "book_fingerprint",
        "source_fingerprint",
        "ingest_stage_key",
        "parent_plan_fingerprint",
        "section_ids",
        "parent_total_units",
        "included_units",
        "excluded_units",
    }:
        raise EvaluationStop("invalid section manifest schema")
    ids = manifest["section_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise EvaluationStop("section scope requires canonical section ID strings")
    try:
        plan = whole_book.plan_for(book, section_ids=tuple(ids))
    except ValueError:
        raise EvaluationStop("section scope does not match canonical parent text") from None
    expected = {
        "schema": SECTION_MANIFEST_SCHEMA,
        "book_fingerprint": book.fingerprint,
        "source_fingerprint": book.source_fingerprint,
        "ingest_stage_key": ref.ingest_stage_key,
        "parent_plan_fingerprint": plan.parent_plan_fingerprint,
        "section_ids": list(plan.section_ids),
        "parent_total_units": plan.parent_total_units,
        "included_units": len(plan.units),
        "excluded_units": plan.excluded_units,
    }
    if manifest != expected:
        raise EvaluationStop("section manifest bounds or parent source identity mismatch")
    digest = _sha(raw)
    if selected_scope_path is not None and digest != verified.section_manifest_sha256:
        raise EvaluationStop("selected original scope digest mismatch")
    root = runner.root / "book-candidates"
    acquisition._private_write(
        root / "section-scopes" / f"{digest}.json",
        raw,
        private_root=root,
    )
    return plan, digest


def _capture_private_generator(
    private_root: Path,
    log_dir: Path,
    producer: Callable[[], Any],
) -> Any:
    """Suppress canonical script logs on public stderr; retain private diagnostics."""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as sink:
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                return producer()
        finally:
            sink.flush()
            sink.seek(0)
            output = sink.read(262145)
            if output:
                _store(
                    private_root,
                    log_dir / f"log-{uuid4().hex}.json",
                    {
                        "diagnostics": output[:262144],
                        "truncated": len(output) > 262144,
                    },
                )


def _mined_export(
    runner: BookOrchestrator,
    ref_path: Path,
    progress_path: Path,
    *,
    section_plan: whole_book.WorkPlan | None = None,
    section_manifest_sha256: str | None = None,
    selected_scope_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    """Bind raw mining/export evidence to verified retained or selected source."""
    ref: whole_book.CandidateAnalysisRef | selected_scope.SelectedRef
    book: BookText | selected_scope.SelectedSource
    if selected_scope_path is not None:
        if ref_path != selected_scope_path:
            raise EvaluationStop("selected source cannot masquerade as ordinary candidate")
        chosen = selected_scope.import_chapter(runner.root, selected_scope_path)
        ref, book = chosen.ref, chosen.book
        if section_plan != chosen.work or section_manifest_sha256 != chosen.section_manifest_sha256:
            raise EvaluationStop("selected progress no longer matches original scoped source")
    else:
        ref, book = runner.load_candidate_for_analysis(ref_path)
    base = runner.root / "book-candidates"
    directory = base / "analysis" / "progress" / ref.book_id
    if progress_path.is_symlink() or progress_path.parent.resolve() != directory.resolve():
        raise EvaluationStop("evaluation progress path is not canonical")
    progress_raw = _read(progress_path, 20_000_000)
    progress: object = whole_book.load_strict_reply(progress_raw)
    work = section_plan if section_plan is not None else whole_book.plan_for(book)
    if (section_plan is None) != (section_manifest_sha256 is None):
        raise EvaluationStop("section manifest is required for scoped evaluation")
    scope_fields = {"section_scope", "section_manifest_sha256", "parent_book_fingerprint"}
    if section_plan is None and isinstance(progress, dict) and scope_fields & set(progress):
        raise EvaluationStop("scoped progress requires its exact section manifest")
    if not isinstance(progress, dict) or any(
        progress.get(k) != v
        for k, v in {
            "schema": whole_book.RESULT_SCHEMA,
            "book_id": ref.book_id,
            "book_fingerprint": book.fingerprint,
            "candidate_fingerprint": ref.candidate_fingerprint,
            "source_fingerprint": book.source_fingerprint,
            "ingest_stage_key": ref.ingest_stage_key,
            "plan_fingerprint": work.fingerprint,
            **(
                {
                    "section_scope": work.section_scope,
                    "section_manifest_sha256": section_manifest_sha256,
                    "parent_book_fingerprint": book.fingerprint,
                }
                if section_plan is not None
                else {}
            ),
            "recovery_status": book.recovery_status,
            "authorization_status": "unverified",
            "review_status": "MINED_UNREVIEWED",
            "publish_eligible": False,
            "status": "complete",
            **(
                {
                    "source_boundary": "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT",
                    "selected_scope_bundle_sha256": chosen.bundle_sha256,
                    "parent_original_reverified_here": False,
                    "parent_excluded_page_count": cast(
                        selected_scope.SelectedSource, book
                    ).parent_excluded_page_count,
                }
                if selected_scope_path is not None
                else {}
            ),
        }.items()
    ):
        raise EvaluationStop("mined progress is incomplete or does not match pinned source")
    records_progress = progress.get("progress")
    if (
        not isinstance(records_progress, list)
        or len(records_progress) != len(work.units)
        or progress.get("total_work_units") != len(work.units)
        or progress.get("candidate_count")
        != sum(x.get("candidate_count", -1) for x in records_progress if isinstance(x, dict))
    ):
        raise EvaluationStop("mining ledger is missing included section units")
    for record, unit in zip(records_progress, work.units, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("status") not in {"complete", "zero_candidate"}
            or record.get("unit") != unit.ordinal
            or record.get("unit_fingerprint") != unit.fingerprint
            or record.get("anchor") != unit.anchor
            or record.get("source_path") != unit.source_path
            or record.get("text_sha256") != unit.text_sha256
            or record.get("evidence_field") != unit.evidence_field
        ):
            raise EvaluationStop("mining progress does not cover canonical scope units")
    counts = progress.get("counts")
    if (
        not isinstance(counts, dict)
        or any(
            counts.get(key) != sum(x["status"] == key for x in records_progress)
            for key in ("complete", "zero_candidate", "unresolved", "failed")
        )
        or counts.get("skipped") != len(work.skipped)
    ):
        raise EvaluationStop("mining progress count or scope coverage mismatch")
    export_path = base / "analysis" / "exports" / ref.book_id / progress_path.name
    export: object = json.loads(_read(export_path, 20_000_000))
    if not isinstance(export, dict) or any(
        export.get(k) != v
        for k, v in {
            "schema": "clipper-book-editorial-catalog-v1",
            "input_kind": (
                "mined_unreviewed_selected_chapter_evidence"
                if selected_scope_path is not None
                else "mined_separate_unreviewed_whole_book"
            ),
            "review_status": "MINED_UNREVIEWED",
            "publish_eligible": False,
            "analysis_plan_fingerprint": progress["plan_fingerprint"],
            "progress_sha256": _sha(progress_raw),
            "ingest_fingerprint": book.fingerprint,
            **(
                {
                    "source_boundary": "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT",
                    "selected_scope_bundle_sha256": chosen.bundle_sha256,
                    "parent_original_reverified_here": False,
                    "parent_excluded_page_count": cast(
                        selected_scope.SelectedSource, book
                    ).parent_excluded_page_count,
                }
                if selected_scope_path is not None
                else {}
            ),
            "recovery_status": book.recovery_status,
            **(
                {
                    "section_scope": work.section_scope,
                    "section_manifest_sha256": section_manifest_sha256,
                    "parent_full_book_complete": False,
                }
                if section_plan is not None
                else {}
            ),
        }.items()
    ):
        raise EvaluationStop("mined export identity or review gate mismatch")
    if section_plan is None and isinstance(export, dict) and "section_scope" in export:
        raise EvaluationStop("unscoped export cannot contain section coverage")
    records = export.get("books")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise EvaluationStop("mined export book identity mismatch")
    record = records[0]
    if any(
        record.get(k) != v
        for k, v in {
            "id": ref.book_id,
            "source_fingerprint": book.source_fingerprint,
            "authorization_status": "unverified",
            **(
                {"status": "selected_original_scope_exporter_attested"}
                if selected_scope_path is not None
                else {}
            ),
            "review_status": "MINED_UNREVIEWED",
            "publish_eligible": False,
        }.items()
    ):
        raise EvaluationStop("mined export provenance mismatch")
    lessons = record.get("lessons")
    if not isinstance(lessons, list) or len(lessons) != progress.get("candidate_count"):
        raise EvaluationStop("mined export candidate count mismatch")
    units = work.units
    by_anchor = {(u.anchor, u.word_offset, u.text_sha256): u for u in units}
    seen: set[str] = set()
    for item in lessons:
        if not isinstance(item, dict) or not isinstance(item.get("evidence_anchor"), dict):
            raise EvaluationStop("mined candidate evidence is malformed")
        evidence = item["evidence_anchor"]
        candidate_unit = by_anchor.get(
            (evidence.get("anchor"), evidence.get("word_offset"), evidence.get("text_sha256"))
        )
        if (
            candidate_unit is None
            or item.get("review_status") != "MINED_UNREVIEWED"
            or item.get("publish_eligible") is not False
            or item.get("claim_source_url") != book.source.source_uri
            or item.get("id") in seen
            or any(
                evidence.get(k) != v
                for k, v in {
                    "section_id": candidate_unit.section_id,
                    "source_path": candidate_unit.source_path,
                    "field": candidate_unit.evidence_field,
                    "origin": candidate_unit.origin,
                    "review_status": candidate_unit.review_status,
                    "research_link_status": candidate_unit.research_link_status,
                    "source_sha256": book.content_sha256,
                }.items()
            )
        ):
            raise EvaluationStop("mined candidate identity or source context mismatch")
        occurrence = evidence.get("selected_occurrence")
        if occurrence is None:
            # Recorded responses have no local_span sidecar. Independently verify
            # the raw proposed ID/quote against the exact canonical span menu;
            # do not rewrite or promote any mined original field.
            try:
                request = whole_book._request_for_identity(book, candidate_unit, book.fingerprint)
                declared = constrained.selected_span(
                    {
                        "id": item["proposed_id"],
                        "evidence": evidence["quote"],
                        "evidence_field": evidence["field"],
                    },
                    request,
                )
            except (ValueError, KeyError, TypeError):
                raise EvaluationStop(
                    "recorded candidate quote pairing is not source-exact"
                ) from None
            occurrence = {
                "unit_start": declared["start"],
                "unit_end": declared["end"],
                "id": declared["id"],
                "quote_sha256": declared["sha256"],
            }
        if not isinstance(occurrence, dict):
            raise EvaluationStop("mined candidate occurrence is malformed")
        start, end = occurrence.get("unit_start"), occurrence.get("unit_end")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(candidate_unit.text)
            or candidate_unit.text[start:end] != evidence.get("quote")
            or occurrence.get("id") != item.get("proposed_id")
            or occurrence.get("quote_sha256")
            != _sha(candidate_unit.text[start:end].encode("utf-8"))
        ):
            raise EvaluationStop("mined source occurrence is inconsistent")
        item["_evaluation_context"] = {
            "anchor": candidate_unit.anchor,
            "section_id": candidate_unit.section_id,
            "source_path": candidate_unit.source_path,
            "unit_word_offset": candidate_unit.word_offset,
            "evidence_field": candidate_unit.evidence_field,
            "origin": candidate_unit.origin,
            "source_review_status": candidate_unit.review_status,
            "research_link_status": candidate_unit.research_link_status,
            "evidence_quote": evidence["quote"],
            "source_context": candidate_unit.text[
                max(0, start - 250) : min(len(candidate_unit.text), end + 250)
            ],
            "recovery_status": book.recovery_status,
            "authorization_status": book.source.authorization_status,
        }
        seen.add(cast(str, item["id"]))
    return progress, lessons, book


def _review(lessons: list[dict[str, Any]], raw: object, progress_sha: str) -> dict[str, Any] | None:
    if (
        not isinstance(raw, dict)
        or set(raw)
        != {"schema", "progress_sha256", "reviewer", "selected_candidate_id", "decisions"}
        or raw["schema"] != REVIEW_SCHEMA
        or raw["progress_sha256"] != progress_sha
    ):
        raise EvaluationStop("source-support review schema or mining identity mismatch")
    if (
        not isinstance(raw["reviewer"], str)
        or not 2 <= len(raw["reviewer"]) <= 80
        or not isinstance(raw["decisions"], list)
    ):
        raise EvaluationStop("source-support review must identify a human reviewer")
    candidates = {item["id"]: item for item in lessons}
    if len(candidates) != len(lessons):
        raise EvaluationStop("duplicate mined candidate IDs")
    decisions: dict[str, dict[str, str]] = {}
    for record in raw["decisions"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"candidate_id", "decision", "reason", "qualifier_quote"}
            or not all(isinstance(x, str) for x in record.values())
        ):
            raise EvaluationStop("invalid human source-support decision")
        cid = record["candidate_id"]
        if (
            cid not in candidates
            or cid in decisions
            or record["decision"] not in {"accept_direct", "reject"}
            or not 8 <= len(record["reason"].strip()) <= 600
        ):
            raise EvaluationStop("missing, duplicate or unsupported source-review decision")
        qualifier = record["qualifier_quote"]
        if len(qualifier) > 400 or (
            qualifier and qualifier not in candidates[cid]["_evaluation_context"]["source_context"]
        ):
            raise EvaluationStop("review qualifier must quote the exact source context")
        decisions[cid] = cast(dict[str, str], record)
    if set(decisions) != set(candidates):
        raise EvaluationStop("every mined candidate requires an explicit human decision")
    selected = raw["selected_candidate_id"]
    if selected is None and all(d["decision"] == "reject" for d in decisions.values()):
        return None
    if (
        not isinstance(selected, str)
        or selected not in decisions
        or decisions[selected]["decision"] != "accept_direct"
        or sum(d["decision"] == "accept_direct" for d in decisions.values()) != 1
    ):
        raise EvaluationStop("evaluation requires exactly one reviewer-selected direct claim")
    return {
        "selected": candidates[selected],
        "decision": decisions[selected],
        "reviewer": raw["reviewer"],
        "decisions": list(decisions.values()),
    }


def _script_config(
    book: object, candidate: dict[str, Any], options: EvaluationOptions, review_fingerprint: str
) -> scripting.NicheConfig:
    source = cast(Any, book).source
    lesson_id = cast(str, candidate["proposed_id"])
    book_id = "-".join(re.findall(r"[a-z0-9]+", source.title.casefold()))[:60] or "source"
    # A private evaluation identity cannot collide with a catalog-created production episode.
    return scripting.NicheConfig(
        niche="private_book_evaluation",
        audience="Adults seeking one testable lesson",
        brand="CLIPPER EVALUATION ONLY",
        voice=options.voice,
        facts=(),
        topic=candidate["topic"],
        format="book_lesson",
        pillar=options.pillar,
        book_title=source.title,
        book_author=source.author,
        source_url=source.source_uri,
        book_id=book_id,
        lesson_id=lesson_id,
        angle_id="assisted-eval",
        episode_id=f"{book_id}--{lesson_id}--assisted-eval",
        script_version=scripting.BOOK_EDIT_CONTRACT,
        selection_fingerprint=review_fingerprint,
        hook_angle=candidate["hook_angle"],
        visual_action=candidate["visual_action"],
    )


def _shot_request(
    script: scripting.Episode, source_context: dict[str, object], timings: list[dict[str, object]]
) -> dict[str, object]:
    scenes = [
        {
            "scene_id": f"scene-{i:02d}",
            "headline": s.headline,
            "narration": s.narration,
            "timing": timings[i],
        }
        for i, s in enumerate(script.scenes)
    ]
    return {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 960,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return exactly one JSON object with a 'shots' array, exactly one shot per "
                    "input scene in the same order. Each shot has exactly scene_id, purpose, "
                    "subject, action, environment, mechanism, continuity, evidence_id. "
                    "Every value is specific nonempty text, <= 220 characters. scene_id and "
                    "evidence_id must be copied verbatim. Ground the shot in the actual "
                    "narration and provided source context. Show a physical causal mechanism "
                    "rather than generic text, charts or decoration. Preserve qualifications. "
                    "Do not claim any images or video already exist. JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "schema": DIRECTION_SCHEMA,
                        "scenes": scenes,
                        "source_context": source_context,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ],
    }


def _shots(
    raw: str, script: scripting.Episode, evidence_id: str, timings: list[dict[str, object]]
) -> list[dict[str, object]]:
    if len(raw.encode("utf-8")) > MAX_DIRECTION_BYTES:
        raise EvaluationStop("direction reply exceeds bounded raw limit")
    try:
        result: object = whole_book.load_strict_reply(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EvaluationStop("direction output is not strict JSON") from exc
    if (
        not isinstance(result, dict)
        or set(result) != {"shots"}
        or not isinstance(result["shots"], list)
    ):
        raise EvaluationStop("direction output schema mismatch")
    shots = result["shots"]
    if len(shots) != len(script.scenes):
        raise EvaluationStop("direction does not cover every script scene")
    validated = []
    for i, shot in enumerate(shots):
        if (
            not isinstance(shot, dict)
            or set(shot) != SHOT_FIELDS
            or any(not isinstance(v, str) or not v.strip() or len(v) > 220 for v in shot.values())
            or shot["scene_id"] != f"scene-{i:02d}"
            or shot["evidence_id"] != evidence_id
        ):
            raise EvaluationStop("direction shot identity, text, or source mapping mismatch")
        validated.append(
            {**shot, "scene_headline": script.scenes[i].headline, "timing": timings[i]}
        )
    return validated


def evaluate_candidate(
    runner: BookOrchestrator,
    ref_path: Path,
    *,
    progress_path: Path | None = None,
    section_manifest: Path | None = None,
    review_file: Path | None = None,
    options: EvaluationOptions = EvaluationOptions(),
    recorded_responses: dict[str, object] | None = None,
    response_file_sha256: str | None = None,
    mine_completion: Callable[[dict[str, object]], str] | None = None,
    script_completion: Callable[[dict[str, object]], str] | None = None,
    direction_completion: Callable[[dict[str, object]], str] | None = None,
    narration_producer: Callable[[str, str], bytes] | None = None,
) -> dict[str, object]:
    """Mine/replay, gate human source support, generate script/audio/text direction.

    Every provider is lazy and independently opt-in. No catalog, visuals or ffmpeg.
    Completed output is still assisted, unverified and never publishable.
    """
    section_plan: whole_book.WorkPlan | None = None
    section_manifest_sha256: str | None = None
    if section_manifest is not None:
        section_plan, section_manifest_sha256 = _section_scope(runner, ref_path, section_manifest)
    if progress_path is not None:
        if options.mine_cpu or recorded_responses is not None or mine_completion is not None:
            raise EvaluationStop("reused progress and new mining are mutually exclusive")
    elif options.mine_cpu:
        if recorded_responses is not None or response_file_sha256 is not None:
            raise EvaluationStop("recorded mining and real CPU mining are mutually exclusive")
        mined = runner.mine_candidate(
            ref_path,
            local_cpu=True,
            max_new_units=options.max_new_units,
            local_completion=mine_completion,
            section_ids=section_plan.section_ids if section_plan else None,
            section_manifest_sha256=section_manifest_sha256,
        )
        progress_path = Path(mined["progress_path"])
    elif recorded_responses is not None and response_file_sha256 is not None:
        mined = runner.mine_candidate(
            ref_path,
            responses=recorded_responses,
            response_file_sha256=response_file_sha256,
            section_ids=section_plan.section_ids if section_plan else None,
            section_manifest_sha256=section_manifest_sha256,
        )
        progress_path = Path(mined["progress_path"])
    else:
        raise EvaluationStop("evaluation requires pinned progress or explicit mining inputs")
    if progress_path is None:
        raise EvaluationStop("mining progress was not recorded")
    _progress, lessons, book = _mined_export(
        runner,
        ref_path,
        progress_path,
        section_plan=section_plan,
        section_manifest_sha256=section_manifest_sha256,
    )
    progress_sha = _sha(_read(progress_path, 20_000_000))
    private_root = runner.root / "book-candidates"
    evaluation_key = content_fingerprint(
        {"schema": EVALUATION_SCHEMA, "progress_sha256": progress_sha}
    )
    evaluation_root = private_root / "evaluations" / evaluation_key
    _store(
        private_root,
        evaluation_root / "mined-candidates.json",
        {
            "schema": EVALUATION_SCHEMA,
            "progress_sha256": progress_sha,
            "source_fingerprint": book.source_fingerprint,
            "candidate_count": len(lessons),
            "section_scope": section_plan.section_scope if section_plan else None,
            "candidates": lessons,
            "review_status": "MINED_UNREVIEWED",
            "publish_eligible": False,
        },
    )
    summary: dict[str, object] = {
        "schema": EVALUATION_SCHEMA,
        "progress_sha256": progress_sha,
        "candidate_count": len(lessons),
        "section_scope": section_plan.section_scope if section_plan else None,
        "scope_sha256": section_manifest_sha256,
        "scope_section_count": len(section_plan.section_ids) if section_plan else 0,
        "scope_included_units": len(section_plan.units) if section_plan else 0,
        "scope_excluded_units": section_plan.excluded_units if section_plan else 0,
        "parent_full_book_complete": False if section_plan else None,
        "publish_eligible": False,
        "source_authorization": book.source.authorization_status,
        "recovery_status": book.recovery_status,
        "execution": "ASSISTED_EVALUATION",
        "model_calls_authorized": {
            "mining_new_units_max": options.max_new_units if options.mine_cpu else 0,
            "script_attempts_max": scripting.MAX_SCRIPT_ATTEMPTS if options.script_cpu else 0,
            "direction_attempts_max": MAX_DIRECTION_ATTEMPTS if options.direction_cpu else 0,
            "kokoro_scenes_max": 5 if options.kokoro_cpu else 0,
        },
    }
    if review_file is None:
        return {**summary, "status": "AWAITING_HUMAN_SOURCE_SUPPORT_REVIEW"}
    review_bytes = _read(review_file)
    review = _review(lessons, json.loads(review_bytes), progress_sha)
    review_sha = _sha(review_bytes)
    _store(
        private_root,
        evaluation_root / f"review-{review_sha}.json",
        {
            "review_sha256": review_sha,
            "review": json.loads(review_bytes),
            "assisted_only": True,
            "publish_eligible": False,
        },
    )
    if review is None:
        return {
            **summary,
            "status": "ABSTAINED_ZERO_SOURCE_SUPPORTED_LESSONS",
            "review_sha256": review_sha,
            "accepted_lessons": 0,
        }
    selected = review["selected"]
    source_context = {
        **selected["_evaluation_context"],
        "book_title": book.source.title,
        "book_author": book.source.author,
        "source_sha256": book.content_sha256,
        "candidate_id": selected["id"],
        "claim": selected["claim"],
        "reviewer_support_reason": review["decision"]["reason"],
        "reviewer_qualifier_quote": review["decision"]["qualifier_quote"],
        "editorial_status": "HUMAN_ASSISTED_EVALUATION_NOT_APPROVED",
        "publish_eligible": False,
    }
    _store(
        private_root,
        evaluation_root / f"selection-{review_sha}.json",
        {
            "selected_candidate_id": selected["id"],
            "source_context": source_context,
            "decisions": review["decisions"],
            "review_sha256": review_sha,
            "human_assisted": True,
            "publish_eligible": False,
        },
    )
    status = {
        **summary,
        "selected_candidate_id": selected["id"],
        "review_sha256": review_sha,
        "accepted_lessons": 1,
    }
    if not options.script_cpu:
        return {**status, "status": "AWAITING_EXPLICIT_CPU_SCRIPT_PERMISSION"}
    config = _script_config(
        book,
        selected,
        options,
        content_fingerprint(
            {"progress": progress_sha, "review": review_sha, "candidate": selected["id"]}
        ),
    )
    if script_completion is None:
        script_completion = scripting._local_script_completion
    expected_script_key = scripting._script_fingerprint(config, evaluation_context=source_context)
    script_root = evaluation_root / f"script-{expected_script_key}"
    acquisition._private_directory(script_root, private_root=private_root)
    cached_script = script_root / "scripts" / f"{expected_script_key}.json"
    if not cached_script.is_file() and (
        any(script_root.glob("started-*.json"))
        or any(script_root.glob("raw-*.json"))
        or (script_root / "script_diagnostics" / f"{expected_script_key}.json").exists()
    ):
        raise EvaluationStop("previous script attempt incomplete; retry requires new authority")
    script_attempts = 0

    def script_produce(request: dict[str, object]) -> str:
        nonlocal script_attempts
        if script_attempts >= scripting.MAX_SCRIPT_ATTEMPTS:
            raise EvaluationStop("script model attempt budget exhausted")
        script_attempts += 1
        _store(
            private_root,
            script_root / f"started-{script_attempts}.json",
            {
                "attempt": script_attempts,
                "request_sha256": content_fingerprint(request),
                "status": "STARTED_UNRESOLVED",
            },
        )
        try:
            raw = script_completion(request)
        except Exception as exc:
            _store(
                private_root,
                script_root / f"failed-{script_attempts}.json",
                {
                    "attempt": script_attempts,
                    "status": "PRODUCER_FAILED",
                    "category": type(exc).__name__,
                },
            )
            raise EvaluationStop("script producer failed; private attempts retained") from None
        if not isinstance(raw, str):
            _store(
                private_root,
                script_root / f"failed-{script_attempts}.json",
                {
                    "attempt": script_attempts,
                    "status": "NON_TEXT_REPLY",
                },
            )
            raise EvaluationStop("script producer returned nontext output")
        _store(
            private_root,
            script_root / f"raw-{script_attempts}.json",
            {
                "attempt": script_attempts,
                "raw": raw,
                "sha256": _sha(raw.encode("utf-8")),
            },
        )
        return raw

    try:
        episode, script_key = _capture_private_generator(
            private_root,
            script_root,
            lambda: scripting.generate_episode(
                config,
                script_root,
                completion=script_produce,
                evaluation_context=source_context,
            ),
        )
    except Exception:
        raise EvaluationStop("script generation failed; private attempts retained") from None
    # Original drafts are private; canonical script diagnostics record edits.
    script_audit = script_root / "script_diagnostics" / f"{script_key}.json"
    attempts = json.loads(_read(script_audit))
    scene_text = [s.narration for s in episode.scenes]
    words = [len(t.split()) for t in scene_text]
    track_data: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []
    elapsed = 0.0
    for index, scene in enumerate(episode.scenes):
        if options.kokoro_cpu:
            audio_root = evaluation_root / f"audio-{review_sha}"
            acquisition._private_directory(audio_root, private_root=private_root)
            producer = narration_producer or narration._kokoro_wav
            wav_cache = narration.tts_cache_path(scene, config, audio_root)
            started_path = audio_root / "attempts" / f"{wav_cache.stem}.json"
            if not wav_cache.is_file():
                if started_path.exists():
                    raise EvaluationStop(
                        "previous Kokoro attempt incomplete; retry requires new authority"
                    )
                _store(
                    private_root,
                    started_path,
                    {
                        "scene_id": f"scene-{index:02d}",
                        "request_sha256": content_fingerprint(narration.tts_request(scene, config)),
                        "status": "STARTED_UNRESOLVED",
                    },
                )
            try:
                wav_path = narration.synthesize_scene(scene, config, audio_root, producer=producer)
                wav = _read(wav_path, 40_000_000)
                channels, rate, frames = narration.validated_wav(wav, cached=True)
            except Exception:
                raise EvaluationStop("Kokoro narration failed or WAV invalid") from None
            seconds = frames / rate
            track_data.append(
                {
                    "scene_id": f"scene-{index:02d}",
                    "sha256": _sha(wav),
                    "samples": frames,
                    "sample_rate": rate,
                    "channels": channels,
                    "duration_seconds": round(seconds, 4),
                    "source": "actual_cpu_kokoro_wav",
                }
            )
            precision = "measured_wav"
        else:
            seconds = words[index] * 60 / 140
            precision = "estimated_no_audio"
        timings.append(
            {
                "start_seconds": round(elapsed, 4),
                "duration_seconds": round(seconds, 4),
                "basis": precision,
            }
        )
        elapsed += seconds
    _store(
        private_root,
        evaluation_root / f"script-report-{script_key}.json",
        {
            "script_fingerprint": script_key,
            "script": episode.to_dict(),
            "model": scripting.SCRIPT_MODEL,
            "revision": scripting.SCRIPT_REVISION,
            "attempts": attempts,
            "raw_attempt_count": len(attempts),
            "publish_eligible": False,
            "script_semantic_review": "PENDING",
        },
    )
    audio_identity = content_fingerprint(
        {
            "script_fingerprint": script_key,
            "voice": options.voice,
            "mode": "kokoro_wav" if options.kokoro_cpu else "estimated",
            "narration": track_data,
            "timings": timings,
        }
    )
    _store(
        private_root,
        evaluation_root / f"narration-report-{audio_identity}.json",
        {
            "script_fingerprint": script_key,
            "estimated_or_actual_duration_seconds": round(elapsed, 4),
            "narration": track_data,
            "voice_id": options.voice,
            "voice_model": scripting.VOICE_MODEL,
            "voice_revision_label": scripting.VOICE_REVISION,
            "voice_revision_resolved_commit": None,
            "voice_identity_verified": False,
            "timing_basis": "measured_wav" if options.kokoro_cpu else "estimated_no_audio",
            "publish_eligible": False,
        },
    )
    status.update(
        {
            "script_scene_count": len(episode.scenes),
            "script_attempts": script_attempts,
            "narration_included": options.kokoro_cpu,
            "measured_duration_seconds": round(elapsed, 4) if options.kokoro_cpu else None,
            "timing_basis": "measured_wav" if options.kokoro_cpu else "estimated_no_audio",
        }
    )
    if not options.direction_cpu:
        return {**status, "status": "AWAITING_EXPLICIT_CPU_TEXT_DIRECTION_PERMISSION"}
    if direction_completion is None:
        direction_completion = scripting._local_script_completion
    request = _shot_request(episode, source_context, timings)
    request_key = content_fingerprint(
        {
            "request": request,
            "script": episode.to_dict(),
            "review": review_sha,
            "contract": DIRECTION_SCHEMA,
        }
    )
    direction_root = evaluation_root / f"direction-{request_key}"
    completed_direction = direction_root / "validated-text-shots.json"
    if completed_direction.is_file():
        cached: object = json.loads(_read(completed_direction))
        if (
            not isinstance(cached, dict)
            or cached.get("schema") != DIRECTION_SCHEMA
            or cached.get("script_fingerprint") != script_key
            or cached.get("review_sha256") != review_sha
            or cached.get("source_context") != source_context
            or cached.get("timing_basis") != status["timing_basis"]
            or cached.get("publish_eligible") is not False
            or not isinstance(cached.get("shots"), list)
        ):
            raise EvaluationStop("cached text direction review or script identity changed")
        simplified = [
            {key: shot.get(key) for key in SHOT_FIELDS}
            for shot in cached["shots"]
            if isinstance(shot, dict)
        ]
        validated = _shots(json.dumps({"shots": simplified}), episode, selected["id"], timings)
        if validated != cached["shots"]:
            raise EvaluationStop("cached shot timing or content changed")
        return {
            **status,
            "status": "PREVISUAL_TEXT_READY_UNREVIEWED",
            "direction_attempts": 0,
            "direction_cache_reused": True,
            "text_shot_count": len(validated),
            "visual_generation_calls": 0,
        }
    if direction_root.is_dir() and (
        any(direction_root.glob("started-*.json")) or any(direction_root.glob("raw-*.json"))
    ):
        raise EvaluationStop("previous direction attempt incomplete; retry requires new authority")
    for index in range(1, MAX_DIRECTION_ATTEMPTS + 1):
        _store(
            private_root,
            direction_root / f"started-{index}.json",
            {
                "attempt": index,
                "request_sha256": content_fingerprint(request),
                "status": "STARTED_UNRESOLVED",
            },
        )
        try:
            raw = _capture_private_generator(
                private_root, direction_root, lambda: direction_completion(request)
            )
        except Exception as exc:
            _store(
                private_root,
                direction_root / f"failed-{index}.json",
                {
                    "attempt": index,
                    "status": "PRODUCER_FAILED",
                    "category": type(exc).__name__,
                },
            )
            raise EvaluationStop("text direction producer failed") from None
        if not isinstance(raw, str):
            _store(
                private_root,
                direction_root / f"failed-{index}.json",
                {
                    "attempt": index,
                    "status": "NON_TEXT_REPLY",
                },
            )
            raise EvaluationStop("direction producer returned nontext output")
        _store(
            private_root,
            direction_root / f"raw-{index}.json",
            {
                "attempt": index,
                "raw": raw,
                "sha256": _sha(raw.encode("utf-8")),
            },
        )
        try:
            shots = _shots(raw, episode, selected["id"], timings)
        except EvaluationStop:
            _store(
                private_root,
                direction_root / f"failed-{index}.json",
                {"attempt": index, "status": "REJECTED_SCHEMA_OR_SCENE_MAPPING"},
            )
            if index == MAX_DIRECTION_ATTEMPTS:
                raise EvaluationStop("text direction failed after two recorded attempts") from None
            continue
        _store(
            private_root,
            direction_root / "validated-text-shots.json",
            {
                "schema": DIRECTION_SCHEMA,
                "shots": shots,
                "script_fingerprint": script_key,
                "review_sha256": review_sha,
                "source_context": source_context,
                "timing_basis": status["timing_basis"],
                "semantic_visual_review": "PENDING",
                "publish_eligible": False,
            },
        )
        return {
            **status,
            "status": "PREVISUAL_TEXT_READY_UNREVIEWED",
            "direction_attempts": index,
            "text_shot_count": len(shots),
            "visual_generation_calls": 0,
        }
    raise AssertionError("unreachable direction attempt limit")

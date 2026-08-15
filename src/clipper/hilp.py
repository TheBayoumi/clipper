from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .models import (
    BeatType,
    EditPlan,
    EditorialBeat,
    HookMode,
    SourceSpan,
    TranscriptSegment,
    TranscriptWord,
)
from .qc import run_technical_qc

HilpDecision = Literal["APPROVE", "REJECT", "REVISE"]
QcRunner = Callable[..., dict[str, Any]]


class HilpSimulationError(RuntimeError):
    """Raised when the production HILP simulation cannot prove all review branches."""


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HilpSimulationError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _word(raw: object) -> TranscriptWord:
    if not isinstance(raw, dict):
        raise HilpSimulationError("transcript word is not an object")
    return TranscriptWord(float(raw["start"]), float(raw["end"]), str(raw["text"]))


def _segment(raw: object) -> TranscriptSegment:
    if not isinstance(raw, dict):
        raise HilpSimulationError("transcript segment is not an object")
    words = tuple(_word(item) for item in raw.get("words") or [])
    return TranscriptSegment(
        float(raw["start"]),
        float(raw["end"]),
        str(raw["text"]),
        words,
        str(raw["speaker_id"]) if raw.get("speaker_id") is not None else None,
    )


def _plan(raw: object) -> EditPlan:
    if not isinstance(raw, dict):
        raise HilpSimulationError("edit plan is not an object")
    spans = tuple(
        SourceSpan(float(item["start"]), float(item["end"]))
        for item in raw.get("source_spans") or []
        if isinstance(item, dict)
    )
    beats = tuple(
        EditorialBeat(
            float(item["start"]),
            float(item["end"]),
            cast(BeatType, str(item["beat_type"])),
            float(item.get("strength") or 0.0),
            str(item["text"]) if item.get("text") is not None else None,
        )
        for item in raw.get("beats") or []
        if isinstance(item, dict)
    )
    return EditPlan(
        plan_id=str(raw["plan_id"]),
        video_id=str(raw["video_id"]),
        concept_id=str(raw["concept_id"]),
        variant_id=str(raw["variant_id"]),
        hook_mode=cast(HookMode, str(raw["hook_mode"])),
        source_spans=spans,
        hook_text=str(raw["hook_text"]) if raw.get("hook_text") is not None else None,
        beats=beats,
        caption_platform=str(raw["caption_platform"]),
        score=float(raw["score"]),
        transcript_fingerprint=str(raw["transcript_fingerprint"]),
        caption_start_source_time=(
            float(raw["caption_start_source_time"])
            if raw.get("caption_start_source_time") is not None
            else None
        ),
        caption_start_word=(
            str(raw["caption_start_word"]) if raw.get("caption_start_word") is not None else None
        ),
    )


def _first_source_word(plan: EditPlan, segments: Sequence[TranscriptSegment]) -> TranscriptWord:
    if not plan.source_spans:
        raise HilpSimulationError(f"plan {plan.plan_id} has no source span")
    span = plan.source_spans[0]
    words = [
        word
        for segment in segments
        for word in segment.words
        if word.start >= span.start - 1e-6 and word.end <= span.end + 1e-6
    ]
    if not words:
        raise HilpSimulationError(f"plan {plan.plan_id} has no word-level source evidence")
    return min(words, key=lambda item: item.start)


def _delayed_anchor(
    plan: EditPlan,
    segments: Sequence[TranscriptSegment],
    *,
    minimum_delay: float = 0.75,
) -> tuple[TranscriptWord, float]:
    first = _first_source_word(plan, segments)
    if not plan.source_spans:
        raise HilpSimulationError(f"plan {plan.plan_id} has no source span")
    span = plan.source_spans[0]
    later_words = sorted(
        (
            word
            for segment in segments
            for word in segment.words
            if word.start >= first.start + minimum_delay and word.end <= span.end - 0.05
        ),
        key=lambda item: item.start,
    )
    if not later_words:
        raise HilpSimulationError(
            f"plan {plan.plan_id} is too short to inject a caption-delay review defect"
        )
    return first, later_words[0].start


def _review_caption_anchor(expected_word: TranscriptWord, plan: EditPlan) -> dict[str, Any]:
    anchor = plan.caption_start_source_time
    if anchor is None:
        anchor = plan.source_spans[0].start if plan.source_spans else 0.0
    delay = float(anchor) - expected_word.start
    decision: HilpDecision = "REVISE" if delay > 0.35 else "APPROVE"
    return {
        "decision": decision,
        "issue": "caption_start_delayed" if decision == "REVISE" else None,
        "expected_first_word": expected_word.text,
        "expected_first_word_time": expected_word.start,
        "caption_anchor_time": anchor,
        "delay_seconds": round(delay, 3),
    }


def _event(
    decision: HilpDecision,
    *,
    plan_id: str,
    concept_id: str,
    reason: str,
    round_number: int,
    **extra: object,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "plan_id": plan_id,
        "concept_id": concept_id,
        "reason": reason,
        "round": round_number,
        "reviewer": "simulated-human",
        "at": datetime.now(UTC).isoformat(),
        **extra,
    }


def _qc(
    qc_runner: QcRunner,
    output: Path,
    plan: EditPlan,
    *,
    watermark_required: bool,
    watermark_present: bool,
) -> dict[str, Any]:
    report = qc_runner(
        output,
        expected_duration=plan.duration,
        caption_path=output.with_suffix(".ass"),
        tracking_path=output.with_suffix(".tracking.json"),
        caption_platform=plan.caption_platform,
        watermark_required=watermark_required,
        watermark_present=watermark_present,
        caption_audit_path=output.with_suffix(".caption-audit.json"),
    )
    if report.get("status") != "PASS":
        raise HilpSimulationError(f"repaired HILP render failed technical QC: {report}")
    return report


def validate_hilp_evidence(payload: dict[str, Any]) -> None:
    if payload.get("status") != "PASS":
        raise HilpSimulationError(f"HILP status is not PASS: {payload.get('status')}")
    events = payload.get("events")
    if not isinstance(events, list):
        raise HilpSimulationError("HILP events are missing")
    decisions = {str(item.get("decision")) for item in events if isinstance(item, dict)}
    if not {"APPROVE", "REJECT", "REVISE"} <= decisions:
        raise HilpSimulationError(f"HILP did not exercise all decision branches: {decisions}")
    final = payload.get("final_shortlist")
    if not isinstance(final, list) or len(final) < 3:
        raise HilpSimulationError("HILP final shortlist is incomplete")
    concepts = {str(item.get("concept_id")) for item in final if isinstance(item, dict)}
    if len(concepts) != len(final):
        raise HilpSimulationError("HILP final shortlist lost concept diversity")
    revision = payload.get("revision_proof")
    if not isinstance(revision, dict):
        raise HilpSimulationError("HILP revision proof is missing")
    if revision.get("before_sha256") == revision.get("after_sha256"):
        raise HilpSimulationError("HILP revise branch did not produce a changed render")
    if revision.get("after_qc") != "PASS":
        raise HilpSimulationError("HILP repaired render did not pass technical QC")


def simulate_hilp_cycle(
    run_dir: str | Path,
    *,
    source_path: Path,
    renderer: Any,
    watermark_path: Path | None = None,
    qc_runner: QcRunner = run_technical_qc,
) -> dict[str, Any]:
    """Exercise approve, reject/replace, and revise/rerender against real render evidence."""

    root = Path(run_dir)
    manifest = _load_object(root / "manifest.json")
    rendered = manifest.get("rendered_clips")
    shortlist = manifest.get("submission_shortlist")
    plans_raw = manifest.get("edit_plans")
    transcript_raw = _load_object(root / "transcript.json")
    if not isinstance(rendered, list) or len(rendered) < 4:
        raise HilpSimulationError("HILP requires at least four technically accepted finalists")
    if not isinstance(shortlist, list) or len(shortlist) < 3:
        raise HilpSimulationError("HILP requires a three-clip initial shortlist")
    if not isinstance(plans_raw, list):
        raise HilpSimulationError("HILP requires edit-plan evidence")

    plan_index = {
        str(item.get("plan_id")): _plan(item)
        for item in plans_raw
        if isinstance(item, dict) and item.get("plan_id")
    }
    transcripts = {
        str(video_id): [_segment(item) for item in items]
        for video_id, items in transcript_raw.items()
        if isinstance(items, list)
    }
    initial = [item for item in shortlist[:3] if isinstance(item, dict)]
    if len(initial) != 3:
        raise HilpSimulationError("initial shortlist contains invalid entries")

    approve_clip, reject_clip, revise_clip = initial
    approve_concept = str(approve_clip.get("concept_id") or "")
    revise_concept = str(revise_clip.get("concept_id") or "")
    initial_plan_ids = {str(item.get("plan_id") or "") for item in initial}
    replacement = next(
        (
            item
            for item in rendered
            if isinstance(item, dict)
            and str(item.get("plan_id") or "") not in initial_plan_ids
            and str(item.get("concept_id") or "") not in {approve_concept, revise_concept}
        ),
        None,
    )
    if replacement is None:
        raise HilpSimulationError("no distinct accepted finalist is available for reject/replace")

    events: list[dict[str, Any]] = []
    events.append(
        _event(
            "APPROVE",
            plan_id=str(approve_clip["plan_id"]),
            concept_id=approve_concept,
            reason="candidate accepted without requested changes",
            round_number=1,
            render_sha256=approve_clip.get("render_sha256"),
        )
    )
    events.append(
        _event(
            "REJECT",
            plan_id=str(reject_clip["plan_id"]),
            concept_id=str(reject_clip.get("concept_id") or ""),
            reason="simulated human prefers a materially different finalist",
            round_number=1,
            replacement_plan_id=replacement.get("plan_id"),
        )
    )
    events.append(
        _event(
            "APPROVE",
            plan_id=str(replacement["plan_id"]),
            concept_id=str(replacement.get("concept_id") or ""),
            reason="replacement finalist accepted",
            round_number=2,
            render_sha256=replacement.get("render_sha256"),
        )
    )

    revise_plan_id = str(revise_clip.get("plan_id") or "")
    try:
        original_plan = plan_index[revise_plan_id]
        segments = transcripts[original_plan.video_id]
    except KeyError as exc:
        raise HilpSimulationError("revise candidate lacks plan/transcript evidence") from exc
    expected_word, bad_anchor = _delayed_anchor(original_plan, segments)
    bad_plan = replace(
        original_plan,
        caption_start_source_time=bad_anchor,
        caption_start_word=None,
    )
    hilp_dir = root / "hilp"
    hilp_dir.mkdir(parents=True, exist_ok=True)
    before = hilp_dir / f"revise-before-{revise_plan_id}.mp4"
    renderer.render(
        source_path,
        before,
        bad_plan.to_clip_candidate("simulated HILP revision candidate"),
        segments,
        watermark_path,
        bad_plan,
    )
    before_review = _review_caption_anchor(expected_word, bad_plan)
    if before_review["decision"] != "REVISE":
        raise HilpSimulationError("controlled HILP defect did not trigger REVISE")
    before_sha = _sha256(before)
    events.append(
        _event(
            "REVISE",
            plan_id=revise_plan_id,
            concept_id=revise_concept,
            reason="caption_start_delayed",
            round_number=1,
            render_sha256=before_sha,
            review=before_review,
            repair_stage="captions",
        )
    )

    repaired_anchor = original_plan.caption_start_source_time
    if repaired_anchor is None or repaired_anchor > expected_word.start + 0.35:
        repaired_anchor = expected_word.start
    repaired_plan = replace(original_plan, caption_start_source_time=repaired_anchor)
    after = hilp_dir / f"revise-after-{revise_plan_id}.mp4"
    renderer.render(
        source_path,
        after,
        repaired_plan.to_clip_candidate("simulated HILP repaired candidate"),
        segments,
        watermark_path,
        repaired_plan,
    )
    after_review = _review_caption_anchor(expected_word, repaired_plan)
    if after_review["decision"] != "APPROVE":
        raise HilpSimulationError(f"caption repair did not resolve review issue: {after_review}")
    after_sha = _sha256(after)
    after_qc = _qc(
        qc_runner,
        after,
        repaired_plan,
        watermark_required=watermark_path is not None,
        watermark_present=watermark_path is not None and watermark_path.is_file(),
    )
    events.append(
        _event(
            "APPROVE",
            plan_id=revise_plan_id,
            concept_id=revise_concept,
            reason="requested caption repair verified",
            round_number=2,
            render_sha256=after_sha,
            review=after_review,
            technical_qc="PASS",
        )
    )

    revised_final = dict(revise_clip)
    revised_final["output_path"] = str(after)
    revised_final["render_sha256"] = after_sha
    revised_final["hilp_revised"] = True
    final_shortlist = [dict(approve_clip), dict(replacement), revised_final]
    payload: dict[str, Any] = {
        "status": "PASS",
        "mode": "simulated_human_in_the_loop",
        "initial_shortlist": initial,
        "events": events,
        "final_shortlist": final_shortlist,
        "branches_exercised": ["APPROVE", "REJECT", "REVISE"],
        "revision_proof": {
            "plan_id": revise_plan_id,
            "before_path": str(before),
            "after_path": str(after),
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "before_review": before_review,
            "after_review": after_review,
            "after_qc": str(after_qc.get("status") or ""),
        },
    }
    validate_hilp_evidence(payload)
    (root / "hilp-simulation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "editorial-review.json").write_text(
        json.dumps(
            {
                "status": "SIMULATED_HILP_COMPLETE",
                "required": True,
                "simulated": True,
                "events": events,
                "final_shortlist": final_shortlist,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload

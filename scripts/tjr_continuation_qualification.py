"""Pinned six-case continuation qualification; never runs source discovery or selectors.

Reads a completed exact-source 111-candidate editorial checkpoint. Both blinded
materiality checks use the real production implementation and record source
pointers. Transcript-provisional labels are not audio-independent gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from clipper.editorial_engine import _validate_cached_boundary, review_optional_contrast
from clipper.models import TranscriptSegment, WordTiming
from scripts.tjr_semantic_editor import (
    LocalSourceReviewer,
    ReviewRequestCache,
    _review_context,
    _review_model_profile,
    _thought_units,
)

ARCHIVED_SOURCE_SHA = "db13e67b993ff1e56d33721ce64a0349b695b71165ab6b36791b9a5e95b9e080"
ARCHIVED_TRANSCRIPT_SHA = "d8dce573061aabb1c349faf89877b9a56996179bc749e0c8f408051be30205f0"
ARCHIVED_REVIEWER_SHA = "7d8b600bdc063c5ddec1c2cee44f76537abcc5c7260e9e6566b719d5e69f7490"
# Source-transcript provisional expectations, NOT independent audio annotation.
CASE_STARTS = {
    588.80: True,  # Coaching salary cap; next question is a new topic.
    2229.11: True,  # League neutrality answered; next question is separate.
    2474.51: True,  # Yardage comparison followed by restatement.
    3054.71: True,  # Purdy transition answered; next sentences repeat it.
    2063.88: False,  # DeMarcus Ware tackle result occurs after the cut.
    1313.25: False,  # Fan-base punchline occurs after the cut.
}


def prepare_cases(
    saved: dict[str, Any], raw_segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    identity = saved.get("identity", {})
    if (
        saved.get("complete") is not True
        or identity.get("source_sha256") != ARCHIVED_SOURCE_SHA
        or identity.get("reviewer_sha256") != ARCHIVED_REVIEWER_SHA
        or identity.get("transcript_sha256") != ARCHIVED_TRANSCRIPT_SHA
        or identity.get("reviewer_model_profile") != _review_model_profile()
        or len(saved.get("audit", {}).get("assessments", [])) != 111
    ):
        raise RuntimeError("CONTINUATION_ARCHIVE_IDENTITY_MISMATCH: refusing unpinned review")
    raw_digest = hashlib.sha256(json.dumps(raw_segments, sort_keys=True).encode()).hexdigest()
    if raw_digest != ARCHIVED_TRANSCRIPT_SHA:
        raise RuntimeError("CONTINUATION_TRANSCRIPT_HASH_MISMATCH: source changed")
    segments = [
        TranscriptSegment(
            float(item["start"]),
            float(item["end"]),
            item["text"],
            tuple(WordTiming(**word) for word in item.get("words", [])),
        )
        for item in raw_segments
    ]
    units = _thought_units(segments)
    assessments = saved["audit"]["assessments"]
    result: list[dict[str, Any]] = []
    for start, expected in CASE_STARTS.items():
        matches = [
            record
            for record in assessments
            if isinstance(record.get("reviewed_start"), (float, int))
            and abs(float(record["reviewed_start"]) - start) < 0.02
            and record.get("rejection") == "SPAN_REVIEW_REJECTED"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"CONTINUATION_MISSING_OR_DUPLICATED_CONTROL: {start}")
        record = matches[0]
        selection = record.get("decision", {})
        context = _review_context(units, selection["start_unit"], selection["end_unit"])
        digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        if digest != record.get("review_input_sha256"):
            raise RuntimeError(f"CONTINUATION_BOUNDARY_CONTEXT_MISMATCH: {start}")
        review = record["exchange_review"]
        _validate_cached_boundary(context, review)
        relation = review.get("continuation_review") or {}
        if (
            relation.get("relation") != "missing_contrast"
            or review.get("opening_standalone") is not True
            or review.get("contains_promotion_or_intro") is not False
            or review.get("exchange_has_payoff") is not True
            or not context["after"]
        ):
            raise RuntimeError(f"CONTINUATION_CONTROL_NOT_ELIGIBLE: {start}")
        resolution = review["source_quote_spans"]["resolution_quote"]
        if resolution is None:
            raise RuntimeError(f"CONTINUATION_MISSING_RESOLUTION: {start}")
        result.append(
            {
                "start": start,
                "expected_optional": expected,
                "selected": context["selected_units"],
                "after": context["after"],
                "final_unit": resolution["last_unit"],
                "review_context_sha256": digest,
            }
        )
    return result


def qualify(
    archive: Path,
    transcript: Path,
    output: Path,
    *,
    dry_run: bool = False,
    reuse_request_cache: Path | None = None,
) -> int:
    saved = json.loads(archive.read_text(encoding="utf-8"))
    raw = json.loads(transcript.read_text(encoding="utf-8"))
    cases = prepare_cases(saved, raw)
    report: dict[str, Any] = {
        "schema": "clipper-targeted-continuation-qualification-v1",
        "source_sha256": ARCHIVED_SOURCE_SHA,
        "transcript_sha256": ARCHIVED_TRANSCRIPT_SHA,
        "assessments_reused": 111,
        "selectors_rerun": 0,
        "full_review_rerun": 0,
        "source_acquired": False,
        "render_invoked": False,
        "annotation_status": "transcript_provisional_not_independent_audio_gold",
        "cases": [],
        "qualified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        report["prepared_case_count"] = len(cases)
        report["status"] = "SOURCE_CONTEXT_VERIFIED__NO_MODEL_CALLS"
        output.write_text(json.dumps(report, indent=2) + "\n")
        return 0
    editor: LocalSourceReviewer | None = None

    def source_reviewer() -> LocalSourceReviewer:
        nonlocal editor
        if editor is None:
            editor = LocalSourceReviewer(_review_model_profile())
        return editor

    request_cache = ReviewRequestCache(
        output.with_name("review-request-cache.json"),
        source_reviewer,
        {
            "source_sha256": ARCHIVED_SOURCE_SHA,
            "transcript_sha256": ARCHIVED_TRANSCRIPT_SHA,
            "reviewer_profile": _review_model_profile(),
            "stage": "targeted_optional_contrast_v1",
        },
        reuse_path=reuse_request_cache,
    )
    try:
        for case in cases:
            record = {
                "start": case["start"],
                "expected_optional": case["expected_optional"],
                "review_context_sha256": case["review_context_sha256"],
            }
            try:
                proof = review_optional_contrast(
                    request_cache, case["selected"], case["after"], case["final_unit"]
                )
                record["proof"] = proof
                record["actual_optional"] = proof["optional_verified"]
                record["correct"] = record["actual_optional"] == case["expected_optional"]
            except (RuntimeError, ValueError, TypeError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["correct"] = False
            report["cases"].append(record)
            report["request_cache_metrics"] = request_cache.metrics
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                "CONTINUATION_CONTROL "
                + json.dumps(
                    {
                        key: record.get(key)
                        for key in (
                            "start",
                            "expected_optional",
                            "actual_optional",
                            "correct",
                            "error",
                        )
                    }
                ),
                flush=True,
            )
            if "error" in record:
                # A schema/transport failure is not a semantic classification.
                # Abort before consuming model time on the remaining controls.
                break
    finally:
        if editor is not None:
            editor.close()
    positives = [case for case in report["cases"] if case["expected_optional"]]
    negatives = [case for case in report["cases"] if not case["expected_optional"]]
    report["positive_recall"] = sum(case.get("actual_optional") is True for case in positives)
    report["negative_controls_preserved"] = sum(
        case.get("actual_optional") is False for case in negatives
    )
    report["qualified"] = (
        len(report["cases"]) == len(CASE_STARTS)
        and report["positive_recall"] >= 3
        and report["negative_controls_preserved"] == len(negatives)
        and not any("error" in case for case in report["cases"])
    )
    report["status"] = (
        "TRANSCRIPT_CONTROL_QUALIFIED__AUDIO_VALIDATION_STILL_REQUIRED"
        if report["qualified"]
        else "CONTINUATION_QUALIFICATION_FAILED__DO_NOT_RENDER_RECOVERED"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["qualified"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editorial-cache", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-request-cache", type=Path)
    args = parser.parse_args()
    return qualify(
        args.editorial_cache,
        args.transcript,
        args.output,
        dry_run=args.dry_run,
        reuse_request_cache=args.reuse_request_cache,
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import build_identity
from .gameplay import contract


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_batch(
    root: Path,
    config: dict[str, Any],
    allocation: dict[str, Any],
) -> dict[str, Any]:
    contract.validate_configuration(config)
    build_identity.validate_artifact_sha(
        allocation.get("execution_sha"), stage="complete batch allocation"
    )
    summaries = [_load(path) for path in sorted(root.rglob("*_pipeline_summary.json"))]
    source_order = tuple(str(item) for item in allocation.get("source_order") or [])
    expected = set(source_order)
    actual = {str(item.get("source_key", "")) for item in summaries}
    failures: list[str] = []

    if actual != expected:
        failures.append(f"batch summaries cover {sorted(actual)}, expected {sorted(expected)}")
    declared_sources = [
        str(item.get("source_key") or "")
        for item in ((config.get("source_profile") or {}).get("assets") or [])
    ]
    if declared_sources and (
        len(set(declared_sources)) != len(declared_sources) or set(declared_sources) != expected
    ):
        failures.append("batch allocation does not include every configured source exactly once")
    if len(summaries) != len(expected):
        failures.append(
            f"expected exactly {len(expected)} source summaries, found {len(summaries)}"
        )

    total_selected = 0
    total_rendered = 0
    total_verified_finishers = 0
    total_selected_finishers = 0

    for item in summaries:
        source = str(item.get("source_key", ""))
        try:
            build_identity.validate_artifact_sha(
                item.get("execution_sha"), stage=f"{source} batch summary"
            )
        except RuntimeError as exc:
            failures.append(str(exc))
        if item.get("failure"):
            failures.append(f"{source}: {item['failure']}")
        selected = int(item.get("selected_count", 0))
        rendered = int(item.get("rendered_count", 0))
        expected_count = int(
            allocation.get("source_allocations", {}).get(source, {}).get("count", -1)
        )
        if rendered != selected:
            failures.append(f"{source}: rendered {rendered} != selected {selected}")
        if selected != expected_count:
            failures.append(f"{source}: rendered count {selected} != allocation {expected_count}")
        if not bool(item.get("technical_qa_passed", False)):
            failures.append(f"{source}: technical QA did not pass")
        if not bool(item.get("ffv1_nut_transport_contract_passed", False)):
            failures.append(f"{source}: lossless FFV1/NUT source-fidelity contract did not pass")
        if (
            str((config.get("source_profile") or {}).get("analysis_derivative") or "") == "proxy"
            and item.get("analysis_alignment_verified") is not True
        ):
            failures.append(f"{source}: original-to-proxy source alignment failed")
        if int(item.get("unplanned_source_cut_count", -1)) != 0:
            failures.append(f"{source}: unplanned source cuts present")

        source_allocation = dict(allocation.get("source_allocations", {}).get(source) or {})
        if int(source_allocation.get("selected_pair_conflict_count", -1)) != 0:
            failures.append(f"{source}: allocation reports selected candidate conflicts")
        dispositions = list(
            source_allocation.get("anchor_dispositions")
            or source_allocation.get("outcome_dispositions")
            or []
        )
        required_count = int(source_allocation.get("required_anchor_count", -1))
        if len(dispositions) != required_count:
            failures.append(
                f"{source}: required semantic anchor dispositions missing "
                f"({len(dispositions)} of {required_count})"
            )
        anchors = [str(entry.get("anchor")) for entry in dispositions]
        if len(anchors) != len(set(anchors)):
            failures.append(f"{source}: duplicated semantic anchor dispositions")
        qualified = [
            entry
            for entry in dispositions
            if entry.get("disposition") == "covered_by_selected_clip"
        ]
        if len(qualified) != int(source_allocation.get("qualified_anchor_count", -1)):
            failures.append(f"{source}: qualified required anchor coverage does not match")
        approved = set(source_allocation.get("plan_keys") or [])
        if any(
            not entry.get("selected_plan_keys")
            or not set(entry.get("selected_plan_keys") or []).issubset(approved)
            for entry in qualified
        ):
            failures.append(f"{source}: covered anchor lacks source-approved plan keys")
        if [
            entry
            for entry in dispositions
            if entry.get("disposition")
            not in {"covered_by_selected_clip", "no_admissible_candidate"}
        ]:
            failures.append(f"{source}: unresolved semantic-anchor dispositions")
        outcome_dispositions = list(source_allocation.get("outcome_dispositions") or [])
        if {str(e.get("anchor")) for e in outcome_dispositions} != {
            str(e.get("anchor")) for e in dispositions if isinstance(e.get("anchor"), float)
        }:
            failures.append(f"{source}: incomplete explicit outcome dispositions")

        total_selected += selected
        total_rendered += rendered
        total_verified_finishers += int(item.get("verified_finishing_move_count", 0))
        total_selected_finishers += int(item.get("selected_finishing_move_count", 0))

    derived_target = int(allocation.get("target_count", -1))
    if total_selected != derived_target:
        failures.append(f"batch selected {total_selected}; allocation target is {derived_target}")
    if total_rendered != total_selected:
        failures.append("batch rendered count does not equal selected count")
    if int(allocation.get("selected_count", -1)) != total_selected:
        failures.append("rendered batch does not match pre-render allocation total")
    if total_verified_finishers > 0 and total_selected_finishers < 1:
        failures.append("verified Finishing Move exists in batch but none was rendered")

    if failures:
        raise AssertionError("\n".join(failures))

    return {
        "schema_version": 1,
        "execution_sha": build_identity.current_execution_sha(),
        "sources": list(source_order),
        "selected_count": total_selected,
        "rendered_count": total_rendered,
        "selection_mode": "coverage_conflict_before_render",
        "derived_target_count": derived_target,
        "verified_finishing_move_count": total_verified_finishers,
        "selected_finishing_move_count": total_selected_finishers,
        "allocation_enforced": True,
        "status": "PASS",
    }


def validate_neural_source_review(review: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed independent six-source semantic gold/playback acceptance.

    GPU output may propose findings but cannot substitute for independently
    reviewed source frames, audio, and negative footage. A pending review
    always produces explicit, machine-readable blockers, never a green status.
    """
    import math
    import re
    from collections import Counter
    from fractions import Fraction

    expected_assets = {
        str(item.get("source_key") or ""): str(item.get("title") or "")
        for item in ((config.get("source_profile") or {}).get("assets") or [])
    }
    sources = review.get("sources")
    if not expected_assets or not isinstance(sources, list):
        raise AssertionError("neural review requires declared campaign sources")
    actual_keys = [str(item.get("source_key") or "") for item in sources if isinstance(item, dict)]
    blockers: list[dict[str, str]] = []

    def reject(source: str, reason: str) -> None:
        blockers.append({"source_key": source, "reason": reason})

    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != set(expected_assets):
        reject("*", "incomplete_or_duplicate_six_source_review")
    if review.get("independent_review_complete") is not True:
        reject("*", "independent_playback_review_incomplete")
    if review.get("gold_qualification") != "PASS":
        reject("*", "independent_gold_not_qualified")
    if review.get("type") == "benchmark_annotation_seed_NOT_model_ground_truth":
        reject("*", "annotation_seed_is_not_verified_gold")
    counts: Counter[str] = Counter()
    for item in sources:
        if not isinstance(item, dict):
            reject("*", "invalid_review_source_entry")
            continue
        source = str(item.get("source_key") or "")
        if item.get("source_title") != expected_assets.get(source):
            reject(source, "campaign_source_title_mismatch")
        if item.get("review_status") != "INDEPENDENTLY_REVIEWED":
            reject(source, "source_playback_review_pending")
        playback = item.get("playback_review") or {}
        if not isinstance(playback, dict) or (
            not str(playback.get("reviewer_id") or "").strip()
            or playback.get("reviewer_role") != "human"
            or playback.get("full_video_and_audio_playback") is not True
            or not str(playback.get("reviewed_at_utc") or "").strip()
        ):
            reject(source, "independent_full_audio_video_playback_evidence_missing")
        source_hash = str(item.get("certified_proxy_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            reject(source, "certified_proxy_sha256_missing")
        for anchor in item.get("prior_campaign_verified_cinematic_anchors") or []:
            if anchor.get("independent_re_review") != "VERIFIED":
                reject(source, "previously_verified_cinematic_anchor_not_re_reviewed")
        raw_rate = str(item.get("nominal_frame_rate") or "")
        try:
            rate = Fraction(raw_rate)
        except (ValueError, ZeroDivisionError):
            rate = Fraction(0)
        if rate <= 0:
            reject(source, "certified_source_frame_rate_missing")
        annotations = item.get("adjudicated_source_frame_annotations") or []
        if not isinstance(annotations, list) or not annotations:
            reject(source, "no_independently_adjudicated_source_events")
            continue
        allowed = set(item.get("required_review_categories") or [])
        for annotation in annotations:
            if not isinstance(annotation, dict):
                reject(source, "malformed_adjudication")
                continue
            category = str(annotation.get("category") or "")
            label = str(annotation.get("adjudication") or "")
            frame = annotation.get("source_frame_index")
            time = annotation.get("source_time_seconds")
            if (
                category not in allowed
                or label not in {"PRESENT", "ABSENT"}
                or isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame < 0
                or isinstance(time, bool)
                or not isinstance(time, int | float)
                or not math.isfinite(time)
                or time < 0
                or rate <= 0
                or abs(frame - time * float(rate)) > 0.51
                or annotation.get("reviewer_id") != playback.get("reviewer_id")
                or not str(annotation.get("source_evidence") or "").strip()
            ):
                reject(source, "invalid_source_frame_adjudication")
            else:
                counts[f"{label}:{category}"] += 1
    if not any(k.startswith("PRESENT:") for k in counts):
        reject("*", "no_positive_source_gold_examples")
    if not any(k.startswith("ABSENT:") for k in counts):
        reject("*", "no_negative_source_gold_examples")
    report = {
        "schema_version": 1,
        "execution_sha": build_identity.current_execution_sha(),
        "status": "PASS" if not blockers else "REJECTED",
        "source_count": len(actual_keys),
        "expected_sources": sorted(expected_assets),
        "adjudications": dict(sorted(counts.items())),
        "blockers": blockers,
        "independent_gold_required": True,
        "model_proposals_are_not_gold": True,
    }
    if blockers:
        raise AssertionError(json.dumps(report, sort_keys=True))
    return report

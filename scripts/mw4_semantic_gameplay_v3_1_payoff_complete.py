from __future__ import annotations

from collections import defaultdict
from typing import Any

import mw4_semantic_gameplay_v3_1_source_span_proposals as canonical
import mw4_semantic_gameplay_v3_1_quality as quality

SemanticPlanV31 = canonical.SemanticPlanV31
EditSegment = canonical.EditSegment
_EPS = 1e-3

plan_integrity_violations = canonical.plan_integrity_violations
finishing_open_plans = canonical.finishing_open_plans


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _verified_payoff_anchors(timeline: Any, config: dict[str, Any]) -> tuple[tuple[int, Any], ...]:
    seen: set[tuple[int, float]] = set()
    anchors: list[tuple[int, Any]] = []
    for engagement in timeline.engagements:
        shot_index = int(engagement.shot_index)
        for event in quality.verified_payoff_events(engagement, config):
            key = (shot_index, round(float(event.time), 3))
            if key in seen:
                continue
            seen.add(key)
            anchors.append((shot_index, event))
    return tuple(sorted(anchors, key=lambda item: float(item[1].time)))


def _anchor_variants(
    timeline: Any,
    shot_index: int,
    anchor_time: float,
    config: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    editor = config["semantic_editor"]
    minimum = _f(editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(editor.get("maximum_output_seconds", 20.0), 20.0)
    preferred = min(maximum, _f(editor.get("preferred_output_seconds", 12.5), 12.5))
    preferred_tail = _f(editor["ending"].get("preferred_payoff_tail_seconds", 0.45), 0.45)
    maximum_tail = _f(editor["ending"].get("maximum_payoff_tail_seconds", 0.75), 0.75)
    shot = timeline.shots[shot_index]
    shot_start, shot_end = float(shot.start), float(shot.end)

    candidates: set[tuple[float, float]] = set()
    end_seeds = {
        min(shot_end, anchor_time + preferred_tail),
        min(shot_end, anchor_time + maximum_tail),
        min(shot_end, anchor_time + 1.25),
        min(shot_end, anchor_time + 2.0),
    }
    for target in (minimum, preferred, maximum):
        for end_seed in end_seeds:
            start = max(shot_start, end_seed - target)
            end = min(shot_end, start + target)
            if end - start < minimum - _EPS:
                continue
            if not (start - _EPS <= anchor_time <= end + _EPS):
                continue
            candidates.add((round(start, 3), round(end, 3)))

    action_start = max(shot_start, anchor_time - 0.30)
    for target in (minimum, preferred):
        end = min(shot_end, action_start + target)
        if end - action_start >= minimum - _EPS:
            candidates.add((round(action_start, 3), round(end, 3)))

    ordered = sorted(
        candidates,
        key=lambda item: (
            abs((item[1] - item[0]) - preferred),
            abs(item[1] - (anchor_time + preferred_tail)),
            item[0],
        ),
    )
    return tuple(ordered[:12])


def _plan_for_span(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
    shot_index: int,
    start: float,
    end: float,
    anchor_time: float,
) -> SemanticPlanV31 | None:
    minimum = _f(config["semantic_editor"].get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(config["semantic_editor"].get("maximum_output_seconds", 20.0), 20.0)
    output_duration = end - start
    if not (minimum - _EPS <= output_duration <= maximum + _EPS):
        return None
    if canonical.core._intersects_excluded(start, end, excluded):
        return None
    if base._protected_finishing_overlap(timeline, start, end, config):
        return None

    proposal = canonical._span_engagement(timeline, shot_index, start, end, config)
    if proposal is None:
        return None
    payoff_times = {round(float(item.time), 3) for item in quality.verified_payoff_events(proposal, config)}
    if round(anchor_time, 3) not in payoff_times:
        return None

    metrics = canonical._span_metrics(timeline, proposal, config)
    evidence = canonical._span_evidence(timeline, shot_index, start, end) or (proposal,)
    story, profile, effects, reasons = quality._route_story(timeline, evidence)
    if not quality._passes_story_gates(
        story,
        metrics["opening"],
        metrics["ending"],
        metrics["retention"],
        metrics["payoff"],
        metrics["weakest"],
        metrics["low_fraction"],
        metrics["residual"],
        config,
    ):
        return None

    segment = EditSegment(round(start, 3), round(end, 3), 1.0, "verified_combat_story")
    plan = SemanticPlanV31(
        segment.start,
        segment.end,
        round(segment.end - segment.start, 3),
        round(output_duration, 3),
        round(
            quality._score_plan(
                metrics["retention"],
                metrics["payoff"],
                metrics["opening"],
                metrics["ending"],
                metrics["coherence"],
                metrics["weakest"],
                False,
                config,
            ),
            5,
        ),
        round(metrics["retention"], 4),
        round(metrics["payoff"], 4),
        round(metrics["opening"], 4),
        round(metrics["ending"], 4),
        round(metrics["coherence"], 4),
        round(metrics["weakest"], 4),
        round(metrics["low_fraction"], 4),
        round(metrics["residual"], 3),
        story,
        profile,
        (segment,),
        effects,
        evidence,
        None,
        reasons
        + (
            f"terminal verified payoff anchor {anchor_time:.3f}s receives an independent same-shot proposal search",
            "all unchanged source-integrity and story-quality gates still apply",
        ),
    )
    if plan_integrity_violations(plan, timeline, config, source_key):
        return None
    return plan


def _anchor_plans(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    anchors = _verified_payoff_anchors(timeline, config)
    qualified: dict[float, list[SemanticPlanV31]] = defaultdict(list)
    attempted = 0
    for shot_index, event in anchors:
        anchor_time = round(float(event.time), 3)
        for start, end in _anchor_variants(timeline, shot_index, anchor_time, config):
            attempted += 1
            plan = _plan_for_span(
                base,
                timeline,
                config,
                excluded,
                source_key,
                shot_index,
                start,
                end,
                anchor_time,
            )
            if plan is not None:
                qualified[anchor_time].append(plan)

    selected: list[SemanticPlanV31] = []
    qualified_anchor_count = 0
    for anchor_time in sorted(qualified):
        unique: list[SemanticPlanV31] = []
        seen: set[tuple[float, float]] = set()
        for plan in sorted(qualified[anchor_time], key=lambda item: item.score, reverse=True):
            key = (round(float(plan.segments[0].start), 3), round(float(plan.segments[0].end), 3))
            if key in seen:
                continue
            seen.add(key)
            unique.append(plan)
        if unique:
            qualified_anchor_count += 1
            selected.extend(unique[:3])

    existing = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    existing.update(
        {
            "verified_payoff_anchor_count": len(anchors),
            "payoff_anchor_span_variant_count": attempted,
            "payoff_anchor_qualified_anchor_count": qualified_anchor_count,
            "payoff_anchor_unrepresented_count": len(anchors) - qualified_anchor_count,
            "payoff_anchor_independent_search": True,
        }
    )
    setattr(timeline, "_proposal_diagnostics", existing)
    return selected


def normal_plans(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    anchored = _anchor_plans(base, timeline, config, excluded, source_key)
    legacy = canonical.normal_plans(base, timeline, config, excluded, source_key)
    merged: list[SemanticPlanV31] = []
    seen: set[tuple[float, float]] = set()
    for plan in sorted(anchored + legacy, key=lambda item: item.score, reverse=True):
        key = (round(float(plan.segments[0].start), 3), round(float(plan.segments[-1].end), 3))
        if key in seen:
            continue
        seen.add(key)
        merged.append(plan)
    return merged


def _one_per_finishing_move(plans: list[SemanticPlanV31]) -> list[SemanticPlanV31]:
    best: dict[tuple[float, float, float], SemanticPlanV31] = {}
    for plan in plans:
        span = plan.finishing_move
        if span is None:
            continue
        key = (round(float(span.start), 3), round(float(span.payoff), 3), round(float(span.end), 3))
        if key not in best or plan.score > best[key].score:
            best[key] = plan
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def build_plans_for_source(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    base.validate_configuration(config)
    normal = normal_plans(base, timeline, config, excluded, source_key)
    finishing = _one_per_finishing_move(
        canonical.finishing_open_plans(base, timeline, config, excluded, source_key)
    )
    plans = sorted(finishing + normal, key=lambda item: item.score, reverse=True)
    diagnostics = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    diagnostics.update(
        {
            "proposal_architecture": "every_verified_payoff_anchor_to_same_shot_quality_gated_source_span",
            "finishing_move_variants_collapsed_per_verified_move": True,
            "support_components_are_not_final_clip_bounds": True,
            "semantic_gap_is_not_source_cut": True,
            "semantic_montage_enabled": False,
            "fallback_planner_enabled": False,
            "threshold_reduction_used": False,
            "qualified_plan_count": len(plans),
        }
    )
    setattr(timeline, "_proposal_diagnostics", diagnostics)
    return plans


def self_test(base: Any) -> None:
    canonical.self_test(base)

    class Shot:
        start = 0.0
        end = 30.0

    class Timeline:
        shots = [Shot()]

    config = {
        "semantic_editor": {
            "minimum_output_seconds": 10.0,
            "maximum_output_seconds": 20.0,
            "preferred_output_seconds": 12.5,
            "ending": {"preferred_payoff_tail_seconds": 0.45, "maximum_payoff_tail_seconds": 0.75},
        }
    }
    variants = _anchor_variants(Timeline(), 0, 10.0, config)
    if not variants or not any(start <= 10.0 <= end and end - start >= 10.0 - _EPS for start, end in variants):
        raise AssertionError("verified payoff anchor did not receive an independent campaign-length proposal")
    print("MW4 payoff-complete same-shot proposal self-test: PASS")

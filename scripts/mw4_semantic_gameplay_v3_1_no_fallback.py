from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

import mw4_semantic_gameplay_v3_1_final_legacy as legacy_module


PAYOFF_KINDS = {"outcome_like", "impact"}
FORBIDDEN_STORY_TYPES = {"semantic_montage"}
_EPS = 1e-3


@dataclass(frozen=True)
class StrictHostileDecision:
    hostile: bool
    score: float
    reason: str
    actor_state: str = "unknown"


@dataclass(frozen=True)
class StrictBodyMoment:
    segment: Any
    engagement: Any
    effect_events: tuple[Any, ...]
    retention_quality: float
    payoff_quality: float
    opening_quality: float
    ending_quality: float
    story_coherence: float
    weakest_quarter_interest: float
    low_interest_fraction: float
    max_unexplained_low_interest_run_seconds: float
    score: float


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def assert_policy(config: dict[str, Any]) -> None:
    """Abort before analysis/planning if any fallback route is enabled."""
    errors: list[str] = []
    policy = config.get("fallback_policy", {})
    if policy.get("enabled") is not False:
        errors.append("fallback_policy.enabled must be false")

    editorial = config.get("editorial", {})
    if bool(editorial.get("finishing_move_allow_semantic_montage_continuation", False)):
        errors.append("semantic-montage Finishing Move continuation is forbidden")

    source_integrity = config.get("source_integrity", {})
    if bool(source_integrity.get("allow_planned_transition_for_semantic_montage", False)):
        errors.append("semantic-montage source bridges are forbidden")

    montage = config.get("semantic_editor", {}).get("semantic_montage", {})
    if bool(montage.get("enabled", False)):
        errors.append("semantic_montage fallback must be disabled")

    detector = config.get("finishing_move_detector", {})
    if bool(detector.get("allow_unverified_automatic", False)):
        errors.append("unverified automatic Finishing Move acceptance is forbidden")

    verifier = config.get("combat_state_verifier", {})
    local = verifier.get("local_interaction_verifier", {})
    if not bool(verifier.get("enabled", False)):
        errors.append("combat-state verifier must be enabled")
    if not bool(local.get("enabled", False)):
        errors.append("local direct-interaction verifier must be enabled")

    continuation = verifier.get("finishing_continuation", {})
    if continuation.get("require_verified_payoff") is not True:
        errors.append("Finishing Move continuation must require a verified payoff")
    forbidden_legacy_keys = {
        "minimum_verified_hostile_anchors_without_payoff",
        "minimum_retention_without_payoff",
    }
    present = sorted(forbidden_legacy_keys.intersection(continuation))
    if present:
        errors.append("no-payoff continuation fallback keys are forbidden: " + ", ".join(present))

    if errors:
        raise RuntimeError("MW4 V3.1 no-fallback policy violation: " + "; ".join(errors))


def _local_confirmation(evidence: dict[str, Any], config: dict[str, Any]) -> tuple[bool, float]:
    local_cfg = config["combat_state_verifier"]["local_interaction_verifier"]
    minimum = _f(local_cfg.get("minimum_hitmarker_score", 0.34), 0.34)
    attempted = _f(evidence.get("local_refine_attempted")) >= 0.5
    score = _f(evidence.get("local_hitmarker_score"))
    return bool(attempted and score >= minimum), score


def strict_event_hostile_decision(event: Any, config: dict[str, Any]) -> StrictHostileDecision:
    """Hostile requires local direct-interaction confirmation in every case."""
    assert_policy(config)
    cfg = config["combat_state_verifier"]
    kinds = set(getattr(event, "kinds", ()) or ())
    evidence = dict(getattr(event, "evidence", {}) or {})
    confidence = _f(getattr(event, "confidence", 0.0))
    combat = _f(evidence.get("combat"))
    outcome = _f(evidence.get("outcome"))
    impact = _f(evidence.get("impact"))
    audio_transient = _f(evidence.get("audio_transient"))
    center_motion = _f(evidence.get("center_motion"))
    local_ok, local_score = _local_confirmation(evidence, config)

    if not local_ok:
        return StrictHostileDecision(False, local_score, "missing_direct_interaction_confirmation", "unknown")

    min_score = _f(cfg.get("minimum_hostile_event_score", 0.64), 0.64)
    outcome_min = _f(cfg.get("outcome_minimum", 0.52), 0.52)
    payoff_combat_min = _f(cfg.get("payoff_combat_minimum", 0.42), 0.42)
    impact_min = _f(cfg.get("impact_minimum", 0.64), 0.64)
    impact_combat_min = _f(cfg.get("impact_combat_minimum", 0.55), 0.55)
    burst_min = _f(cfg.get("strong_combat_minimum", 0.70), 0.70)
    burst_audio_min = _f(cfg.get("strong_combat_audio_transient_minimum", 0.62), 0.62)
    burst_center_min = _f(cfg.get("strong_combat_center_motion_minimum", 0.40), 0.40)

    if "outcome_like" in kinds and outcome >= outcome_min and max(combat, impact) >= payoff_combat_min:
        score = min(
            1.0,
            0.44 * max(confidence, outcome)
            + 0.30 * max(combat, impact)
            + 0.14 * max(audio_transient, center_motion)
            + 0.12 * local_score,
        )
        hostile = score >= min_score
        return StrictHostileDecision(hostile, score, "directly_confirmed_outcome", "hostile" if hostile else "unknown")

    if "impact" in kinds and impact >= impact_min and combat >= impact_combat_min:
        score = min(
            1.0,
            0.40 * max(confidence, impact)
            + 0.32 * combat
            + 0.16 * max(audio_transient, center_motion)
            + 0.12 * local_score,
        )
        hostile = score >= min_score
        return StrictHostileDecision(hostile, score, "directly_confirmed_impact", "hostile" if hostile else "unknown")

    if (
        "combat_burst" in kinds
        and combat >= burst_min
        and audio_transient >= burst_audio_min
        and center_motion >= burst_center_min
    ):
        score = min(
            1.0,
            0.40 * max(confidence, combat)
            + 0.22 * audio_transient
            + 0.18 * center_motion
            + 0.20 * local_score,
        )
        hostile = score >= min_score
        return StrictHostileDecision(hostile, score, "directly_confirmed_strong_combat", "hostile" if hostile else "unknown")

    return StrictHostileDecision(False, 0.0, "insufficient_hostile_evidence", "unknown")


def strict_continuation_failures(plan: Any, config: dict[str, Any]) -> list[str]:
    """Finishing Move bodies require an actual verified payoff; no alternate path."""
    assert_policy(config)
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    failures: list[str] = []
    min_retention = max(
        _f(config.get("performance_targets", {}).get("retention_quality_min", 0.36), 0.36),
        _f(cfg.get("minimum_retention_quality", 0.40), 0.40),
    )
    min_payoff = max(
        _f(config.get("performance_targets", {}).get("payoff_quality_min", 0.34), 0.34),
        _f(cfg.get("minimum_payoff_quality", 0.40), 0.40),
    )
    min_ending = _f(cfg.get("minimum_ending_quality", 0.45), 0.45)
    min_weakest = _f(cfg.get("minimum_weakest_quarter_interest", 0.30), 0.30)
    max_residual = _f(cfg.get("maximum_unexplained_low_interest_run_seconds", 0.75), 0.75)

    hostile_events = [
        event
        for engagement in getattr(plan, "engagements", ())
        for event in getattr(engagement, "events", ())
    ]
    has_verified_payoff = any(
        any(kind in PAYOFF_KINDS for kind in getattr(event, "kinds", ()))
        for event in hostile_events
    )

    if not has_verified_payoff:
        failures.append("continuation lacks a verified hostile payoff")
    if _f(plan.retention_quality) < min_retention:
        failures.append("continuation retention below independent floor")
    if _f(plan.payoff_quality) < min_payoff:
        failures.append("continuation payoff below independent floor")
    if _f(plan.ending_quality) < min_ending:
        failures.append("continuation ending below independent floor")
    if _f(plan.weakest_quarter_interest) < min_weakest:
        failures.append("continuation weakest quarter below independent floor")
    if _f(plan.max_unexplained_low_interest_run_seconds) > max_residual:
        failures.append("continuation contains excessive unexplained inactivity")
    if any(getattr(segment, "reason", "") == "compressed_traversal" for segment in getattr(plan, "segments", ())):
        failures.append("continuation relies on compressed traversal")
    return failures


def _longest_true_run(mask: np.ndarray, fps: float) -> float:
    longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / fps


def _residual_for_segment(legacy: Any, timeline: Any, engagement: Any, start: float, end: float, config: dict[str, Any]) -> float:
    fps = float(timeline.fps)
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(timeline.times), int(math.ceil(end * fps)))
    if i1 <= i0:
        return float("inf")
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    explained = legacy._explained_mask(timeline, (engagement,), None, config)
    low = np.asarray(timeline.signals["interest"] < low_threshold, dtype=bool)
    return _longest_true_run((low & ~explained)[i0:i1], fps)


def _strict_body_moment(
    legacy: Any,
    combat: Any,
    islands: Any,
    timeline: Any,
    engagement: Any,
    config: dict[str, Any],
) -> tuple[StrictBodyMoment | None, list[str]]:
    failures: list[str] = []
    if not engagement.events:
        return None, ["empty_verified_combat_island"]

    payoff_events = [
        event
        for event in engagement.events
        if any(kind in PAYOFF_KINDS for kind in getattr(event, "kinds", ()))
    ]
    if not payoff_events:
        return None, ["verified_combat_island_has_no_payoff"]

    shot = timeline.shots[engagement.shot_index]
    tail = float(config["semantic_editor"]["ending"].get("preferred_payoff_tail_seconds", 0.45))
    first_event = float(engagement.events[0].time)
    start = max(float(shot.start), float(engagement.start), first_event - 0.30)
    end = min(float(shot.end), float(engagement.end), float(payoff_events[-1].time) + tail)
    if end <= start + _EPS:
        return None, ["invalid_verified_combat_island_bounds"]

    segment = legacy.EditSegment(round(start, 3), round(end, 3), 1.0, "verified_combat_island_body")
    duration = legacy._segment_duration(segment)
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    minimum_moment = _f(cfg.get("minimum_verified_island_moment_seconds", 1.0), 1.0)
    maximum_moment = _f(cfg.get("maximum_verified_island_moment_seconds", 6.5), 6.5)
    if duration < minimum_moment - _EPS:
        failures.append("verified_combat_island_too_short")
    if duration > maximum_moment + _EPS:
        failures.append("verified_combat_island_too_long")

    residual = _residual_for_segment(legacy, timeline, engagement, start, end, config)
    opening, ending, coherence, retention, payoff, weakest, low_fraction = legacy._quality_metrics(
        timeline, start, end, (engagement,), None, residual, config
    )
    opening_floor = _f(
        cfg.get("minimum_verified_island_opening_quality", config["semantic_editor"]["opening"].get("minimum_quality", 0.42)),
        0.42,
    )
    if opening < opening_floor:
        failures.append("verified_combat_island_opening_below_floor")

    effect_events = tuple(sorted(
        legacy.refined.core._raw_effect_events(timeline, (engagement,)),
        key=lambda item: item.time,
    ))
    probe = legacy.SemanticPlanV31(
        start=segment.start,
        end=segment.end,
        raw_duration=round(segment.end - segment.start, 3),
        output_duration=round(duration, 3),
        score=round(legacy._score_plan(retention, payoff, opening, ending, coherence, weakest, False, config), 5),
        retention_quality=round(retention, 4),
        payoff_quality=round(payoff, 4),
        opening_quality=round(opening, 4),
        ending_quality=round(ending, 4),
        story_coherence=round(coherence, 4),
        weakest_quarter_interest=round(weakest, 4),
        low_interest_fraction=round(low_fraction, 4),
        max_unexplained_low_interest_run_seconds=round(residual, 3),
        story_type="verified_combat_island_continuation",
        effect_profile="clean_pressure",
        segments=(segment,),
        effect_events=effect_events,
        engagements=(engagement,),
        finishing_move=None,
        editorial_reasons=("strict verified payoff combat island",),
    )
    failures.extend(combat.continuation_failures(probe, config))
    failures.extend(islands.plan_island_failures(probe, timeline, config))
    failures = list(dict.fromkeys(failures))
    if failures:
        return None, failures

    return StrictBodyMoment(
        segment=segment,
        engagement=engagement,
        effect_events=effect_events,
        retention_quality=float(retention),
        payoff_quality=float(payoff),
        opening_quality=float(opening),
        ending_quality=float(ending),
        story_coherence=float(coherence),
        weakest_quarter_interest=float(weakest),
        low_interest_fraction=float(low_fraction),
        max_unexplained_low_interest_run_seconds=float(residual),
        score=float(probe.score),
    ), []


def _assemble_body_group(
    legacy: Any,
    combat: Any,
    islands: Any,
    timeline: Any,
    group: tuple[StrictBodyMoment, ...],
    config: dict[str, Any],
    body_minimum: float,
    body_maximum: float,
    maximum_gap: float,
) -> tuple[Any | None, list[str]]:
    segments = tuple(item.segment for item in group)
    if any(float(right.start) < float(left.end) - _EPS for left, right in zip(segments, segments[1:])):
        return None, ["verified_body_segments_overlap_or_reverse"]
    if any(float(right.start) - float(left.end) > maximum_gap + _EPS for left, right in zip(segments, segments[1:])):
        return None, ["verified_body_segment_gap_exceeds_contract"]

    output_duration = sum(legacy._segment_duration(segment) for segment in segments)
    if output_duration < body_minimum - _EPS:
        return None, ["verified_body_too_short"]
    if output_duration > body_maximum + _EPS:
        return None, ["verified_body_too_long"]

    durations = np.asarray([legacy._segment_duration(item.segment) for item in group], dtype=float)
    retention = float(np.average([item.retention_quality for item in group], weights=durations))
    payoff_values = [float(item.payoff_quality) for item in group]
    payoff = float(0.55 * max(payoff_values) + 0.45 * np.mean(payoff_values))
    opening = float(group[0].opening_quality)
    ending = float(group[-1].ending_quality)
    coherence = float(np.average([item.story_coherence for item in group], weights=durations))
    weakest = float(min(item.weakest_quarter_interest for item in group))
    low_fraction = float(np.average([item.low_interest_fraction for item in group], weights=durations))
    residual = float(max(item.max_unexplained_low_interest_run_seconds for item in group))

    effect_events = tuple(sorted(
        [max(item.effect_events, key=lambda event: event.confidence) for item in group if item.effect_events],
        key=lambda event: event.time,
    ))
    plan = legacy.SemanticPlanV31(
        start=segments[0].start,
        end=segments[-1].end,
        raw_duration=round(sum(segment.end - segment.start for segment in segments), 3),
        output_duration=round(output_duration, 3),
        score=round(legacy._score_plan(retention, payoff, opening, ending, coherence, weakest, False, config), 5),
        retention_quality=round(retention, 4),
        payoff_quality=round(payoff, 4),
        opening_quality=round(opening, 4),
        ending_quality=round(ending, 4),
        story_coherence=round(coherence, 4),
        weakest_quarter_interest=round(weakest, 4),
        low_interest_fraction=round(low_fraction, 4),
        max_unexplained_low_interest_run_seconds=round(residual, 3),
        story_type="verified_combat_island_continuation",
        effect_profile="clean_pressure",
        segments=segments,
        effect_events=effect_events,
        engagements=tuple(item.engagement for item in group),
        finishing_move=None,
        editorial_reasons=(
            "strict verified-payoff combat-island body",
            "hard cuts remove reload/search gaps without alternate continuation paths",
        ),
    )
    failures = combat.continuation_failures(plan, config) + islands.plan_island_failures(plan, timeline, config)
    failures = list(dict.fromkeys(failures))
    return (None, failures) if failures else (plan, [])


def strict_build_continuations(
    legacy: Any,
    combat: Any,
    islands: Any,
    timeline: Any,
    span: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> tuple[list[Any], dict[str, Any]]:
    assert_policy(config)
    editorial = config["editorial"]
    shot = timeline.shots[span.shot_index]
    lead = _f(editorial.get("finishing_move_opening_lead_seconds", 0.28), 0.28)
    hold = _f(editorial.get("finishing_move_payoff_hold_seconds", 0.45), 0.45)
    hero_start = max(float(shot.start), float(span.start) - lead)
    hero_end = min(float(shot.end), float(span.end) + hold)
    hero_duration = max(0.0, hero_end - hero_start)
    final_minimum = _f(config["semantic_editor"].get("minimum_output_seconds", 10.0), 10.0)
    final_maximum = min(
        _f(config["semantic_editor"].get("maximum_output_seconds", 20.0), 20.0),
        _f(editorial.get("finishing_move_montage_max_output_seconds", 15.5), 15.5),
    )
    body_minimum = max(0.0, final_minimum - hero_duration)
    body_maximum = max(0.0, final_maximum - hero_duration)
    maximum_gap = _f(editorial.get("finishing_move_max_continuation_gap_seconds", 18.0), 18.0)
    later_floor = max(hero_end, float(span.end) + 0.25)
    latest_first_start = hero_end + maximum_gap

    eligible_islands = [
        engagement for engagement in timeline.engagements
        if engagement.events
        and float(engagement.start) >= later_floor - _EPS
        and float(engagement.start) <= latest_first_start + _EPS
        and not legacy.refined.core._intersects_excluded(float(engagement.start), float(engagement.end), excluded)
    ]

    moments: list[StrictBodyMoment] = []
    island_diagnostics: list[dict[str, Any]] = []
    for engagement in eligible_islands:
        moment, failures = _strict_body_moment(legacy, combat, islands, timeline, engagement, config)
        island_diagnostics.append({
            "start": round(float(engagement.start), 3),
            "end": round(float(engagement.end), 3),
            "event_times": [round(float(event.time), 3) for event in engagement.events],
            "qualified": moment is not None,
            "failures": failures,
        })
        if moment is not None:
            moments.append(moment)

    maximum_moments = int(config["combat_state_verifier"]["finishing_continuation"].get("maximum_verified_island_moments_per_body", 3))
    candidates: list[Any] = []
    group_rejections: list[dict[str, Any]] = []
    for count in range(1, max(1, maximum_moments) + 1):
        for group in itertools.combinations(moments, count):
            candidate, failures = _assemble_body_group(
                legacy, combat, islands, timeline, group, config,
                body_minimum, body_maximum, maximum_gap,
            )
            if candidate is not None:
                candidates.append(candidate)
            elif failures:
                group_rejections.append({
                    "segments": [[round(float(item.segment.start), 3), round(float(item.segment.end), 3)] for item in group],
                    "failures": failures,
                })

    unique: list[Any] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        key = tuple((round(float(segment.start), 3), round(float(segment.end), 3)) for segment in candidate.segments)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return unique, {
        "span_start": round(float(span.start), 3),
        "span_end": round(float(span.end), 3),
        "hero_start": round(hero_start, 3),
        "hero_end": round(hero_end, 3),
        "hero_output_duration": round(hero_duration, 3),
        "required_body_minimum_seconds": round(body_minimum, 3),
        "allowed_body_maximum_seconds": round(body_maximum, 3),
        "later_floor": round(later_floor, 3),
        "latest_first_start": round(latest_first_start, 3),
        "eligible_combat_islands": len(eligible_islands),
        "individually_qualified_combat_islands": len(moments),
        "dedicated_continuation_count": len(unique),
        "island_diagnostics": island_diagnostics,
        "group_rejections": group_rejections[:12],
        "fallback_execution_allowed": False,
        "policy": "verified local-interaction hostile payoff islands only; no alternate continuation path",
    }


def strict_finishing_assembler(
    timeline: Any,
    continuations: list[Any],
    config: dict[str, Any],
    source_key: str,
) -> list[Any]:
    """Assemble Finishing Move clips only from strict verified-island bodies."""
    assert_policy(config)
    legacy = legacy_module
    if not timeline.finishing_moves:
        return []
    if any(getattr(item, "story_type", "") != "verified_combat_island_continuation" for item in continuations):
        raise RuntimeError("MW4 V3.1 no-fallback policy: non-verified continuation supplied to Finishing Move assembler")

    editorial = config["editorial"]
    minimum = _f(config["semantic_editor"].get("minimum_output_seconds", 10.0), 10.0)
    maximum = min(
        _f(config["semantic_editor"].get("maximum_output_seconds", 20.0), 20.0),
        _f(editorial.get("finishing_move_montage_max_output_seconds", 15.5), 15.5),
    )
    lead = _f(editorial.get("finishing_move_opening_lead_seconds", 0.28), 0.28)
    hold = _f(editorial.get("finishing_move_payoff_hold_seconds", 0.45), 0.45)
    max_gap = _f(editorial.get("finishing_move_max_continuation_gap_seconds", 18.0), 18.0)
    results: list[Any] = []

    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero = legacy.EditSegment(
            round(max(float(shot.start), float(span.start) - lead), 3),
            round(min(float(shot.end), float(span.end) + hold), 3),
            1.0,
            "finishing_move_open_hero",
        )
        hero_duration = legacy._segment_duration(hero)
        later_floor = max(float(hero.end), float(span.end) + 0.25)

        for continuation in continuations:
            if not continuation.segments:
                continue
            first_start = float(continuation.segments[0].start)
            if first_start < later_floor - _EPS:
                continue
            if first_start - float(hero.end) > max_gap + _EPS:
                continue
            if any(max(float(hero.start), float(seg.start)) < min(float(hero.end), float(seg.end)) for seg in continuation.segments):
                continue
            if strict_continuation_failures(continuation, config):
                continue

            segments = (hero,) + tuple(continuation.segments)
            output_duration = sum(legacy._segment_duration(segment) for segment in segments)
            if not (minimum <= output_duration <= maximum):
                continue

            opening = max(0.90, min(1.0, 0.74 + 0.22 * float(span.confidence)))
            ending = float(continuation.ending_quality)
            coherence = min(1.0, 0.10 + 0.88 * float(continuation.story_coherence))
            retention = float(0.18 * opening + 0.82 * float(continuation.retention_quality))
            hero_payoff = min(1.0, float(span.confidence) + 0.12)
            payoff = float(0.45 * hero_payoff + 0.55 * float(continuation.payoff_quality))
            weakest = float(continuation.weakest_quarter_interest)
            low_fraction = float(continuation.low_interest_fraction) * (float(continuation.output_duration) / output_duration)
            residual = float(continuation.max_unexplained_low_interest_run_seconds)

            if not legacy._passes_story_gates(
                "finishing_move_open", opening, ending, retention, payoff,
                weakest, low_fraction, residual, config,
            ):
                continue

            plan = legacy.SemanticPlanV31(
                start=hero.start,
                end=continuation.end,
                raw_duration=round(sum(segment.end - segment.start for segment in segments), 3),
                output_duration=round(output_duration, 3),
                score=round(legacy._score_plan(retention, payoff, opening, ending, coherence, weakest, True, config), 5),
                retention_quality=round(retention, 4),
                payoff_quality=round(payoff, 4),
                opening_quality=round(opening, 4),
                ending_quality=round(ending, 4),
                story_coherence=round(coherence, 4),
                weakest_quarter_interest=round(weakest, 4),
                low_interest_fraction=round(low_fraction, 4),
                max_unexplained_low_interest_run_seconds=round(residual, 3),
                story_type="finishing_move_open",
                effect_profile="finishing_move_hero",
                segments=segments,
                effect_events=continuation.effect_events,
                engagements=continuation.engagements,
                finishing_move=span,
                editorial_reasons=(
                    "verified Finishing Move opens the clip",
                    "continuation contains only strict verified-payoff combat islands",
                    "no semantic or no-payoff fallback path exists",
                ) + tuple(continuation.editorial_reasons),
            )
            if not legacy.plan_integrity_violations(plan, timeline, config, source_key):
                results.append(plan)

    return sorted(results, key=lambda item: item.score, reverse=True)


def _forbidden_semantic_montage(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("MW4 V3.1 no-fallback policy: semantic montage execution is forbidden")


def install(legacy: Any, combat: Any, islands: Any, finishing_body: Any) -> None:
    if bool(getattr(legacy, "_mw4_no_fallback_installed", False)):
        return

    original_analyze = legacy.analyze_source
    original_select = legacy.select_plans

    def analyze_source(source: Any, config: dict[str, Any]) -> Any:
        assert_policy(config)
        return original_analyze(source, config)

    def build_plans_for_source(timeline: Any, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> list[Any]:
        assert_policy(config)
        normal = legacy._normal_plans(timeline, config, excluded, source_key)
        if any(getattr(plan, "story_type", "") in FORBIDDEN_STORY_TYPES for plan in normal):
            raise RuntimeError("MW4 V3.1 no-fallback policy: forbidden story leaked from normal planner")
        finishing = legacy._finishing_open_plans(timeline, [], config, source_key)
        plans = sorted(finishing + normal, key=lambda item: item.score, reverse=True)
        if any(getattr(plan, "story_type", "") in FORBIDDEN_STORY_TYPES for plan in plans):
            raise RuntimeError("MW4 V3.1 no-fallback policy: forbidden semantic montage plan produced")
        return plans

    def select_plans(plans: list[Any], config: dict[str, Any]) -> list[Any]:
        assert_policy(config)
        if any(getattr(plan, "story_type", "") in FORBIDDEN_STORY_TYPES for plan in plans):
            raise RuntimeError("MW4 V3.1 no-fallback policy: forbidden plan reached selection")
        return original_select(plans, config)

    combat._event_hostile_decision = strict_event_hostile_decision
    combat.continuation_failures = strict_continuation_failures
    combat.self_test = self_test
    finishing_body.build_continuations = strict_build_continuations
    finishing_body.self_test = finishing_body_self_test
    legacy._semantic_montage_plans = _forbidden_semantic_montage
    legacy._montage_moments = _forbidden_semantic_montage
    legacy.analyze_source = analyze_source
    legacy.build_plans_for_source = build_plans_for_source
    legacy.select_plans = select_plans
    setattr(legacy, "_mw4_no_fallback_installed", True)


def _test_config() -> dict[str, Any]:
    return {
        "fallback_policy": {"enabled": False},
        "editorial": {
            "finishing_move_allow_semantic_montage_continuation": False,
            "finishing_move_allow_verified_combat_island_continuation": True,
        },
        "source_integrity": {"allow_planned_transition_for_semantic_montage": False},
        "semantic_editor": {
            "semantic_montage": {"enabled": False},
            "opening": {"minimum_quality": 0.42},
        },
        "finishing_move_detector": {"allow_unverified_automatic": False},
        "combat_state_verifier": {
            "enabled": True,
            "minimum_hostile_event_score": 0.64,
            "outcome_minimum": 0.52,
            "payoff_combat_minimum": 0.42,
            "impact_minimum": 0.64,
            "impact_combat_minimum": 0.55,
            "strong_combat_minimum": 0.70,
            "strong_combat_audio_transient_minimum": 0.62,
            "strong_combat_center_motion_minimum": 0.40,
            "local_interaction_verifier": {"enabled": True, "minimum_hitmarker_score": 0.34},
            "finishing_continuation": {
                "require_verified_payoff": True,
                "minimum_retention_quality": 0.40,
                "minimum_payoff_quality": 0.40,
                "minimum_ending_quality": 0.45,
                "minimum_weakest_quarter_interest": 0.30,
                "maximum_unexplained_low_interest_run_seconds": 0.75,
                "minimum_verified_island_moment_seconds": 1.0,
                "maximum_verified_island_moment_seconds": 6.5,
                "maximum_verified_island_moments_per_body": 3,
            },
        },
        "performance_targets": {"retention_quality_min": 0.36, "payoff_quality_min": 0.34},
    }


def self_test() -> None:
    cfg = _test_config()
    assert_policy(cfg)

    bad_cfg = _test_config()
    bad_cfg["semantic_editor"]["semantic_montage"]["enabled"] = True
    try:
        assert_policy(bad_cfg)
    except RuntimeError:
        pass
    else:
        raise AssertionError("semantic montage fallback configuration did not fail closed")

    dual_without_local = SimpleNamespace(
        kinds=("combat_burst", "impact", "outcome_like"),
        confidence=0.90,
        evidence={
            "combat": 0.82,
            "impact": 0.82,
            "outcome": 0.80,
            "audio_transient": 0.90,
            "center_motion": 0.90,
            "local_refine_attempted": 1.0,
            "local_hitmarker_score": 0.0,
        },
    )
    if strict_event_hostile_decision(dual_without_local, cfg).hostile:
        raise AssertionError("dual-payoff event bypassed direct-interaction verification")

    direct_payoff = SimpleNamespace(
        kinds=("combat_burst", "impact", "outcome_like"),
        confidence=0.90,
        evidence={
            "combat": 0.82,
            "impact": 0.82,
            "outcome": 0.80,
            "audio_transient": 0.90,
            "center_motion": 0.90,
            "local_refine_attempted": 1.0,
            "local_hitmarker_score": 0.90,
        },
    )
    if not strict_event_hostile_decision(direct_payoff, cfg).hostile:
        raise AssertionError("directly verified hostile payoff was rejected")

    no_payoff_body = SimpleNamespace(
        retention_quality=0.90,
        payoff_quality=0.90,
        ending_quality=0.90,
        weakest_quarter_interest=0.90,
        max_unexplained_low_interest_run_seconds=0.0,
        engagements=(SimpleNamespace(events=(SimpleNamespace(kinds=("combat_burst",)), SimpleNamespace(kinds=("combat_burst",)))),),
        segments=(SimpleNamespace(reason="verified_combat_island_body"),),
    )
    failures = strict_continuation_failures(no_payoff_body, cfg)
    if "continuation lacks a verified hostile payoff" not in failures:
        raise AssertionError("no-payoff continuation fallback remained reachable")

    payoff_body = SimpleNamespace(
        retention_quality=0.90,
        payoff_quality=0.90,
        ending_quality=0.90,
        weakest_quarter_interest=0.90,
        max_unexplained_low_interest_run_seconds=0.0,
        engagements=(SimpleNamespace(events=(SimpleNamespace(kinds=("outcome_like",)),)),),
        segments=(SimpleNamespace(reason="verified_combat_island_body"),),
    )
    if strict_continuation_failures(payoff_body, cfg):
        raise AssertionError("strict verified-payoff continuation was rejected")

    print("MW4 no-fallback semantic policy self-test: PASS")


def finishing_body_self_test() -> None:
    cfg = _test_config()
    assert_policy(cfg)
    continuation = cfg["combat_state_verifier"]["finishing_continuation"]
    if continuation.get("require_verified_payoff") is not True:
        raise AssertionError("verified payoff requirement is not locked")
    if "minimum_verified_hostile_anchors_without_payoff" in continuation:
        raise AssertionError("no-payoff continuation key reappeared")
    print("MW4 strict Finishing Move body policy self-test: PASS")

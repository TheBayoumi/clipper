from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1e-3


@dataclass(frozen=True)
class _BodyMoment:
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


def duration_contract(legacy: Any, timeline: Any, span: Any, config: dict[str, Any]) -> tuple[Any, float, float, float]:
    editorial = config["editorial"]
    shot = timeline.shots[span.shot_index]
    lead = float(editorial.get("finishing_move_opening_lead_seconds", 0.28))
    hold = float(editorial.get("finishing_move_payoff_hold_seconds", 0.45))
    hero = legacy.EditSegment(
        round(max(float(shot.start), float(span.start) - lead), 3),
        round(min(float(shot.end), float(span.end) + hold), 3),
        1.0,
        "finishing_move_open_hero",
    )
    hero_duration = legacy._segment_duration(hero)
    final_minimum = float(config["semantic_editor"].get("minimum_output_seconds", 10.0))
    final_maximum = min(
        float(config["semantic_editor"].get("maximum_output_seconds", 20.0)),
        float(editorial.get("finishing_move_montage_max_output_seconds", 15.5)),
    )
    return (
        hero,
        hero_duration,
        max(0.0, final_minimum - hero_duration),
        max(0.0, final_maximum - hero_duration),
    )


def _residual_for_segment(legacy: Any, timeline: Any, engagement: Any, start: float, end: float, config: dict[str, Any]) -> float:
    fps = float(timeline.fps)
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(timeline.times), int(math.ceil(end * fps)))
    if i1 <= i0:
        return float("inf")
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    explained = legacy._explained_mask(timeline, (engagement,), None, config)
    low = np.asarray(timeline.signals["interest"] < low_threshold, dtype=bool)
    return float(legacy.refined.core._longest_true_run((low & ~explained)[i0:i1], fps))


def _moment_bounds(legacy: Any, timeline: Any, engagement: Any, config: dict[str, Any]) -> tuple[float, float] | None:
    if not engagement.events:
        return None
    shot = timeline.shots[engagement.shot_index]
    tail = float(config["semantic_editor"]["ending"].get("preferred_payoff_tail_seconds", 0.45))
    first_event = float(engagement.events[0].time)
    payoff_events = [
        event for event in engagement.events
        if any(kind in legacy.PAYOFF_KINDS for kind in getattr(event, "kinds", ()))
    ]
    start = max(float(shot.start), float(engagement.start), first_event - 0.30)
    end = (
        min(float(shot.end), float(engagement.end), float(payoff_events[-1].time) + tail)
        if payoff_events
        else min(float(shot.end), float(engagement.end))
    )
    return (start, end) if end > start + _EPS else None


def _island_moment(
    legacy: Any,
    combat: Any,
    islands: Any,
    timeline: Any,
    engagement: Any,
    config: dict[str, Any],
) -> tuple[_BodyMoment | None, list[str]]:
    failures: list[str] = []
    bounds = _moment_bounds(legacy, timeline, engagement, config)
    if bounds is None:
        return None, ["invalid_or_empty_island_bounds"]
    start, end = bounds
    segment = legacy.EditSegment(round(start, 3), round(end, 3), 1.0, "verified_combat_island_body")
    duration = legacy._segment_duration(segment)
    cfg = config["combat_state_verifier"].get("finishing_continuation", {})
    minimum_moment = float(cfg.get("minimum_verified_island_moment_seconds", 1.0))
    maximum_moment = float(config["semantic_editor"].get("semantic_montage", {}).get("maximum_moment_seconds", 6.5))
    if duration < minimum_moment - _EPS:
        failures.append("island_moment_too_short")
    if duration > maximum_moment + _EPS:
        failures.append("island_moment_too_long")

    residual = _residual_for_segment(legacy, timeline, engagement, start, end, config)
    opening, ending, coherence, retention, payoff, weakest, low_fraction = legacy._quality_metrics(
        timeline, start, end, (engagement,), None, residual, config
    )
    effect_events = tuple(sorted(
        legacy.refined.core._raw_effect_events(timeline, (engagement,)),
        key=lambda item: item.time,
    ))
    probe = legacy.SemanticPlanV31(
        start=segment.start,
        end=segment.end,
        raw_duration=round(segment.end - segment.start, 3),
        output_duration=round(duration, 3),
        score=round(legacy._score_plan(
            retention, payoff, opening, ending, coherence, weakest, False, config
        ), 5),
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
        editorial_reasons=("independently qualified verified combat island",),
    )

    opening_floor = float(
        config.get("story_quality_gates", {}).get("opening_min", {}).get(
            "semantic_montage",
            config["semantic_editor"].get("opening", {}).get("minimum_quality", 0.42),
        )
    )
    if opening < opening_floor:
        failures.append("island_opening_below_existing_floor")
    failures.extend(combat.continuation_failures(probe, config))
    failures.extend(islands.plan_island_failures(probe, timeline, config))
    failures = list(dict.fromkeys(failures))
    if failures:
        return None, failures
    return _BodyMoment(
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


def _single_island_candidate(
    legacy: Any,
    moment: _BodyMoment,
    body_minimum: float,
    body_maximum: float,
) -> Any | None:
    output_duration = legacy._segment_duration(moment.segment)
    if not (body_minimum - _EPS <= output_duration <= body_maximum + _EPS):
        return None
    return legacy.SemanticPlanV31(
        start=moment.segment.start,
        end=moment.segment.end,
        raw_duration=round(moment.segment.end - moment.segment.start, 3),
        output_duration=round(output_duration, 3),
        score=round(moment.score, 5),
        retention_quality=round(moment.retention_quality, 4),
        payoff_quality=round(moment.payoff_quality, 4),
        opening_quality=round(moment.opening_quality, 4),
        ending_quality=round(moment.ending_quality, 4),
        story_coherence=round(moment.story_coherence, 4),
        weakest_quarter_interest=round(moment.weakest_quarter_interest, 4),
        low_interest_fraction=round(moment.low_interest_fraction, 4),
        max_unexplained_low_interest_run_seconds=round(moment.max_unexplained_low_interest_run_seconds, 3),
        story_type="verified_combat_island_continuation",
        effect_profile="clean_pressure",
        segments=(moment.segment,),
        effect_events=moment.effect_events,
        engagements=(moment.engagement,),
        finishing_move=None,
        editorial_reasons=(
            "single independently qualified verified hostile combat island supplies the Finishing Move body",
            "body duration is evaluated against remaining final-clip duration rather than a standalone 10-second minimum",
            "reload/search combat-island breaks remain excluded",
        ),
    )


def _multi_island_candidate(
    legacy: Any,
    combat: Any,
    islands: Any,
    group: tuple[_BodyMoment, ...],
    timeline: Any,
    config: dict[str, Any],
    body_minimum: float,
    body_maximum: float,
    max_source_jump: float,
) -> tuple[Any | None, list[str]]:
    failures: list[str] = []
    segments = tuple(item.segment for item in group)
    if any(float(right.start) < float(left.end) - _EPS for left, right in zip(segments, segments[1:])):
        return None, ["body_moments_overlap_or_reverse"]
    if any(float(right.start) - float(left.end) > max_source_jump + _EPS for left, right in zip(segments, segments[1:])):
        return None, ["body_source_jump_exceeds_existing_gap_contract"]
    output_duration = sum(legacy._segment_duration(segment) for segment in segments)
    if output_duration < body_minimum - _EPS:
        return None, ["combined_body_too_short"]
    if output_duration > body_maximum + _EPS:
        return None, ["combined_body_too_long"]

    durations = np.asarray([legacy._segment_duration(item.segment) for item in group], dtype=float)
    if float(np.sum(durations)) <= 0.0:
        return None, ["combined_body_zero_duration"]
    retention = float(np.average([item.retention_quality for item in group], weights=durations))
    payoff_values = [float(item.payoff_quality) for item in group]
    payoff = float(0.55 * max(payoff_values) + 0.45 * np.mean(payoff_values))
    opening = float(group[0].opening_quality)
    ending = float(group[-1].ending_quality)
    coherence = float(np.average([item.story_coherence for item in group], weights=durations))
    weakest = float(min(item.weakest_quarter_interest for item in group))
    low_fraction = float(np.average([item.low_interest_fraction for item in group], weights=durations))
    residual = float(max(item.max_unexplained_low_interest_run_seconds for item in group))

    effect_events: list[Any] = []
    for item in group:
        if item.effect_events:
            effect_events.append(max(item.effect_events, key=lambda event: event.confidence))
    engagements = tuple(item.engagement for item in group)
    plan = legacy.SemanticPlanV31(
        start=segments[0].start,
        end=segments[-1].end,
        raw_duration=round(sum(segment.end - segment.start for segment in segments), 3),
        output_duration=round(output_duration, 3),
        score=round(legacy._score_plan(
            retention, payoff, opening, ending, coherence, weakest, False, config
        ), 5),
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
        effect_events=tuple(sorted(effect_events, key=lambda event: event.time)),
        engagements=engagements,
        finishing_move=None,
        editorial_reasons=(
            "dedicated Finishing Move body uses only independently qualified verified combat islands",
            "hard cuts skip reload/search gaps while generic semantic-montage continuation remains disabled",
            "body duration is evaluated against remaining final-clip duration rather than a standalone 10-second minimum",
        ),
    )
    failures.extend(combat.continuation_failures(plan, config))
    failures.extend(islands.plan_island_failures(plan, timeline, config))
    failures = list(dict.fromkeys(failures))
    return (None, failures) if failures else (plan, [])


def build_continuations(
    legacy: Any,
    combat: Any,
    islands: Any,
    timeline: Any,
    span: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> tuple[list[Any], dict[str, Any]]:
    hero, hero_duration, body_minimum, body_maximum = duration_contract(legacy, timeline, span, config)
    max_gap = float(config["editorial"].get("finishing_move_max_continuation_gap_seconds", 18.0))
    later_floor = max(float(hero.end), float(span.end) + 0.25)
    latest_first_start = float(hero.end) + max_gap
    candidates: list[Any] = []

    eligible_islands = [
        engagement for engagement in timeline.engagements
        if engagement.events
        and float(engagement.start) >= later_floor - _EPS
        and float(engagement.start) <= latest_first_start + _EPS
        and not legacy.refined.core._intersects_excluded(
            float(engagement.start), float(engagement.end), excluded
        )
    ]
    moments: list[_BodyMoment] = []
    island_diagnostics: list[dict[str, Any]] = []
    for engagement in eligible_islands:
        moment, failures = _island_moment(
            legacy, combat, islands, timeline, engagement, config
        )
        island_diagnostics.append({
            "start": round(float(engagement.start), 3),
            "end": round(float(engagement.end), 3),
            "event_times": [round(float(event.time), 3) for event in engagement.events],
            "qualified": moment is not None,
            "failures": failures,
            **({
                "moment_start": round(float(moment.segment.start), 3),
                "moment_end": round(float(moment.segment.end), 3),
                "moment_duration": round(float(legacy._segment_duration(moment.segment)), 3),
                "retention": round(float(moment.retention_quality), 4),
                "payoff": round(float(moment.payoff_quality), 4),
                "ending": round(float(moment.ending_quality), 4),
                "weakest": round(float(moment.weakest_quarter_interest), 4),
            } if moment is not None else {}),
        })
        if moment is not None:
            moments.append(moment)
            single = _single_island_candidate(legacy, moment, body_minimum, body_maximum)
            if single is not None:
                candidates.append(single)

    maximum_moments = int(
        config["semantic_editor"].get("semantic_montage", {}).get("maximum_moments_per_clip", 3)
    )
    group_rejections: list[dict[str, Any]] = []
    for count in range(2, max(2, maximum_moments) + 1):
        for group in itertools.combinations(moments, count):
            candidate, failures = _multi_island_candidate(
                legacy, combat, islands, group, timeline, config,
                body_minimum, body_maximum, max_gap,
            )
            if candidate is not None:
                candidates.append(candidate)
            elif failures:
                group_rejections.append({
                    "segments": [
                        [round(float(item.segment.start), 3), round(float(item.segment.end), 3)]
                        for item in group
                    ],
                    "failures": failures,
                })

    unique: list[Any] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        key = tuple(
            (round(float(segment.start), 3), round(float(segment.end), 3))
            for segment in candidate.segments
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return unique, {
        "span_start": round(float(span.start), 3),
        "span_end": round(float(span.end), 3),
        "hero_start": round(float(hero.start), 3),
        "hero_end": round(float(hero.end), 3),
        "hero_output_duration": round(float(hero_duration), 3),
        "required_body_minimum_seconds": round(float(body_minimum), 3),
        "allowed_body_maximum_seconds": round(float(body_maximum), 3),
        "later_floor": round(float(later_floor), 3),
        "latest_first_start": round(float(latest_first_start), 3),
        "excluded_window_count": len(excluded),
        "eligible_combat_islands": len(eligible_islands),
        "individually_qualified_combat_islands": len(moments),
        "dedicated_continuation_count": len(unique),
        "island_diagnostics": island_diagnostics,
        "group_rejections": group_rejections[:12],
        "policy": "verified combat islands only; each island independently passes continuation gates; body minimum applies to the assembled body, not each island; hostile/payoff-or-sustained-pressure/retention/reload-search/exclusion gates are unchanged",
    }


def self_test() -> None:
    class Legacy:
        class EditSegment:
            def __init__(self, start: float, end: float, speed: float, reason: str):
                self.start, self.end, self.speed, self.reason = start, end, speed, reason

        @staticmethod
        def _segment_duration(segment: Any) -> float:
            return (segment.end - segment.start) / segment.speed

    timeline = type("Timeline", (), {"shots": (type("Shot", (), {"start": 0.0, "end": 30.0})(),)})()
    span = type("Span", (), {"start": 1.0, "end": 3.0, "shot_index": 0})()
    config = {
        "editorial": {
            "finishing_move_opening_lead_seconds": 0.28,
            "finishing_move_payoff_hold_seconds": 0.45,
            "finishing_move_montage_max_output_seconds": 15.5,
        },
        "semantic_editor": {"minimum_output_seconds": 10.0, "maximum_output_seconds": 20.0},
    }
    _, hero_duration, body_minimum, body_maximum = duration_contract(Legacy, timeline, span, config)
    if abs(hero_duration - 2.73) > 0.01:
        raise AssertionError(f"unexpected hero duration {hero_duration}")
    if abs(body_minimum - 7.27) > 0.01:
        raise AssertionError(f"body still inherits standalone 10-second minimum: {body_minimum}")
    if body_maximum <= body_minimum:
        raise AssertionError("invalid hero-aware body duration window")
    if not (hero_duration + 7.80 >= 10.0 and hero_duration + 7.00 < 10.0):
        raise AssertionError("hero-aware duration acceptance regression")

    a = Legacy.EditSegment(10.0, 14.0, 1.0, "verified_combat_island_body")
    b = Legacy.EditSegment(16.0, 19.5, 1.0, "verified_combat_island_body")
    if Legacy._segment_duration(a) >= body_minimum or Legacy._segment_duration(b) >= body_minimum:
        raise AssertionError("synthetic individual islands unexpectedly meet body minimum")
    if Legacy._segment_duration(a) + Legacy._segment_duration(b) < body_minimum:
        raise AssertionError("synthetic combined islands do not exercise assembled-body minimum")
    print("MW4 Finishing Move hero-aware/combat-island continuation self-test: PASS")

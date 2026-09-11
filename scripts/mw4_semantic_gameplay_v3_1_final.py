from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import mw4_semantic_gameplay_v3_1_refined as refined

ShotSpan = refined.ShotSpan
ConsolidatedEvent = refined.ConsolidatedEvent
Engagement = refined.Engagement
FinishingMoveSpan = refined.FinishingMoveSpan
EditSegment = refined.EditSegment
SemanticPlanV31 = refined.SemanticPlanV31
SemanticTimelineV31 = refined.SemanticTimelineV31
PAYOFF_KINDS = refined.PAYOFF_KINDS


@dataclass(frozen=True)
class _MontageMoment:
    segment: EditSegment
    engagement: Engagement
    score: float
    retention_quality: float
    payoff_quality: float
    opening_quality: float
    ending_quality: float
    story_coherence: float
    weakest_quarter_interest: float
    low_interest_fraction: float
    max_unexplained_low_interest_run_seconds: float
    effect_events: tuple[Any, ...]


def _source_key(source: Path) -> str:
    stem = source.stem.lower()
    for key in ("batch2", "week2", "r1"):
        if key in stem:
            return key
    return stem


def _verified_cut_windows(
    config: dict[str, Any],
    source_key: str,
) -> tuple[tuple[float, float], ...]:
    raw = config.get("source_integrity", {}).get("verified_cut_windows", {}).get(source_key, [])
    windows: list[tuple[float, float]] = []
    for item in raw:
        if len(item) != 2:
            continue
        left, right = sorted((float(item[0]), float(item[1])))
        if right > left:
            windows.append((left, right))
    return tuple(sorted(windows))


def _build_hardened_shots(
    source: Path,
    duration: float,
    config: dict[str, Any],
) -> tuple[ShotSpan, ...]:
    """Build one source-structure map used by every V3.1 planning path.

    The automatic FFmpeg scene detector remains deliberately conservative because
    explosions, smoke and damage overlays can look like cuts. Visually verified
    missed stringout edits are therefore merged into the same shot map rather than
    handled as clip exclusions later.
    """
    source_key = _source_key(source)
    automatic = list(refined._scene_cut_times(source, config))
    verified = [
        (left + right) / 2.0
        for left, right in _verified_cut_windows(config, source_key)
    ]
    cuts: list[float] = []
    for value in sorted(automatic + verified):
        if value <= 0.35 or value >= duration - 0.35:
            continue
        if all(abs(value - old) >= 0.18 for old in cuts):
            cuts.append(value)

    guard = float(config.get("semantic_analysis", {}).get("scene_change_guard_seconds", 0.12))
    boundaries = [0.0] + cuts + [duration]
    shots: list[ShotSpan] = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        start = left if index == 0 else min(right, left + guard)
        end = right if index == len(boundaries) - 2 else max(start, right - guard)
        if end - start >= 1.0:
            shots.append(ShotSpan(round(start, 3), round(end, 3)))
    if not shots:
        shots.append(ShotSpan(0.0, duration))
    return tuple(shots)


def _verified_finishing_moves(
    shots: tuple[ShotSpan, ...],
    heuristic: tuple[FinishingMoveSpan, ...],
    source: Path,
    config: dict[str, Any],
) -> tuple[FinishingMoveSpan, ...]:
    """Fail closed on Finishing Moves until automatic classification is proven."""
    cfg = config.get("finishing_move_detector", {})
    key = _source_key(source)
    verified: list[FinishingMoveSpan] = []

    for item in cfg.get("verified_spans", {}).get(key, []):
        start = float(item["start"])
        payoff = float(item["payoff"])
        end = float(item["end"])
        midpoint = (start + end) / 2.0
        shot_index = refined.core._shot_index(shots, midpoint)
        shot = shots[shot_index]
        if not (shot.start <= midpoint <= shot.end):
            continue
        verified.append(
            FinishingMoveSpan(
                start=round(max(shot.start, start), 3),
                payoff=round(payoff, 3),
                end=round(min(shot.end, end), 3),
                shot_index=shot_index,
                confidence=0.99,
                evidence={
                    "visual_verification": 1.0,
                    "third_person_execution_regime": 1.0,
                },
            )
        )

    if bool(cfg.get("allow_unverified_automatic", False)):
        min_conf = float(cfg.get("automatic_high_precision_confidence", 0.86))
        min_duration = float(cfg.get("automatic_minimum_span_seconds", 1.25))
        max_duration = float(cfg.get("automatic_maximum_span_seconds", 3.80))
        for span in heuristic:
            duration = span.end - span.start
            if span.confidence < min_conf or not (min_duration <= duration <= max_duration):
                continue
            if any(max(span.start, old.start) < min(span.end, old.end) for old in verified):
                continue
            verified.append(span)

    return tuple(sorted(verified, key=lambda item: item.start))


def analyze_source(source: Path, config: dict[str, Any]) -> SemanticTimelineV31:
    base_timeline = refined.core.base.analyze_source(source, config)
    shots = _build_hardened_shots(source, base_timeline.duration, config)
    consolidated = refined.core.consolidate_events(base_timeline, shots)
    engagements = refined.core.cluster_engagements(base_timeline, shots, consolidated, config)
    heuristic = refined.core.detect_finishing_moves(base_timeline, shots, engagements, config)
    finishing = _verified_finishing_moves(shots, heuristic, source, config)
    timeline = SemanticTimelineV31(
        base_timeline,
        shots,
        consolidated,
        engagements,
        finishing,
    )
    setattr(timeline, "_source_key", _source_key(source))
    return timeline


def _explained_mask(
    timeline: SemanticTimelineV31,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    config: dict[str, Any],
) -> np.ndarray:
    cfg = config["semantic_editor"]["dull"]
    pre = float(cfg.get("engagement_explanation_lead_seconds", 0.35))
    post = float(cfg.get("engagement_explanation_tail_seconds", 0.30))
    mask = np.zeros(len(timeline.times), dtype=bool)
    for engagement in engagements:
        i0 = max(0, int(math.floor((engagement.start - pre) * timeline.fps)))
        i1 = min(len(mask), int(math.ceil((engagement.end + post) * timeline.fps)))
        mask[i0:i1] = True
    if finishing is not None:
        i0 = max(0, int(math.floor((finishing.start - 0.12) * timeline.fps)))
        i1 = min(len(mask), int(math.ceil((finishing.end + 0.18) * timeline.fps)))
        mask[i0:i1] = True
    return mask


def _ranges(
    mask: np.ndarray,
    i0: int,
    i1: int,
    fps: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    run: int | None = None
    for index in range(i0, i1):
        if bool(mask[index]) and run is None:
            run = index
        elif not bool(mask[index]) and run is not None:
            out.append((run / fps, index / fps))
            run = None
    if run is not None:
        out.append((run / fps, i1 / fps))
    return out


def _segment_duration(segment: EditSegment) -> float:
    return (segment.end - segment.start) / segment.speed


def _source_segments_overlap(a: SemanticPlanV31, b: SemanticPlanV31) -> bool:
    for left in a.segments:
        for right in b.segments:
            if max(left.start, right.start) < min(left.end, right.end):
                return True
    return False


def _editable_segments(
    timeline: SemanticTimelineV31,
    start: float,
    end: float,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    config: dict[str, Any],
) -> tuple[tuple[EditSegment, ...], tuple[str, ...], float] | None:
    """Edit only unexplained inactivity, never meaningful low-motion gameplay."""
    cfg = config["semantic_editor"]["dull"]
    fps = timeline.fps
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(timeline.times), int(math.ceil(end * fps)))
    explained = _explained_mask(timeline, engagements, finishing, config)
    low = timeline.signals["interest"] < float(cfg.get("low_interest_threshold", 0.30))
    unexplained_low = low & ~explained
    traversal = (
        (timeline.signals["traversal"] > 0.58)
        & (timeline.signals["combat"] < 0.42)
        & (timeline.signals["outcome"] < 0.40)
        & ~explained
    )

    min_action = float(cfg.get("minimum_action_seconds", 0.70))
    max_action = float(cfg.get("maximum_single_action_seconds", 1.60))
    max_speed_source = float(cfg.get("maximum_speedup_source_seconds", 0.85))
    speed = float(cfg.get("traversal_speed", 1.35))
    actions: list[tuple[float, float, str]] = []

    for a, b in _ranges(unexplained_low, i0, i1, fps):
        a, b = max(start, a), min(end, b)
        if min_action <= b - a <= max_action:
            actions.append((a, b, "cut_dull"))
    for a, b in _ranges(traversal, i0, i1, fps):
        a, b = max(start, a), min(end, b)
        if min_action <= b - a <= max_speed_source:
            actions.append((a, b, "compress_traversal"))

    actions.sort(key=lambda item: item[1] - item[0], reverse=True)
    chosen: list[tuple[float, float, str]] = []
    for action in actions:
        if any(max(action[0], old[0]) < min(action[1], old[1]) for old in chosen):
            continue
        chosen.append(action)
        if len(chosen) >= int(cfg.get("maximum_editorial_actions", 2)):
            break
    chosen.sort()

    boundaries = sorted({start, end, *[point for action in chosen for point in action[:2]]})
    segments: list[EditSegment] = []
    reasons: list[str] = []
    removed_equivalent = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left + right) / 2.0
        action = next((item for item in chosen if item[0] <= midpoint <= item[1]), None)
        if action is None:
            segments.append(EditSegment(round(left, 3), round(right, 3), 1.0, "keep"))
        elif action[2] == "cut_dull":
            removed_equivalent += right - left
            reasons.append(f"cut unexplained dull {left:.2f}-{right:.2f}s")
        else:
            removed_equivalent += (right - left) - (right - left) / speed
            segments.append(
                EditSegment(
                    round(left, 3),
                    round(right, 3),
                    round(speed, 3),
                    "compressed_traversal",
                )
            )
            reasons.append(f"compress unexplained traversal {left:.2f}-{right:.2f}s at {speed:.2f}x")

    if removed_equivalent > float(cfg.get("maximum_total_removed_equivalent_seconds", 2.0)):
        return None
    output_duration = sum(_segment_duration(segment) for segment in segments)
    if output_duration < float(config["semantic_editor"].get("minimum_output_seconds", 10.0)):
        return None

    kept_runs: list[float] = []
    for segment in segments:
        s0 = max(i0, int(math.floor(segment.start * fps)))
        s1 = min(i1, int(math.ceil(segment.end * fps)))
        kept_runs.append(
            refined.core._longest_true_run(unexplained_low[s0:s1], fps) / segment.speed
        )
    residual = max(kept_runs, default=0.0)
    if residual > float(cfg.get("maximum_unexplained_low_interest_run_seconds", 0.90)):
        return None
    return tuple(segments), tuple(reasons), residual


def _quality_metrics(
    timeline: SemanticTimelineV31,
    start: float,
    end: float,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    residual_run: float,
    config: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    sig = timeline.signals
    fps = timeline.fps
    i0 = max(0, int(math.floor(start * fps)))
    i1 = min(len(timeline.times), int(math.ceil(end * fps)))
    raw_interest = sig["interest"][i0:i1]
    if raw_interest.size == 0:
        return (0.0,) * 7

    explained = _explained_mask(timeline, engagements, finishing, config)[i0:i1]
    floor = float(config["semantic_editor"]["dull"].get("explained_interest_floor", 0.26))
    effective_interest = np.where(explained, np.maximum(raw_interest, floor), raw_interest)
    quarters = np.array_split(effective_interest, 4)
    weakest = min(float(np.mean(part)) for part in quarters if len(part))
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    low_fraction = float(np.mean((raw_interest < low_threshold) & ~explained))

    first_event = finishing.start if finishing is not None else engagements[0].events[0].time
    delay = max(0.0, first_event - start)
    opening_mean = refined.core._mean(
        sig["interest"], timeline, start, min(end, start + 1.25)
    )
    opening_slice = sig["interest"][
        i0:min(i1, i0 + max(1, int(round(1.1 * fps))))
    ]
    opening_explained = explained[:len(opening_slice)]
    opening_dead = refined.core._longest_true_run(
        (opening_slice < low_threshold) & ~opening_explained,
        fps,
    )
    opening_quality = np.clip(
        0.43 * opening_mean
        + 0.34 * (1.0 - min(1.0, delay / 0.90))
        + 0.23 * (1.0 - min(1.0, opening_dead / 0.50)),
        0,
        1,
    )
    if finishing is not None:
        opening_quality = min(1.0, float(opening_quality) + 0.20 * finishing.confidence)

    payoff_events = [
        event
        for engagement in engagements
        for event in engagement.events
        if any(kind in PAYOFF_KINDS for kind in event.kinds)
    ]
    last_payoff_time = payoff_events[-1].time if payoff_events else None
    if finishing is not None and (
        last_payoff_time is None or finishing.payoff > last_payoff_time
    ):
        last_payoff_time = finishing.payoff

    if last_payoff_time is not None:
        tail = max(0.0, end - last_payoff_time)
        tail_quality = float(np.clip(1.0 - abs(tail - 0.45) / 0.75, 0, 1))
        ending_interest = refined.core._mean(
            sig["interest"], timeline, max(start, end - 1.0), end
        )
        ending_quality = 0.66 * tail_quality + 0.34 * ending_interest
    else:
        active = max(
            refined.core._mean(sig["combat"], timeline, max(start, end - 0.8), end),
            refined.core._mean(sig["contact"], timeline, max(start, end - 0.8), end),
        )
        ending_quality = 0.72 * active + 0.28 * refined.core._mean(
            sig["interest"], timeline, max(start, end - 1.0), end
        )

    payoff_strength = max(
        [event.confidence for event in payoff_events]
        + ([min(1.0, finishing.confidence + 0.12)] if finishing is not None else [0.0])
    )
    coherence = np.clip(
        0.54 * float(np.mean([eng.confidence for eng in engagements]))
        + 0.26 * min(1.0, len(engagements) / 4.0)
        + 0.20 * (1.0 - min(1.0, low_fraction / 0.45)),
        0,
        1,
    )
    retention = np.clip(
        0.27 * float(np.mean(effective_interest))
        + 0.18 * weakest
        + 0.22 * opening_quality
        + 0.18 * ending_quality
        + 0.09 * coherence
        + 0.06 * (1.0 - min(1.0, residual_run / 0.90)),
        0,
        1,
    )
    payoff_quality = np.clip(
        0.72 * payoff_strength + 0.28 * ending_quality,
        0,
        1,
    )
    return (
        float(opening_quality),
        float(ending_quality),
        float(coherence),
        float(retention),
        float(payoff_quality),
        weakest,
        low_fraction,
    )


def _story_gate(
    story: str,
    metric: str,
    config: dict[str, Any],
    default: float,
) -> float:
    gates = config.get("story_quality_gates", {})
    return float(gates.get(metric, {}).get(story, gates.get(metric, {}).get("default", default)))


def _score_plan(
    retention: float,
    payoff: float,
    opening: float,
    ending: float,
    coherence: float,
    weakest: float,
    finishing: bool,
    config: dict[str, Any],
) -> float:
    weights = config["semantic_editor"]["selection"]
    score = (
        float(weights.get("retention_quality_weight", 0.24)) * retention
        + float(weights.get("payoff_quality_weight", 0.17)) * payoff
        + float(weights.get("opening_weight", 0.20)) * opening
        + float(weights.get("ending_weight", 0.17)) * ending
        + float(weights.get("story_coherence_weight", 0.12)) * coherence
        + float(weights.get("weakest_section_weight", 0.10)) * weakest
    )
    if finishing:
        score += float(weights.get("finishing_move_bonus", 0.14))
    return float(score)


def _shot_for_time(timeline: SemanticTimelineV31, value: float) -> int:
    return refined.core._shot_index(timeline.shots, value)


def _planned_cross_shot_bridge(
    plan: SemanticPlanV31,
    left: EditSegment,
    right: EditSegment,
    bridge_index: int,
) -> bool:
    if (
        plan.story_type == "finishing_move_open"
        and bridge_index == 0
        and left.reason == "finishing_move_open_hero"
    ):
        return True
    if (
        plan.story_type == "semantic_montage"
        and left.reason == "semantic_montage_moment"
        and right.reason == "semantic_montage_moment"
    ):
        return True
    if (
        plan.story_type == "finishing_move_open"
        and left.reason == "semantic_montage_moment"
        and right.reason == "semantic_montage_moment"
    ):
        return True
    return False


def plan_integrity_violations(
    plan: SemanticPlanV31,
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    source_key: str,
) -> list[str]:
    """Reject accidental stringout continuity; permit only explicit montage joins."""
    windows = _verified_cut_windows(config, source_key)
    violations: list[str] = []

    for segment in plan.segments:
        inside_one_shot = any(
            shot.start - 1e-3 <= segment.start
            and segment.end <= shot.end + 1e-3
            for shot in timeline.shots
        )
        if not inside_one_shot:
            violations.append(
                f"segment {segment.start:.3f}-{segment.end:.3f} crosses a source-shot boundary"
            )
        for left, right in windows:
            midpoint = (left + right) / 2.0
            if segment.start < midpoint < segment.end:
                violations.append(
                    f"segment {segment.start:.3f}-{segment.end:.3f} crosses verified source cut {left:.3f}-{right:.3f}"
                )

    for index, (left_seg, right_seg) in enumerate(zip(plan.segments, plan.segments[1:])):
        if right_seg.start <= left_seg.end + 0.02:
            continue
        left_shot = _shot_for_time(timeline, max(left_seg.start, left_seg.end - 1e-3))
        right_shot = _shot_for_time(timeline, min(right_seg.end, right_seg.start + 1e-3))
        if left_shot != right_shot and not _planned_cross_shot_bridge(
            plan, left_seg, right_seg, index
        ):
            violations.append(
                f"unplanned cross-shot bridge {left_seg.end:.3f}->{right_seg.start:.3f}"
            )

    if plan.finishing_move is not None:
        if (
            plan.story_type != "finishing_move_open"
            or plan.effect_profile != "finishing_move_hero"
        ):
            violations.append(
                "verified Finishing Move is not routed as finishing_move_open/finishing_move_hero"
            )
        if not plan.segments:
            violations.append("Finishing Move plan has no source segments")
        else:
            first = plan.segments[0]
            if not (first.start <= plan.finishing_move.start <= first.end):
                violations.append("Finishing Move is not contained in the first output segment")
            opening_delay = (plan.finishing_move.start - first.start) / first.speed
            if opening_delay > 0.48:
                violations.append(f"Finishing Move opening delay is {opening_delay:.3f}s")
    return violations


def _passes_story_gates(
    story: str,
    opening: float,
    ending: float,
    retention: float,
    payoff: float,
    weakest: float,
    low_fraction: float,
    residual: float,
    config: dict[str, Any],
) -> bool:
    editor = config["semantic_editor"]
    if opening < _story_gate(
        story,
        "opening_min",
        config,
        float(editor["opening"].get("minimum_quality", 0.42)),
    ):
        return False
    if ending < _story_gate(
        story,
        "ending_min",
        config,
        float(editor["ending"].get("minimum_quality", 0.40)),
    ):
        return False
    if retention < _story_gate(
        story,
        "retention_min",
        config,
        float(config["performance_targets"].get("retention_quality_min", 0.36)),
    ):
        return False
    if payoff < _story_gate(
        story,
        "payoff_min",
        config,
        float(config["performance_targets"].get("payoff_quality_min", 0.34)),
    ):
        return False
    if weakest < float(editor["dull"].get("minimum_weak_quarter_interest", 0.27)):
        return False
    if low_fraction > float(editor["dull"].get("maximum_low_interest_fraction", 0.38)):
        return False
    if residual > float(editor["dull"].get("maximum_unexplained_low_interest_run_seconds", 0.90)):
        return False
    return True


def _normal_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    plans: list[SemanticPlanV31] = []
    for chain in refined._candidate_engagement_chains(timeline, config):
        if refined._find_finishing_for_chain(timeline, chain) is not None:
            continue
        boundary = refined._boundary_for_chain(timeline, chain, config, None)
        if boundary is None:
            continue
        start, end = boundary
        if refined.core._intersects_excluded(start, end, excluded):
            continue
        edited = _editable_segments(timeline, start, end, chain, None, config)
        if edited is None:
            continue
        segments, edit_reasons, residual = edited
        output_duration = sum(_segment_duration(segment) for segment in segments)
        opening, ending, coherence, retention, payoff, weakest, low_fraction = _quality_metrics(
            timeline,
            start,
            end,
            chain,
            None,
            residual,
            config,
        )
        story, profile, effect_events, route_reasons = refined.core._route(
            timeline,
            chain,
            None,
        )
        if not _passes_story_gates(
            story,
            opening,
            ending,
            retention,
            payoff,
            weakest,
            low_fraction,
            residual,
            config,
        ):
            continue

        plan = SemanticPlanV31(
            start=start,
            end=end,
            raw_duration=round(end - start, 3),
            output_duration=round(output_duration, 3),
            score=round(
                _score_plan(
                    retention,
                    payoff,
                    opening,
                    ending,
                    coherence,
                    weakest,
                    False,
                    config,
                ),
                5,
            ),
            retention_quality=round(retention, 4),
            payoff_quality=round(payoff, 4),
            opening_quality=round(opening, 4),
            ending_quality=round(ending, 4),
            story_coherence=round(coherence, 4),
            weakest_quarter_interest=round(weakest, 4),
            low_interest_fraction=round(low_fraction, 4),
            max_unexplained_low_interest_run_seconds=round(residual, 3),
            story_type=story,
            effect_profile=profile,
            segments=segments,
            effect_events=effect_events,
            engagements=chain,
            finishing_move=None,
            editorial_reasons=tuple(route_reasons) + tuple(edit_reasons),
        )
        if not plan_integrity_violations(plan, timeline, config, source_key):
            plans.append(plan)

    plans.sort(key=lambda item: item.score, reverse=True)
    unique: list[SemanticPlanV31] = []
    for plan in plans:
        anchor = plan.engagements[0].events[0].time
        if any(
            plan.engagements[0].shot_index == prior.engagements[0].shot_index
            and abs(anchor - prior.engagements[0].events[0].time) < 2.4
            for prior in unique
        ):
            continue
        unique.append(plan)
    return unique


def _montage_moments(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[_MontageMoment]:
    cfg = config["semantic_editor"].get("semantic_montage", {})
    if not bool(cfg.get("enabled", True)):
        return []

    minimum = float(cfg.get("minimum_moment_seconds", 2.4))
    maximum = float(cfg.get("maximum_moment_seconds", 6.5))
    min_open = float(cfg.get("minimum_moment_opening_quality", 0.38))
    min_retention = float(cfg.get("minimum_moment_retention_quality", 0.30))
    min_payoff = float(cfg.get("minimum_moment_payoff_confidence", 0.58))
    min_weakest = float(cfg.get("minimum_moment_weakest_quarter_interest", 0.24))
    max_low_fraction = float(cfg.get("maximum_moment_low_interest_fraction", 0.42))
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    moments: list[_MontageMoment] = []

    for engagement in timeline.engagements:
        payoffs = [
            event
            for event in engagement.events
            if any(kind in PAYOFF_KINDS for kind in event.kinds)
        ]
        if not payoffs:
            continue
        payoff_confidence = max(event.confidence for event in payoffs)
        if payoff_confidence < min_payoff:
            continue

        shot = timeline.shots[engagement.shot_index]
        first_event = engagement.events[0].time
        last_payoff = max(event.time for event in payoffs)
        start = max(shot.start, min(engagement.start - 0.12, first_event - 0.28))
        end = min(shot.end, max(engagement.end + 0.12, last_payoff + 0.45))
        duration = end - start
        if duration < minimum:
            continue
        if duration > maximum:
            start = max(shot.start, end - maximum)
            duration = end - start
        if not (minimum <= duration <= maximum):
            continue
        if refined.core._intersects_excluded(start, end, excluded):
            continue

        explained = _explained_mask(timeline, (engagement,), None, config)
        i0 = max(0, int(math.floor(start * timeline.fps)))
        i1 = min(len(timeline.times), int(math.ceil(end * timeline.fps)))
        low = timeline.signals["interest"] < low_threshold
        residual = refined.core._longest_true_run(
            (low & ~explained)[i0:i1],
            timeline.fps,
        )
        opening, ending, coherence, retention, payoff, weakest, low_fraction = _quality_metrics(
            timeline,
            start,
            end,
            (engagement,),
            None,
            residual,
            config,
        )
        if opening < min_open or retention < min_retention:
            continue
        if weakest < min_weakest or low_fraction > max_low_fraction:
            continue
        if residual > float(config["semantic_editor"]["dull"].get("maximum_unexplained_low_interest_run_seconds", 0.90)):
            continue

        effect_events = tuple(
            sorted(
                refined.core._raw_effect_events(timeline, (engagement,)),
                key=lambda item: item.time,
            )
        )
        if not effect_events:
            continue
        segment = EditSegment(
            round(start, 3),
            round(end, 3),
            1.0,
            "semantic_montage_moment",
        )
        moment_score = (
            0.28 * opening
            + 0.28 * payoff
            + 0.20 * retention
            + 0.12 * ending
            + 0.12 * coherence
        )
        moments.append(
            _MontageMoment(
                segment=segment,
                engagement=engagement,
                score=round(float(moment_score), 5),
                retention_quality=retention,
                payoff_quality=max(payoff, payoff_confidence),
                opening_quality=opening,
                ending_quality=ending,
                story_coherence=coherence,
                weakest_quarter_interest=weakest,
                low_interest_fraction=low_fraction,
                max_unexplained_low_interest_run_seconds=residual,
                effect_events=effect_events,
            )
        )

    # Keep a broad but finite pool to avoid combinatorial noise while preserving
    # different source shots and payoff types.
    moments.sort(key=lambda item: item.score, reverse=True)
    pool = moments[:30]
    return sorted(pool, key=lambda item: item.segment.start)


def _montage_plan_from_group(
    group: tuple[_MontageMoment, ...],
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    source_key: str,
) -> SemanticPlanV31 | None:
    segments = tuple(item.segment for item in group)
    output_duration = sum(_segment_duration(segment) for segment in segments)
    cfg = config["semantic_editor"].get("semantic_montage", {})
    minimum = float(config["semantic_editor"].get("minimum_output_seconds", 10.0))
    maximum = min(
        float(config["semantic_editor"].get("maximum_output_seconds", 20.0)),
        float(cfg.get("maximum_output_seconds", 15.5)),
    )
    if not (minimum <= output_duration <= maximum):
        return None

    durations = np.asarray([_segment_duration(item.segment) for item in group], dtype=float)
    total = float(np.sum(durations))
    retention = float(np.average([item.retention_quality for item in group], weights=durations))
    payoff_values = [item.payoff_quality for item in group]
    payoff = float(0.55 * max(payoff_values) + 0.45 * np.mean(payoff_values))
    opening = float(group[0].opening_quality)
    ending = float(group[-1].ending_quality)
    coherence_base = float(np.average([item.story_coherence for item in group], weights=durations))
    coherence = float(np.clip(0.90 * coherence_base + 0.04 * min(1.0, len(group) / 3.0), 0, 1))
    weakest = float(min(item.weakest_quarter_interest for item in group))
    low_fraction = float(np.average([item.low_interest_fraction for item in group], weights=durations))
    residual = float(max(item.max_unexplained_low_interest_run_seconds for item in group))
    story = "semantic_montage"

    if not _passes_story_gates(
        story,
        opening,
        ending,
        retention,
        payoff,
        weakest,
        low_fraction,
        residual,
        config,
    ):
        return None

    effect_events: list[Any] = []
    for item in group:
        if item.effect_events:
            strongest = max(item.effect_events, key=lambda event: event.confidence)
            effect_events.append(strongest)
    effect_events.sort(key=lambda event: event.time)
    if len(effect_events) < 2:
        return None

    score = _score_plan(
        retention,
        payoff,
        opening,
        ending,
        coherence,
        weakest,
        False,
        config,
    ) - float(cfg.get("selection_penalty", 0.015))
    plan = SemanticPlanV31(
        start=segments[0].start,
        end=segments[-1].end,
        raw_duration=round(sum(segment.end - segment.start for segment in segments), 3),
        output_duration=round(output_duration, 3),
        score=round(score, 5),
        retention_quality=round(retention, 4),
        payoff_quality=round(payoff, 4),
        opening_quality=round(opening, 4),
        ending_quality=round(ending, 4),
        story_coherence=round(coherence, 4),
        weakest_quarter_interest=round(weakest, 4),
        low_interest_fraction=round(low_fraction, 4),
        max_unexplained_low_interest_run_seconds=round(residual, 3),
        story_type=story,
        effect_profile="semantic_montage",
        segments=segments,
        effect_events=tuple(effect_events[:3]),
        engagements=tuple(item.engagement for item in group),
        finishing_move=None,
        editorial_reasons=(
            f"deliberate {len(group)}-moment semantic montage built only from individually qualifying payoff moments",
            "every montage source segment remains inside one hardened source shot",
            "cross-shot joins are explicit editor transitions rather than accidental stringout continuation",
        ),
    )
    if plan_integrity_violations(plan, timeline, config, source_key):
        return None
    return plan


def _semantic_montage_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    cfg = config["semantic_editor"].get("semantic_montage", {})
    if not bool(cfg.get("enabled", True)):
        return []
    moments = _montage_moments(timeline, config, excluded)
    maximum_moments = max(2, min(3, int(cfg.get("maximum_moments_per_clip", 3))))
    plans: list[SemanticPlanV31] = []

    for left_index, first in enumerate(moments):
        for right_index in range(left_index + 1, len(moments)):
            second = moments[right_index]
            if max(first.segment.start, second.segment.start) < min(first.segment.end, second.segment.end):
                continue
            pair = (first, second)
            pair_plan = _montage_plan_from_group(pair, timeline, config, source_key)
            if pair_plan is not None:
                plans.append(pair_plan)
            if maximum_moments < 3:
                continue
            pair_duration = _segment_duration(first.segment) + _segment_duration(second.segment)
            if pair_duration >= float(config["semantic_editor"].get("minimum_output_seconds", 10.0)):
                continue
            for third_index in range(right_index + 1, len(moments)):
                third = moments[third_index]
                if third.segment.start < second.segment.end:
                    continue
                triple_plan = _montage_plan_from_group(
                    (first, second, third),
                    timeline,
                    config,
                    source_key,
                )
                if triple_plan is not None:
                    plans.append(triple_plan)

    plans.sort(key=lambda item: item.score, reverse=True)
    unique: list[SemanticPlanV31] = []
    seen: set[tuple[float, ...]] = set()
    for plan in plans:
        key = tuple(round(segment.start, 1) for segment in plan.segments)
        if key in seen:
            continue
        seen.add(key)
        unique.append(plan)
        if len(unique) >= 30:
            break
    return unique


def _finishing_open_plans(
    timeline: SemanticTimelineV31,
    continuations: list[SemanticPlanV31],
    config: dict[str, Any],
    source_key: str,
) -> list[SemanticPlanV31]:
    if not timeline.finishing_moves:
        return []

    editorial = config["editorial"]
    minimum = float(config["semantic_editor"].get("minimum_output_seconds", 10.0))
    maximum = min(
        float(config["semantic_editor"].get("maximum_output_seconds", 20.0)),
        float(editorial.get("finishing_move_montage_max_output_seconds", 15.5)),
    )
    lead = float(editorial.get("finishing_move_opening_lead_seconds", 0.28))
    hold = float(editorial.get("finishing_move_payoff_hold_seconds", 0.45))
    results: list[SemanticPlanV31] = []

    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero_start = max(shot.start, span.start - lead)
        hero_end = min(shot.end, span.end + hold)
        if hero_end <= hero_start:
            continue
        hero = EditSegment(
            round(hero_start, 3),
            round(hero_end, 3),
            1.0,
            "finishing_move_open_hero",
        )
        hero_duration = _segment_duration(hero)
        candidates: list[tuple[float, SemanticPlanV31]] = []

        for continuation in continuations:
            if continuation.story_type == "finishing_move_open":
                continue
            if any(
                max(hero.start, segment.start) < min(hero.end, segment.end)
                for segment in continuation.segments
            ):
                continue
            combined_duration = hero_duration + continuation.output_duration
            if not (minimum <= combined_duration <= maximum):
                continue
            later = continuation.segments[0].start >= span.end + 0.25
            distance = (
                max(0.0, continuation.segments[0].start - span.end)
                if later
                else abs(span.start - continuation.segments[-1].end) + 120.0
            )
            preference = (
                continuation.score
                + (0.08 if later else 0.0)
                - min(distance, 120.0) * 0.00035
                - (0.02 if continuation.story_type == "semantic_montage" else 0.0)
            )
            candidates.append((preference, continuation))

        for _, continuation in sorted(candidates, key=lambda item: item[0], reverse=True)[:5]:
            segments = (hero,) + continuation.segments
            output_duration = sum(_segment_duration(segment) for segment in segments)
            opening = max(0.90, min(1.0, 0.74 + 0.22 * span.confidence))
            ending = continuation.ending_quality
            coherence = min(1.0, 0.10 + 0.88 * continuation.story_coherence)
            retention = min(1.0, 0.18 * opening + 0.82 * continuation.retention_quality)
            payoff = max(0.92, continuation.payoff_quality)
            weakest = continuation.weakest_quarter_interest
            low_fraction = continuation.low_interest_fraction * (
                continuation.output_duration / output_duration
            )
            residual = continuation.max_unexplained_low_interest_run_seconds
            story = "finishing_move_open"
            if not _passes_story_gates(
                story,
                opening,
                ending,
                retention,
                payoff,
                weakest,
                low_fraction,
                residual,
                config,
            ):
                continue

            score = _score_plan(
                retention,
                payoff,
                opening,
                ending,
                coherence,
                weakest,
                True,
                config,
            )
            plan = SemanticPlanV31(
                start=hero.start,
                end=continuation.end,
                raw_duration=round(sum(segment.end - segment.start for segment in segments), 3),
                output_duration=round(output_duration, 3),
                score=round(score, 5),
                retention_quality=round(retention, 4),
                payoff_quality=round(payoff, 4),
                opening_quality=round(opening, 4),
                ending_quality=round(ending, 4),
                story_coherence=round(coherence, 4),
                weakest_quarter_interest=round(weakest, 4),
                low_interest_fraction=round(low_fraction, 4),
                max_unexplained_low_interest_run_seconds=round(residual, 3),
                story_type=story,
                effect_profile="finishing_move_hero",
                segments=segments,
                effect_events=continuation.effect_events,
                engagements=continuation.engagements,
                finishing_move=span,
                editorial_reasons=(
                    "verified Finishing Move opens the clip as the protected hero event",
                    "one deliberate semantic continuation supplies campaign duration instead of dead padding",
                    "every continuation segment remains source-integrity checked",
                ) + continuation.editorial_reasons,
            )
            if not plan_integrity_violations(plan, timeline, config, source_key):
                results.append(plan)

    return sorted(results, key=lambda item: item.score, reverse=True)


def build_plans_for_source(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    normal = _normal_plans(timeline, config, excluded, source_key)
    montage = _semantic_montage_plans(timeline, config, excluded, source_key)
    finishing = _finishing_open_plans(
        timeline,
        normal + montage,
        config,
        source_key,
    )
    return sorted(finishing + normal + montage, key=lambda item: item.score, reverse=True)


def build_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[SemanticPlanV31]:
    return build_plans_for_source(
        timeline,
        config,
        excluded,
        getattr(timeline, "_source_key", ""),
    )


def select_plans(
    plans: list[SemanticPlanV31],
    config: dict[str, Any],
) -> list[SemanticPlanV31]:
    """Select strong contiguous stories first; semantic montages only fill unused slots."""
    maximum = int(config.get("count_per_source_max", 4))
    minimum = int(config.get("minimum_count_per_source", 2))
    selected: list[SemanticPlanV31] = []

    finishing = [plan for plan in plans if plan.story_type == "finishing_move_open"]
    if finishing:
        selected.append(finishing[0])

    normal = [
        plan
        for plan in plans
        if plan.story_type not in {"finishing_move_open", "semantic_montage"}
    ]
    montage = [plan for plan in plans if plan.story_type == "semantic_montage"]

    seen_story = {plan.story_type for plan in selected}
    for plan in normal:
        if plan.story_type in seen_story:
            continue
        if any(_source_segments_overlap(plan, prior) for prior in selected):
            continue
        selected.append(plan)
        seen_story.add(plan.story_type)
        if len(selected) >= maximum:
            break

    for plan in normal:
        if len(selected) >= maximum:
            break
        if plan in selected:
            continue
        if any(_source_segments_overlap(plan, prior) for prior in selected):
            continue
        selected.append(plan)

    for plan in montage:
        if len(selected) >= maximum:
            break
        if any(_source_segments_overlap(plan, prior) for prior in selected):
            continue
        selected.append(plan)

    if len(selected) < minimum:
        raise RuntimeError(
            f"Only {len(selected)} hardened V3.1 candidates passed source-integrity/opening/ending/no-dull gates; minimum is {minimum}."
        )
    return sorted(
        selected,
        key=lambda item: (
            item.story_type != "finishing_move_open",
            item.segments[0].start,
        ),
    )


def diagnose_source(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str | None = None,
) -> dict[str, Any]:
    key = source_key or getattr(timeline, "_source_key", "")
    plans = build_plans_for_source(timeline, config, excluded, key)
    normal = [item for item in plans if item.story_type not in {"finishing_move_open", "semantic_montage"}]
    montage = [item for item in plans if item.story_type == "semantic_montage"]
    finishing = [item for item in plans if item.story_type == "finishing_move_open"]
    return {
        "shot_count": len(timeline.shots),
        "engagement_count": len(timeline.engagements),
        "verified_finishing_move_count": len(timeline.finishing_moves),
        "finishing_moves": [asdict(item) for item in timeline.finishing_moves],
        "verified_source_cut_windows": [
            list(item) for item in _verified_cut_windows(config, key)
        ],
        "counts": {
            "pass": len(plans),
            "normal": len(normal),
            "semantic_montage": len(montage),
            "finishing_move_open": len(finishing),
        },
        "candidate_boundaries": [
            [
                item.start,
                item.end,
                item.story_type,
                item.score,
                [[seg.start, seg.end, seg.reason] for seg in item.segments],
            ]
            for item in plans[:30]
        ],
        "dull_policy": "only unexplained low-interest footage is eligible for dull rejection/cutting",
        "source_integrity_policy": (
            "contiguous stories cannot cross hardened source boundaries; "
            "only explicit finishing_move_open or semantic_montage joins may bridge shots"
        ),
        "montage_policy": (
            "semantic montage is a controlled filler after contiguous stories and uses only individually qualifying payoff moments"
        ),
    }


def _self_test() -> None:
    assert _story_gate(
        "sustained_pressure",
        "payoff_min",
        {"story_quality_gates": {"payoff_min": {"sustained_pressure": 0.16}}},
        0.34,
    ) == 0.16
    ordinary_left = EditSegment(0.0, 4.0, 1.0, "keep")
    ordinary_right = EditSegment(10.0, 14.0, 1.0, "keep")
    montage_left = EditSegment(0.0, 4.0, 1.0, "semantic_montage_moment")
    montage_right = EditSegment(10.0, 14.0, 1.0, "semantic_montage_moment")

    ordinary = SemanticPlanV31(
        start=0.0,
        end=14.0,
        raw_duration=8.0,
        output_duration=8.0,
        score=0.7,
        retention_quality=0.5,
        payoff_quality=0.5,
        opening_quality=0.5,
        ending_quality=0.5,
        story_coherence=0.5,
        weakest_quarter_interest=0.4,
        low_interest_fraction=0.1,
        max_unexplained_low_interest_run_seconds=0.2,
        story_type="precision_outcome",
        effect_profile="precision_punch",
        segments=(ordinary_left, ordinary_right),
        effect_events=(),
        engagements=(),
        finishing_move=None,
        editorial_reasons=(),
    )
    montage = SemanticPlanV31(
        start=0.0,
        end=14.0,
        raw_duration=8.0,
        output_duration=8.0,
        score=0.7,
        retention_quality=0.5,
        payoff_quality=0.5,
        opening_quality=0.5,
        ending_quality=0.5,
        story_coherence=0.5,
        weakest_quarter_interest=0.4,
        low_interest_fraction=0.1,
        max_unexplained_low_interest_run_seconds=0.2,
        story_type="semantic_montage",
        effect_profile="semantic_montage",
        segments=(montage_left, montage_right),
        effect_events=(),
        engagements=(),
        finishing_move=None,
        editorial_reasons=(),
    )
    assert not _planned_cross_shot_bridge(ordinary, ordinary_left, ordinary_right, 0)
    assert _planned_cross_shot_bridge(montage, montage_left, montage_right, 0)
    print(
        json.dumps(
            {
                "self_test": "PASS",
                "candidate_mode": "engagement_driven_hardened_source_integrity",
                "finishing_move_policy": "verified-opening-hero-with-deliberate-semantic-continuation",
                "semantic_montage_policy": "controlled-fallback-from-individually-qualified-payoff-moments",
                "normal_cross_stringout_cut_allowed": False,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    parser.error("Use this module through the V3.1 shadow or production renderer")


if __name__ == "__main__":
    main()

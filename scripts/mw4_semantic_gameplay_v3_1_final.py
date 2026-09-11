from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
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


def _source_key(source: Path) -> str:
    stem = source.stem.lower()
    for key in ("batch2", "week2", "r1"):
        if key in stem:
            return key
    return stem


def _verified_cut_windows(config: dict[str, Any], source_key: str) -> tuple[tuple[float, float], ...]:
    raw = config.get("source_integrity", {}).get("verified_cut_windows", {}).get(source_key, [])
    windows: list[tuple[float, float]] = []
    for item in raw:
        if len(item) != 2:
            continue
        left, right = sorted((float(item[0]), float(item[1])))
        if right > left:
            windows.append((left, right))
    return tuple(sorted(windows))


def _build_hardened_shots(source: Path, duration: float, config: dict[str, Any]) -> tuple[ShotSpan, ...]:
    source_key = _source_key(source)
    automatic = list(refined._scene_cut_times(source, config))
    verified = [(left + right) / 2.0 for left, right in _verified_cut_windows(config, source_key)]
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
    base_timeline: Any,
    shots: tuple[ShotSpan, ...],
    heuristic: tuple[FinishingMoveSpan, ...],
    source: Path,
    config: dict[str, Any],
) -> tuple[FinishingMoveSpan, ...]:
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
                evidence={"visual_verification": 1.0, "third_person_execution_regime": 1.0},
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
            if any(max(span.start, item.start) < min(span.end, item.end) for item in verified):
                continue
            verified.append(span)
    return tuple(sorted(verified, key=lambda item: item.start))


def analyze_source(source: Path, config: dict[str, Any]) -> SemanticTimelineV31:
    base_timeline = refined.core.base.analyze_source(source, config)
    shots = _build_hardened_shots(source, base_timeline.duration, config)
    consolidated = refined.core.consolidate_events(base_timeline, shots)
    engagements = refined.core.cluster_engagements(base_timeline, shots, consolidated, config)
    heuristic = refined.core.detect_finishing_moves(base_timeline, shots, engagements, config)
    finishing = _verified_finishing_moves(base_timeline, shots, heuristic, source, config)
    timeline = SemanticTimelineV31(base_timeline, shots, consolidated, engagements, finishing)
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


def _ranges(mask: np.ndarray, i0: int, i1: int, fps: float) -> list[tuple[float, float]]:
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


def _editable_segments(
    timeline: SemanticTimelineV31,
    start: float,
    end: float,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    config: dict[str, Any],
) -> tuple[tuple[EditSegment, ...], tuple[str, ...], float] | None:
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
            segments.append(EditSegment(round(left, 3), round(right, 3), round(speed, 3), "compressed_traversal"))
            reasons.append(f"compress unexplained traversal {left:.2f}-{right:.2f}s at {speed:.2f}x")
    if removed_equivalent > float(cfg.get("maximum_total_removed_equivalent_seconds", 2.0)):
        return None
    output_duration = sum((segment.end - segment.start) / segment.speed for segment in segments)
    if output_duration < float(config["semantic_editor"].get("minimum_output_seconds", 10.0)):
        return None
    kept_runs: list[float] = []
    for segment in segments:
        s0 = max(i0, int(math.floor(segment.start * fps)))
        s1 = min(i1, int(math.ceil(segment.end * fps)))
        kept_runs.append(refined.core._longest_true_run(unexplained_low[s0:s1], fps) / segment.speed)
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
    opening_mean = refined.core._mean(sig["interest"], timeline, start, min(end, start + 1.25))
    opening_slice = sig["interest"][i0:min(i1, i0 + max(1, int(round(1.1 * fps))))]
    opening_explained = explained[:len(opening_slice)]
    opening_dead = refined.core._longest_true_run((opening_slice < low_threshold) & ~opening_explained, fps)
    opening_quality = np.clip(
        0.43 * opening_mean
        + 0.34 * (1.0 - min(1.0, delay / 0.90))
        + 0.23 * (1.0 - min(1.0, opening_dead / 0.50)),
        0, 1,
    )
    if finishing is not None:
        opening_quality = min(1.0, float(opening_quality) + 0.20 * finishing.confidence)
    payoff_events = [
        event for engagement in engagements for event in engagement.events
        if any(kind in PAYOFF_KINDS for kind in event.kinds)
    ]
    last_payoff_time = payoff_events[-1].time if payoff_events else None
    if finishing is not None and (last_payoff_time is None or finishing.payoff > last_payoff_time):
        last_payoff_time = finishing.payoff
    if last_payoff_time is not None:
        tail = max(0.0, end - last_payoff_time)
        tail_quality = float(np.clip(1.0 - abs(tail - 0.45) / 0.75, 0, 1))
        ending_interest = refined.core._mean(sig["interest"], timeline, max(start, end - 1.0), end)
        ending_quality = 0.66 * tail_quality + 0.34 * ending_interest
    else:
        active = max(
            refined.core._mean(sig["combat"], timeline, max(start, end - 0.8), end),
            refined.core._mean(sig["contact"], timeline, max(start, end - 0.8), end),
        )
        ending_quality = 0.72 * active + 0.28 * refined.core._mean(sig["interest"], timeline, max(start, end - 1.0), end)
    payoff_strength = max(
        [event.confidence for event in payoff_events]
        + ([min(1.0, finishing.confidence + 0.12)] if finishing is not None else [0.0])
    )
    coherence = np.clip(
        0.54 * float(np.mean([eng.confidence for eng in engagements]))
        + 0.26 * min(1.0, len(engagements) / 4.0)
        + 0.20 * (1.0 - min(1.0, low_fraction / 0.45)),
        0, 1,
    )
    retention = np.clip(
        0.27 * float(np.mean(effective_interest))
        + 0.18 * weakest
        + 0.22 * opening_quality
        + 0.18 * ending_quality
        + 0.09 * coherence
        + 0.06 * (1.0 - min(1.0, residual_run / 0.90)),
        0, 1,
    )
    payoff_quality = np.clip(0.72 * payoff_strength + 0.28 * ending_quality, 0, 1)
    return float(opening_quality), float(ending_quality), float(coherence), float(retention), float(payoff_quality), weakest, low_fraction


def _story_gate(story: str, metric: str, config: dict[str, Any], default: float) -> float:
    gates = config.get("story_quality_gates", {})
    return float(gates.get(metric, {}).get(story, gates.get(metric, {}).get("default", default)))


def _segment_duration(segment: EditSegment) -> float:
    return (segment.end - segment.start) / segment.speed


def _source_segments_overlap(a: SemanticPlanV31, b: SemanticPlanV31) -> bool:
    for left in a.segments:
        for right in b.segments:
            if max(left.start, right.start) < min(left.end, right.end):
                return True
    return False


def plan_integrity_violations(
    plan: SemanticPlanV31,
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    source_key: str,
) -> list[str]:
    windows = _verified_cut_windows(config, source_key)
    violations: list[str] = []
    for segment in plan.segments:
        inside_one_shot = any(
            shot.start - 1e-3 <= segment.start and segment.end <= shot.end + 1e-3
            for shot in timeline.shots
        )
        if not inside_one_shot:
            violations.append(f"segment {segment.start:.3f}-{segment.end:.3f} crosses a source-shot boundary")
        for left, right in windows:
            midpoint = (left + right) / 2.0
            if segment.start < midpoint < segment.end:
                violations.append(f"segment {segment.start:.3f}-{segment.end:.3f} crosses verified source cut {left:.3f}-{right:.3f}")
    for index, (left_seg, right_seg) in enumerate(zip(plan.segments, plan.segments[1:])):
        if right_seg.start <= left_seg.end + 0.02:
            continue
        bridge_has_verified_cut = any(
            left_seg.end < (left + right) / 2.0 < right_seg.start
            for left, right in windows
        )
        planned_finishing_montage = (
            plan.story_type == "finishing_move_open"
            and index == 0
            and left_seg.reason == "finishing_move_open_hero"
        )
        if bridge_has_verified_cut and not planned_finishing_montage:
            violations.append(
                f"normal editorial bridge {left_seg.end:.3f}->{right_seg.start:.3f} jumps across a verified source cut"
            )
    if plan.finishing_move is not None:
        if plan.story_type != "finishing_move_open" or plan.effect_profile != "finishing_move_hero":
            violations.append("verified Finishing Move is not routed as finishing_move_open/finishing_move_hero")
        first = plan.segments[0]
        if not (first.start <= plan.finishing_move.start <= first.end):
            violations.append("Finishing Move is not contained in the first output segment")
        opening_delay = (plan.finishing_move.start - first.start) / first.speed
        if opening_delay > 0.48:
            violations.append(f"Finishing Move opening delay is {opening_delay:.3f}s")
    return violations


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


def _normal_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    plans: list[SemanticPlanV31] = []
    for chain in refined._candidate_engagement_chains(timeline, config):
        finishing = refined._find_finishing_for_chain(timeline, chain)
        if finishing is not None:
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
            timeline, start, end, chain, None, residual, config
        )
        story, profile, effect_events, route_reasons = refined.core._route(timeline, chain, None)
        editor = config["semantic_editor"]
        if opening < _story_gate(story, "opening_min", config, float(editor["opening"].get("minimum_quality", 0.42))):
            continue
        if ending < _story_gate(story, "ending_min", config, float(editor["ending"].get("minimum_quality", 0.40))):
            continue
        if retention < _story_gate(story, "retention_min", config, float(config["performance_targets"].get("retention_quality_min", 0.36))):
            continue
        if payoff < _story_gate(story, "payoff_min", config, float(config["performance_targets"].get("payoff_quality_min", 0.34))):
            continue
        if weakest < float(editor["dull"].get("minimum_weak_quarter_interest", 0.27)):
            continue
        if low_fraction > float(editor["dull"].get("maximum_low_interest_fraction", 0.38)):
            continue
        plan = SemanticPlanV31(
            start=start, end=end,
            raw_duration=round(end - start, 3),
            output_duration=round(output_duration, 3),
            score=round(_score_plan(retention, payoff, opening, ending, coherence, weakest, False, config), 5),
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
        if plan_integrity_violations(plan, timeline, config, source_key):
            continue
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


def _finishing_open_plans(
    timeline: SemanticTimelineV31,
    normal_plans: list[SemanticPlanV31],
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
        hero = EditSegment(round(hero_start, 3), round(hero_end, 3), 1.0, "finishing_move_open_hero")
        hero_duration = _segment_duration(hero)
        candidates: list[tuple[float, SemanticPlanV31]] = []
        for continuation in normal_plans:
            if any(max(hero.start, seg.start) < min(hero.end, seg.end) for seg in continuation.segments):
                continue
            combined_duration = hero_duration + continuation.output_duration
            if not (minimum <= combined_duration <= maximum):
                continue
            later = continuation.start >= span.end + 0.25
            distance = max(0.0, continuation.start - span.end) if later else abs(span.start - continuation.end) + 120.0
            preference = continuation.score + (0.08 if later else 0.0) - min(distance, 120.0) * 0.00035
            candidates.append((preference, continuation))
        for _, continuation in sorted(candidates, key=lambda item: item[0], reverse=True)[:3]:
            segments = (hero,) + continuation.segments
            output_duration = sum(_segment_duration(segment) for segment in segments)
            opening = max(0.90, min(1.0, 0.74 + 0.22 * span.confidence))
            ending = continuation.ending_quality
            coherence = min(1.0, 0.10 + 0.88 * continuation.story_coherence)
            retention = min(1.0, 0.18 * opening + 0.82 * continuation.retention_quality)
            payoff = max(0.92, continuation.payoff_quality)
            weakest = continuation.weakest_quarter_interest
            low_fraction = continuation.low_interest_fraction * (continuation.output_duration / output_duration)
            score = _score_plan(retention, payoff, opening, ending, coherence, weakest, True, config)
            plan = SemanticPlanV31(
                start=hero.start,
                end=continuation.end,
                raw_duration=round(sum(seg.end - seg.start for seg in segments), 3),
                output_duration=round(output_duration, 3),
                score=round(score, 5),
                retention_quality=round(retention, 4),
                payoff_quality=round(payoff, 4),
                opening_quality=round(opening, 4),
                ending_quality=round(ending, 4),
                story_coherence=round(coherence, 4),
                weakest_quarter_interest=round(weakest, 4),
                low_interest_fraction=round(low_fraction, 4),
                max_unexplained_low_interest_run_seconds=continuation.max_unexplained_low_interest_run_seconds,
                story_type="finishing_move_open",
                effect_profile="finishing_move_hero",
                segments=segments,
                effect_events=continuation.effect_events,
                engagements=continuation.engagements,
                finishing_move=span,
                editorial_reasons=(
                    "verified Finishing Move opens the clip as the protected hero event",
                    "one deliberate montage transition supplies a qualifying semantic continuation instead of dead padding",
                ) + continuation.editorial_reasons,
            )
            if not plan_integrity_violations(plan, timeline, config, source_key):
                results.append(plan)
    return sorted(results, key=lambda item: item.score, reverse=True)


def build_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[SemanticPlanV31]:
    source_key = getattr(timeline, "_source_key", "")
    normal = _normal_plans(timeline, config, excluded, source_key)
    finishing = _finishing_open_plans(timeline, normal, config, source_key)
    return sorted(finishing + normal, key=lambda item: item.score, reverse=True)


def build_plans_for_source(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    normal = _normal_plans(timeline, config, excluded, source_key)
    finishing = _finishing_open_plans(timeline, normal, config, source_key)
    return sorted(finishing + normal, key=lambda item: item.score, reverse=True)


def select_plans(plans: list[SemanticPlanV31], config: dict[str, Any]) -> list[SemanticPlanV31]:
    maximum = int(config.get("count_per_source_max", 4))
    minimum = int(config.get("minimum_count_per_source", 2))
    selected: list[SemanticPlanV31] = []
    finishing = [plan for plan in plans if plan.story_type == "finishing_move_open"]
    if finishing:
        selected.append(finishing[0])
    seen_story = {plan.story_type for plan in selected}
    for plan in plans:
        if plan in selected or plan.story_type in seen_story:
            continue
        if any(_source_segments_overlap(plan, prior) for prior in selected):
            continue
        selected.append(plan)
        seen_story.add(plan.story_type)
        if len(selected) >= maximum:
            break
    for plan in plans:
        if plan in selected:
            continue
        if any(_source_segments_overlap(plan, prior) for prior in selected):
            continue
        selected.append(plan)
        if len(selected) >= maximum:
            break
    if len(selected) < minimum:
        raise RuntimeError(
            f"Only {len(selected)} hardened V3.1 candidates passed source-integrity/opening/ending/no-dull gates; minimum is {minimum}."
        )
    return sorted(selected, key=lambda item: (item.story_type != "finishing_move_open", item.segments[0].start))


def diagnose_source(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str | None = None,
) -> dict[str, Any]:
    key = source_key or getattr(timeline, "_source_key", "")
    plans = build_plans_for_source(timeline, config, excluded, key)
    return {
        "shot_count": len(timeline.shots),
        "engagement_count": len(timeline.engagements),
        "verified_finishing_move_count": len(timeline.finishing_moves),
        "finishing_moves": [asdict(item) for item in timeline.finishing_moves],
        "verified_source_cut_windows": [list(item) for item in _verified_cut_windows(config, key)],
        "counts": {"pass": len(plans)},
        "candidate_boundaries": [[item.start, item.end, item.story_type, item.score] for item in plans[:20]],
        "dull_policy": "only unexplained low-interest footage is eligible for dull rejection/cutting",
        "source_integrity_policy": "normal plans cannot cross automatic or visually verified source-stringout boundaries",
    }


def _self_test() -> None:
    assert _story_gate("sustained_pressure", "payoff_min", {"story_quality_gates": {"payoff_min": {"sustained_pressure": 0.16}}}, 0.34) == 0.16
    a = SemanticPlanV31(
        0, 10, 10, 10, .7, .5, .5, .5, .5, .5, .4, .1, .2,
        "precision_outcome", "precision_punch", (EditSegment(0, 10, 1, "keep"),), (), (), None, (),
    )
    b = SemanticPlanV31(
        20, 30, 10, 10, .7, .5, .5, .5, .5, .5, .4, .1, .2,
        "impact_payoff", "impact_flash", (EditSegment(20, 30, 1, "keep"),), (), (), None, (),
    )
    assert not _source_segments_overlap(a, b)
    print(json.dumps({
        "self_test": "PASS",
        "candidate_mode": "engagement_driven_hardened_source_integrity",
        "finishing_move_policy": "verified-opening-hero-with-one-deliberate-montage-transition",
        "normal_cross_stringout_cut_allowed": False,
    }))


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

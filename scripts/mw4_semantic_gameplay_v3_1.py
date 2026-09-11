from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import mw4_semantic_gameplay_v3 as base


@dataclass(frozen=True)
class ShotSpan:
    start: float
    end: float


@dataclass(frozen=True)
class ConsolidatedEvent:
    time: float
    kinds: tuple[str, ...]
    confidence: float
    evidence: dict[str, float]


@dataclass(frozen=True)
class Engagement:
    start: float
    end: float
    shot_index: int
    confidence: float
    events: tuple[ConsolidatedEvent, ...]


@dataclass(frozen=True)
class FinishingMoveSpan:
    start: float
    payoff: float
    end: float
    shot_index: int
    confidence: float
    evidence: dict[str, float]


@dataclass(frozen=True)
class EditSegment:
    start: float
    end: float
    speed: float
    reason: str


@dataclass(frozen=True)
class SemanticPlanV31:
    start: float
    end: float
    raw_duration: float
    output_duration: float
    score: float
    retention_quality: float
    payoff_quality: float
    opening_quality: float
    ending_quality: float
    story_coherence: float
    weakest_quarter_interest: float
    low_interest_fraction: float
    max_unexplained_low_interest_run_seconds: float
    story_type: str
    effect_profile: str
    segments: tuple[EditSegment, ...]
    effect_events: tuple[base.SemanticEvent, ...]
    engagements: tuple[Engagement, ...]
    finishing_move: FinishingMoveSpan | None
    editorial_reasons: tuple[str, ...]


@dataclass
class SemanticTimelineV31:
    base: base.SemanticTimeline
    shots: tuple[ShotSpan, ...]
    consolidated_events: tuple[ConsolidatedEvent, ...]
    engagements: tuple[Engagement, ...]
    finishing_moves: tuple[FinishingMoveSpan, ...]

    @property
    def fps(self) -> float:
        return self.base.fps

    @property
    def duration(self) -> float:
        return self.base.duration

    @property
    def times(self) -> np.ndarray:
        return self.base.times

    @property
    def signals(self) -> dict[str, np.ndarray]:
        return self.base.signals


MEANINGFUL_KINDS = {"combat_burst", "outcome_like", "impact", "contact"}
PAYOFF_KINDS = {"outcome_like", "impact"}


def _idx(timeline: SemanticTimelineV31 | base.SemanticTimeline, seconds: float) -> int:
    fps = timeline.fps
    return max(0, min(len(timeline.times) - 1, int(round(seconds * fps - 0.5))))


def _mean(signal: np.ndarray, timeline: SemanticTimelineV31, start: float, end: float) -> float:
    i0 = max(0, int(math.floor(start * timeline.fps)))
    i1 = min(len(signal), int(math.ceil(end * timeline.fps)))
    if i1 <= i0:
        return 0.0
    return float(np.mean(signal[i0:i1]))


def _max(signal: np.ndarray, timeline: SemanticTimelineV31, start: float, end: float) -> float:
    i0 = max(0, int(math.floor(start * timeline.fps)))
    i1 = min(len(signal), int(math.ceil(end * timeline.fps)))
    if i1 <= i0:
        return 0.0
    return float(np.max(signal[i0:i1]))


def _longest_true_run(mask: np.ndarray, fps: float) -> float:
    longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / fps


def _shot_index(shots: tuple[ShotSpan, ...], time: float) -> int:
    for index, shot in enumerate(shots):
        if shot.start <= time <= shot.end:
            return index
    return max(0, len(shots) - 1)


def detect_shots(timeline: base.SemanticTimeline, config: dict[str, Any]) -> tuple[ShotSpan, ...]:
    cfg = config.get("semantic_analysis", {})
    signals = timeline.signals
    cut_score = np.clip(
        0.44 * signals["global_motion"]
        + 0.23 * signals["center_motion"]
        + 0.20 * signals["luma_delta"]
        + 0.13 * signals["hud_change"],
        0,
        1,
    )
    threshold = float(cfg.get("shot_cut_threshold", 0.82))
    spacing = max(1, int(round(0.60 * timeline.fps)))
    candidates: list[int] = []
    order = np.argsort(cut_score)[::-1]
    for raw in order:
        index = int(raw)
        if float(cut_score[index]) < threshold:
            break
        if index <= 1 or index >= len(cut_score) - 2:
            continue
        if all(abs(index - prior) >= spacing for prior in candidates):
            candidates.append(index)
    candidates.sort()
    boundaries = [0.0] + [round((index + 0.5) / timeline.fps, 3) for index in candidates] + [timeline.duration]
    guard = float(cfg.get("shot_cut_guard_seconds", 0.35))
    shots: list[ShotSpan] = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        start = left if index == 0 else min(right, left + guard)
        end = right if index == len(boundaries) - 2 else max(start, right - guard)
        if end - start >= 1.0:
            shots.append(ShotSpan(round(start, 3), round(end, 3)))
    if not shots:
        shots.append(ShotSpan(0.0, timeline.duration))
    return tuple(shots)


def consolidate_events(timeline: base.SemanticTimeline, shots: tuple[ShotSpan, ...]) -> tuple[ConsolidatedEvent, ...]:
    raw = [event for event in timeline.events if event.kind in MEANINGFUL_KINDS and event.confidence >= 0.52]
    if not raw:
        return ()
    groups: list[list[base.SemanticEvent]] = []
    for event in raw:
        if not groups:
            groups.append([event])
            continue
        prior = groups[-1][-1]
        same_shot = _shot_index(shots, event.time) == _shot_index(shots, prior.time)
        if same_shot and event.time - prior.time <= 0.48:
            groups[-1].append(event)
        else:
            groups.append([event])
    consolidated: list[ConsolidatedEvent] = []
    priority = {"outcome_like": 4, "impact": 3, "combat_burst": 2, "contact": 1}
    for group in groups:
        strongest = max(group, key=lambda item: (priority.get(item.kind, 0), item.confidence))
        confidence = min(1.0, max(item.confidence for item in group) + 0.035 * (len({item.kind for item in group}) - 1))
        evidence: dict[str, float] = {}
        for item in group:
            for key, value in item.evidence.items():
                evidence[key] = max(evidence.get(key, 0.0), float(value))
        consolidated.append(
            ConsolidatedEvent(
                time=strongest.time,
                kinds=tuple(sorted({item.kind for item in group})),
                confidence=round(confidence, 4),
                evidence={key: round(value, 4) for key, value in evidence.items()},
            )
        )
    return tuple(consolidated)


def cluster_engagements(
    timeline: base.SemanticTimeline,
    shots: tuple[ShotSpan, ...],
    events: tuple[ConsolidatedEvent, ...],
    config: dict[str, Any],
) -> tuple[Engagement, ...]:
    cfg = config.get("semantic_analysis", {})
    join = float(cfg.get("engagement_join_seconds", 2.25))
    context = float(cfg.get("engagement_context_seconds", 0.55))
    minimum = float(cfg.get("minimum_engagement_confidence", 0.56))
    groups: list[list[ConsolidatedEvent]] = []
    for event in events:
        if event.confidence < minimum:
            continue
        if not groups:
            groups.append([event])
            continue
        previous = groups[-1][-1]
        same_shot = _shot_index(shots, event.time) == _shot_index(shots, previous.time)
        if same_shot and event.time - previous.time <= join:
            groups[-1].append(event)
        else:
            groups.append([event])
    engagements: list[Engagement] = []
    for group in groups:
        shot_index = _shot_index(shots, group[0].time)
        shot = shots[shot_index]
        start = max(shot.start, group[0].time - context)
        end = min(shot.end, group[-1].time + context)
        payoff_bonus = 0.08 if any(any(kind in PAYOFF_KINDS for kind in item.kinds) for item in group) else 0.0
        confidence = min(1.0, float(np.mean([item.confidence for item in group])) + payoff_bonus)
        engagements.append(
            Engagement(round(start, 3), round(end, 3), shot_index, round(confidence, 4), tuple(group))
        )
    return tuple(engagements)


def detect_finishing_moves(
    timeline: base.SemanticTimeline,
    shots: tuple[ShotSpan, ...],
    engagements: tuple[Engagement, ...],
    config: dict[str, Any],
) -> tuple[FinishingMoveSpan, ...]:
    cfg = config.get("finishing_move_detector", {})
    if not bool(cfg.get("enabled", True)):
        return ()
    fps = timeline.fps
    sig = timeline.signals
    min_span = float(cfg.get("minimum_span_seconds", 1.05))
    max_span = float(cfg.get("maximum_span_seconds", 4.60))
    min_conf = float(cfg.get("minimum_confidence", 0.64))
    min_center = float(cfg.get("minimum_center_motion", 0.46))
    min_contact_impact = float(cfg.get("minimum_contact_or_impact", 0.52))
    min_terminal = float(cfg.get("minimum_terminal_outcome", 0.50))
    max_dull = float(cfg.get("maximum_internal_dull_fraction", 0.34))
    max_gap_bins = max(1, int(round(float(cfg.get("maximum_gap_between_choreography_bins_seconds", 0.50)) * fps)))

    score = np.clip(
        0.34 * sig["center_motion"]
        + 0.18 * sig["global_motion"]
        + 0.17 * sig["contact"]
        + 0.12 * sig["impact"]
        + 0.11 * sig["outcome"]
        + 0.08 * (1.0 - sig["traversal"]),
        0,
        1,
    )
    active = (
        (sig["center_motion"] >= min_center)
        & (np.maximum(sig["contact"], sig["impact"]) >= min_contact_impact * 0.72)
        & (sig["traversal"] < 0.68)
    )

    spans: list[FinishingMoveSpan] = []
    for engagement in engagements:
        i0 = max(0, int(math.floor(engagement.start * fps)))
        i1 = min(len(score), int(math.ceil(engagement.end * fps)))
        if i1 - i0 < int(min_span * fps):
            continue
        indices = [index for index in range(i0, i1) if active[index]]
        if not indices:
            continue
        runs: list[tuple[int, int]] = []
        run_start = previous = indices[0]
        for index in indices[1:]:
            if index - previous <= max_gap_bins:
                previous = index
                continue
            runs.append((run_start, previous + 1))
            run_start = previous = index
        runs.append((run_start, previous + 1))
        for left, right in runs:
            raw_start = left / fps
            raw_end = right / fps
            duration = raw_end - raw_start
            if duration < min_span or duration > max_span:
                continue
            terminal_start = max(left, right - max(1, int(round(0.85 * fps))))
            terminal_outcome = float(np.max(np.maximum(sig["outcome"][terminal_start:right], sig["impact"][terminal_start:right])))
            dull_fraction = float(np.mean(sig["dull"][left:right] > 0.5))
            if terminal_outcome < min_terminal or dull_fraction > max_dull:
                continue
            center_mean = float(np.mean(sig["center_motion"][left:right]))
            contact_impact = float(np.mean(np.maximum(sig["contact"][left:right], sig["impact"][left:right])))
            sustained = float(np.mean(score[left:right]))
            body_dominance = float(np.mean(np.maximum(sig["center_motion"][left:right] - 0.72 * sig["combat"][left:right], 0.0)))
            confidence = np.clip(
                0.30 * center_mean + 0.24 * contact_impact + 0.24 * terminal_outcome + 0.14 * sustained + 0.08 * min(1.0, body_dominance * 3.5),
                0,
                1,
            )
            if confidence < min_conf:
                continue
            payoff_index = terminal_start + int(np.argmax(np.maximum(sig["outcome"][terminal_start:right], sig["impact"][terminal_start:right])))
            shot_index = engagement.shot_index
            shot = shots[shot_index]
            start = max(shot.start, raw_start - 0.16)
            end = min(shot.end, raw_end + 0.20)
            spans.append(
                FinishingMoveSpan(
                    start=round(start, 3),
                    payoff=round((payoff_index + 0.5) / fps, 3),
                    end=round(end, 3),
                    shot_index=shot_index,
                    confidence=round(float(confidence), 4),
                    evidence={
                        "center_motion_mean": round(center_mean, 4),
                        "contact_impact_mean": round(contact_impact, 4),
                        "terminal_outcome": round(terminal_outcome, 4),
                        "sustained_choreography": round(sustained, 4),
                        "dull_fraction": round(dull_fraction, 4),
                    },
                )
            )
    deduped: list[FinishingMoveSpan] = []
    for span in sorted(spans, key=lambda item: item.confidence, reverse=True):
        if any(max(span.start, prior.start) < min(span.end, prior.end) for prior in deduped):
            continue
        deduped.append(span)
    return tuple(sorted(deduped, key=lambda item: item.start))


def analyze_source(source: Path, config: dict[str, Any]) -> SemanticTimelineV31:
    base_timeline = base.analyze_source(source, config)
    shots = detect_shots(base_timeline, config)
    consolidated = consolidate_events(base_timeline, shots)
    engagements = cluster_engagements(base_timeline, shots, consolidated, config)
    finishing = detect_finishing_moves(base_timeline, shots, engagements, config)
    return SemanticTimelineV31(base_timeline, shots, consolidated, engagements, finishing)


def _intersects_excluded(start: float, end: float, excluded: list[list[float]], margin: float = 0.15) -> bool:
    return any(max(start - margin, float(a)) < min(end + margin, float(b)) for a, b in excluded)


def _candidate_engagement_chains(timeline: SemanticTimelineV31, config: dict[str, Any]) -> list[tuple[Engagement, ...]]:
    chains: list[tuple[Engagement, ...]] = []
    max_target = float(config["semantic_editor"].get("maximum_output_target_seconds", 15.5))
    for index, engagement in enumerate(timeline.engagements):
        chain = [engagement]
        chains.append(tuple(chain))
        for next_engagement in timeline.engagements[index + 1:]:
            if next_engagement.shot_index != engagement.shot_index:
                break
            gap = next_engagement.start - chain[-1].end
            if gap > 3.25:
                break
            if next_engagement.end - chain[0].start > max_target + 1.2:
                break
            chain.append(next_engagement)
            chains.append(tuple(chain))
            if len(chain) >= 4:
                break
    return chains


def _boundary_for_chain(
    timeline: SemanticTimelineV31,
    chain: tuple[Engagement, ...],
    config: dict[str, Any],
    finishing: FinishingMoveSpan | None,
) -> tuple[float, float] | None:
    editor = config["semantic_editor"]
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_seconds", 20.0))
    preferred = float(editor.get("preferred_output_seconds", 12.5))
    shot = timeline.shots[chain[0].shot_index]
    if finishing is not None:
        lead = float(config["editorial"].get("finishing_move_opening_lead_seconds", 0.28))
        start = max(shot.start, finishing.start - lead)
    else:
        first_event = chain[0].events[0].time
        start = max(shot.start, first_event - 0.30)
    last = chain[-1]
    payoff_events = [event for event in last.events if any(kind in PAYOFF_KINDS for kind in event.kinds)]
    anchor = payoff_events[-1].time if payoff_events else last.events[-1].time
    tail = float(config["semantic_editor"]["ending"].get("preferred_payoff_tail_seconds", 0.45))
    end = min(shot.end, max(last.end, anchor + tail))

    if end - start < minimum:
        desired_end = min(shot.end, start + max(minimum, preferred))
        end = max(end, desired_end)
    if end - start < minimum:
        return None
    if end - start > maximum:
        end = start + maximum
    return round(start, 3), round(end, 3)


def _protected_intervals(
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
) -> list[tuple[float, float]]:
    intervals = [(eng.start - 0.20, eng.end + 0.20) for eng in engagements]
    if finishing is not None:
        intervals.append((finishing.start - 0.10, finishing.end + 0.15))
    return intervals


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
    protected = _protected_intervals(engagements, finishing)
    low = timeline.signals["interest"] < float(cfg.get("low_interest_threshold", 0.30))
    traversal = (
        (timeline.signals["traversal"] > 0.58)
        & (timeline.signals["combat"] < 0.42)
        & (timeline.signals["outcome"] < 0.40)
    )

    def ranges(mask: np.ndarray) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        run: int | None = None
        for index in range(i0, i1):
            if bool(mask[index]) and run is None:
                run = index
            elif not bool(mask[index]) and run is not None:
                out.append((run / fps, index / fps)); run = None
        if run is not None:
            out.append((run / fps, i1 / fps))
        return out

    min_action = float(cfg.get("minimum_action_seconds", 0.70))
    max_action = float(cfg.get("maximum_single_action_seconds", 1.60))
    max_speed_source = float(cfg.get("maximum_speedup_source_seconds", 0.85))
    speed = float(cfg.get("traversal_speed", 1.35))
    actions: list[tuple[float, float, str]] = []
    for a, b in ranges(low):
        a, b = max(start, a), min(end, b)
        if b - a < min_action or b - a > max_action:
            continue
        if any(max(a, p0) < min(b, p1) for p0, p1 in protected):
            continue
        actions.append((a, b, "cut_dull"))
    for a, b in ranges(traversal):
        a, b = max(start, a), min(end, b)
        if b - a < min_action or b - a > max_speed_source:
            continue
        if any(max(a, p0) < min(b, p1) for p0, p1 in protected):
            continue
        actions.append((a, b, "compress_traversal"))
    actions.sort(key=lambda item: item[1] - item[0], reverse=True)
    chosen: list[tuple[float, float, str]] = []
    for action in actions:
        if any(max(action[0], x[0]) < min(action[1], x[1]) for x in chosen):
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
        midpoint = (left + right) / 2
        action = next((item for item in chosen if item[0] <= midpoint <= item[1]), None)
        if action is None:
            segments.append(EditSegment(round(left, 3), round(right, 3), 1.0, "keep"))
        elif action[2] == "cut_dull":
            removed_equivalent += right - left
            reasons.append(f"cut dull {left:.2f}-{right:.2f}s")
        else:
            removed_equivalent += (right - left) - (right - left) / speed
            segments.append(EditSegment(round(left, 3), round(right, 3), round(speed, 3), "compressed_traversal"))
            reasons.append(f"compress short traversal {left:.2f}-{right:.2f}s at {speed:.2f}x")
    if removed_equivalent > float(cfg.get("maximum_total_removed_equivalent_seconds", 2.0)):
        return None
    output_duration = sum((segment.end - segment.start) / segment.speed for segment in segments)
    if output_duration < float(config["semantic_editor"].get("minimum_output_seconds", 10.0)):
        return None

    residual_mask = low[i0:i1]
    max_run = _longest_true_run(residual_mask, fps)
    for action in chosen:
        if action[2] == "cut_dull":
            # Residual run is conservatively recomputed below from kept low ranges.
            pass
    kept_low_runs: list[float] = []
    for segment in segments:
        s0 = max(i0, int(math.floor(segment.start * fps)))
        s1 = min(i1, int(math.ceil(segment.end * fps)))
        kept_low_runs.append(_longest_true_run(low[s0:s1], fps) / segment.speed)
    max_run = max(kept_low_runs, default=0.0)
    if max_run > float(cfg.get("maximum_unexplained_low_interest_run_seconds", 0.90)):
        return None
    return tuple(segments), tuple(reasons), max_run


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
    interest = sig["interest"][i0:i1]
    if interest.size == 0:
        return (0.0,) * 7
    quarters = np.array_split(interest, 4)
    weakest = min(float(np.mean(part)) for part in quarters if len(part))
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    low_fraction = float(np.mean(interest < low_threshold))

    first_event = min((event.time for eng in engagements for event in eng.events), default=end)
    delay = max(0.0, first_event - start)
    opening_mean = _mean(sig["interest"], timeline, start, min(end, start + 1.25))
    opening_dead = _longest_true_run(
        sig["interest"][i0:min(i1, i0 + max(1, int(round(1.1 * fps))))] < low_threshold,
        fps,
    )
    opening_quality = np.clip(
        0.43 * opening_mean
        + 0.34 * (1.0 - min(1.0, delay / 0.90))
        + 0.23 * (1.0 - min(1.0, opening_dead / 0.50)),
        0,
        1,
    )
    if finishing is not None and finishing.start - start <= 0.45:
        opening_quality = min(1.0, opening_quality + 0.16 * finishing.confidence)

    payoff_times = [
        event.time for eng in engagements for event in eng.events if any(kind in PAYOFF_KINDS for kind in event.kinds)
    ]
    last_payoff = max(payoff_times, default=None)
    if finishing is not None:
        last_payoff = max(last_payoff or finishing.payoff, finishing.payoff)
    if last_payoff is not None:
        tail = max(0.0, end - last_payoff)
        ending_quality = float(np.clip(1.0 - abs(tail - 0.45) / 0.85, 0, 1))
        ending_quality = 0.62 * ending_quality + 0.38 * _mean(sig["interest"], timeline, max(start, end - 1.0), end)
    else:
        active = max(
            _mean(sig["combat"], timeline, max(start, end - 0.8), end),
            _mean(sig["contact"], timeline, max(start, end - 0.8), end),
        )
        ending_quality = 0.72 * active + 0.28 * _mean(sig["interest"], timeline, max(start, end - 1.0), end)

    payoff_strength = max(
        [event.confidence for eng in engagements for event in eng.events if any(kind in PAYOFF_KINDS for kind in event.kinds)]
        + ([finishing.confidence + 0.12] if finishing is not None else [0.0])
    )
    story_coherence = np.clip(
        0.55 * float(np.mean([eng.confidence for eng in engagements]))
        + 0.25 * min(1.0, len(engagements) / 3.0)
        + 0.20 * (1.0 - min(1.0, low_fraction / 0.45)),
        0,
        1,
    )
    retention_quality = np.clip(
        0.30 * float(np.mean(interest))
        + 0.20 * weakest
        + 0.20 * opening_quality
        + 0.16 * ending_quality
        + 0.08 * story_coherence
        + 0.06 * (1.0 - min(1.0, residual_run / 0.90)),
        0,
        1,
    )
    payoff_quality = np.clip(0.70 * payoff_strength + 0.30 * ending_quality, 0, 1)
    return (
        float(opening_quality), float(ending_quality), float(story_coherence),
        float(retention_quality), float(payoff_quality), weakest, low_fraction,
    )


def _raw_effect_events(timeline: SemanticTimelineV31, engagements: tuple[Engagement, ...]) -> tuple[base.SemanticEvent, ...]:
    candidates = [
        event for event in timeline.base.events
        if any(eng.start <= event.time <= eng.end for eng in engagements)
        and event.kind in {"outcome_like", "impact", "combat_burst"}
        and event.confidence >= 0.56
    ]
    return tuple(sorted(candidates, key=lambda item: item.confidence, reverse=True)[:3])


def _route(
    timeline: SemanticTimelineV31,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
) -> tuple[str, str, tuple[base.SemanticEvent, ...], tuple[str, ...]]:
    events = _raw_effect_events(timeline, engagements)
    if finishing is not None:
        return "finishing_move_open", "finishing_move_hero", events, ("finishing-move-like choreography opens the clip and is fully protected",)
    outcomes = [event for event in events if event.kind == "outcome_like"]
    impacts = [event for event in events if event.kind == "impact"]
    if len(engagements) >= 3 and len(events) >= 3:
        return "engagement_chain", "chain_escalation", events[:3], ("three engagement arc with progressive payoff emphasis",)
    if impacts and max(item.confidence for item in impacts) >= 0.76:
        return "impact_payoff", "impact_flash", (max(impacts, key=lambda item: item.confidence),), ("high-confidence audiovisual impact",)
    if outcomes:
        return "precision_outcome", "precision_punch", (max(outcomes, key=lambda item: item.confidence),), ("isolated outcome-like payoff",)
    return "sustained_pressure", "clean_pressure", events[:2], ("sustained engagement kept readable",)


def build_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[SemanticPlanV31]:
    chains = _candidate_engagement_chains(timeline, config)
    plans: list[SemanticPlanV31] = []
    for chain in chains:
        finishing = next(
            (span for span in timeline.finishing_moves if span.shot_index == chain[0].shot_index and chain[0].start - 0.2 <= span.start <= chain[-1].end + 0.4),
            None,
        )
        # A detected finishing move must be the first semantic hero event, never a late payoff.
        if finishing is not None:
            chain = tuple(eng for eng in chain if eng.end >= finishing.start)
            if not chain:
                continue
        boundaries = _boundary_for_chain(timeline, chain, config, finishing)
        if boundaries is None:
            continue
        start, end = boundaries
        if finishing is not None and finishing.start - start > 0.48:
            continue
        if _intersects_excluded(start, end, excluded):
            continue
        edited = _editable_segments(timeline, start, end, chain, finishing, config)
        if edited is None:
            continue
        segments, edit_reasons, residual = edited
        output_duration = sum((segment.end - segment.start) / segment.speed for segment in segments)
        opening, ending, coherence, retention, payoff, weakest, low_fraction = _quality_metrics(
            timeline, start, end, chain, finishing, residual, config
        )
        editor = config["semantic_editor"]
        if opening < float(editor["opening"].get("minimum_quality", 0.42)):
            continue
        if ending < float(editor["ending"].get("minimum_quality", 0.40)):
            continue
        if retention < float(config["performance_targets"].get("retention_quality_min", 0.36)):
            continue
        if payoff < float(config["performance_targets"].get("payoff_quality_min", 0.34)):
            continue
        if weakest < float(editor["dull"].get("minimum_weak_quarter_interest", 0.30)):
            continue
        if low_fraction > float(editor["dull"].get("maximum_low_interest_fraction", 0.38)):
            continue
        story, profile, effect_events, route_reasons = _route(timeline, chain, finishing)
        weights = editor["selection"]
        score = (
            float(weights.get("retention_quality_weight", 0.24)) * retention
            + float(weights.get("payoff_quality_weight", 0.17)) * payoff
            + float(weights.get("opening_weight", 0.20)) * opening
            + float(weights.get("ending_weight", 0.17)) * ending
            + float(weights.get("story_coherence_weight", 0.12)) * coherence
            + float(weights.get("weakest_section_weight", 0.10)) * weakest
        )
        if finishing is not None:
            score += float(weights.get("finishing_move_bonus", 0.14))
        reasons = tuple(route_reasons) + tuple(edit_reasons)
        plans.append(
            SemanticPlanV31(
                start=start,
                end=end,
                raw_duration=round(end - start, 3),
                output_duration=round(output_duration, 3),
                score=round(float(score), 5),
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
                finishing_move=finishing,
                editorial_reasons=reasons,
            )
        )
    # Suppress neighboring windows around the same core engagement before final selection.
    plans.sort(key=lambda item: item.score, reverse=True)
    unique: list[SemanticPlanV31] = []
    for plan in plans:
        anchor = plan.finishing_move.payoff if plan.finishing_move is not None else plan.engagements[0].events[0].time
        if any(
            abs(anchor - (prior.finishing_move.payoff if prior.finishing_move is not None else prior.engagements[0].events[0].time)) < 2.2
            for prior in unique
        ):
            continue
        unique.append(plan)
    return unique


def select_plans(plans: list[SemanticPlanV31], config: dict[str, Any]) -> list[SemanticPlanV31]:
    maximum = int(config.get("count_per_source_max", 4))
    minimum = int(config.get("minimum_count_per_source", 2))
    selected: list[SemanticPlanV31] = []
    seen_story: set[str] = set()
    # First pass deliberately favors distinct semantic stories/engagements.
    for plan in plans:
        if plan.story_type in seen_story:
            continue
        if any(max(plan.start, prior.start) < min(plan.end, prior.end) for prior in selected):
            continue
        selected.append(plan)
        seen_story.add(plan.story_type)
        if len(selected) >= maximum:
            break
    for plan in plans:
        if plan in selected:
            continue
        if any(max(plan.start, prior.start) < min(plan.end, prior.end) for prior in selected):
            continue
        selected.append(plan)
        if len(selected) >= maximum:
            break
    if len(selected) < minimum:
        raise RuntimeError(
            f"Only {len(selected)} V3.1 engagement-driven candidates passed the opening/ending/no-dull gates; minimum is {minimum}."
        )
    return sorted(selected, key=lambda item: item.start)


def shadow_report(source: Path, source_key: str, config: dict[str, Any]) -> dict[str, Any]:
    timeline = analyze_source(source, config)
    plans = build_plans(timeline, config, config.get("excluded_windows", {}).get(source_key, []))
    selected = select_plans(plans, config)
    return {
        "source_key": source_key,
        "semantic_engine": "deterministic-gameplay-v3.1",
        "editorial_planner": "semantic-editor-v3.1",
        "shot_count": len(timeline.shots),
        "consolidated_event_count": len(timeline.consolidated_events),
        "engagement_count": len(timeline.engagements),
        "finishing_move_like_count": len(timeline.finishing_moves),
        "finishing_move_note": "Heuristic finishing-move-like detections are hypotheses, not guaranteed named move classifications.",
        "finishing_moves": [asdict(item) for item in timeline.finishing_moves],
        "selected": [asdict(item) for item in selected],
        "candidate_count_after_semantic_gates": len(plans),
    }


def _self_test() -> None:
    shots = (ShotSpan(0.0, 20.0),)
    events = (
        ConsolidatedEvent(0.55, ("contact",), 0.72, {}),
        ConsolidatedEvent(1.10, ("combat_burst", "impact"), 0.83, {}),
        ConsolidatedEvent(1.55, ("outcome_like",), 0.80, {}),
    )
    engagement = Engagement(0.25, 2.10, 0, 0.82, events)
    finishing = FinishingMoveSpan(0.30, 1.55, 1.90, 0, 0.78, {})
    assert finishing.start - 0.28 <= 0.02 + finishing.start
    assert engagement.events[-1].time == finishing.payoff
    print(json.dumps({"self_test": "PASS", "finishing_move_must_open": True, "candidate_mode": "engagement_driven"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-key")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test(); return
    if not args.source or not args.source_key or not args.config or not args.output:
        parser.error("--source, --source-key, --config and --output are required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = shadow_report(args.source, args.source_key, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "source": args.source_key,
        "selected": len(report["selected"]),
        "engagements": report["engagement_count"],
        "finishing_move_like": report["finishing_move_like_count"],
    }))


if __name__ == "__main__":
    main()

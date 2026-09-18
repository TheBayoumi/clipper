from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import mw4_semantic_gameplay_v3 as base


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
    return max(0, min(len(timeline.times) - 1, round(seconds * fps - 0.5)))


def _mean(signal: np.ndarray, timeline: SemanticTimelineV31, start: float, end: float) -> float:
    i0 = max(0, math.floor(start * timeline.fps))
    i1 = min(len(signal), math.ceil(end * timeline.fps))
    if i1 <= i0:
        return 0.0
    return float(np.mean(signal[i0:i1]))


def _max(signal: np.ndarray, timeline: SemanticTimelineV31, start: float, end: float) -> float:
    i0 = max(0, math.floor(start * timeline.fps))
    i1 = min(len(signal), math.ceil(end * timeline.fps))
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


def _intersects_excluded(
    start: float,
    end: float,
    excluded: list[list[float]],
    margin: float = 0.15,
) -> bool:
    return any(
        max(start - margin, float(left)) < min(end + margin, float(right))
        for left, right in excluded
    )


def scene_cut_times(source: Path, config: dict[str, Any]) -> list[float]:
    """Return deterministic source-stringout cut hypotheses only.

    Scene detection is evidence construction, not editorial planning. The low-rate
    analysis stream intentionally separates source-cut evidence from gameplay motion.
    """
    cfg = config.get("semantic_analysis", {})
    threshold = float(cfg.get("scene_change_threshold", 0.34))
    analysis_fps = float(cfg.get("scene_change_fps", 6.0))
    vf = (
        f"fps={analysis_fps},scale=320:-1:flags=fast_bilinear,"
        f"select='gt(scene,{threshold})',showinfo"
    )
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(source),
            "-an",
            "-vf",
            vf,
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    times: list[float] = []
    for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr):
        value = float(match.group(1))
        if value > 0.35 and all(abs(value - prior) >= 0.70 for prior in times):
            times.append(value)
    return sorted(times)


def consolidate_events(
    timeline: base.SemanticTimeline,
    shots: tuple[ShotSpan, ...],
) -> tuple[ConsolidatedEvent, ...]:
    raw = [
        event
        for event in timeline.events
        if event.kind in MEANINGFUL_KINDS and event.confidence >= 0.52
    ]
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
        strongest = max(
            group,
            key=lambda item: (priority.get(item.kind, 0), item.confidence),
        )
        confidence = min(
            1.0,
            max(item.confidence for item in group)
            + 0.035 * (len({item.kind for item in group}) - 1),
        )
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
    """Build coarse evidence envelopes only.

    These envelopes are not canonical hostile engagements. They are intentionally
    broad inputs to direct-interaction verification and exact combat-island splitting.
    """
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
        payoff_bonus = (
            0.08 if any(any(kind in PAYOFF_KINDS for kind in item.kinds) for item in group) else 0.0
        )
        confidence = min(
            1.0,
            float(np.mean([item.confidence for item in group])) + payoff_bonus,
        )
        engagements.append(
            Engagement(
                round(start, 3),
                round(end, 3),
                shot_index,
                round(confidence, 4),
                tuple(group),
            )
        )
    return tuple(engagements)


def discover_finishing_moves(
    timeline: base.SemanticTimeline,
    shots: tuple[ShotSpan, ...],
    engagements: tuple[Engagement, ...],
    config: dict[str, Any],
) -> tuple[FinishingMoveSpan, ...]:
    """Return heuristic Finishing Move hypotheses for diagnostics/review only.

    Production planning never calls this function to establish a verified move.
    Canonical eligibility comes exclusively from manually verified campaign spans.
    """
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
    max_gap_bins = max(
        1,
        round(float(cfg.get("maximum_gap_between_choreography_bins_seconds", 0.50)) * fps),
    )

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
        i0 = max(0, math.floor(engagement.start * fps))
        i1 = min(len(score), math.ceil(engagement.end * fps))
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
            terminal_start = max(left, right - max(1, round(0.85 * fps)))
            terminal_outcome = float(
                np.max(
                    np.maximum(
                        sig["outcome"][terminal_start:right],
                        sig["impact"][terminal_start:right],
                    )
                )
            )
            dull_fraction = float(np.mean(sig["dull"][left:right] > 0.5))
            if terminal_outcome < min_terminal or dull_fraction > max_dull:
                continue
            center_mean = float(np.mean(sig["center_motion"][left:right]))
            contact_impact = float(
                np.mean(
                    np.maximum(
                        sig["contact"][left:right],
                        sig["impact"][left:right],
                    )
                )
            )
            sustained = float(np.mean(score[left:right]))
            body_dominance = float(
                np.mean(
                    np.maximum(
                        sig["center_motion"][left:right] - 0.72 * sig["combat"][left:right],
                        0.0,
                    )
                )
            )
            confidence = np.clip(
                0.30 * center_mean
                + 0.24 * contact_impact
                + 0.24 * terminal_outcome
                + 0.14 * sustained
                + 0.08 * min(1.0, body_dominance * 3.5),
                0,
                1,
            )
            if confidence < min_conf:
                continue
            payoff_index = terminal_start + int(
                np.argmax(
                    np.maximum(
                        sig["outcome"][terminal_start:right],
                        sig["impact"][terminal_start:right],
                    )
                )
            )
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


def architecture_self_test() -> None:
    forbidden = {
        "analyze_source",
        "build_plans",
        "select_plans",
        "shadow_report",
        "_candidate_engagement_chains",
        "_boundary_for_chain",
        "_editable_segments",
    }
    present = forbidden.intersection(globals())
    if present:
        raise AssertionError(f"evidence module exposes planner route: {sorted(present)}")

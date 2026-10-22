"""Canonical Clipper beat-composition search for verified gameplay payoffs."""

from __future__ import annotations

import math
from collections import Counter
from itertools import pairwise
from typing import Any

import numpy as np

from . import analysis, combat, interaction, quality
from . import semantics as core

EditSegment = core.EditSegment
Engagement = core.Engagement
SemanticPlan = core.SemanticPlan
_EPS = 1e-3


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config["semantic_editor"].get("beat_editing") or {})


def _verified_events(timeline: Any, config: dict[str, Any]) -> list[Any]:
    return [
        event
        for event in timeline.consolidated_events
        if interaction.hostile_decision(event, config).hostile
    ]


def _safe_beat(
    timeline: Any,
    segment: EditSegment,
    config: dict[str, Any],
    source_key: str,
    excluded: list[list[float]],
) -> str | None:
    if not any(
        shot.start - _EPS <= segment.start and segment.end <= shot.end + _EPS
        for shot in timeline.shots
    ):
        return "source_shot_boundary"
    if any(
        segment.start < (left + right) / 2 < segment.end
        for left, right in quality._verified_cut_windows(config, source_key)
    ):
        return "certified_stringout_cut"
    if any(segment.start < right and segment.end > left for left, right in excluded):
        return "excluded_source_region"
    if analysis._protected_finishing_overlap(timeline, segment.start, segment.end, config):
        return "protected_finishing_move_choreography"
    if combat.segment_crosses_gap(timeline, segment.start, segment.end):
        return "unexplained_break_inside_beat"
    death = np.asarray(timeline.signals.get("player_death", []), dtype=np.float32)
    if death.size:
        left = max(0, math.floor(segment.start * timeline.fps))
        right = min(death.size, math.ceil(segment.end * timeline.fps))
        if right > left and float(death[left:right].max()) >= 0.5:
            return "confirmed_death_inside_beat"
    return None


def _evidence(
    events: list[Any], segment: EditSegment, timeline: Any, config: dict[str, Any]
) -> Engagement | None:
    verified = tuple(
        event for event in events if segment.start - _EPS <= float(event.time) <= segment.end + _EPS
    )
    if not verified:
        return None
    return Engagement(
        segment.start,
        segment.end,
        int(core._shot_index(timeline.shots, float(verified[0].time))),
        round(float(np.mean([interaction.hostile_decision(e, config).score for e in verified])), 4),
        verified,
    )


def _edited_metrics(
    timeline: Any,
    segments: tuple[EditSegment, ...],
    engagements: tuple[Engagement, ...],
    anchor_time: float,
    config: dict[str, Any],
) -> dict[str, float]:
    """Evaluate the actual retained time sequence, not discarded source footage."""
    fps = float(timeline.fps)
    signal = np.asarray(timeline.signals["interest"], dtype=np.float32)
    parts = [
        signal[max(0, math.floor(s.start * fps)) : min(len(signal), math.ceil(s.end * fps))]
        for s in segments
    ]
    if any(len(part) == 0 for part in parts):
        raise ValueError("empty source beat")
    interest = np.concatenate(parts)
    explained = np.zeros(interest.size, dtype=bool)
    cursor = 0
    for part, segment, engagement in zip(parts, segments, engagements, strict=True):
        for event in engagement.events:
            at = float(event.time) - segment.start
            left = max(0, math.floor((at - 0.35) * fps))
            right = min(len(part), math.ceil((at + 0.55) * fps))
            explained[cursor + left : cursor + right] = True
        cursor += len(part)
    dull = config["semantic_editor"]["dull"]
    threshold = float(dull.get("low_interest_threshold", 0.30))
    floor = float(dull.get("explained_interest_floor", 0.26))
    edited_interest = np.where(explained, np.maximum(interest, floor), interest)
    low = (interest < threshold) & ~explained
    quarters = np.array_split(edited_interest, 4)
    weakest = min(float(np.mean(q)) for q in quarters if q.size)
    residual = core._longest_true_run(low, fps)
    low_fraction = float(np.mean(low))
    first_action = min(float(event.time) for event in engagements[0].events)
    opening_delay = max(0.0, first_action - segments[0].start)
    opening = float(
        np.clip(
            0.48 * float(np.mean(edited_interest[: max(1, round(1.25 * fps))]))
            + 0.35 * (1 - min(1.0, opening_delay / 0.9))
            + 0.17 * (1 - min(1.0, core._longest_true_run(low[: max(1, round(fps))], fps) / 0.5)),
            0,
            1,
        )
    )
    tail = segments[-1].end - anchor_time
    ending = float(
        np.clip(
            0.65 * (1 - min(1.0, abs(tail - 0.45) / 0.85))
            + 0.35 * float(np.mean(edited_interest[-max(1, round(fps)) :])),
            0,
            1,
        )
    )
    confirmed = [
        event for event in engagements[-1].events if abs(float(event.time) - anchor_time) <= 0.002
    ]
    if not confirmed:
        raise ValueError("payoff was not retained in the final beat")
    payoff = float(
        np.clip(0.73 * max(float(event.confidence) for event in confirmed) + 0.27 * ending, 0, 1)
    )
    gaps = [right.start - left.end for left, right in pairwise(segments)]
    gap = max(gaps)
    limit = float(_settings(config).get("max_source_gap_seconds", 18.0))
    coherence = float(
        np.clip(
            0.62 * float(np.mean([e.confidence for e in engagements]))
            + 0.30 * (1 - min(1.0, gap / limit))
            + 0.08 * (1 - low_fraction),
            0,
            1,
        )
    )
    retention = float(
        np.clip(
            0.29 * float(np.mean(edited_interest))
            + 0.21 * weakest
            + 0.19 * opening
            + 0.15 * ending
            + 0.10 * coherence
            + 0.06 * (1 - min(1.0, residual / 0.9)),
            0,
            1,
        )
    )
    return dict(
        opening=opening,
        ending=ending,
        payoff=payoff,
        retention=retention,
        coherence=coherence,
        weakest=weakest,
        low_fraction=low_fraction,
        residual=residual,
    )


def montage_for_anchor(
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
    anchor_time: float,
    region: tuple[float, float],
) -> tuple[list[SemanticPlan], dict[str, int]]:
    """Search frame-preserving two- and three-beat edits for one verified payoff.

    The edit grid is a calibrated search space, not a fixed duration template;
    its candidates pass the same source, evidence and duration contracts as
    continuous gameplay proposals. No source footage is fabricated or reordered.
    """
    settings = _settings(config)
    if settings.get("enabled") is not True:
        return [], {"beat_editing_disabled": 1}
    max_gap = float(settings.get("max_source_gap_seconds", 18.0))
    lookback = float(settings.get("candidate_lookback_seconds", 25.0))
    step = float(settings.get("duration_grid_step_seconds", 0.5))
    min_payoff = float(settings.get("min_payoff_beat_seconds", 5.0))
    max_payoff = float(settings.get("max_payoff_beat_seconds", 7.0))
    max_segments = int(settings.get("maximum_segments", 2))
    if (
        not 0.5 <= max_gap <= 30
        or not 5 <= lookback <= 45
        or not 0.25 <= step <= 1.0
        or not 3.5 <= min_payoff <= max_payoff <= 8.0
        or max_segments not in (2, 3)
    ):
        raise ValueError("beat editing budget outside safe Clipper bounds")
    events = _verified_events(timeline, config)
    anchor = next(
        (
            e
            for e in events
            if abs(float(e.time) - anchor_time) <= 0.002
            and set(e.kinds).intersection(interaction.PAYOFF_KINDS)
        ),
        None,
    )
    if anchor is None:
        return [], {"unverified_target_payoff": 1}
    setup_events = [
        e
        for e in events
        if region[0] <= float(e.time) < anchor_time - 3.0
        and float(e.time) >= anchor_time - lookback
    ]
    editor = config["semantic_editor"]
    minimum = float(editor["minimum_output_seconds"])
    maximum = float(editor["maximum_output_seconds"])
    reject: Counter[str] = Counter()
    proposals: list[SemanticPlan] = []
    seen: set[tuple[float, ...]] = set()

    def submit(spans: tuple[EditSegment, ...]) -> None:
        key = tuple(t for seg in spans for t in (seg.start, seg.end))
        if key in seen:
            return
        seen.add(key)
        duration = sum(seg.end - seg.start for seg in spans)
        if duration < minimum - _EPS or duration > maximum + _EPS:
            reject["duration_outside_final_edit_contract"] += 1
            return
        gaps = [right.start - left.end for left, right in pairwise(spans)]
        if any(not 0.05 < gap <= max_gap for gap in gaps):
            reject["invalid_source_cut_distance"] += 1
            return
        if spans[0].start < region[0] - _EPS or spans[-1].end > region[1] + _EPS:
            reject["outside_certified_source_region"] += 1
            return
        reason = next(
            (
                failure
                for seg in spans
                if (failure := _safe_beat(timeline, seg, config, source_key, excluded))
            ),
            None,
        )
        if reason:
            reject[reason] += 1
            return
        engagements = tuple(_evidence(events, seg, timeline, config) for seg in spans)
        if any(item is None for item in engagements):
            reject["not_all_retained_beats_directly_verified"] += 1
            return
        # The assertion follows the explicit non-None checks above.
        verified = tuple(item for item in engagements if item is not None)
        if anchor not in verified[-1].events:
            reject["verified_payoff_not_retained_in_terminal_beat"] += 1
            return
        if float(verified[0].events[0].time) - spans[0].start > float(
            editor["opening"].get("max_time_to_meaningful_action_seconds", 0.9)
        ):
            reject["edited_opening_action_latency"] += 1
            return
        metrics = _edited_metrics(timeline, spans, verified, anchor_time, config)
        score = quality._score_plan(
            metrics["retention"],
            metrics["payoff"],
            metrics["opening"],
            metrics["ending"],
            metrics["coherence"],
            metrics["weakest"],
            False,
            config,
        )
        plan = SemanticPlan(
            spans[0].start,
            spans[-1].end,
            round(duration, 3),
            round(duration, 3),
            round(score, 5),
            round(metrics["retention"], 4),
            round(metrics["payoff"], 4),
            round(metrics["opening"], 4),
            round(metrics["ending"], 4),
            round(metrics["coherence"], 4),
            round(metrics["weakest"], 4),
            round(metrics["low_fraction"], 4),
            round(metrics["residual"], 3),
            "combat_montage",
            "source_native_combat_hard_cuts",
            spans,
            quality._effect_events_for_engagements(timeline, verified),
            verified,
            None,
            (
                "source-grounded independently verified combat-beat composition",
                "uninteresting source time omitted using explicit native hard cuts",
                "editorial strength measured on actual retained source frames",
            ),
            round(anchor_time, 3),
            None,
        )
        violations = quality.plan_integrity_violations(plan, timeline, config, source_key)
        if violations:
            reject["integrity." + violations[0]] += 1
            return
        proposals.append(plan)

    def grid(low: float, high: float) -> tuple[float, ...]:
        return tuple(
            round(low + i * step, 3) for i in range(max(0, math.floor((high - low) / step)) + 1)
        )

    for cue in setup_events:
        for target in grid(minimum, maximum):
            for payoff_length in grid(min_payoff, min(max_payoff, target - 2.0)):
                setup_length = round(target - payoff_length, 3)
                for lead in (0.25, 0.50):
                    begin = round(max(region[0], float(cue.time) - lead), 3)
                    for tail in (0.35, 0.60):
                        payoff_end = round(min(region[1], anchor_time + tail), 3)
                        spans = (
                            EditSegment(
                                begin, round(begin + setup_length, 3), 1.0, "verified_combat_setup"
                            ),
                            EditSegment(
                                round(payoff_end - payoff_length, 3),
                                payoff_end,
                                1.0,
                                "verified_combat_payoff",
                            ),
                        )
                        submit(spans)

    # When a third distinct verified interaction lies between setup and payoff,
    # also search short escalation beats. This is an additional framing choice,
    # never another independent outcome detector or allocator.
    if max_segments == 3:
        for first_index, first in enumerate(setup_events):
            for middle in setup_events[first_index + 1 :]:
                if float(middle.time) - float(first.time) < 3.0:
                    continue
                for target in grid(minimum, maximum):
                    for start_length in (2.0, 2.5, 3.0):
                        for middle_length in (2.0, 2.5, 3.0):
                            payoff_length = round(target - start_length - middle_length, 3)
                            if not min_payoff <= payoff_length <= max_payoff:
                                continue
                            begin = round(max(region[0], float(first.time) - 0.35), 3)
                            middle_start = round(float(middle.time) - 0.30, 3)
                            payoff_end = round(min(region[1], anchor_time + 0.45), 3)
                            submit(
                                (
                                    EditSegment(
                                        begin,
                                        round(begin + start_length, 3),
                                        1.0,
                                        "verified_combat_setup",
                                    ),
                                    EditSegment(
                                        middle_start,
                                        round(middle_start + middle_length, 3),
                                        1.0,
                                        "verified_combat_escalation",
                                    ),
                                    EditSegment(
                                        round(payoff_end - payoff_length, 3),
                                        payoff_end,
                                        1.0,
                                        "verified_combat_payoff",
                                    ),
                                )
                            )
    proposals.sort(
        key=lambda plan: (
            -plan.score,
            plan.start,
            plan.end,
            tuple((seg.start, seg.end) for seg in plan.segments),
        )
    )
    two_beats = [plan for plan in proposals if len(plan.segments) == 2]
    three_beats = [plan for plan in proposals if len(plan.segments) == 3]
    # Preserve the editor's composition alternatives in the allocator's pool;
    # a dense two-beat grid must not erase every valid three-beat framing.
    return two_beats[:6] + three_beats[:6], dict(sorted(reject.items()))

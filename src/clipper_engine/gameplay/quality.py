from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from . import combat, interaction
from . import semantics as core

Engagement = core.Engagement
FinishingMoveSpan = core.FinishingMoveSpan
EditSegment = core.EditSegment
SemanticPlan = core.SemanticPlan
SemanticTimeline = core.SemanticTimeline
PAYOFF_KINDS = core.PAYOFF_KINDS
_EPS = 1e-3


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _verified_cut_windows(
    config: dict[str, Any],
    source_key: str,
) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for item in (
        config.get("source_integrity", {}).get("verified_cut_windows", {}).get(source_key, [])
    ):
        if len(item) != 2:
            continue
        left, right = sorted((float(item[0]), float(item[1])))
        if right > left:
            out.append((left, right))
    return tuple(sorted(out))


def _segment_duration(segment: EditSegment) -> float:
    return (float(segment.end) - float(segment.start)) / float(segment.speed)


def _source_segments_overlap(
    left: SemanticPlan,
    right: SemanticPlan,
) -> bool:
    return any(
        max(a.start, b.start) < min(a.end, b.end) for a in left.segments for b in right.segments
    )


def _explained_mask(
    timeline: SemanticTimeline,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    config: dict[str, Any],
) -> np.ndarray:
    cfg = config["semantic_editor"]["dull"]
    pre = float(cfg.get("engagement_explanation_lead_seconds", 0.35))
    post = float(cfg.get("engagement_explanation_tail_seconds", 0.30))
    mask = np.zeros(len(timeline.times), dtype=bool)
    for engagement in engagements:
        i0 = max(
            0,
            math.floor((float(engagement.start) - pre) * timeline.fps),
        )
        i1 = min(
            len(mask),
            math.ceil((float(engagement.end) + post) * timeline.fps),
        )
        mask[i0:i1] = True
    if finishing is not None:
        i0 = max(
            0,
            math.floor((float(finishing.start) - 0.12) * timeline.fps),
        )
        i1 = min(
            len(mask),
            math.ceil((float(finishing.end) + 0.18) * timeline.fps),
        )
        mask[i0:i1] = True
    return mask


def _longest_unexplained_low_run(
    timeline: SemanticTimeline,
    start: float,
    end: float,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    config: dict[str, Any],
) -> float:
    i0 = max(0, math.floor(start * timeline.fps))
    i1 = min(len(timeline.times), math.ceil(end * timeline.fps))
    if i1 <= i0:
        return float("inf")
    threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    explained = _explained_mask(timeline, engagements, finishing, config)
    low = np.asarray(timeline.signals["interest"] < threshold, dtype=bool)
    return core._longest_true_run(
        (low & ~explained)[i0:i1],
        float(timeline.fps),
    )


def _quality_metrics(
    timeline: SemanticTimeline,
    start: float,
    end: float,
    engagements: tuple[Engagement, ...],
    finishing: FinishingMoveSpan | None,
    residual_run: float,
    config: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float]:
    sig = timeline.signals
    fps = timeline.fps
    i0 = max(0, math.floor(start * fps))
    i1 = min(len(timeline.times), math.ceil(end * fps))
    raw_interest = np.asarray(sig["interest"][i0:i1], dtype=np.float32)
    if raw_interest.size == 0:
        return (0.0,) * 7

    explained = _explained_mask(timeline, engagements, finishing, config)[i0:i1]
    floor = float(config["semantic_editor"]["dull"].get("explained_interest_floor", 0.26))
    effective = np.where(explained, np.maximum(raw_interest, floor), raw_interest)
    quarters = np.array_split(effective, 4)
    weakest = min(float(np.mean(part)) for part in quarters if len(part))
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    low_fraction = float(np.mean((raw_interest < low_threshold) & ~explained))

    first_event = finishing.start if finishing is not None else engagements[0].events[0].time
    delay = max(0.0, float(first_event) - start)
    opening_mean = core._mean(
        sig["interest"],
        timeline,
        start,
        min(end, start + 1.25),
    )
    opening_slice = np.asarray(
        sig["interest"][i0 : min(i1, i0 + max(1, round(1.1 * fps)))],
        dtype=np.float32,
    )
    opening_explained = explained[: len(opening_slice)]
    opening_dead = core._longest_true_run(
        (opening_slice < low_threshold) & ~opening_explained,
        fps,
    )
    opening = float(
        np.clip(
            0.43 * opening_mean
            + 0.34 * (1.0 - min(1.0, delay / 0.90))
            + 0.23 * (1.0 - min(1.0, opening_dead / 0.50)),
            0,
            1,
        )
    )
    if finishing is not None:
        opening = min(1.0, opening + 0.20 * float(finishing.confidence))

    payoff_events = [
        event
        for engagement in engagements
        for event in engagement.events
        if any(kind in PAYOFF_KINDS for kind in event.kinds)
    ]
    last_payoff = payoff_events[-1].time if payoff_events else None
    if finishing is not None and (last_payoff is None or finishing.payoff > last_payoff):
        last_payoff = finishing.payoff

    if last_payoff is not None:
        tail = max(0.0, end - float(last_payoff))
        tail_quality = float(np.clip(1.0 - abs(tail - 0.45) / 0.75, 0, 1))
        ending_interest = core._mean(
            sig["interest"],
            timeline,
            max(start, end - 1.0),
            end,
        )
        ending = 0.66 * tail_quality + 0.34 * ending_interest
    else:
        active = max(
            core._mean(sig["combat"], timeline, max(start, end - 0.8), end),
            core._mean(sig["contact"], timeline, max(start, end - 0.8), end),
        )
        ending = 0.72 * active + 0.28 * core._mean(
            sig["interest"],
            timeline,
            max(start, end - 1.0),
            end,
        )

    payoff_strength = max(
        [float(event.confidence) for event in payoff_events]
        + ([min(1.0, float(finishing.confidence) + 0.12)] if finishing is not None else [0.0])
    )
    coherence = float(
        np.clip(
            0.54 * float(np.mean([eng.confidence for eng in engagements]))
            + 0.26 * min(1.0, len(engagements) / 4.0)
            + 0.20 * (1.0 - min(1.0, low_fraction / 0.45)),
            0,
            1,
        )
    )
    retention = float(
        np.clip(
            0.27 * float(np.mean(effective))
            + 0.18 * weakest
            + 0.22 * opening
            + 0.18 * ending
            + 0.09 * coherence
            + 0.06 * (1.0 - min(1.0, residual_run / 0.90)),
            0,
            1,
        )
    )
    payoff = float(np.clip(0.72 * payoff_strength + 0.28 * ending, 0, 1))
    return (
        opening,
        float(ending),
        coherence,
        retention,
        payoff,
        weakest,
        low_fraction,
    )


def story_admissibility_failures(
    timeline: SemanticTimeline,
    start: float,
    end: float,
    engagements: tuple[Engagement, ...],
    config: dict[str, Any],
) -> list[str]:
    """Return hard semantic reasons a contiguous gameplay story cannot be used."""
    failures: list[str] = []
    editor = config["semantic_editor"]
    opening_cfg = editor["opening"]
    hostile_events = sorted(
        (
            event
            for engagement in engagements
            for event in engagement.events
            if start - _EPS <= float(event.time) <= end + _EPS
            and interaction.hostile_decision(event, config).hostile
        ),
        key=lambda event: float(event.time),
    )
    if not hostile_events:
        return ["no_verified_hostile_evidence"]

    first_action = float(hostile_events[0].time)
    action_delay = max(0.0, first_action - start)
    maximum_delay = float(opening_cfg.get("max_time_to_meaningful_action_seconds", 0.90))
    if action_delay > maximum_delay + _EPS:
        failures.append("opening_action_latency")

    low_threshold = float(editor["dull"].get("low_interest_threshold", 0.30))
    i0 = max(0, math.floor(start * timeline.fps))
    i1 = min(len(timeline.times), math.ceil(min(first_action, end) * timeline.fps))
    opening_dead = (
        core._longest_true_run(
            np.asarray(timeline.signals["interest"][i0:i1] < low_threshold, dtype=bool),
            float(timeline.fps),
        )
        if i1 > i0
        else 0.0
    )
    maximum_dead_open = float(opening_cfg.get("max_unexplained_dead_open_seconds", 0.50))
    if opening_dead > maximum_dead_open + _EPS:
        failures.append("unexplained_dead_open")

    active = [
        engagement
        for engagement in sorted(engagements, key=lambda item: float(item.start))
        if any(interaction.hostile_decision(event, config).hostile for event in engagement.events)
    ]
    for left, right in itertools.pairwise(active):
        gap_start = float(left.end)
        gap_end = float(right.start)
        if gap_end <= gap_start + _EPS:
            continue
        if not combat.bridge_supported(timeline, gap_start, gap_end, config):
            failures.append("unsupported_combat_continuity")
            break

    death = np.asarray(timeline.signals.get("player_death", []), dtype=np.float32)
    if death.size:
        d0 = max(0, math.floor(start * timeline.fps))
        d1 = min(len(death), math.ceil(end * timeline.fps))
        if d1 > d0 and float(np.max(death[d0:d1])) >= 0.5:
            failures.append("confirmed_player_death")

    return list(dict.fromkeys(failures))


def story_quality_diagnostics(
    story: str,
    opening: float,
    ending: float,
    retention: float,
    payoff: float,
    weakest: float,
    low_fraction: float,
    residual: float,
    config: dict[str, Any],
) -> dict[str, dict[str, float | bool]]:
    editor = config["semantic_editor"]
    checks = {
        "opening": (
            opening,
            _story_gate(
                story,
                "opening_min",
                config,
                float(editor["opening"].get("minimum_quality", 0.42)),
            ),
            "minimum",
        ),
        "ending": (
            ending,
            _story_gate(
                story,
                "ending_min",
                config,
                float(editor["ending"].get("minimum_quality", 0.40)),
            ),
            "minimum",
        ),
        "retention": (
            retention,
            _story_gate(
                story,
                "retention_min",
                config,
                float(config["performance_targets"].get("retention_quality_min", 0.36)),
            ),
            "minimum",
        ),
        "payoff": (
            payoff,
            _story_gate(
                story,
                "payoff_min",
                config,
                float(config["performance_targets"].get("payoff_quality_min", 0.34)),
            ),
            "minimum",
        ),
        "weakest_quarter": (
            weakest,
            float(editor["dull"].get("minimum_weak_quarter_interest", 0.27)),
            "minimum",
        ),
        "low_interest_fraction": (
            low_fraction,
            float(editor["dull"].get("maximum_low_interest_fraction", 0.38)),
            "maximum",
        ),
        "residual_low_interest_run_seconds": (
            residual,
            float(editor["dull"].get("maximum_unexplained_low_interest_run_seconds", 0.90)),
            "maximum",
        ),
    }
    result: dict[str, dict[str, float | bool]] = {}
    for name, (measured, threshold, direction) in checks.items():
        passed = measured >= threshold if direction == "minimum" else measured <= threshold
        result[name] = {
            "measured": round(float(measured), 6),
            "threshold": round(float(threshold), 6),
            "passed": bool(passed),
        }
    return result


def _story_gate(
    story: str,
    metric: str,
    config: dict[str, Any],
    default: float,
) -> float:
    table = config.get("story_quality_gates", {}).get(metric, {})
    return float(table.get(story, table.get("default", default)))


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
    return bool(
        opening
        >= _story_gate(
            story,
            "opening_min",
            config,
            float(editor["opening"].get("minimum_quality", 0.42)),
        )
        and ending
        >= _story_gate(
            story,
            "ending_min",
            config,
            float(editor["ending"].get("minimum_quality", 0.40)),
        )
        and retention
        >= _story_gate(
            story,
            "retention_min",
            config,
            float(config["performance_targets"].get("retention_quality_min", 0.36)),
        )
        and payoff
        >= _story_gate(
            story,
            "payoff_min",
            config,
            float(config["performance_targets"].get("payoff_quality_min", 0.34)),
        )
        and weakest >= float(editor["dull"].get("minimum_weak_quarter_interest", 0.27))
        and low_fraction <= float(editor["dull"].get("maximum_low_interest_fraction", 0.38))
        and residual
        <= float(editor["dull"].get("maximum_unexplained_low_interest_run_seconds", 0.90))
    )


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


def _effect_events_for_engagements(
    timeline: SemanticTimeline,
    engagements: tuple[Engagement, ...],
) -> tuple[Any, ...]:
    verified_times = [
        float(event.time) for engagement in engagements for event in engagement.events
    ]
    raw = [
        event
        for event in timeline.base.events
        if event.kind in {"outcome_like", "impact", "combat_burst"}
        and event.confidence >= 0.56
        and any(abs(float(event.time) - time) <= 0.30 for time in verified_times)
    ]
    unique: list[Any] = []
    for event in sorted(
        raw,
        key=lambda item: (-float(item.confidence), float(item.time)),
    ):
        if any(
            abs(float(event.time) - float(old.time)) <= 0.12 and event.kind == old.kind
            for old in unique
        ):
            continue
        unique.append(event)
    return tuple(sorted(unique[:3], key=lambda item: float(item.time)))


def _route_story(
    timeline: SemanticTimeline,
    chain: tuple[Engagement, ...],
) -> tuple[str, str, tuple[Any, ...], tuple[str, ...]]:
    effects = _effect_events_for_engagements(timeline, chain)
    verified = [event for engagement in chain for event in engagement.events]
    impact = [event for event in verified if "impact" in event.kinds]
    outcome = [event for event in verified if "outcome_like" in event.kinds]
    if len(chain) >= 3 and len(verified) >= 3:
        return (
            "engagement_chain",
            "chain_escalation",
            effects,
            ("continuous directly verified hostile engagement chain",),
        )
    if impact and max(float(item.confidence) for item in impact) >= 0.76:
        return (
            "impact_payoff",
            "impact_flash",
            effects,
            ("directly verified high-confidence impact payoff",),
        )
    if outcome:
        return (
            "precision_outcome",
            "precision_punch",
            effects,
            ("directly verified outcome payoff",),
        )
    return (
        "sustained_pressure",
        "clean_pressure",
        effects,
        ("continuous directly verified hostile pressure",),
    )


def verified_payoff_events(
    engagement: Engagement,
    config: dict[str, Any],
) -> tuple[Any, ...]:
    """Return payoff anchors that independently satisfy hostile verification."""
    return tuple(
        event
        for event in engagement.events
        if any(kind in PAYOFF_KINDS for kind in event.kinds)
        and interaction.hostile_decision(event, config).hostile
    )


def finishing_body_hard_cut_gap(config: dict[str, Any]) -> float:
    editorial = config["editorial"]
    first_gap = _f(
        editorial.get("finishing_move_max_continuation_gap_seconds", 18.0),
        18.0,
    )
    join_gap = _f(
        editorial.get(
            "finishing_move_body_hard_cut_max_source_gap_seconds",
            first_gap,
        ),
        first_gap,
    )
    return max(0.0, min(first_gap, join_gap))


def _chronology_failures(plan: SemanticPlan) -> list[str]:
    segments = list(plan.segments)
    if not segments:
        return ["plan has no source segments"]
    failures: list[str] = []
    if abs(float(plan.start) - float(segments[0].start)) > _EPS:
        failures.append("plan start does not match first segment")
    if abs(float(plan.end) - float(segments[-1].end)) > _EPS:
        failures.append("plan end does not match last segment")
    if float(plan.end) <= float(plan.start):
        failures.append("plan bounds are non-monotonic")
    for index, segment in enumerate(segments):
        if float(segment.end) <= float(segment.start):
            failures.append(f"segment {index + 1} has invalid bounds")
        if abs(float(segment.speed) - 1.0) > 1e-6:
            failures.append(f"segment {index + 1} is not source-native 1.0x")
        if index and float(segment.start) < float(segments[index - 1].end) - _EPS:
            failures.append(f"source chronology reversal/overlap at segment {index + 1}")
    return failures


def _planned_cross_shot_bridge(
    plan: SemanticPlan,
    left: EditSegment,
    right: EditSegment,
    bridge_index: int,
) -> bool:
    if plan.story_type != "finishing_move_open":
        return False
    if (
        bridge_index == 0
        and left.reason == "finishing_move_open_hero"
        and right.reason == "verified_combat_island_body"
    ):
        return True
    return (
        left.reason == "verified_combat_island_body"
        and right.reason == "verified_combat_island_body"
    )


def plan_integrity_violations(
    plan: SemanticPlan,
    timeline: SemanticTimeline,
    config: dict[str, Any],
    source_key: str,
) -> list[str]:
    failures = _chronology_failures(plan)
    windows = _verified_cut_windows(config, source_key)

    for index, segment in enumerate(plan.segments):
        inside = any(
            shot.start - _EPS <= segment.start and segment.end <= shot.end + _EPS
            for shot in timeline.shots
        )
        if not inside:
            failures.append(
                f"segment {segment.start:.3f}-{segment.end:.3f} crosses a source-shot boundary"
            )
        for left, right in windows:
            midpoint = (left + right) / 2.0
            if segment.start < midpoint < segment.end:
                failures.append(
                    f"segment {segment.start:.3f}-{segment.end:.3f} crosses verified source cut "
                    f"{left:.3f}-{right:.3f}"
                )
        if segment.reason != "finishing_move_open_hero" and combat.segment_crosses_gap(
            timeline,
            float(segment.start),
            float(segment.end),
        ):
            failures.append(f"segment {index + 1} crosses a sustained reload/search/recovery break")

    for index, (left, right) in enumerate(zip(plan.segments, plan.segments[1:], strict=False)):
        if right.start <= left.end + 0.02:
            continue
        left_shot = core._shot_index(
            timeline.shots,
            max(left.start, left.end - _EPS),
        )
        right_shot = core._shot_index(
            timeline.shots,
            min(right.end, right.start + _EPS),
        )
        if left_shot != right_shot and not _planned_cross_shot_bridge(
            plan,
            left,
            right,
            index,
        ):
            failures.append(f"unplanned cross-shot bridge {left.end:.3f}->{right.start:.3f}")

    for engagement in plan.engagements:
        for event in engagement.events:
            if not interaction.hostile_decision(event, config).hostile:
                failures.append(f"plan includes non-hostile/ambiguous anchor at {event.time:.3f}s")

    if plan.finishing_move is not None:
        if plan.story_type != "finishing_move_open" or plan.effect_profile != "finishing_move_hero":
            failures.append(
                "verified Finishing Move is not routed as finishing_move_open/finishing_move_hero"
            )
        if not plan.segments or plan.segments[0].reason != "finishing_move_open_hero":
            failures.append("Finishing Move first segment is not the protected hero")
            return list(dict.fromkeys(failures))

        hero = plan.segments[0]
        if not hero.start <= plan.finishing_move.start <= hero.end:
            failures.append("Finishing Move is not contained in first output segment")
        if (plan.finishing_move.start - hero.start) / hero.speed > 0.48:
            failures.append("Finishing Move opening delay exceeds 0.48s")

        body = list(plan.segments[1:])
        engagements = list(plan.engagements)
        if not body:
            failures.append("Finishing Move has no verified combat-island body")
        if len(body) != len(engagements):
            failures.append("Finishing Move body/engagement cardinality mismatch")
        else:
            later_floor = max(
                float(hero.end),
                float(plan.finishing_move.end) + 0.25,
            )
            max_first_gap = _f(
                config["editorial"].get(
                    "finishing_move_max_continuation_gap_seconds",
                    18.0,
                ),
                18.0,
            )
            hard_cut_gap = finishing_body_hard_cut_gap(config)
            if body and float(body[0].start) < later_floor - _EPS:
                failures.append(
                    "Finishing Move continuation starts before strict later-content floor"
                )
            if body and float(body[0].start) - float(hero.end) > max_first_gap + _EPS:
                failures.append("Finishing Move first continuation starts beyond configured reach")

            for index, (segment, engagement) in enumerate(zip(body, engagements, strict=False)):
                if segment.reason != "verified_combat_island_body":
                    failures.append("Finishing Move body contains a non-island segment")
                    continue
                if (
                    abs(float(segment.start) - float(engagement.start)) > _EPS
                    or abs(float(segment.end) - float(engagement.end)) > _EPS
                ):
                    failures.append(
                        "Finishing Move body segment does not preserve exact canonical "
                        "combat-island bounds"
                    )
                if index:
                    gap = float(segment.start) - float(body[index - 1].end)
                    if gap < -_EPS:
                        failures.append("Finishing Move body segments overlap/reverse")
                    elif gap > hard_cut_gap + _EPS:
                        failures.append("Finishing Move body hard-cut source gap exceeds contract")

    return list(dict.fromkeys(failures))

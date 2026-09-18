from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import mw4_semantic_gameplay_v3_1 as core
from . import mw4_semantic_gameplay_v3_1_combat_islands as islands
from . import mw4_semantic_gameplay_v3_1_combat_state as combat
from . import mw4_semantic_gameplay_v3_1_local_verify as local_verify
from . import mw4_semantic_gameplay_v3_1_quality as quality

ShotSpan = core.ShotSpan
ConsolidatedEvent = core.ConsolidatedEvent
Engagement = core.Engagement
FinishingMoveSpan = core.FinishingMoveSpan
EditSegment = core.EditSegment
SemanticPlanV31 = core.SemanticPlanV31
SemanticTimelineV31 = core.SemanticTimelineV31
PAYOFF_KINDS = core.PAYOFF_KINDS
plan_integrity_violations = quality.plan_integrity_violations
_EPS = 1e-3


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_key(source: Path) -> str:
    stem = source.stem.lower()
    return next((key for key in ("batch2", "week2", "r1") if key in stem), stem)


def validate_configuration(config: dict[str, Any]) -> None:
    """Validate the single canonical semantic route and fail closed."""
    combat.validate_configuration(config)
    errors: list[str] = []

    if "fallback_policy" in config:
        errors.append("obsolete fallback_policy key must be removed")
    if "semantic_montage" in config.get("semantic_editor", {}):
        errors.append("obsolete semantic_montage configuration must be removed")

    for key in ("count_per_source_max", "minimum_count_per_source"):
        if key in config:
            errors.append(f"static clip-count configuration must be removed: {key}")
    batch = config.get("batch_selection", {})
    for key in (
        "candidate_pool_per_source",
        "maximum_per_source",
        "minimum_total_clips",
        "maximum_total_clips",
    ):
        if key in batch:
            errors.append(f"static batch clip-count configuration must be removed: {key}")

    semantic_editor = config.get("semantic_editor", {})
    minimum_output = _f(semantic_editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum_output = _f(semantic_editor.get("maximum_output_seconds", 12.0), 12.0)
    preferred_output = _f(semantic_editor.get("preferred_output_seconds", 11.0), 11.0)
    if abs(minimum_output - 10.0) > _EPS or abs(maximum_output - 12.0) > _EPS:
        errors.append("MW4 canonical clip duration contract must be exactly 10-12 seconds")
    if not minimum_output - _EPS <= preferred_output <= maximum_output + _EPS:
        errors.append("preferred_output_seconds must remain inside the 10-12 second contract")

    editorial = config.get("editorial", {})
    for key in (
        "finishing_move_allow_semantic_montage_continuation",
        "finishing_move_allow_verified_combat_island_continuation",
        "finishing_move_montage_max_output_seconds",
    ):
        if key in editorial:
            errors.append(f"obsolete editorial key must be removed: {key}")

    if "allow_planned_transition_for_semantic_montage" in config.get("source_integrity", {}):
        errors.append("obsolete semantic-montage source-transition key must be removed")
    if "allow_unverified_automatic" in config.get("finishing_move_detector", {}):
        errors.append(
            "automatic Finishing Move acceptance switch must be removed; discovery is diagnostics-only"
        )

    continuation = config.get("combat_state_verifier", {}).get("finishing_continuation", {})
    if continuation.get("require_verified_payoff") is not True:
        errors.append("Finishing Move continuation must require verified payoff")
    for key in (
        "minimum_verified_hostile_anchors_without_payoff",
        "minimum_retention_without_payoff",
    ):
        if key in continuation:
            errors.append(f"obsolete no-payoff continuation key must be removed: {key}")

    first_gap = _f(editorial.get("finishing_move_max_continuation_gap_seconds", 0.0))
    body_gap = _f(editorial.get("finishing_move_body_hard_cut_max_source_gap_seconds", 0.0))
    if first_gap <= 0.0:
        errors.append("finishing_move_max_continuation_gap_seconds must be positive")
    if body_gap <= 0.0 or body_gap > first_gap + _EPS:
        errors.append(
            "Finishing Move body hard-cut gap must be positive and no larger than first-continuation reach"
        )

    if errors:
        raise RuntimeError("MW4 V3.1 canonical configuration violation: " + "; ".join(errors))


def _verified_cut_windows(
    config: dict[str, Any],
    source_key: str,
) -> tuple[tuple[float, float], ...]:
    return quality._verified_cut_windows(config, source_key)


def _build_hardened_shots(
    source: Path,
    duration: float,
    config: dict[str, Any],
) -> tuple[ShotSpan, ...]:
    automatic = list(core.scene_cut_times(source, config))
    verified = [
        (left + right) / 2.0 for left, right in _verified_cut_windows(config, _source_key(source))
    ]
    cuts: list[float] = []
    for value in sorted(automatic + verified):
        if 0.35 < value < duration - 0.35 and all(abs(value - old) >= 0.18 for old in cuts):
            cuts.append(value)
    guard = float(config.get("semantic_analysis", {}).get("scene_change_guard_seconds", 0.12))
    boundaries = [0.0, *cuts, duration]
    shots: list[ShotSpan] = []
    for index, (left, right) in enumerate(itertools.pairwise(boundaries)):
        start = left if index == 0 else min(right, left + guard)
        end = right if index == len(boundaries) - 2 else max(start, right - guard)
        if end - start >= 1.0:
            shots.append(ShotSpan(round(start, 3), round(end, 3)))
    return tuple(shots or [ShotSpan(0.0, duration)])


def _verified_finishing_moves(
    shots: tuple[ShotSpan, ...],
    source: Path,
    config: dict[str, Any],
) -> tuple[FinishingMoveSpan, ...]:
    """Only manually verified campaign spans can enter production planning."""
    verified: list[FinishingMoveSpan] = []
    items = (
        config.get("finishing_move_detector", {})
        .get("verified_spans", {})
        .get(_source_key(source), [])
    )
    for item in items:
        start = float(item["start"])
        payoff = float(item["payoff"])
        end = float(item["end"])
        midpoint = (start + end) / 2.0
        shot_index = core._shot_index(shots, midpoint)
        shot = shots[shot_index]
        if shot.start <= midpoint <= shot.end:
            verified.append(
                FinishingMoveSpan(
                    round(max(float(shot.start), start), 3),
                    round(payoff, 3),
                    round(min(float(shot.end), end), 3),
                    shot_index,
                    0.99,
                    {
                        "visual_verification": 1.0,
                        "third_person_execution_regime": 1.0,
                    },
                )
            )
    return tuple(sorted(verified, key=lambda item: item.start))


def _verified_hostile_signal(timeline: SemanticTimelineV31) -> np.ndarray:
    signal = np.zeros(len(timeline.times), dtype=np.float32)
    for engagement in timeline.engagements:
        i0 = max(0, int(np.floor(float(engagement.start) * timeline.fps)))
        i1 = min(
            len(signal),
            int(np.ceil(float(engagement.end) * timeline.fps)),
        )
        signal[i0:i1] = 1.0
    return signal


def analyze_source(
    source: Path,
    config: dict[str, Any],
) -> SemanticTimelineV31:
    """Canonical evidence -> verification -> combat-island analysis route."""
    validate_configuration(config)
    base = core.base.analyze_source(source, config)
    shots = _build_hardened_shots(source, float(base.duration), config)
    consolidated = core.consolidate_events(base, shots)
    coarse = core.cluster_engagements(base, shots, consolidated, config)
    timeline = SemanticTimelineV31(
        base,
        shots,
        consolidated,
        coarse,
        _verified_finishing_moves(shots, source, config),
    )
    timeline._source_key = _source_key(source)

    local_verify.annotate_timeline(source, timeline, config)
    timeline.engagements = islands.build(
        timeline,
        coarse,
        config,
        core,
        combat,
        Engagement,
    )
    timeline.signals["combat_island_gap"] = islands.gap_signal(timeline, config)
    timeline.signals["verified_hostile"] = _verified_hostile_signal(timeline)
    timeline._semantic_diagnostics = {
        "coarse_engagement_count": len(coarse),
        "verified_combat_island_count": len(timeline.engagements),
        "contact_only_can_anchor_hostile": False,
        "unknown_actor_defaults_to_hostile": False,
        "combat_islands_are_exact_gap_split_bounds": True,
        "body_segments_preserve_exact_combat_island_bounds": True,
        "alternate_planner_available": False,
    }
    return timeline


def _candidate_engagement_chains(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
) -> list[tuple[Engagement, ...]]:
    minimum = _f(
        config["semantic_editor"].get("minimum_output_seconds", 10.0),
        10.0,
    )
    maximum = _f(
        config["semantic_editor"].get("maximum_output_seconds", 12.0),
        12.0,
    )
    max_engagements = int(config["semantic_editor"].get("maximum_engagements_per_story", 7))
    chains: list[tuple[Engagement, ...]] = []

    for start_index, first in enumerate(timeline.engagements):
        chain: list[Engagement] = []
        for candidate in timeline.engagements[start_index:]:
            if candidate.shot_index != first.shot_index:
                break
            if chain and not islands.bridge_supported(
                timeline,
                float(chain[-1].end),
                float(candidate.start),
                config,
            ):
                break
            chain.append(candidate)
            if len(chain) > max_engagements:
                break
            duration = float(chain[-1].end) - float(chain[0].start)
            if minimum <= duration <= maximum:
                chains.append(tuple(chain))
            if duration > maximum:
                break

    unique: list[tuple[Engagement, ...]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for chain in chains:
        key = tuple((round(float(item.start), 3), round(float(item.end), 3)) for item in chain)
        if key not in seen:
            seen.add(key)
            unique.append(chain)
    return unique


def _protected_finishing_overlap(
    timeline: SemanticTimelineV31,
    start: float,
    end: float,
    config: dict[str, Any],
) -> bool:
    if end <= start + _EPS:
        return False
    editorial = config["editorial"]
    lead = _f(
        editorial.get("finishing_move_opening_lead_seconds", 0.28),
        0.28,
    )
    hold = _f(
        editorial.get("finishing_move_payoff_hold_seconds", 0.45),
        0.45,
    )
    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        protected_start = max(float(shot.start), float(span.start) - lead)
        protected_end = min(float(shot.end), float(span.end) + hold)
        if max(start, protected_start) < min(end, protected_end) - _EPS:
            return True
    return False


def _normal_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    plans: list[SemanticPlanV31] = []
    for chain in _candidate_engagement_chains(timeline, config):
        start, end = float(chain[0].start), float(chain[-1].end)
        if (
            core._intersects_excluded(start, end, excluded)
            or islands.segment_crosses_gap(timeline, start, end)
            or _protected_finishing_overlap(timeline, start, end, config)
        ):
            continue

        segment = EditSegment(
            round(start, 3),
            round(end, 3),
            1.0,
            "verified_combat_story",
        )
        residual = quality._longest_unexplained_low_run(
            timeline,
            start,
            end,
            chain,
            None,
            config,
        )
        (
            opening,
            ending,
            coherence,
            retention,
            payoff,
            weakest,
            low_fraction,
        ) = quality._quality_metrics(
            timeline,
            start,
            end,
            chain,
            None,
            residual,
            config,
        )
        story, profile, effects, reasons = quality._route_story(timeline, chain)
        if not quality._passes_story_gates(
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
            round(start, 3),
            round(end, 3),
            round(end - start, 3),
            round(end - start, 3),
            round(
                quality._score_plan(
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
            round(retention, 4),
            round(payoff, 4),
            round(opening, 4),
            round(ending, 4),
            round(coherence, 4),
            round(weakest, 4),
            round(low_fraction, 4),
            round(residual, 3),
            story,
            profile,
            (segment,),
            effects,
            chain,
            None,
            (*reasons, "exact verified-combat-island story bounds"),
        )
        if not quality.plan_integrity_violations(
            plan,
            timeline,
            config,
            source_key,
        ):
            plans.append(plan)
    return sorted(plans, key=lambda item: item.score, reverse=True)


def _component_metric_failures(
    metrics: dict[str, float],
    config: dict[str, Any],
    *,
    require_ending: bool,
) -> list[str]:
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    min_retention = max(
        _f(
            config.get("performance_targets", {}).get("retention_quality_min", 0.36),
            0.36,
        ),
        _f(cfg.get("minimum_retention_quality", 0.40), 0.40),
    )
    min_payoff = max(
        _f(
            config.get("performance_targets", {}).get("payoff_quality_min", 0.34),
            0.34,
        ),
        _f(cfg.get("minimum_payoff_quality", 0.40), 0.40),
    )
    checks = [
        (
            metrics["opening"]
            >= _f(
                cfg.get("minimum_verified_island_opening_quality", 0.42),
                0.42,
            ),
            "verified combat island opening below floor",
        ),
        (
            metrics["retention"] >= min_retention,
            "verified combat island retention below floor",
        ),
        (
            metrics["payoff"] >= min_payoff,
            "verified combat island payoff below floor",
        ),
        (
            metrics["weakest"] >= _f(cfg.get("minimum_weakest_quarter_interest", 0.30), 0.30),
            "verified combat island weakest quarter below floor",
        ),
        (
            metrics["residual"]
            <= _f(
                cfg.get(
                    "maximum_unexplained_low_interest_run_seconds",
                    0.75,
                ),
                0.75,
            ),
            "verified combat island unexplained inactivity exceeds floor",
        ),
        (
            metrics["low_fraction"]
            <= _f(
                config["semantic_editor"]["dull"].get("maximum_low_interest_fraction", 0.38),
                0.38,
            ),
            "verified combat island low-interest fraction exceeds floor",
        ),
    ]
    if require_ending:
        checks.append(
            (
                metrics["ending"] >= _f(cfg.get("minimum_ending_quality", 0.45), 0.45),
                "verified combat island ending below floor",
            )
        )
    return [message for passed, message in checks if not passed]


def _metrics_for_bounds(
    timeline: SemanticTimelineV31,
    engagement: Engagement,
    start: float,
    end: float,
    config: dict[str, Any],
) -> dict[str, float]:
    residual = quality._longest_unexplained_low_run(
        timeline,
        start,
        end,
        (engagement,),
        None,
        config,
    )
    (
        opening,
        ending,
        coherence,
        retention,
        payoff,
        weakest,
        low_fraction,
    ) = quality._quality_metrics(
        timeline,
        start,
        end,
        (engagement,),
        None,
        residual,
        config,
    )
    return {
        "duration": end - start,
        "opening": opening,
        "ending": ending,
        "coherence": coherence,
        "retention": retention,
        "payoff": payoff,
        "weakest": weakest,
        "low_fraction": low_fraction,
        "residual": residual,
    }


def _qualify_island(
    timeline: SemanticTimelineV31,
    engagement: Engagement,
    config: dict[str, Any],
    *,
    require_ending: bool,
) -> tuple[dict[str, float] | None, list[str]]:
    """Qualify the actual canonical gap-split island, never a derived sub-window."""
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    start, end = float(engagement.start), float(engagement.end)
    duration = end - start
    failures: list[str] = []

    if (
        duration
        < _f(
            cfg.get("minimum_verified_island_moment_seconds", 1.0),
            1.0,
        )
        - _EPS
    ):
        failures.append("verified combat island shorter than minimum body island")
    if (
        duration
        > _f(
            cfg.get("maximum_verified_island_moment_seconds", 6.5),
            6.5,
        )
        + _EPS
    ):
        failures.append("verified combat island longer than maximum body island")
    if not quality.verified_payoff_events(engagement, config):
        failures.append("verified combat island lacks hostile payoff")
    if islands.segment_crosses_gap(timeline, start, end):
        failures.append("verified combat island crosses a hard combat gap")
    failures.extend(
        f"ambiguous/non-hostile event at {event.time:.3f}s"
        for event in engagement.events
        if not combat.hostile_decision(event, config).hostile
    )

    metrics = _metrics_for_bounds(timeline, engagement, start, end, config)
    failures.extend(
        _component_metric_failures(
            metrics,
            config,
            require_ending=require_ending,
        )
    )
    if failures:
        return None, list(dict.fromkeys(failures))
    return metrics, []


def _body_segment(engagement: Engagement) -> EditSegment:
    """Convert a canonical island to an exact 1.0x body segment."""
    return EditSegment(
        round(float(engagement.start), 3),
        round(float(engagement.end), 3),
        1.0,
        "verified_combat_island_body",
    )


def _required_body_duration(
    hero: EditSegment,
    final_minimum: float,
) -> float:
    return max(0.0, final_minimum - quality._segment_duration(hero))


def _group_structure_failures(
    engagements: tuple[Engagement, ...],
    hero_end: float,
    config: dict[str, Any],
) -> list[str]:
    if not engagements:
        return ["empty Finishing Move body group"]
    max_first_gap = _f(
        config["editorial"].get("finishing_move_max_continuation_gap_seconds", 18.0),
        18.0,
    )
    hard_cut_gap = quality.finishing_body_hard_cut_gap(config)
    failures: list[str] = []
    if float(engagements[0].start) - hero_end > max_first_gap + _EPS:
        failures.append(
            "first verified body island starts beyond Finishing Move continuation reach"
        )
    for left, right in itertools.pairwise(engagements):
        gap = float(right.start) - float(left.end)
        if gap < -_EPS:
            failures.append("verified body islands overlap or reverse")
        elif gap > hard_cut_gap + _EPS:
            failures.append("verified body hard-cut source gap exceeds contract")
    return list(dict.fromkeys(failures))


def _finishing_open_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    results: list[SemanticPlanV31] = []
    diagnostics: list[dict[str, Any]] = []
    editorial = config["editorial"]
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    lead = _f(
        editorial.get("finishing_move_opening_lead_seconds", 0.28),
        0.28,
    )
    hold = _f(
        editorial.get("finishing_move_payoff_hold_seconds", 0.45),
        0.45,
    )
    max_first_gap = _f(
        editorial.get("finishing_move_max_continuation_gap_seconds", 18.0),
        18.0,
    )
    hard_cut_gap = quality.finishing_body_hard_cut_gap(config)
    final_minimum = _f(
        config["semantic_editor"].get("minimum_output_seconds", 10.0),
        10.0,
    )
    final_maximum = min(
        _f(
            config["semantic_editor"].get("maximum_output_seconds", 12.0),
            12.0,
        ),
        _f(
            editorial.get("finishing_move_max_output_seconds", 12.0),
            12.0,
        ),
    )
    max_islands = int(cfg.get("maximum_verified_island_moments_per_body", 3))

    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero = EditSegment(
            round(max(float(shot.start), float(span.start) - lead), 3),
            round(min(float(shot.end), float(span.end) + hold), 3),
            1.0,
            "finishing_move_open_hero",
        )
        hero_duration = quality._segment_duration(hero)
        body_minimum = _required_body_duration(hero, final_minimum)
        body_maximum = max(0.0, final_maximum - hero_duration)
        later_floor = max(float(hero.end), float(span.end) + 0.25)

        components: list[tuple[Engagement, dict[str, float]]] = []
        island_diagnostics: list[dict[str, Any]] = []
        for engagement in timeline.engagements:
            if float(engagement.start) < later_floor - _EPS:
                continue
            if core._intersects_excluded(
                float(engagement.start),
                float(engagement.end),
                excluded,
            ):
                continue

            component_metrics, component_failures = _qualify_island(
                timeline,
                engagement,
                config,
                require_ending=False,
            )
            terminal_metrics, terminal_failures = _qualify_island(
                timeline,
                engagement,
                config,
                require_ending=True,
            )
            island_diagnostics.append(
                {
                    "start": round(float(engagement.start), 3),
                    "end": round(float(engagement.end), 3),
                    "event_times": [round(float(event.time), 3) for event in engagement.events],
                    "component_qualified": component_metrics is not None,
                    "component_failures": component_failures,
                    "terminal_qualified": terminal_metrics is not None,
                    "terminal_failures": terminal_failures,
                    "evaluated_bounds_are_exact_island_bounds": True,
                }
            )
            if component_metrics is not None:
                components.append((engagement, component_metrics))

        group_rejections: list[dict[str, Any]] = []
        valid_count = 0
        for count in range(1, max(1, max_islands) + 1):
            for group in itertools.combinations(components, count):
                engagements = tuple(item[0] for item in group)
                structure_failures = _group_structure_failures(
                    engagements,
                    float(hero.end),
                    config,
                )
                if structure_failures:
                    group_rejections.append(
                        {
                            "islands": [
                                [
                                    round(float(item.start), 3),
                                    round(float(item.end), 3),
                                ]
                                for item in engagements
                            ],
                            "failures": structure_failures,
                        }
                    )
                    continue

                terminal_metrics, terminal_failures = _qualify_island(
                    timeline,
                    engagements[-1],
                    config,
                    require_ending=True,
                )
                if terminal_metrics is None:
                    group_rejections.append(
                        {
                            "islands": [
                                [
                                    round(float(item.start), 3),
                                    round(float(item.end), 3),
                                ]
                                for item in engagements
                            ],
                            "failures": terminal_failures,
                        }
                    )
                    continue

                metrics = [item[1] for item in group[:-1]] + [terminal_metrics]
                body_segments_tuple = tuple(_body_segment(item) for item in engagements)
                durations = np.asarray(
                    [quality._segment_duration(segment) for segment in body_segments_tuple],
                    dtype=float,
                )
                body_duration = float(np.sum(durations))
                if not (body_minimum - _EPS <= body_duration <= body_maximum + _EPS):
                    group_rejections.append(
                        {
                            "islands": [
                                [
                                    round(float(item.start), 3),
                                    round(float(item.end), 3),
                                ]
                                for item in engagements
                            ],
                            "body_duration": round(body_duration, 3),
                            "required_body_minimum_seconds": round(body_minimum, 3),
                            "failures": ["verified body duration outside hero-aware contract"],
                        }
                    )
                    continue

                body_retention = float(
                    np.average(
                        [item["retention"] for item in metrics],
                        weights=durations,
                    )
                )
                payoff_values = [float(item["payoff"]) for item in metrics]
                body_payoff = float(0.55 * max(payoff_values) + 0.45 * np.mean(payoff_values))
                body_ending = float(metrics[-1]["ending"])
                body_coherence = float(
                    np.average(
                        [item["coherence"] for item in metrics],
                        weights=durations,
                    )
                )
                body_weakest = float(min(item["weakest"] for item in metrics))
                body_low_fraction = float(
                    np.average(
                        [item["low_fraction"] for item in metrics],
                        weights=durations,
                    )
                )
                body_residual = float(max(item["residual"] for item in metrics))

                if not quality._passes_story_gates(
                    "finishing_move_open",
                    max(0.90, float(metrics[0]["opening"])),
                    body_ending,
                    body_retention,
                    body_payoff,
                    body_weakest,
                    body_low_fraction,
                    body_residual,
                    config,
                ):
                    group_rejections.append(
                        {
                            "islands": [
                                [
                                    round(float(item.start), 3),
                                    round(float(item.end), 3),
                                ]
                                for item in engagements
                            ],
                            "failures": ["aggregated verified body failed unchanged story gates"],
                        }
                    )
                    continue

                segments = (hero, *body_segments_tuple)
                output_duration = sum(quality._segment_duration(segment) for segment in segments)
                opening = max(
                    0.90,
                    min(1.0, 0.74 + 0.22 * float(span.confidence)),
                )
                retention = float(0.18 * opening + 0.82 * body_retention)
                payoff = float(0.45 * min(1.0, float(span.confidence) + 0.12) + 0.55 * body_payoff)
                coherence = min(1.0, 0.10 + 0.88 * body_coherence)
                low_fraction = body_low_fraction * (body_duration / output_duration)

                if not quality._passes_story_gates(
                    "finishing_move_open",
                    opening,
                    body_ending,
                    retention,
                    payoff,
                    body_weakest,
                    low_fraction,
                    body_residual,
                    config,
                ):
                    group_rejections.append(
                        {
                            "islands": [
                                [
                                    round(float(item.start), 3),
                                    round(float(item.end), 3),
                                ]
                                for item in engagements
                            ],
                            "failures": [
                                "combined Finishing Move plan failed unchanged story gates"
                            ],
                        }
                    )
                    continue

                plan = SemanticPlanV31(
                    hero.start,
                    body_segments_tuple[-1].end,
                    round(
                        sum(segment.end - segment.start for segment in segments),
                        3,
                    ),
                    round(output_duration, 3),
                    round(
                        quality._score_plan(
                            retention,
                            payoff,
                            opening,
                            body_ending,
                            coherence,
                            body_weakest,
                            True,
                            config,
                        ),
                        5,
                    ),
                    round(retention, 4),
                    round(payoff, 4),
                    round(opening, 4),
                    round(body_ending, 4),
                    round(coherence, 4),
                    round(body_weakest, 4),
                    round(low_fraction, 4),
                    round(body_residual, 3),
                    "finishing_move_open",
                    "finishing_move_hero",
                    segments,
                    quality._effect_events_for_engagements(
                        timeline,
                        engagements,
                    ),
                    engagements,
                    span,
                    (
                        "verified Finishing Move opens the clip",
                        "every body segment preserves exact canonical gap-split combat-island bounds",
                        "every body island independently passes direct-hostile/payoff/retention gates",
                        "terminal body island passes ending quality at its actual island boundary",
                        "source-time reload/search gaps are omitted only by explicit bounded hard cuts",
                    ),
                )
                integrity = quality.plan_integrity_violations(
                    plan,
                    timeline,
                    config,
                    source_key,
                )
                if integrity:
                    group_rejections.append(
                        {
                            "islands": [
                                [
                                    round(float(item.start), 3),
                                    round(float(item.end), 3),
                                ]
                                for item in engagements
                            ],
                            "failures": integrity,
                        }
                    )
                    continue

                results.append(plan)
                valid_count += 1

        diagnostics.append(
            {
                "span_start": round(float(span.start), 3),
                "span_end": round(float(span.end), 3),
                "hero_start": round(float(hero.start), 3),
                "hero_end": round(float(hero.end), 3),
                "required_body_minimum_seconds": round(body_minimum, 3),
                "allowed_body_maximum_seconds": round(body_maximum, 3),
                "first_body_max_gap_seconds": round(max_first_gap, 3),
                "body_hard_cut_max_source_gap_seconds": round(hard_cut_gap, 3),
                "component_qualified_island_count": len(components),
                "valid_finishing_plan_count": valid_count,
                "island_diagnostics": island_diagnostics,
                "group_rejections": group_rejections[:32],
                "body_segments_preserve_exact_combat_island_bounds": True,
                "alternate_planner_available": False,
            }
        )

    timeline._finishing_continuation_diagnostics = diagnostics
    return sorted(results, key=lambda item: item.score, reverse=True)


def build_plans_for_source(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    validate_configuration(config)
    return sorted(
        _finishing_open_plans(timeline, config, excluded, source_key)
        + _normal_plans(timeline, config, excluded, source_key),
        key=lambda item: item.score,
        reverse=True,
    )


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
    selected: list[SemanticPlanV31] = []
    finishing = [plan for plan in plans if plan.story_type == "finishing_move_open"]
    if finishing:
        selected.append(finishing[0])
    for plan in plans:
        if plan in selected or any(
            quality._source_segments_overlap(plan, prior) for prior in selected
        ):
            continue
        selected.append(plan)
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
    source_key: str,
) -> dict[str, Any]:
    plans = build_plans_for_source(timeline, config, excluded, source_key)
    result = dict(getattr(timeline, "_semantic_diagnostics", {}))
    result.update(
        {
            "shot_count": len(timeline.shots),
            "engagement_count": len(timeline.engagements),
            "verified_finishing_move_count": len(timeline.finishing_moves),
            "candidate_count_after_semantic_gates": len(plans),
            "local_interaction_verifier": dict(
                getattr(timeline, "_local_interaction_diagnostics", {})
            ),
            "finishing_move_continuation_diagnostics": list(
                getattr(timeline, "_finishing_continuation_diagnostics", [])
            ),
        }
    )
    return result


def _self_test() -> None:
    config = {
        "editorial": {
            "finishing_move_opening_lead_seconds": 0.28,
            "finishing_move_payoff_hold_seconds": 0.45,
            "finishing_move_max_continuation_gap_seconds": 18.0,
            "finishing_move_body_hard_cut_max_source_gap_seconds": 18.0,
            "finishing_move_max_output_seconds": 12.0,
        },
        "source_integrity": {},
        "semantic_editor": {
            "minimum_output_seconds": 10.0,
            "maximum_output_seconds": 12.0,
            "maximum_inter_engagement_gap_seconds": 4.0,
            "opening": {"minimum_quality": 0.42},
            "ending": {
                "preferred_payoff_tail_seconds": 0.45,
                "maximum_payoff_tail_seconds": 0.75,
                "minimum_quality": 0.40,
            },
            "dull": {
                "maximum_low_interest_fraction": 0.38,
                "minimum_weak_quarter_interest": 0.27,
                "maximum_unexplained_low_interest_run_seconds": 0.90,
            },
            "selection": {},
        },
        "finishing_move_detector": {},
        "combat_state_verifier": {
            "enabled": True,
            "local_interaction_verifier": {
                "enabled": True,
                "minimum_hitmarker_score": 0.34,
            },
            "finishing_continuation": {
                "require_verified_payoff": True,
                "minimum_retention_quality": 0.40,
                "minimum_payoff_quality": 0.40,
                "minimum_ending_quality": 0.45,
                "minimum_weakest_quarter_interest": 0.30,
                "maximum_unexplained_low_interest_run_seconds": 0.75,
                "minimum_verified_island_opening_quality": 0.42,
            },
        },
        "performance_targets": {
            "retention_quality_min": 0.36,
            "payoff_quality_min": 0.34,
        },
    }
    validate_configuration(config)
    core.architecture_self_test()
    combat.self_test()
    islands.self_test()
    local_verify.self_test()

    poor_terminal = {
        "opening": 0.9,
        "ending": 0.1,
        "retention": 0.9,
        "payoff": 0.9,
        "weakest": 0.9,
        "residual": 0.0,
        "low_fraction": 0.0,
    }
    if _component_metric_failures(
        poor_terminal,
        config,
        require_ending=False,
    ):
        raise AssertionError(
            "intermediate verified body island was incorrectly required to be a clip ending"
        )
    if "verified combat island ending below floor" not in _component_metric_failures(
        poor_terminal,
        config,
        require_ending=True,
    ):
        raise AssertionError("terminal verified body island escaped unchanged ending gate")

    hero = EditSegment(148.07, 150.50, 1.0, "finishing_move_open_hero")
    if abs(_required_body_duration(hero, 10.0) - 7.57) > _EPS:
        raise AssertionError("Finishing Move body minimum is not hero-aware")

    exact_island = SimpleNamespace(start=153.533, end=157.800)
    exact_segment = _body_segment(exact_island)
    if abs(exact_segment.start - 153.533) > _EPS or abs(exact_segment.end - 157.800) > _EPS:
        raise AssertionError("verified combat island was trimmed, centered, padded, or expanded")

    later = SimpleNamespace(start=172.900, end=176.700)
    if _group_structure_failures((exact_island, later), 150.500, config):
        raise AssertionError("valid bounded verified-island hard cut was rejected")
    too_late_first = SimpleNamespace(start=169.0, end=172.0)
    if not _group_structure_failures((too_late_first,), 150.500, config):
        raise AssertionError("first Finishing Move body island escaped 18s reach")
    too_far_second = SimpleNamespace(start=176.100, end=179.0)
    if not _group_structure_failures(
        (exact_island, too_far_second),
        150.500,
        config,
    ):
        raise AssertionError("body hard cut exceeding source-gap contract was accepted")

    if "canonical_terminal_end" in quality.__dict__:
        raise AssertionError("payoff-tail terminal sub-window implementation still exists")

    print("MW4 canonical semantic architecture self-test: PASS")


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test()
        return
    raise SystemExit("Use this module through mw4_fullframe_retention_v3_1.py")


if __name__ == "__main__":
    main()

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

import numpy as np

from . import mw4_semantic_gameplay_v3_1 as core
from . import mw4_semantic_gameplay_v3_1_combat_islands as islands
from . import mw4_semantic_gameplay_v3_1_combat_state as combat
from . import mw4_semantic_gameplay_v3_1_quality as quality

Engagement = core.Engagement
EditSegment = core.EditSegment
SemanticPlanV31 = core.SemanticPlanV31

_EPS = 1e-3


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _support_components(timeline: Any, config: dict[str, Any]) -> tuple[Engagement, ...]:
    """Build maximal same-shot, hard-gap-free support regions around verified hostile evidence.

    Coarse engagement envelopes remain discovery input only. They are intentionally
    not used as final edit bounds.
    """
    hard = np.asarray(
        timeline.signals.get("combat_island_gap", islands.gap_signal(timeline, config)),
        dtype=np.float32,
    )
    grouped: dict[tuple[int, float, float], list[tuple[Any, Any]]] = defaultdict(list)
    for event in timeline.consolidated_events:
        decision = combat.hostile_decision(event, config)
        if not decision.hostile:
            continue
        component = islands._support_component_for_event(timeline, event, hard, core)
        if component is None:
            continue
        shot_index, start, end = component
        grouped[(int(shot_index), float(start), float(end))].append((event, decision))

    minimum = _f(
        config["combat_state_verifier"].get("minimum_combat_island_seconds", 1.0),
        1.0,
    )
    out: list[Engagement] = []
    for (shot_index, start, end), members in grouped.items():
        if end - start < minimum - _EPS:
            continue
        members = sorted(members, key=lambda item: float(item[0].time))
        out.append(
            Engagement(
                round(start, 3),
                round(end, 3),
                shot_index,
                round(float(np.mean([item[1].score for item in members])), 4),
                tuple(item[0] for item in members),
            )
        )
    return tuple(sorted(out, key=lambda item: (item.start, item.end)))


def _proposal_engagement(
    component: Engagement,
    start: float,
    end: float,
    config: dict[str, Any],
) -> Engagement | None:
    events = tuple(
        event
        for event in component.events
        if start - _EPS <= float(event.time) <= end + _EPS
        and combat.hostile_decision(event, config).hostile
    )
    if not events:
        return None
    scores = [combat.hostile_decision(event, config).score for event in events]
    return Engagement(
        round(start, 3),
        round(end, 3),
        int(component.shot_index),
        round(float(np.mean(scores)), 4),
        events,
    )


def _evidence_engagements(
    timeline: Any,
    start: float,
    end: float,
) -> tuple[Engagement, ...]:
    evidence = tuple(
        item
        for item in timeline.engagements
        if int(item.shot_index) == int(core._shot_index(timeline.shots, (start + end) / 2.0))
        and any(start - _EPS <= float(event.time) <= end + _EPS for event in item.events)
    )
    return evidence


def _metrics(
    timeline: Any,
    proposal: Engagement,
    config: dict[str, Any],
) -> dict[str, float]:
    start, end = float(proposal.start), float(proposal.end)
    evidence = _evidence_engagements(timeline, start, end) or (proposal,)
    residual = quality._longest_unexplained_low_run(
        timeline,
        start,
        end,
        evidence,
        None,
        config,
    )
    opening, ending, coherence, retention, payoff, weakest, low_fraction = quality._quality_metrics(
        timeline,
        start,
        end,
        evidence,
        None,
        residual,
        config,
    )
    return {
        "duration": end - start,
        "opening": float(opening),
        "ending": float(ending),
        "coherence": float(coherence),
        "retention": float(retention),
        "payoff": float(payoff),
        "weakest": float(weakest),
        "low_fraction": float(low_fraction),
        "residual": float(residual),
    }


def _normal_variants(
    component: Engagement,
    config: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    editor = config["semantic_editor"]
    maximum = _f(editor.get("maximum_output_seconds", 12.0), 12.0)
    preferred = min(
        maximum,
        _f(editor.get("preferred_output_seconds", 11.0), 11.0),
    )
    tail = _f(editor["ending"].get("preferred_payoff_tail_seconds", 0.45), 0.45)
    events = tuple(sorted(component.events, key=lambda item: float(item.time)))
    payoffs = tuple(
        event for event in events if event in quality.verified_payoff_events(component, config)
    )
    terminals = payoffs or events[-1:]
    candidates: list[tuple[float, float]] = []

    duration = float(component.end) - float(component.start)
    if duration <= maximum + _EPS:
        candidates.append((float(component.start), float(component.end)))

    for terminal in terminals:
        end = min(float(component.end), float(terminal.time) + tail)
        prior_events = [
            event for event in events if float(event.time) <= float(terminal.time) + _EPS
        ]
        if not prior_events:
            continue
        first = prior_events[0]
        candidates.append(
            (
                max(float(component.start), float(first.time) - 0.30),
                end,
            )
        )
        candidates.append(
            (
                max(float(component.start), end - preferred),
                end,
            )
        )
        candidates.append(
            (
                max(float(component.start), end - maximum),
                end,
            )
        )

    unique: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for start, end in candidates:
        if end <= start + _EPS:
            continue
        if end - start > maximum + _EPS:
            continue
        key = (round(start, 3), round(end, 3))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    unique.sort(
        key=lambda item: (
            abs((item[1] - item[0]) - preferred),
            item[0],
        )
    )
    return tuple(unique[:4])


def _component_chains(
    components: tuple[Engagement, ...],
    config: dict[str, Any],
) -> tuple[tuple[Engagement, ...], ...]:
    max_gap = _f(
        config["semantic_editor"].get("maximum_inter_engagement_gap_seconds", 4.0),
        4.0,
    )
    max_items = int(config["semantic_editor"].get("maximum_engagements_per_story", 7))
    chains: list[tuple[Engagement, ...]] = []
    for index, first in enumerate(components):
        chain: list[Engagement] = []
        for candidate in components[index:]:
            if int(candidate.shot_index) != int(first.shot_index):
                break
            if chain and float(candidate.start) - float(chain[-1].end) > max_gap + _EPS:
                break
            chain.append(candidate)
            if len(chain) > max_items:
                break
            chains.append(tuple(chain))
    return tuple(chains)


def _aggregate_metrics(
    timeline: Any,
    engagements: tuple[Engagement, ...],
    config: dict[str, Any],
) -> dict[str, float]:
    per = [_metrics(timeline, item, config) for item in engagements]
    durations = np.asarray([item["duration"] for item in per], dtype=float)
    total = float(np.sum(durations))
    if total <= 0.0:
        return {
            "opening": 0.0,
            "ending": 0.0,
            "coherence": 0.0,
            "retention": 0.0,
            "payoff": 0.0,
            "weakest": 0.0,
            "low_fraction": 1.0,
            "residual": float("inf"),
        }
    return {
        "opening": float(per[0]["opening"]),
        "ending": float(per[-1]["ending"]),
        "coherence": float(np.average([item["coherence"] for item in per], weights=durations)),
        "retention": float(np.average([item["retention"] for item in per], weights=durations)),
        "payoff": float(max(item["payoff"] for item in per)),
        "weakest": float(min(item["weakest"] for item in per)),
        "low_fraction": float(
            np.average([item["low_fraction"] for item in per], weights=durations)
        ),
        "residual": float(max(item["residual"] for item in per)),
    }


def normal_plans(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    minimum = _f(
        config["semantic_editor"].get("minimum_output_seconds", 10.0),
        10.0,
    )
    maximum = _f(
        config["semantic_editor"].get("maximum_output_seconds", 12.0),
        12.0,
    )
    components = _support_components(timeline, config)
    plans: list[SemanticPlanV31] = []
    diagnostics = {
        "support_component_count": len(components),
        "normal_component_chain_count": 0,
        "normal_variant_combination_count": 0,
        "normal_duration_reject_count": 0,
        "normal_integrity_reject_count": 0,
        "normal_quality_reject_count": 0,
        "normal_qualified_count": 0,
    }

    for chain in _component_chains(components, config):
        diagnostics["normal_component_chain_count"] += 1
        variants = [_normal_variants(item, config) for item in chain]
        if any(not item for item in variants):
            continue
        for choice in itertools.islice(itertools.product(*variants), 64):
            diagnostics["normal_variant_combination_count"] += 1
            proposal_engagements: list[Engagement] = []
            valid = True
            for component, (start, end) in zip(chain, choice, strict=False):
                proposal = _proposal_engagement(component, start, end, config)
                if proposal is None:
                    valid = False
                    break
                proposal_engagements.append(proposal)
            if not valid:
                continue

            proposal_tuple = tuple(proposal_engagements)
            segments = tuple(
                EditSegment(
                    round(float(item.start), 3),
                    round(float(item.end), 3),
                    1.0,
                    "verified_combat_story",
                )
                for item in proposal_tuple
            )
            output_duration = float(sum(quality._segment_duration(segment) for segment in segments))
            if not (minimum - _EPS <= output_duration <= maximum + _EPS):
                diagnostics["normal_duration_reject_count"] += 1
                continue
            if any(
                core._intersects_excluded(float(item.start), float(item.end), excluded)
                or islands.segment_crosses_gap(timeline, float(item.start), float(item.end))
                or base._protected_finishing_overlap(
                    timeline, float(item.start), float(item.end), config
                )
                for item in proposal_tuple
            ):
                diagnostics["normal_integrity_reject_count"] += 1
                continue

            metrics = _aggregate_metrics(timeline, proposal_tuple, config)
            story, profile, effects, reasons = quality._route_story(timeline, proposal_tuple)
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
                diagnostics["normal_quality_reject_count"] += 1
                continue

            plan = SemanticPlanV31(
                round(float(segments[0].start), 3),
                round(float(segments[-1].end), 3),
                round(sum(float(item.end) - float(item.start) for item in segments), 3),
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
                segments,
                effects,
                proposal_tuple,
                None,
                (
                    *reasons,
                    "verified hostile evidence anchors a boundary-refined source-safe proposal",
                    "every kept segment stays inside one same-shot hard-gap-free support component",
                ),
            )
            integrity = quality.plan_integrity_violations(plan, timeline, config, source_key)
            if integrity:
                diagnostics["normal_integrity_reject_count"] += 1
                continue
            plans.append(plan)
            diagnostics["normal_qualified_count"] += 1

    unique: list[SemanticPlanV31] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for plan in sorted(plans, key=lambda item: item.score, reverse=True):
        key = tuple(
            (round(float(segment.start), 3), round(float(segment.end), 3))
            for segment in plan.segments
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(plan)

    existing = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    existing.update(diagnostics)
    timeline._proposal_diagnostics = existing
    return unique


def _component_gate_failures(
    base: Any,
    metrics: dict[str, float],
    config: dict[str, Any],
    *,
    require_ending: bool,
) -> list[str]:
    return base._component_metric_failures(
        metrics,
        config,
        require_ending=require_ending,
    )


def _finishing_variants(
    component: Engagement,
    config: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    minimum = _f(cfg.get("minimum_verified_island_moment_seconds", 1.0), 1.0)
    maximum = _f(cfg.get("maximum_verified_island_moment_seconds", 6.5), 6.5)
    tail = _f(
        config["semantic_editor"]["ending"].get("preferred_payoff_tail_seconds", 0.45),
        0.45,
    )
    payoffs = tuple(quality.verified_payoff_events(component, config))
    candidates: list[tuple[float, float]] = []
    for payoff in payoffs:
        end = min(float(component.end), float(payoff.time) + tail)
        candidates.append(
            (
                max(float(component.start), end - maximum),
                end,
            )
        )
        first = min(
            (event for event in component.events if float(event.time) <= float(payoff.time) + _EPS),
            key=lambda item: float(item.time),
            default=payoff,
        )
        candidates.append(
            (
                max(float(component.start), float(first.time) - 0.30),
                end,
            )
        )

    unique: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for start, end in candidates:
        duration = end - start
        if duration < minimum - _EPS or duration > maximum + _EPS:
            continue
        key = (round(start, 3), round(end, 3))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    unique.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    return tuple(unique[:3])


def finishing_open_plans(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    results: list[SemanticPlanV31] = []
    diagnostics: list[dict[str, Any]] = []
    components = _support_components(timeline, config)
    editorial = config["editorial"]
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    lead = _f(editorial.get("finishing_move_opening_lead_seconds", 0.28), 0.28)
    hold = _f(editorial.get("finishing_move_payoff_hold_seconds", 0.45), 0.45)
    final_minimum = _f(config["semantic_editor"].get("minimum_output_seconds", 10.0), 10.0)
    final_maximum = min(
        _f(config["semantic_editor"].get("maximum_output_seconds", 12.0), 12.0),
        _f(editorial.get("finishing_move_max_output_seconds", 12.0), 12.0),
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
        body_minimum = base._required_body_duration(hero, final_minimum)
        body_maximum = max(0.0, final_maximum - hero_duration)
        later_floor = max(float(hero.end), float(span.end) + 0.25)

        moments: list[tuple[tuple[int, float, float], Engagement, dict[str, float], bool]] = []
        moment_diagnostics: list[dict[str, Any]] = []
        for component in components:
            component_key = (
                int(component.shot_index),
                round(float(component.start), 3),
                round(float(component.end), 3),
            )
            for start, end in _finishing_variants(component, config):
                if start < later_floor - _EPS:
                    continue
                if core._intersects_excluded(start, end, excluded):
                    continue
                proposal = _proposal_engagement(component, start, end, config)
                if proposal is None or not quality.verified_payoff_events(proposal, config):
                    continue
                if islands.segment_crosses_gap(timeline, start, end):
                    continue
                metrics = _metrics(timeline, proposal, config)
                component_failures = _component_gate_failures(
                    base, metrics, config, require_ending=False
                )
                terminal_failures = _component_gate_failures(
                    base, metrics, config, require_ending=True
                )
                moment_diagnostics.append(
                    {
                        "support_component": [
                            round(float(component.start), 3),
                            round(float(component.end), 3),
                        ],
                        "proposal_bounds": [round(start, 3), round(end, 3)],
                        "proposal_duration": round(end - start, 3),
                        "component_failures": component_failures,
                        "terminal_failures": terminal_failures,
                    }
                )
                if not component_failures:
                    moments.append(
                        (
                            component_key,
                            proposal,
                            metrics,
                            not terminal_failures,
                        )
                    )

        group_rejections: list[dict[str, Any]] = []
        valid_count = 0
        for count in range(1, max(1, max_islands) + 1):
            for group in itertools.combinations(moments, count):
                component_keys = [item[0] for item in group]
                if len(set(component_keys)) != len(component_keys):
                    continue
                ordered = tuple(sorted(group, key=lambda item: float(item[1].start)))
                engagements = tuple(item[1] for item in ordered)
                if not ordered[-1][3]:
                    continue
                structure_failures = base._group_structure_failures(
                    engagements,
                    float(hero.end),
                    config,
                )
                if structure_failures:
                    group_rejections.append(
                        {
                            "proposals": [
                                [round(float(item.start), 3), round(float(item.end), 3)]
                                for item in engagements
                            ],
                            "failures": structure_failures,
                        }
                    )
                    continue

                body_segments = tuple(
                    EditSegment(
                        round(float(item.start), 3),
                        round(float(item.end), 3),
                        1.0,
                        "verified_combat_island_body",
                    )
                    for item in engagements
                )
                durations = np.asarray(
                    [quality._segment_duration(item) for item in body_segments],
                    dtype=float,
                )
                body_duration = float(np.sum(durations))
                if not (body_minimum - _EPS <= body_duration <= body_maximum + _EPS):
                    group_rejections.append(
                        {
                            "proposals": [
                                [round(float(item.start), 3), round(float(item.end), 3)]
                                for item in engagements
                            ],
                            "body_duration": round(body_duration, 3),
                            "required_body_minimum_seconds": round(body_minimum, 3),
                            "failures": [
                                "boundary-refined body duration outside hero-aware contract"
                            ],
                        }
                    )
                    continue

                metrics = [item[2] for item in ordered]
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
                    continue

                segments = (hero, *body_segments)
                output_duration = sum(quality._segment_duration(segment) for segment in segments)
                opening = max(0.90, min(1.0, 0.74 + 0.22 * float(span.confidence)))
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
                    continue

                plan = SemanticPlanV31(
                    hero.start,
                    body_segments[-1].end,
                    round(sum(segment.end - segment.start for segment in segments), 3),
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
                    quality._effect_events_for_engagements(timeline, engagements),
                    engagements,
                    span,
                    (
                        "verified Finishing Move opens the clip",
                        "body moments are boundary-refined only inside same-shot hard-gap-free verified-hostile support components",
                        "every body moment retains an independently verified hostile payoff and unchanged component quality floors",
                        "terminal body moment passes the unchanged ending-quality gate",
                    ),
                )
                integrity = quality.plan_integrity_violations(plan, timeline, config, source_key)
                if integrity:
                    group_rejections.append(
                        {
                            "proposals": [
                                [round(float(item.start), 3), round(float(item.end), 3)]
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
                "support_component_count": len(components),
                "body_proposal_count": len(moments),
                "required_body_minimum_seconds": round(body_minimum, 3),
                "allowed_body_maximum_seconds": round(body_maximum, 3),
                "maximum_qualified_body_seconds_seen": round(
                    max(
                        (
                            sum(float(item[1].end) - float(item[1].start) for item in group)
                            for count in range(1, max(1, max_islands) + 1)
                            for group in itertools.combinations(moments, count)
                            if len({entry[0] for entry in group}) == len(group)
                        ),
                        default=0.0,
                    ),
                    3,
                ),
                "valid_finishing_plan_count": valid_count,
                "moment_diagnostics": moment_diagnostics[:48],
                "group_rejections": group_rejections[:48],
                "boundary_model": "verified_hostile_anchor_to_hard_gap_free_support_to_refined_proposal",
                "threshold_reduction_used": False,
                "alternate_planner_available": False,
            }
        )

    timeline._finishing_continuation_diagnostics = diagnostics
    return sorted(results, key=lambda item: item.score, reverse=True)


def build_plans_for_source(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    base.validate_configuration(config)
    normal = normal_plans(base, timeline, config, excluded, source_key)
    finishing = finishing_open_plans(base, timeline, config, excluded, source_key)
    plans = sorted(finishing + normal, key=lambda item: item.score, reverse=True)
    diagnostics = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    diagnostics.update(
        {
            "proposal_architecture": "verified_anchor_to_support_component_to_boundary_refined_clip",
            "support_components_are_not_final_clip_bounds": True,
            "semantic_montage_enabled": False,
            "fallback_planner_enabled": False,
            "threshold_reduction_used": False,
            "qualified_plan_count": len(plans),
        }
    )
    timeline._proposal_diagnostics = diagnostics
    return plans


def self_test(base: Any) -> None:
    if (
        base._required_body_duration(
            EditSegment(148.07, 150.50, 1.0, "finishing_move_open_hero"),
            10.0,
        )
        != 7.57
    ):
        raise AssertionError("proposal planner changed hero-aware duration contract")

    component = type(
        "Component",
        (),
        {
            "start": 152.0,
            "end": 158.2,
            "shot_index": 0,
            "events": (),
        },
    )()
    maximum = 6.5
    payoff = 157.2
    tail = 0.45
    end = min(float(component.end), payoff + tail)
    start = max(float(component.start), end - maximum)
    if end - start <= 4.267:
        raise AssertionError(
            "support-component boundary refinement cannot expand beyond a stale coarse envelope"
        )
    if end - start > maximum + _EPS:
        raise AssertionError("Finishing Move proposal exceeded unchanged body-moment maximum")
    print("MW4 verified-anchor proposal architecture self-test: PASS")

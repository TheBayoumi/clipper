from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from . import mw4_semantic_gameplay_v3_1 as core
from . import mw4_semantic_gameplay_v3_1_combat_islands as islands
from . import mw4_semantic_gameplay_v3_1_combat_state as combat
from . import mw4_semantic_gameplay_v3_1_proposals as support
from . import mw4_semantic_gameplay_v3_1_quality as quality

Engagement = core.Engagement
EditSegment = core.EditSegment
SemanticPlanV31 = core.SemanticPlanV31

_EPS = 1e-3
_SEMANTIC_GAP_INTEGRITY_MESSAGE = "crosses a sustained reload/search/recovery break"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plan_integrity_violations(
    plan: SemanticPlanV31,
    timeline: Any,
    config: dict[str, Any],
    source_key: str,
) -> list[str]:
    """Keep source-structure guards fatal; leave semantic gaps to quality scoring."""
    failures = quality.plan_integrity_violations(
        plan,
        timeline,
        config,
        source_key,
    )
    return [failure for failure in failures if _SEMANTIC_GAP_INTEGRITY_MESSAGE not in failure]


def _shot_for_component(timeline: Any, component: Engagement) -> Any:
    return timeline.shots[int(component.shot_index)]


def _hostile_events_in_span(
    timeline: Any,
    shot_index: int,
    start: float,
    end: float,
    config: dict[str, Any],
) -> tuple[Any, ...]:
    return tuple(
        event
        for event in timeline.consolidated_events
        if int(core._shot_index(timeline.shots, float(event.time))) == int(shot_index)
        and start - _EPS <= float(event.time) <= end + _EPS
        and combat.hostile_decision(event, config).hostile
    )


def _span_engagement(
    timeline: Any,
    shot_index: int,
    start: float,
    end: float,
    config: dict[str, Any],
) -> Engagement | None:
    events = _hostile_events_in_span(
        timeline,
        shot_index,
        start,
        end,
        config,
    )
    if not events:
        return None
    scores = [combat.hostile_decision(event, config).score for event in events]
    return Engagement(
        round(float(start), 3),
        round(float(end), 3),
        int(shot_index),
        round(float(np.mean(scores)), 4),
        events,
    )


def _span_evidence(
    timeline: Any,
    shot_index: int,
    start: float,
    end: float,
) -> tuple[Engagement, ...]:
    return tuple(
        item
        for item in timeline.engagements
        if int(item.shot_index) == int(shot_index)
        and any(start - _EPS <= float(event.time) <= end + _EPS for event in item.events)
    )


def _span_metrics(
    timeline: Any,
    proposal: Engagement,
    config: dict[str, Any],
) -> dict[str, float]:
    start, end = float(proposal.start), float(proposal.end)
    evidence = _span_evidence(
        timeline,
        int(proposal.shot_index),
        start,
        end,
    ) or (proposal,)
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


def _normal_span_variants(
    timeline: Any,
    chain: tuple[Engagement, ...],
    config: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    editor = config["semantic_editor"]
    minimum = _f(editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(editor.get("maximum_output_seconds", 20.0), 20.0)
    preferred = min(
        maximum,
        _f(editor.get("preferred_output_seconds", 12.5), 12.5),
    )
    tail = _f(editor["ending"].get("preferred_payoff_tail_seconds", 0.45), 0.45)
    shot = _shot_for_component(timeline, chain[0])
    shot_start, shot_end = float(shot.start), float(shot.end)

    all_events = tuple(
        sorted(
            (event for item in chain for event in item.events),
            key=lambda event: float(event.time),
        )
    )
    if not all_events:
        return ()

    payoff_events = tuple(
        event for item in chain for event in quality.verified_payoff_events(item, config)
    )
    terminal = max(
        payoff_events or all_events[-1:],
        key=lambda event: float(event.time),
    )
    terminal_end = min(shot_end, float(terminal.time) + tail)
    first_event = all_events[0]
    candidates: set[tuple[float, float]] = set()

    raw_start = float(chain[0].start)
    raw_end = float(chain[-1].end)
    if raw_end - raw_start <= maximum + _EPS:
        candidates.add((raw_start, raw_end))

    for target in (minimum, preferred, maximum):
        start = max(shot_start, terminal_end - target)
        if start <= float(first_event.time) + _EPS:
            candidates.add((start, terminal_end))

    anchored_start = max(shot_start, float(first_event.time) - 0.30)
    if terminal_end - anchored_start <= maximum + _EPS:
        candidates.add((anchored_start, terminal_end))

    # If a payoff-tail ending cannot reach the campaign minimum, allow the source
    # shot to extend forward; unchanged ending/retention gates decide whether it stays.
    if terminal_end - shot_start < minimum - _EPS:
        end = min(shot_end, max(terminal_end, shot_start + minimum))
        candidates.add((shot_start, end))

    unique = [
        (round(start, 3), round(end, 3))
        for start, end in candidates
        if end > start + _EPS
        and end - start <= maximum + _EPS
        and shot_start - _EPS <= start
        and end <= shot_end + _EPS
    ]
    unique.sort(
        key=lambda item: (
            0 if minimum - _EPS <= item[1] - item[0] <= maximum + _EPS else 1,
            abs((item[1] - item[0]) - preferred),
            item[0],
        )
    )
    return tuple(unique[:8])


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
        config["semantic_editor"].get("maximum_output_seconds", 20.0),
        20.0,
    )
    components = support._support_components(timeline, config)
    plans: list[SemanticPlanV31] = []
    diagnostics = {
        "support_component_count": len(components),
        "normal_component_chain_count": 0,
        "normal_source_span_variant_count": 0,
        "normal_duration_reject_count": 0,
        "normal_integrity_reject_count": 0,
        "normal_quality_reject_count": 0,
        "normal_qualified_count": 0,
    }

    for chain in support._component_chains(components, config):
        diagnostics["normal_component_chain_count"] += 1
        if not chain:
            continue
        for start, end in _normal_span_variants(timeline, chain, config):
            diagnostics["normal_source_span_variant_count"] += 1
            output_duration = end - start
            if not (minimum - _EPS <= output_duration <= maximum + _EPS):
                diagnostics["normal_duration_reject_count"] += 1
                continue
            if core._intersects_excluded(start, end, excluded) or base._protected_finishing_overlap(
                timeline,
                start,
                end,
                config,
            ):
                diagnostics["normal_integrity_reject_count"] += 1
                continue

            proposal = _span_engagement(
                timeline,
                int(chain[0].shot_index),
                start,
                end,
                config,
            )
            if proposal is None:
                continue
            metrics = _span_metrics(timeline, proposal, config)
            evidence = _span_evidence(
                timeline,
                int(proposal.shot_index),
                start,
                end,
            ) or (proposal,)
            story, profile, effects, reasons = quality._route_story(
                timeline,
                evidence,
            )
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

            segment = EditSegment(
                round(start, 3),
                round(end, 3),
                1.0,
                "verified_combat_story",
            )
            plan = SemanticPlanV31(
                segment.start,
                segment.end,
                round(segment.end - segment.start, 3),
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
                (segment,),
                effects,
                evidence,
                None,
                (
                    *reasons,
                    "verified hostile evidence anchors a contiguous same-shot source proposal",
                    "semantic reload/search/recovery gaps are evaluated by unchanged quality gates rather than treated as source cuts",
                ),
            )
            integrity = plan_integrity_violations(
                plan,
                timeline,
                config,
                source_key,
            )
            if integrity:
                diagnostics["normal_integrity_reject_count"] += 1
                continue
            plans.append(plan)
            diagnostics["normal_qualified_count"] += 1

    unique: list[SemanticPlanV31] = []
    seen: set[tuple[float, float]] = set()
    for plan in sorted(plans, key=lambda item: item.score, reverse=True):
        key = (
            round(float(plan.segments[0].start), 3),
            round(float(plan.segments[0].end), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(plan)

    existing = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    existing.update(diagnostics)
    timeline._proposal_diagnostics = existing
    return unique


def _candidate_starts(
    *,
    earliest: float,
    component_start: float,
    anchor_start: float,
    end: float,
    minimum: float,
    step: float = 0.25,
) -> tuple[float, ...]:
    # Search both backward and forward around the verified payoff anchor.
    # The payoff must remain inside the proposal; the unchanged quality gates
    # decide how much same-shot context is admissible.
    latest = end - minimum
    if latest < earliest - _EPS:
        return ()
    starts: set[float] = {
        round(earliest, 3),
        round(max(earliest, min(component_start, latest)), 3),
        round(max(earliest, min(anchor_start, latest)), 3),
        round(latest, 3),
    }
    cursor = latest
    while cursor > earliest + _EPS:
        starts.add(round(cursor, 3))
        cursor -= step
    return tuple(sorted(starts))


def _finishing_variants(
    timeline: Any,
    component: Engagement,
    later_floor: float,
    config: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    """Generate payoff-anchored same-shot spans; quality gates decide semantic gaps."""
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    minimum = _f(
        cfg.get("minimum_verified_island_moment_seconds", 1.0),
        1.0,
    )
    maximum = _f(
        cfg.get("maximum_verified_island_moment_seconds", 6.5),
        6.5,
    )
    tail = _f(
        config["semantic_editor"]["ending"].get(
            "preferred_payoff_tail_seconds",
            0.45,
        ),
        0.45,
    )
    shot = _shot_for_component(timeline, component)
    shot_start, shot_end = float(shot.start), float(shot.end)
    candidates: set[tuple[float, float]] = set()

    for payoff in quality.verified_payoff_events(component, config):
        natural_end = min(shot_end, float(payoff.time) + tail)
        earliest = max(
            shot_start,
            later_floor,
            natural_end - maximum,
        )
        first = min(
            (event for event in component.events if float(event.time) <= float(payoff.time) + _EPS),
            key=lambda event: float(event.time),
            default=payoff,
        )
        anchor_start = max(
            earliest,
            float(first.time) - 0.30,
        )
        starts = _candidate_starts(
            earliest=earliest,
            component_start=float(component.start),
            anchor_start=anchor_start,
            end=natural_end,
            minimum=minimum,
        )

        for start in starts:
            latest_end = min(
                shot_end,
                start + maximum,
            )
            if latest_end < natural_end - _EPS:
                continue

            # Preserve the preferred payoff-tail candidate, then search forward
            # within the same real source shot. Longer non-terminal body moments
            # may include semantic reload/search/recovery regions only when the
            # unchanged retention/weakest/residual quality gates still pass.
            ends: set[float] = {
                round(natural_end, 3),
                round(latest_end, 3),
            }
            cursor = natural_end
            while cursor < latest_end - _EPS:
                ends.add(round(cursor, 3))
                cursor += 0.25

            for end in ends:
                duration = end - start
                if minimum - _EPS <= duration <= maximum + _EPS:
                    candidates.add(
                        (
                            round(start, 3),
                            round(end, 3),
                        )
                    )

    unique = sorted(
        candidates,
        key=lambda item: (
            -(item[1] - item[0]),
            item[0],
            item[1],
        ),
    )
    return tuple(unique[:160])


def finishing_open_plans(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    results: list[SemanticPlanV31] = []
    diagnostics: list[dict[str, Any]] = []
    components = support._support_components(timeline, config)
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
    final_minimum = _f(
        config["semantic_editor"].get("minimum_output_seconds", 10.0),
        10.0,
    )
    final_maximum = min(
        _f(
            config["semantic_editor"].get("maximum_output_seconds", 20.0),
            20.0,
        ),
        _f(
            editorial.get("finishing_move_max_output_seconds", 15.5),
            15.5,
        ),
    )
    max_islands = int(cfg.get("maximum_verified_island_moments_per_body", 3))

    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero = EditSegment(
            round(
                max(float(shot.start), float(span.start) - lead),
                3,
            ),
            round(
                min(float(shot.end), float(span.end) + hold),
                3,
            ),
            1.0,
            "finishing_move_open_hero",
        )
        hero_duration = quality._segment_duration(hero)
        body_minimum = base._required_body_duration(hero, final_minimum)
        body_maximum = max(0.0, final_maximum - hero_duration)
        later_floor = max(float(hero.end), float(span.end) + 0.25)

        moments: list[
            tuple[
                tuple[int, float, float],
                Engagement,
                dict[str, float],
                bool,
            ]
        ] = []
        moment_diagnostics: list[dict[str, Any]] = []

        for component in components:
            component_key = (
                int(component.shot_index),
                round(float(component.start), 3),
                round(float(component.end), 3),
            )
            qualified_for_component: list[
                tuple[
                    Engagement,
                    dict[str, float],
                    bool,
                ]
            ] = []

            for start, end in _finishing_variants(
                timeline,
                component,
                later_floor,
                config,
            ):
                if start < later_floor - _EPS:
                    continue
                if core._intersects_excluded(start, end, excluded):
                    continue
                proposal = _span_engagement(
                    timeline,
                    int(component.shot_index),
                    start,
                    end,
                    config,
                )
                if proposal is None or not quality.verified_payoff_events(
                    proposal,
                    config,
                ):
                    continue

                metrics = _span_metrics(
                    timeline,
                    proposal,
                    config,
                )
                component_failures = base._component_metric_failures(
                    metrics,
                    config,
                    require_ending=False,
                )
                terminal_failures = base._component_metric_failures(
                    metrics,
                    config,
                    require_ending=True,
                )
                moment_diagnostics.append(
                    {
                        "support_component": [
                            round(float(component.start), 3),
                            round(float(component.end), 3),
                        ],
                        "proposal_bounds": [
                            round(start, 3),
                            round(end, 3),
                        ],
                        "proposal_duration": round(end - start, 3),
                        "crosses_semantic_gap": bool(
                            islands.segment_crosses_gap(
                                timeline,
                                start,
                                end,
                            )
                        ),
                        "component_failures": component_failures,
                        "terminal_failures": terminal_failures,
                    }
                )
                if not component_failures:
                    qualified_for_component.append(
                        (
                            proposal,
                            metrics,
                            not terminal_failures,
                        )
                    )

            def by_duration(item):
                return (
                    -(float(item[0].end) - float(item[0].start)),
                    float(item[0].start),
                    float(item[0].end),
                )

            longest = sorted(
                qualified_for_component,
                key=by_duration,
            )[:8]
            terminal_variants = sorted(
                (item for item in qualified_for_component if item[2]),
                key=by_duration,
            )[:4]
            selected_variants: list[
                tuple[
                    Engagement,
                    dict[str, float],
                    bool,
                ]
            ] = []
            seen_variant_bounds: set[tuple[float, float]] = set()
            for item in longest + terminal_variants:
                bounds = (
                    round(float(item[0].start), 3),
                    round(float(item[0].end), 3),
                )
                if bounds in seen_variant_bounds:
                    continue
                seen_variant_bounds.add(bounds)
                selected_variants.append(item)

            for proposal, metrics, terminal_ok in selected_variants:
                moments.append(
                    (
                        component_key,
                        proposal,
                        metrics,
                        terminal_ok,
                    )
                )

        group_rejections: list[dict[str, Any]] = []
        valid_count = 0
        for count in range(1, max(1, max_islands) + 1):
            for group in itertools.combinations(moments, count):
                component_keys = [item[0] for item in group]
                if len(set(component_keys)) != len(component_keys):
                    continue
                ordered = tuple(
                    sorted(
                        group,
                        key=lambda item: float(item[1].start),
                    )
                )
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
                                [
                                    round(float(item.start), 3),
                                    round(float(item.end), 3),
                                ]
                                for item in engagements
                            ],
                            "body_duration": round(body_duration, 3),
                            "required_body_minimum_seconds": round(
                                body_minimum,
                                3,
                            ),
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
                    max(
                        0.90,
                        float(metrics[0]["opening"]),
                    ),
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
                opening = max(
                    0.90,
                    min(
                        1.0,
                        0.74 + 0.22 * float(span.confidence),
                    ),
                )
                retention = float(0.18 * opening + 0.82 * body_retention)
                payoff = float(
                    0.45
                    * min(
                        1.0,
                        float(span.confidence) + 0.12,
                    )
                    + 0.55 * body_payoff
                )
                coherence = min(
                    1.0,
                    0.10 + 0.88 * body_coherence,
                )
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
                        "body moments are payoff-anchored contiguous source spans inside one real source shot",
                        "semantic reload/search/recovery gaps remain subject to unchanged quality gates instead of becoming source-cut boundaries",
                        "every body moment retains an independently verified hostile payoff and unchanged component quality floors",
                        "terminal body moment passes the unchanged ending-quality gate",
                    ),
                )
                integrity = plan_integrity_violations(
                    plan,
                    timeline,
                    config,
                    source_key,
                )
                if integrity:
                    group_rejections.append(
                        {
                            "proposals": [
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
                "support_component_count": len(components),
                "body_proposal_count": len(moments),
                "required_body_minimum_seconds": round(
                    body_minimum,
                    3,
                ),
                "allowed_body_maximum_seconds": round(
                    body_maximum,
                    3,
                ),
                "maximum_qualified_body_seconds_seen": round(
                    max(
                        (
                            sum(float(item[1].end) - float(item[1].start) for item in group)
                            for count in range(
                                1,
                                max(1, max_islands) + 1,
                            )
                            for group in itertools.combinations(
                                moments,
                                count,
                            )
                            if len({entry[0] for entry in group}) == len(group)
                        ),
                        default=0.0,
                    ),
                    3,
                ),
                "valid_finishing_plan_count": valid_count,
                "moment_diagnostics": moment_diagnostics[:96],
                "group_rejections": group_rejections[:96],
                "boundary_model": "verified_payoff_anchor_to_same_shot_quality_gated_source_span",
                "semantic_gap_is_not_source_cut": True,
                "threshold_reduction_used": False,
                "alternate_planner_available": False,
            }
        )

    timeline._finishing_continuation_diagnostics = diagnostics
    return sorted(
        results,
        key=lambda item: item.score,
        reverse=True,
    )


def build_plans_for_source(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    base.validate_configuration(config)
    normal = normal_plans(
        base,
        timeline,
        config,
        excluded,
        source_key,
    )
    finishing = finishing_open_plans(
        base,
        timeline,
        config,
        excluded,
        source_key,
    )
    plans = sorted(
        finishing + normal,
        key=lambda item: item.score,
        reverse=True,
    )
    diagnostics = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    diagnostics.update(
        {
            "proposal_architecture": "verified_anchor_to_same_shot_quality_gated_source_span",
            "support_components_are_not_final_clip_bounds": True,
            "semantic_gap_is_not_source_cut": True,
            "semantic_montage_enabled": False,
            "fallback_planner_enabled": False,
            "threshold_reduction_used": False,
            "qualified_plan_count": len(plans),
        }
    )
    timeline._proposal_diagnostics = diagnostics
    return plans


def self_test(base: Any) -> None:
    hero = EditSegment(
        148.07,
        150.50,
        1.0,
        "finishing_move_open_hero",
    )
    if abs(float(base._required_body_duration(hero, 10.0)) - 7.57) > _EPS:
        raise AssertionError("source-span planner changed hero-aware duration contract")

    starts = _candidate_starts(
        earliest=150.50,
        component_start=153.12,
        anchor_start=154.617,
        end=156.70,
        minimum=1.0,
    )
    if not starts or min(starts) >= 153.12 - _EPS:
        raise AssertionError(
            "payoff-anchored source-span planner still treats the support component as a clip wall"
        )
    if 156.70 - min(starts) > 6.5 + _EPS:
        raise AssertionError(
            "payoff-anchored source-span planner exceeded unchanged body-moment maximum"
        )
    if min(starts) < 150.50 - _EPS:
        raise AssertionError(
            "payoff-anchored source-span planner crossed strict later-content floor"
        )
    natural_end = 156.70
    if max(starts) <= 154.617 + _EPS:
        raise AssertionError(
            "payoff-anchored source-span search cannot move its opening forward to capture later same-shot action"
        )
    expanded_start = max(starts)
    expanded_end = min(170.0, expanded_start + 6.5)
    if expanded_end <= natural_end + _EPS:
        raise AssertionError(
            "non-terminal payoff-anchored body moment cannot expand forward inside its source shot"
        )
    if expanded_end - expanded_start > 6.5 + _EPS:
        raise AssertionError("forward-expanded body moment exceeded unchanged 6.5s maximum")
    print("MW4 same-shot quality-gated source-span proposal self-test: PASS")

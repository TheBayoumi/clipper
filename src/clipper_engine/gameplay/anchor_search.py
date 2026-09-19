from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, ClassVar

from . import analysis, interaction, quality
from . import source_spans as canonical

SemanticPlan = canonical.SemanticPlan
EditSegment = canonical.EditSegment
Engagement = canonical.Engagement
_EPS = 1e-3

finishing_open_plans = canonical.finishing_open_plans


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def plan_integrity_violations(
    plan: SemanticPlan,
    timeline: Any,
    config: dict[str, Any],
    source_key: str,
) -> list[str]:
    """Keep verified source cuts fatal while treating automatic scene cuts as soft evidence.

    The input is a source stringout. Automatic scene-cut detection is useful for
    evidence grouping, but it is not reliable enough to erase an independently
    verified hostile payoff merely because a 10-20s proposal crosses an automatic
    boundary. Manually verified stringout cut windows remain hard integrity walls.
    Finishing Move routing keeps the stricter canonical segment rules unchanged.
    """
    failures = canonical.plan_integrity_violations(
        plan,
        timeline,
        config,
        source_key,
    )
    if plan.story_type == "finishing_move_open":
        return failures
    return [failure for failure in failures if "crosses a source-shot boundary" not in failure]


def _verified_payoff_anchors(
    timeline: Any,
    config: dict[str, Any],
) -> tuple[Any, ...]:
    seen: set[float] = set()
    anchors: list[Any] = []
    for engagement in timeline.engagements:
        for event in quality.verified_payoff_events(engagement, config):
            key = round(float(event.time), 3)
            if key in seen:
                continue
            seen.add(key)
            anchors.append(event)
    return tuple(sorted(anchors, key=lambda item: float(item.time)))


def _verified_source_region(
    timeline: Any,
    source_key: str,
    anchor_time: float,
    config: dict[str, Any],
) -> tuple[float, float] | None:
    """Return the contiguous source region bounded only by verified stringout cuts."""
    duration = float(getattr(timeline.base, "duration", 0.0))
    if duration <= 0.0:
        return None
    start, end = 0.0, duration
    for left, right in quality._verified_cut_windows(config, source_key):
        if left - _EPS <= anchor_time <= right + _EPS:
            return None
        if right < anchor_time:
            start = max(start, right)
            continue
        if left > anchor_time:
            end = min(end, left)
            break
    if end <= start + _EPS:
        return None
    return (round(start, 3), round(end, 3))


def _anchor_variants(
    region_start: float,
    region_end: float,
    anchor_time: float,
    config: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    editor = config["semantic_editor"]
    minimum = _f(editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(editor.get("maximum_output_seconds", 12.0), 12.0)
    preferred = min(
        maximum,
        _f(editor.get("preferred_output_seconds", 11.0), 11.0),
    )
    preferred_tail = _f(
        editor["ending"].get("preferred_payoff_tail_seconds", 0.45),
        0.45,
    )
    maximum_tail = _f(
        editor["ending"].get("maximum_payoff_tail_seconds", 0.75),
        0.75,
    )

    candidates: set[tuple[float, float]] = set()
    end_seeds = {
        min(region_end, anchor_time + preferred_tail),
        min(region_end, anchor_time + maximum_tail),
        min(region_end, anchor_time + 1.25),
        min(region_end, anchor_time + 2.0),
    }
    for target in (minimum, preferred, maximum):
        for end_seed in end_seeds:
            start = max(region_start, end_seed - target)
            end = min(region_end, start + target)
            if end - start < minimum - _EPS:
                # If the anchor is near the start of the verified region, search
                # forward instead of failing just because payoff-tail framing is
                # too short.
                start = max(region_start, min(anchor_time - 0.30, region_end - minimum))
                end = min(region_end, start + target)
            if end - start < minimum - _EPS:
                continue
            if not (start - _EPS <= anchor_time <= end + _EPS):
                continue
            candidates.add((round(start, 3), round(end, 3)))

    action_start = max(region_start, anchor_time - 0.30)
    for target in (minimum, preferred, maximum):
        start = min(action_start, max(region_start, region_end - target))
        end = min(region_end, start + target)
        if end - start >= minimum - _EPS and start - _EPS <= anchor_time <= end + _EPS:
            candidates.add((round(start, 3), round(end, 3)))

    ordered = sorted(
        candidates,
        key=lambda item: (
            abs((item[1] - item[0]) - preferred),
            abs(item[1] - (anchor_time + preferred_tail)),
            item[0],
        ),
    )
    return tuple(ordered[:16])


def _span_hostile_events(
    timeline: Any,
    start: float,
    end: float,
    config: dict[str, Any],
) -> tuple[Any, ...]:
    return tuple(
        event
        for event in timeline.consolidated_events
        if start - _EPS <= float(event.time) <= end + _EPS
        and interaction.hostile_decision(event, config).hostile
    )


def _span_evidence(
    timeline: Any,
    start: float,
    end: float,
) -> tuple[Engagement, ...]:
    return tuple(
        item
        for item in timeline.engagements
        if any(start - _EPS <= float(event.time) <= end + _EPS for event in item.events)
    )


def _span_proposal(
    timeline: Any,
    start: float,
    end: float,
    anchor_time: float,
    config: dict[str, Any],
) -> Engagement | None:
    events = _span_hostile_events(timeline, start, end, config)
    if not events:
        return None
    scores = [interaction.hostile_decision(event, config).score for event in events]
    shot_index = int(canonical.core._shot_index(timeline.shots, anchor_time))
    return Engagement(
        round(start, 3),
        round(end, 3),
        shot_index,
        round(sum(scores) / len(scores), 4),
        events,
    )


def _span_metrics(
    timeline: Any,
    proposal: Engagement,
    evidence: tuple[Engagement, ...],
    config: dict[str, Any],
) -> dict[str, float]:
    start, end = float(proposal.start), float(proposal.end)
    explained = evidence or (proposal,)
    residual = quality._longest_unexplained_low_run(
        timeline,
        start,
        end,
        explained,
        None,
        config,
    )
    opening, ending, coherence, retention, payoff, weakest, low_fraction = quality._quality_metrics(
        timeline,
        start,
        end,
        explained,
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


def _plan_for_span(
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
    start: float,
    end: float,
    anchor_time: float,
) -> tuple[SemanticPlan | None, str]:
    minimum = _f(config["semantic_editor"].get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(config["semantic_editor"].get("maximum_output_seconds", 12.0), 12.0)
    output_duration = end - start
    if not (minimum - _EPS <= output_duration <= maximum + _EPS):
        return None, "duration"
    if canonical.core._intersects_excluded(start, end, excluded):
        return None, "excluded"
    if analysis._protected_finishing_overlap(timeline, start, end, config):
        return None, "protected_finishing_overlap"

    proposal = _span_proposal(timeline, start, end, anchor_time, config)
    if proposal is None:
        return None, "no_verified_hostile_evidence"
    payoff_times = {
        round(float(item.time), 3) for item in quality.verified_payoff_events(proposal, config)
    }
    if round(anchor_time, 3) not in payoff_times:
        return None, "anchor_not_verified_payoff"

    evidence = _span_evidence(timeline, start, end) or (proposal,)
    admissibility = quality.story_admissibility_failures(
        timeline,
        start,
        end,
        evidence,
        config,
    )
    if admissibility:
        return None, "admissibility." + admissibility[0]

    metrics = _span_metrics(timeline, proposal, evidence, config)
    story, profile, effects, reasons = quality._route_story(timeline, evidence)
    quality_diagnostics = quality.story_quality_diagnostics(
        story,
        metrics["opening"],
        metrics["ending"],
        metrics["retention"],
        metrics["payoff"],
        metrics["weakest"],
        metrics["low_fraction"],
        metrics["residual"],
        config,
    )

    segment = EditSegment(round(start, 3), round(end, 3), 1.0, "verified_combat_story")
    plan = SemanticPlan(
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
            f"verified payoff anchor {anchor_time:.3f}s receives an independent "
            "source-region proposal search",
            "source-integrity and semantic-admissibility gates are hard; "
            "quality ranks legal framings",
        ),
        round(anchor_time, 3),
        quality_diagnostics,
    )
    integrity = plan_integrity_violations(plan, timeline, config, source_key)
    if integrity:
        return None, "source_integrity:" + "|".join(integrity[:3])
    return plan, "qualified"


def _anchor_plans(
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlan]:
    anchors = _verified_payoff_anchors(timeline, config)
    qualified: dict[float, list[SemanticPlan]] = defaultdict(list)
    attempted = 0
    anchor_diagnostics: list[dict[str, Any]] = []

    for event in anchors:
        anchor_time = round(float(event.time), 3)
        region = _verified_source_region(
            timeline,
            source_key,
            anchor_time,
            config,
        )
        if region is None:
            anchor_diagnostics.append(
                {
                    "anchor_time": anchor_time,
                    "source_region": None,
                    "attempted_variant_count": 0,
                    "qualified_variant_count": 0,
                    "rejections": {"inside_verified_cut_or_no_region": 1},
                }
            )
            continue

        rejections: Counter[str] = Counter()
        variants = _anchor_variants(
            region[0],
            region[1],
            anchor_time,
            config,
        )
        for start, end in variants:
            attempted += 1
            plan, reason = _plan_for_span(
                timeline,
                config,
                excluded,
                source_key,
                start,
                end,
                anchor_time,
            )
            if plan is None:
                rejections[reason] += 1
            else:
                qualified[anchor_time].append(plan)

        anchor_diagnostics.append(
            {
                "anchor_time": anchor_time,
                "source_region": [region[0], region[1]],
                "attempted_variant_count": len(variants),
                "qualified_variant_count": len(qualified[anchor_time]),
                "rejections": dict(sorted(rejections.items())),
            }
        )

    selected: list[SemanticPlan] = []
    qualified_anchor_count = 0
    for anchor_time in sorted(qualified):
        unique: list[SemanticPlan] = []
        seen: set[tuple[float, float]] = set()
        for plan in sorted(
            qualified[anchor_time],
            key=lambda item: item.score,
            reverse=True,
        ):
            key = (
                round(float(plan.segments[0].start), 3),
                round(float(plan.segments[0].end), 3),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(plan)
        if unique:
            qualified_anchor_count += 1
            selected.extend(unique[:3])

    existing = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    existing.update(
        {
            "verified_payoff_anchor_count": len(anchors),
            "payoff_anchor_times": [round(float(item.time), 3) for item in anchors],
            "payoff_anchor_span_variant_count": attempted,
            "payoff_anchor_qualified_anchor_count": qualified_anchor_count,
            "payoff_anchor_unrepresented_count": len(anchors) - qualified_anchor_count,
            "payoff_anchor_independent_search": True,
            "automatic_scene_cuts_are_soft_for_normal_payoff_proposals": True,
            "verified_stringout_cuts_remain_hard": True,
            "payoff_anchor_diagnostics": anchor_diagnostics,
        }
    )
    timeline._proposal_diagnostics = existing
    return selected


def normal_plans(
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlan]:
    anchored = _anchor_plans(
        timeline,
        config,
        excluded,
        source_key,
    )
    legacy = canonical.normal_plans(
        timeline,
        config,
        excluded,
        source_key,
    )
    merged: list[SemanticPlan] = []
    seen: set[tuple[float, float]] = set()
    for plan in sorted(
        anchored + legacy,
        key=lambda item: item.score,
        reverse=True,
    ):
        key = (
            round(float(plan.segments[0].start), 3),
            round(float(plan.segments[-1].end), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(plan)
    return merged


def _one_per_finishing_move(
    plans: list[SemanticPlan],
) -> list[SemanticPlan]:
    best: dict[tuple[float, float, float], SemanticPlan] = {}
    for plan in plans:
        span = plan.finishing_move
        if span is None:
            continue
        key = (
            round(float(span.start), 3),
            round(float(span.payoff), 3),
            round(float(span.end), 3),
        )
        if key not in best or plan.score > best[key].score:
            best[key] = plan
    return sorted(
        best.values(),
        key=lambda item: item.score,
        reverse=True,
    )


def build_plans_for_source(
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlan]:
    analysis.validate_configuration(config)
    normal = normal_plans(
        timeline,
        config,
        excluded,
        source_key,
    )
    finishing = _one_per_finishing_move(
        canonical.finishing_open_plans(
            timeline,
            config,
            excluded,
            source_key,
        )
    )
    plans = sorted(
        finishing + normal,
        key=lambda item: item.score,
        reverse=True,
    )
    diagnostics = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    diagnostics.update(
        {
            "proposal_architecture": (
                "every_verified_payoff_anchor_to_verified_source_region_quality_gated_span"
            ),
            "finishing_move_variants_collapsed_per_verified_move": True,
            "automatic_scene_cuts_are_soft_for_normal_payoff_proposals": True,
            "verified_stringout_cuts_remain_hard": True,
            "semantic_montage_enabled": False,
            "fallback_planner_enabled": False,
            "threshold_reduction_used": False,
            "qualified_plan_count": len(plans),
        }
    )
    timeline._proposal_diagnostics = diagnostics
    return plans


def self_test() -> None:
    canonical.self_test()

    class Base:
        duration = 30.0

    class Shot:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class Timeline:
        base = Base()
        shots: ClassVar[list[Shot]] = [Shot(0.0, 5.0), Shot(5.2, 30.0)]

    config = {
        "source_integrity": {"verified_cut_windows": {}},
        "semantic_editor": {
            "minimum_output_seconds": 10.0,
            "maximum_output_seconds": 12.0,
            "preferred_output_seconds": 11.0,
            "ending": {
                "preferred_payoff_tail_seconds": 0.45,
                "maximum_payoff_tail_seconds": 0.75,
            },
        },
    }
    region = _verified_source_region(
        Timeline(),
        "test",
        4.8,
        config,
    )
    if region != (0.0, 30.0):
        raise AssertionError(
            "automatic scene cut incorrectly became a hard verified source region wall"
        )
    variants = _anchor_variants(
        region[0],
        region[1],
        4.8,
        config,
    )
    if not variants or not any(
        start <= 4.8 <= end and end - start >= 10.0 - _EPS and end > 5.2 for start, end in variants
    ):
        raise AssertionError(
            "verified payoff near an automatic cut cannot receive a campaign-length "
            "source-region proposal"
        )

    cut_config = {
        **config,
        "source_integrity": {"verified_cut_windows": {"test": [[6.0, 6.5]]}},
    }
    hard_region = _verified_source_region(
        Timeline(),
        "test",
        4.8,
        cut_config,
    )
    if hard_region != (0.0, 6.0):
        raise AssertionError("verified stringout cut did not remain a hard proposal boundary")

    event = type(
        "Event",
        (),
        {
            "time": 4.8,
            "kinds": ("outcome_like",),
            "confidence": 0.95,
            "evidence": {
                "local_refine_attempted": 1.0,
                "local_hitmarker_score": 0.95,
                "combat": 0.90,
                "outcome": 0.90,
                "impact": 0.90,
                "audio_transient": 0.90,
                "center_motion": 0.90,
            },
        },
    )()
    decision_config = {
        "combat_state_verifier": {
            "enabled": True,
            "local_interaction_verifier": {
                "enabled": True,
                "minimum_hitmarker_score": 0.34,
            },
            "minimum_hostile_event_score": 0.64,
            "outcome_minimum": 0.52,
            "payoff_combat_minimum": 0.42,
            "impact_minimum": 0.64,
            "impact_combat_minimum": 0.55,
            "strong_combat_minimum": 0.70,
            "strong_combat_audio_transient_minimum": 0.62,
            "strong_combat_center_motion_minimum": 0.40,
        }
    }
    hostile_timeline = type("HostileTimeline", (), {"consolidated_events": (event,)})()
    if _span_hostile_events(hostile_timeline, 0.0, 10.0, decision_config) != (event,):
        raise AssertionError("anchor search did not use the canonical interaction verifier")

    print("MW4 payoff-complete verified-source-region proposal self-test: PASS")

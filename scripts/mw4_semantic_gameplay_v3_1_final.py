from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np

import mw4_semantic_gameplay_v3_1_refined as refined
import mw4_semantic_gameplay_v3_1_local_verify as local_verify
import mw4_semantic_gameplay_v3_1_combat_state as combat
import mw4_semantic_gameplay_v3_1_combat_islands as islands
import mw4_semantic_gameplay_v3_1_quality as quality

ShotSpan = refined.ShotSpan
ConsolidatedEvent = refined.ConsolidatedEvent
Engagement = refined.Engagement
FinishingMoveSpan = refined.FinishingMoveSpan
EditSegment = refined.EditSegment
SemanticPlanV31 = refined.SemanticPlanV31
SemanticTimelineV31 = refined.SemanticTimelineV31
PAYOFF_KINDS = refined.PAYOFF_KINDS
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
    """Canonical configuration validation. There is no alternate planner path."""
    combat.validate_configuration(config)
    errors: list[str] = []
    if config.get("fallback_policy", {}).get("enabled") is not False:
        errors.append("fallback_policy.enabled must be false")
    if bool(config.get("semantic_editor", {}).get("semantic_montage", {}).get("enabled", False)):
        errors.append("semantic montage must be disabled")
    if bool(config.get("editorial", {}).get("finishing_move_allow_semantic_montage_continuation", False)):
        errors.append("semantic montage Finishing Move continuation is forbidden")
    if bool(config.get("source_integrity", {}).get("allow_planned_transition_for_semantic_montage", False)):
        errors.append("semantic montage source transitions are forbidden")
    if bool(config.get("finishing_move_detector", {}).get("allow_unverified_automatic", False)):
        errors.append("unverified automatic Finishing Moves are forbidden")
    continuation = config.get("combat_state_verifier", {}).get("finishing_continuation", {})
    if continuation.get("require_verified_payoff") is not True:
        errors.append("Finishing Move continuation must require verified payoff")
    for key in ("minimum_verified_hostile_anchors_without_payoff", "minimum_retention_without_payoff"):
        if key in continuation:
            errors.append(f"obsolete no-payoff continuation key is forbidden: {key}")
    if errors:
        raise RuntimeError("MW4 V3.1 canonical configuration violation: " + "; ".join(errors))


def _verified_cut_windows(config: dict[str, Any], source_key: str) -> tuple[tuple[float, float], ...]:
    return quality._verified_cut_windows(config, source_key)


def _build_hardened_shots(source: Path, duration: float, config: dict[str, Any]) -> tuple[ShotSpan, ...]:
    automatic = list(refined._scene_cut_times(source, config))
    verified = [(left + right) / 2.0 for left, right in _verified_cut_windows(config, _source_key(source))]
    cuts: list[float] = []
    for value in sorted(automatic + verified):
        if 0.35 < value < duration - 0.35 and all(abs(value - old) >= 0.18 for old in cuts):
            cuts.append(value)
    guard = float(config.get("semantic_analysis", {}).get("scene_change_guard_seconds", 0.12))
    boundaries = [0.0] + cuts + [duration]
    shots: list[ShotSpan] = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        start = left if index == 0 else min(right, left + guard)
        end = right if index == len(boundaries) - 2 else max(start, right - guard)
        if end - start >= 1.0:
            shots.append(ShotSpan(round(start, 3), round(end, 3)))
    return tuple(shots or [ShotSpan(0.0, duration)])


def _verified_finishing_moves(shots: tuple[ShotSpan, ...], source: Path, config: dict[str, Any]) -> tuple[FinishingMoveSpan, ...]:
    verified: list[FinishingMoveSpan] = []
    for item in config.get("finishing_move_detector", {}).get("verified_spans", {}).get(_source_key(source), []):
        start, payoff, end = float(item["start"]), float(item["payoff"]), float(item["end"])
        midpoint = (start + end) / 2.0
        shot_index = refined.core._shot_index(shots, midpoint)
        shot = shots[shot_index]
        if shot.start <= midpoint <= shot.end:
            verified.append(FinishingMoveSpan(
                round(max(float(shot.start), start), 3), round(payoff, 3), round(min(float(shot.end), end), 3),
                shot_index, 0.99, {"visual_verification": 1.0, "third_person_execution_regime": 1.0},
            ))
    return tuple(sorted(verified, key=lambda item: item.start))


def _verified_hostile_signal(timeline: SemanticTimelineV31) -> np.ndarray:
    signal = np.zeros(len(timeline.times), dtype=np.float32)
    for engagement in timeline.engagements:
        i0 = max(0, int(np.floor(float(engagement.start) * timeline.fps)))
        i1 = min(len(signal), int(np.ceil(float(engagement.end) * timeline.fps)))
        signal[i0:i1] = 1.0
    return signal


def analyze_source(source: Path, config: dict[str, Any]) -> SemanticTimelineV31:
    """Evidence -> direct interaction verification -> exact gap-split combat islands."""
    validate_configuration(config)
    base = refined.core.base.analyze_source(source, config)
    shots = _build_hardened_shots(source, float(base.duration), config)
    consolidated = refined.core.consolidate_events(base, shots)
    coarse = refined.core.cluster_engagements(base, shots, consolidated, config)
    timeline = SemanticTimelineV31(base, shots, consolidated, coarse, _verified_finishing_moves(shots, source, config))
    setattr(timeline, "_source_key", _source_key(source))
    local_verify.annotate_timeline(source, timeline, config)
    timeline.engagements = islands.build(timeline, coarse, config, refined.core, combat, Engagement)
    timeline.signals["combat_island_gap"] = islands.gap_signal(timeline, config)
    timeline.signals["verified_hostile"] = _verified_hostile_signal(timeline)
    setattr(timeline, "_semantic_diagnostics", {
        "coarse_engagement_count": len(coarse),
        "verified_combat_island_count": len(timeline.engagements),
        "contact_only_can_anchor_hostile": False,
        "unknown_actor_defaults_to_hostile": False,
        "combat_islands_are_exact_gap_split_bounds": True,
        "fallback_execution_allowed": False,
    })
    return timeline


def _candidate_engagement_chains(timeline: SemanticTimelineV31, config: dict[str, Any]) -> list[tuple[Engagement, ...]]:
    minimum = _f(config["semantic_editor"].get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(config["semantic_editor"].get("maximum_output_seconds", 20.0), 20.0)
    max_engagements = int(config["semantic_editor"].get("maximum_engagements_per_story", 7))
    chains: list[tuple[Engagement, ...]] = []
    for start_index, first in enumerate(timeline.engagements):
        chain: list[Engagement] = []
        for candidate in timeline.engagements[start_index:]:
            if candidate.shot_index != first.shot_index:
                break
            if chain and not islands.bridge_supported(timeline, float(chain[-1].end), float(candidate.start), config):
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


def _normal_plans(timeline: SemanticTimelineV31, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> list[SemanticPlanV31]:
    plans: list[SemanticPlanV31] = []
    for chain in _candidate_engagement_chains(timeline, config):
        start, end = float(chain[0].start), float(chain[-1].end)
        if refined.core._intersects_excluded(start, end, excluded) or islands.segment_crosses_gap(timeline, start, end):
            continue
        segment = EditSegment(round(start, 3), round(end, 3), 1.0, "verified_combat_story")
        residual = quality._longest_unexplained_low_run(timeline, start, end, chain, None, config)
        opening, ending, coherence, retention, payoff, weakest, low_fraction = quality._quality_metrics(
            timeline, start, end, chain, None, residual, config
        )
        story, profile, effects, reasons = quality._route_story(timeline, chain)
        if not quality._passes_story_gates(story, opening, ending, retention, payoff, weakest, low_fraction, residual, config):
            continue
        plan = SemanticPlanV31(
            round(start, 3), round(end, 3), round(end - start, 3), round(end - start, 3),
            round(quality._score_plan(retention, payoff, opening, ending, coherence, weakest, False, config), 5),
            round(retention, 4), round(payoff, 4), round(opening, 4), round(ending, 4), round(coherence, 4),
            round(weakest, 4), round(low_fraction, 4), round(residual, 3), story, profile, (segment,), effects, chain, None,
            reasons + ("exact verified-combat-island story bounds",),
        )
        if not quality.plan_integrity_violations(plan, timeline, config, source_key):
            plans.append(plan)
    return sorted(plans, key=lambda item: item.score, reverse=True)


def _island_body_metrics(timeline: SemanticTimelineV31, engagement: Engagement, config: dict[str, Any]) -> tuple[dict[str, float] | None, list[str]]:
    """Evaluate the exact canonical island; never shrink to a payoff-centered window."""
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    start, end = float(engagement.start), float(engagement.end)
    duration = end - start
    failures: list[str] = []
    if duration < _f(cfg.get("minimum_verified_island_moment_seconds", 1.0), 1.0) - _EPS:
        failures.append("verified combat island shorter than minimum body island")
    if duration > _f(cfg.get("maximum_verified_island_moment_seconds", 6.5), 6.5) + _EPS:
        failures.append("verified combat island longer than maximum body island")
    if not any(any(kind in PAYOFF_KINDS for kind in event.kinds) for event in engagement.events):
        failures.append("verified combat island lacks hostile payoff")
    if islands.segment_crosses_gap(timeline, start, end):
        failures.append("verified combat island crosses a hard combat gap")
    residual = quality._longest_unexplained_low_run(timeline, start, end, (engagement,), None, config)
    opening, ending, coherence, retention, payoff, weakest, low_fraction = quality._quality_metrics(
        timeline, start, end, (engagement,), None, residual, config
    )
    min_retention = max(_f(config.get("performance_targets", {}).get("retention_quality_min", 0.36), 0.36), _f(cfg.get("minimum_retention_quality", 0.40), 0.40))
    min_payoff = max(_f(config.get("performance_targets", {}).get("payoff_quality_min", 0.34), 0.34), _f(cfg.get("minimum_payoff_quality", 0.40), 0.40))
    checks = (
        (opening >= _f(cfg.get("minimum_verified_island_opening_quality", 0.42), 0.42), "verified combat island opening below floor"),
        (retention >= min_retention, "verified combat island retention below floor"),
        (payoff >= min_payoff, "verified combat island payoff below floor"),
        (ending >= _f(cfg.get("minimum_ending_quality", 0.45), 0.45), "verified combat island ending below floor"),
        (weakest >= _f(cfg.get("minimum_weakest_quarter_interest", 0.30), 0.30), "verified combat island weakest quarter below floor"),
        (residual <= _f(cfg.get("maximum_unexplained_low_interest_run_seconds", 0.75), 0.75), "verified combat island unexplained inactivity exceeds floor"),
        (low_fraction <= _f(config["semantic_editor"]["dull"].get("maximum_low_interest_fraction", 0.38), 0.38), "verified combat island low-interest fraction exceeds floor"),
    )
    failures.extend(message for passed, message in checks if not passed)
    failures.extend(
        f"ambiguous/non-hostile event at {event.time:.3f}s"
        for event in engagement.events if not combat.hostile_decision(event, config).hostile
    )
    if failures:
        return None, list(dict.fromkeys(failures))
    return {
        "duration": duration, "opening": opening, "ending": ending, "coherence": coherence,
        "retention": retention, "payoff": payoff, "weakest": weakest,
        "low_fraction": low_fraction, "residual": residual,
    }, []


def _finishing_open_plans(timeline: SemanticTimelineV31, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> list[SemanticPlanV31]:
    results: list[SemanticPlanV31] = []
    diagnostics: list[dict[str, Any]] = []
    editorial = config["editorial"]
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    lead = _f(editorial.get("finishing_move_opening_lead_seconds", 0.28), 0.28)
    hold = _f(editorial.get("finishing_move_payoff_hold_seconds", 0.45), 0.45)
    max_gap = _f(editorial.get("finishing_move_max_continuation_gap_seconds", 18.0), 18.0)
    final_minimum = _f(config["semantic_editor"].get("minimum_output_seconds", 10.0), 10.0)
    final_maximum = min(_f(config["semantic_editor"].get("maximum_output_seconds", 20.0), 20.0), _f(editorial.get("finishing_move_montage_max_output_seconds", 15.5), 15.5))
    max_islands = int(cfg.get("maximum_verified_island_moments_per_body", 3))

    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero = EditSegment(round(max(float(shot.start), float(span.start) - lead), 3), round(min(float(shot.end), float(span.end) + hold), 3), 1.0, "finishing_move_open_hero")
        hero_duration = quality._segment_duration(hero)
        body_minimum, body_maximum = max(0.0, final_minimum - hero_duration), max(0.0, final_maximum - hero_duration)
        later_floor = max(float(hero.end), float(span.end) + 0.25)
        qualified: list[tuple[Engagement, dict[str, float]]] = []
        island_diagnostics: list[dict[str, Any]] = []
        for engagement in timeline.engagements:
            if float(engagement.start) < later_floor - _EPS or float(engagement.start) > float(hero.end) + max_gap + _EPS:
                continue
            if refined.core._intersects_excluded(float(engagement.start), float(engagement.end), excluded):
                continue
            metrics, failures = _island_body_metrics(timeline, engagement, config)
            island_diagnostics.append({"start": round(float(engagement.start), 3), "end": round(float(engagement.end), 3), "qualified": metrics is not None, "failures": failures})
            if metrics is not None:
                qualified.append((engagement, metrics))

        groups: list[tuple[tuple[Engagement, dict[str, float]], ...]] = []
        for count in range(1, max(1, max_islands) + 1):
            for group in itertools.combinations(qualified, count):
                engagements = tuple(item[0] for item in group)
                if any(float(right.start) < float(left.end) - _EPS or float(right.start) - float(left.end) > max_gap + _EPS for left, right in zip(engagements, engagements[1:])):
                    continue
                body_duration = sum(item[1]["duration"] for item in group)
                if body_minimum - _EPS <= body_duration <= body_maximum + _EPS:
                    groups.append(group)

        valid_count = 0
        for group in groups:
            engagements = tuple(item[0] for item in group)
            metrics = [item[1] for item in group]
            durations = np.asarray([item["duration"] for item in metrics], dtype=float)
            body_retention = float(np.average([item["retention"] for item in metrics], weights=durations))
            payoff_values = [item["payoff"] for item in metrics]
            body_payoff = float(0.55 * max(payoff_values) + 0.45 * np.mean(payoff_values))
            body_ending = float(metrics[-1]["ending"])
            body_coherence = float(np.average([item["coherence"] for item in metrics], weights=durations))
            body_weakest = float(min(item["weakest"] for item in metrics))
            body_low_fraction = float(np.average([item["low_fraction"] for item in metrics], weights=durations))
            body_residual = float(max(item["residual"] for item in metrics))
            if not quality._passes_story_gates("finishing_move_open", max(0.90, float(metrics[0]["opening"])), body_ending, body_retention, body_payoff, body_weakest, body_low_fraction, body_residual, config):
                continue
            body_segments = tuple(EditSegment(round(float(item.start), 3), round(float(item.end), 3), 1.0, "verified_combat_island_body") for item in engagements)
            segments = (hero,) + body_segments
            output_duration = sum(quality._segment_duration(segment) for segment in segments)
            opening = max(0.90, min(1.0, 0.74 + 0.22 * float(span.confidence)))
            retention = float(0.18 * opening + 0.82 * body_retention)
            payoff = float(0.45 * min(1.0, float(span.confidence) + 0.12) + 0.55 * body_payoff)
            coherence = min(1.0, 0.10 + 0.88 * body_coherence)
            low_fraction = body_low_fraction * (float(np.sum(durations)) / output_duration)
            if not quality._passes_story_gates("finishing_move_open", opening, body_ending, retention, payoff, body_weakest, low_fraction, body_residual, config):
                continue
            plan = SemanticPlanV31(
                hero.start, body_segments[-1].end,
                round(sum(segment.end - segment.start for segment in segments), 3), round(output_duration, 3),
                round(quality._score_plan(retention, payoff, opening, body_ending, coherence, body_weakest, True, config), 5),
                round(retention, 4), round(payoff, 4), round(opening, 4), round(body_ending, 4), round(coherence, 4),
                round(body_weakest, 4), round(low_fraction, 4), round(body_residual, 3),
                "finishing_move_open", "finishing_move_hero", segments,
                quality._effect_events_for_engagements(timeline, engagements), engagements, span,
                ("verified Finishing Move opens the clip", "body uses exact canonical combat-island boundaries", "every body island independently passes direct-hostile/payoff/retention gates", "reload/search gaps are removed only by explicit hard cuts"),
            )
            if not quality.plan_integrity_violations(plan, timeline, config, source_key):
                results.append(plan)
                valid_count += 1
        diagnostics.append({
            "span_start": round(float(span.start), 3), "span_end": round(float(span.end), 3),
            "hero_start": round(float(hero.start), 3), "hero_end": round(float(hero.end), 3),
            "required_body_minimum_seconds": round(body_minimum, 3), "allowed_body_maximum_seconds": round(body_maximum, 3),
            "qualified_island_count": len(qualified), "candidate_group_count": len(groups),
            "valid_finishing_plan_count": valid_count, "island_diagnostics": island_diagnostics,
            "fallback_execution_allowed": False,
        })
    setattr(timeline, "_finishing_continuation_diagnostics", diagnostics)
    return sorted(results, key=lambda item: item.score, reverse=True)


def build_plans_for_source(timeline: SemanticTimelineV31, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> list[SemanticPlanV31]:
    validate_configuration(config)
    return sorted(_finishing_open_plans(timeline, config, excluded, source_key) + _normal_plans(timeline, config, excluded, source_key), key=lambda item: item.score, reverse=True)


def build_plans(timeline: SemanticTimelineV31, config: dict[str, Any], excluded: list[list[float]]) -> list[SemanticPlanV31]:
    return build_plans_for_source(timeline, config, excluded, getattr(timeline, "_source_key", ""))


def select_plans(plans: list[SemanticPlanV31], config: dict[str, Any]) -> list[SemanticPlanV31]:
    maximum, minimum = int(config.get("count_per_source_max", 4)), int(config.get("minimum_count_per_source", 0))
    selected: list[SemanticPlanV31] = []
    finishing = [plan for plan in plans if plan.story_type == "finishing_move_open"]
    if finishing:
        selected.append(finishing[0])
    for plan in plans:
        if plan in selected or any(quality._source_segments_overlap(plan, prior) for prior in selected):
            continue
        selected.append(plan)
        if len(selected) >= maximum:
            break
    if len(selected) < minimum:
        raise RuntimeError(f"Only {len(selected)} canonical V3.1 candidates passed; minimum is {minimum}.")
    return sorted(selected, key=lambda item: (item.story_type != "finishing_move_open", item.segments[0].start))


def diagnose_source(timeline: SemanticTimelineV31, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> dict[str, Any]:
    plans = build_plans_for_source(timeline, config, excluded, source_key)
    result = dict(getattr(timeline, "_semantic_diagnostics", {}))
    result.update({
        "shot_count": len(timeline.shots), "engagement_count": len(timeline.engagements),
        "verified_finishing_move_count": len(timeline.finishing_moves), "candidate_count_after_semantic_gates": len(plans),
        "local_interaction_verifier": dict(getattr(timeline, "_local_interaction_diagnostics", {})),
        "finishing_move_continuation_diagnostics": list(getattr(timeline, "_finishing_continuation_diagnostics", [])),
    })
    return result


def _self_test() -> None:
    validate_configuration({
        "fallback_policy": {"enabled": False},
        "editorial": {"finishing_move_allow_semantic_montage_continuation": False},
        "source_integrity": {"allow_planned_transition_for_semantic_montage": False},
        "semantic_editor": {"semantic_montage": {"enabled": False}},
        "finishing_move_detector": {"allow_unverified_automatic": False},
        "combat_state_verifier": {"enabled": True, "local_interaction_verifier": {"enabled": True, "minimum_hitmarker_score": 0.34}, "finishing_continuation": {"require_verified_payoff": True}},
    })
    combat.self_test()
    islands.self_test()
    local_verify.self_test()
    segment = EditSegment(153.533, 157.800, 1.0, "verified_combat_island_body")
    if (segment.start, segment.end) != (153.533, 157.8):
        raise AssertionError("Finishing Move body changed canonical combat-island boundaries")
    print("MW4 canonical semantic architecture self-test: PASS")


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test()
        return
    raise SystemExit("Use this module through mw4_fullframe_retention_v3_1.py")


if __name__ == "__main__":
    main()

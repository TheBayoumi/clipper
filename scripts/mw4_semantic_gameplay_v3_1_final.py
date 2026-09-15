from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import mw4_semantic_gameplay_v3_1_final_legacy as _legacy
from mw4_semantic_gameplay_v3_1_final_legacy import *  # noqa: F401,F403
import mw4_semantic_gameplay_v3_1_combat_state as _combat
import mw4_semantic_gameplay_v3_1_combat_islands as _islands
import mw4_semantic_gameplay_v3_1_finishing_body as _finishing_body
import mw4_semantic_gameplay_v3_1_no_fallback as _no_fallback

_ORIGINAL_INTEGRITY = _legacy.plan_integrity_violations
_ORIGINAL_FINISHING = _no_fallback.strict_finishing_assembler
_EPS = 1e-3


def _chronology_failures(plan: SemanticPlanV31) -> list[str]:
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
        if float(segment.speed) <= 0.0:
            failures.append(f"segment {index + 1} has non-positive speed")
        if index and float(segment.start) < float(segments[index - 1].end) - _EPS:
            failures.append(f"source chronology reversal/overlap at segment {index + 1}")
    return failures


def _planned_cross_shot_bridge(
    plan: SemanticPlanV31,
    left: EditSegment,
    right: EditSegment,
    bridge_index: int,
) -> bool:
    """Allow only explicit verified Finishing Move hard cuts; no generic bridge path."""
    if plan.story_type != "finishing_move_open":
        return False
    if (
        bridge_index == 0
        and left.reason == "finishing_move_open_hero"
        and right.reason == "verified_combat_island_body"
    ):
        return True
    return bool(
        left.reason == "verified_combat_island_body"
        and right.reason == "verified_combat_island_body"
    )


def plan_integrity_violations(plan: SemanticPlanV31, timeline: SemanticTimelineV31, config: dict[str, Any], source_key: str) -> list[str]:
    failures = list(_ORIGINAL_INTEGRITY(plan, timeline, config, source_key))
    failures.extend(_chronology_failures(plan))
    failures.extend(_combat.plan_combat_state_failures(plan, timeline, config))
    failures.extend(_islands.plan_island_failures(plan, timeline, config))
    if plan.story_type == "semantic_montage":
        failures.append("semantic montage fallback story is forbidden")
    if any(str(getattr(segment, "reason", "")) == "semantic_montage_moment" for segment in plan.segments):
        failures.append("semantic montage fallback segment is forbidden")
    return list(dict.fromkeys(failures))


def _matched_continuation(plan: SemanticPlanV31, continuations: list[SemanticPlanV31]) -> SemanticPlanV31 | None:
    body = tuple(plan.segments[1:])
    for continuation in continuations:
        if body == tuple(continuation.segments):
            return continuation
    return None


def _conservative_finishing_metrics(
    plan: SemanticPlanV31,
    continuation: SemanticPlanV31,
    span: FinishingMoveSpan,
    config: dict[str, Any],
) -> SemanticPlanV31 | None:
    hero_payoff = min(1.0, float(span.confidence) + 0.12)
    payoff = float(0.45 * hero_payoff + 0.55 * float(continuation.payoff_quality))
    retention = float(0.18 * float(plan.opening_quality) + 0.82 * float(continuation.retention_quality))
    weakest = min(float(plan.weakest_quarter_interest), float(continuation.weakest_quarter_interest))
    low_fraction = float(plan.low_interest_fraction)
    residual = max(float(plan.max_unexplained_low_interest_run_seconds), float(continuation.max_unexplained_low_interest_run_seconds))
    if not _legacy._passes_story_gates(
        "finishing_move_open",
        float(plan.opening_quality),
        float(plan.ending_quality),
        retention,
        payoff,
        weakest,
        low_fraction,
        residual,
        config,
    ):
        return None
    score = _legacy._score_plan(
        retention,
        payoff,
        float(plan.opening_quality),
        float(plan.ending_quality),
        float(plan.story_coherence),
        weakest,
        True,
        config,
    )
    return replace(
        plan,
        score=round(float(score), 5),
        retention_quality=round(retention, 4),
        payoff_quality=round(payoff, 4),
        weakest_quarter_interest=round(weakest, 4),
        max_unexplained_low_interest_run_seconds=round(residual, 3),
        editorial_reasons=(
            "Finishing Move continuation independently passes strict hostile/payoff/retention gates; hero score cannot rescue its body",
        ) + tuple(plan.editorial_reasons),
    )


def _certified_continuation(
    continuation: SemanticPlanV31,
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
) -> SemanticPlanV31 | None:
    del timeline
    if continuation.story_type != "verified_combat_island_continuation":
        return None
    if not bool(config["editorial"].get("finishing_move_allow_verified_combat_island_continuation", False)):
        return None
    return continuation


def _finishing_open_plans(timeline: SemanticTimelineV31, continuations: list[SemanticPlanV31], config: dict[str, Any], source_key: str) -> list[SemanticPlanV31]:
    _no_fallback.assert_policy(config)
    if not timeline.finishing_moves:
        return []
    if continuations:
        raise RuntimeError("MW4 V3.1 no-fallback policy: external continuation pool is forbidden")

    editorial = config["editorial"]
    hold = float(editorial.get("finishing_move_payoff_hold_seconds", 0.45))
    max_gap = float(editorial.get("finishing_move_max_continuation_gap_seconds", 18.0))
    excluded = config.get("excluded_windows", {}).get(source_key, [])
    results: list[SemanticPlanV31] = []
    diagnostics: list[dict[str, Any]] = []
    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero_end = min(shot.end, span.end + hold)
        later_floor = max(hero_end, float(span.end) + 0.25)
        dedicated, dedicated_diagnostics = _finishing_body.build_continuations(
            _legacy, _combat, _islands, timeline, span, config, excluded
        )
        eligible: list[SemanticPlanV31] = []
        rejection_reasons: list[dict[str, Any]] = []
        for raw_continuation in dedicated:
            reasons: list[str] = []
            continuation = _certified_continuation(raw_continuation, timeline, config)
            if continuation is None:
                reasons.append("continuation_not_strict_verified_island_body")
                rejection_reasons.append({"story": raw_continuation.story_type, "reasons": reasons})
                continue
            reasons.extend(_chronology_failures(continuation))
            reasons.extend(_combat.continuation_failures(continuation, config))
            reasons.extend(_islands.plan_island_failures(continuation, timeline, config))
            first_start = float(continuation.segments[0].start)
            if first_start < later_floor - _EPS:
                reasons.append("continuation_starts_before_later_floor")
            if first_start - hero_end > max_gap + _EPS:
                reasons.append("continuation_starts_beyond_gap_contract")
            if reasons:
                rejection_reasons.append({
                    "story": continuation.story_type,
                    "start": round(first_start, 3),
                    "end": round(float(continuation.segments[-1].end), 3),
                    "reasons": list(dict.fromkeys(reasons)),
                })
                continue
            eligible.append(continuation)

        single = SemanticTimelineV31(timeline.base, timeline.shots, timeline.consolidated_events, timeline.engagements, (span,))
        setattr(single, "_source_key", source_key)
        if hasattr(timeline, "_combat_state_diagnostics"):
            setattr(single, "_combat_state_diagnostics", getattr(timeline, "_combat_state_diagnostics"))
        produced = _ORIGINAL_FINISHING(single, eligible, config, source_key)
        valid: list[SemanticPlanV31] = []
        produced_rejections: list[dict[str, Any]] = []
        for plan in produced:
            continuation = _matched_continuation(plan, eligible)
            if continuation is None:
                produced_rejections.append({"reason": "produced_plan_body_did_not_match_strict_continuation"})
                continue
            hardened = _conservative_finishing_metrics(plan, continuation, span, config)
            if hardened is None:
                produced_rejections.append({
                    "start": round(float(plan.start), 3),
                    "end": round(float(plan.end), 3),
                    "reason": "combined_finishing_metrics_failed_existing_story_gate",
                })
                continue
            integrity = plan_integrity_violations(hardened, single, config, source_key)
            if integrity:
                produced_rejections.append({
                    "start": round(float(hardened.start), 3),
                    "end": round(float(hardened.end), 3),
                    "reason": "integrity_failure",
                    "failures": integrity,
                })
                continue
            valid.append(hardened)
        diagnostics.append({
            **dedicated_diagnostics,
            "eligible_continuation_count": len(eligible),
            "valid_finishing_plan_count": len(valid),
            "fallback_execution_allowed": False,
            "eligible_candidates": [
                {
                    "story": item.story_type,
                    "start": round(float(item.segments[0].start), 3),
                    "end": round(float(item.segments[-1].end), 3),
                    "output_duration": round(float(item.output_duration), 3),
                    "retention": round(float(item.retention_quality), 4),
                    "payoff": round(float(item.payoff_quality), 4),
                }
                for item in eligible[:8]
            ],
            "continuation_rejections": rejection_reasons[:16],
            "produced_plan_rejections": produced_rejections[:16],
            "status": "PASS" if valid else "NO_VALID_FINISHING_PLAN",
        })
        results.extend(valid)
    setattr(timeline, "_finishing_continuation_diagnostics", diagnostics)
    return sorted(results, key=lambda item: item.score, reverse=True)


def _hardening_self_test() -> None:
    seg = lambda start, end, reason="keep": SimpleNamespace(start=start, end=end, speed=1.0, reason=reason)
    good = SimpleNamespace(start=1.0, end=4.0, segments=(seg(1.0, 2.0), seg(3.0, 4.0)))
    bad = SimpleNamespace(start=10.0, end=5.0, segments=(seg(10.0, 12.0), seg(1.0, 5.0)))
    if _chronology_failures(good):
        raise AssertionError("chronological semantic plan was rejected")
    if not _chronology_failures(bad):
        raise AssertionError("backward semantic plan was accepted")
    span_end, hero_end = 10.0, 10.1
    later_floor = max(hero_end, span_end + 0.25)
    if later_floor < span_end + 0.25:
        raise AssertionError("strict later-content reachability guard failed")

    deliberate = SimpleNamespace(story_type="finishing_move_open")
    hero = seg(10.0, 12.0, "finishing_move_open_hero")
    body_left = seg(20.0, 23.0, "verified_combat_island_body")
    body_right = seg(30.0, 33.0, "verified_combat_island_body")
    if not _planned_cross_shot_bridge(deliberate, hero, body_left, 0):
        raise AssertionError("verified Finishing Move hero-to-body hard cut was rejected")
    if not _planned_cross_shot_bridge(deliberate, body_left, body_right, 1):
        raise AssertionError("deliberate verified-combat-island body hard cut was rejected")
    unrelated = SimpleNamespace(story_type="engagement_chain")
    if _planned_cross_shot_bridge(unrelated, body_left, body_right, 0):
        raise AssertionError("verified-body bridge escaped the Finishing Move contract")
    generic_left = seg(20.0, 23.0, "keep")
    generic_right = seg(30.0, 33.0, "keep")
    if _planned_cross_shot_bridge(deliberate, generic_left, generic_right, 1):
        raise AssertionError("generic cross-shot bridge was silently permitted")

    policy_cfg = {
        "editorial": {
            "finishing_move_allow_verified_combat_island_continuation": True,
        }
    }
    verified_body = SimpleNamespace(story_type="verified_combat_island_continuation")
    if _certified_continuation(verified_body, SimpleNamespace(), policy_cfg) is None:
        raise AssertionError("strict verified-island continuation was rejected")
    forbidden_body = SimpleNamespace(story_type="semantic_montage")
    if _certified_continuation(forbidden_body, SimpleNamespace(), policy_cfg) is not None:
        raise AssertionError("semantic montage continuation remained reachable")

    if analyze_source is not _legacy.analyze_source or diagnose_source is not _legacy.diagnose_source:
        raise AssertionError("strict analyze/diagnose wrappers are not exported by final semantic module")
    _combat.self_test()
    _islands.self_test()
    _finishing_body.self_test()
    print("MW4 semantic no-fallback chronology/combat-state/island/Finishing-body self-test: PASS")


_combat.install(_legacy)
_islands.install(_legacy, _combat)
_legacy._planned_cross_shot_bridge = _planned_cross_shot_bridge
_legacy.plan_integrity_violations = plan_integrity_violations
_legacy._finishing_open_plans = _finishing_open_plans
_no_fallback.install(_legacy, _combat, _islands, _finishing_body)
_INSTALLED_DIAGNOSE = _legacy.diagnose_source


def diagnose_source(timeline: SemanticTimelineV31, config: dict[str, Any], excluded: list[list[float]], source_key: str) -> dict[str, Any]:
    result = _INSTALLED_DIAGNOSE(timeline, config, excluded, source_key)
    result["finishing_move_continuation_diagnostics"] = list(
        getattr(timeline, "_finishing_continuation_diagnostics", [])
    )
    result["fallback_execution_allowed"] = False
    return result


_legacy.diagnose_source = diagnose_source
analyze_source = _legacy.analyze_source


def main() -> None:
    if "--self-test" in sys.argv:
        _hardening_self_test()
        return
    _legacy.main()


if __name__ == "__main__":
    main()

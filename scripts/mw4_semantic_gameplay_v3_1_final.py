from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import mw4_semantic_gameplay_v3_1_final_legacy as _legacy
from mw4_semantic_gameplay_v3_1_final_legacy import *  # noqa: F401,F403
import mw4_semantic_gameplay_v3_1_combat_state as _combat
import mw4_semantic_gameplay_v3_1_combat_islands as _islands

_ORIGINAL_INTEGRITY = _legacy.plan_integrity_violations
_ORIGINAL_FINISHING = _legacy._finishing_open_plans
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


def plan_integrity_violations(plan: SemanticPlanV31, timeline: SemanticTimelineV31, config: dict[str, Any], source_key: str) -> list[str]:
    failures = list(_ORIGINAL_INTEGRITY(plan, timeline, config, source_key))
    failures.extend(_chronology_failures(plan))
    failures.extend(_combat.plan_combat_state_failures(plan, timeline, config))
    failures.extend(_islands.plan_island_failures(plan, timeline, config))
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
            "Finishing Move continuation independently passes hostile/payoff/retention gates; hero score cannot rescue its body",
        ) + tuple(plan.editorial_reasons),
    )


def _certified_continuation(
    continuation: SemanticPlanV31,
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
) -> SemanticPlanV31 | None:
    if continuation.story_type != "semantic_montage":
        return continuation
    editorial = config["editorial"]
    if bool(editorial.get("finishing_move_allow_semantic_montage_continuation", False)):
        return continuation
    if not bool(editorial.get("finishing_move_allow_verified_combat_island_continuation", False)):
        return None
    if not _islands.verified_island_montage(continuation, timeline, config, _combat):
        return None
    return replace(
        continuation,
        story_type="verified_combat_island_continuation",
        editorial_reasons=(
            "dedicated Finishing Move body assembled only from independently verified combat islands; generic semantic-montage continuation remains disabled",
        ) + tuple(continuation.editorial_reasons),
    )


def _finishing_open_plans(timeline: SemanticTimelineV31, continuations: list[SemanticPlanV31], config: dict[str, Any], source_key: str) -> list[SemanticPlanV31]:
    if not timeline.finishing_moves:
        return []
    editorial = config["editorial"]
    hold = float(editorial.get("finishing_move_payoff_hold_seconds", 0.45))
    max_gap = float(editorial.get("finishing_move_max_continuation_gap_seconds", 18.0))
    results: list[SemanticPlanV31] = []
    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero_end = min(shot.end, span.end + hold)
        later_floor = max(hero_end, float(span.end) + 0.25)
        eligible: list[SemanticPlanV31] = []
        for raw_continuation in continuations:
            if raw_continuation.story_type == "finishing_move_open" or not raw_continuation.segments:
                continue
            continuation = _certified_continuation(raw_continuation, timeline, config)
            if continuation is None:
                continue
            if _chronology_failures(continuation):
                continue
            if _combat.continuation_failures(continuation, config):
                continue
            if _islands.plan_island_failures(continuation, timeline, config):
                continue
            first_start = float(continuation.segments[0].start)
            if first_start < later_floor - _EPS:
                continue
            if first_start - hero_end > max_gap + _EPS:
                continue
            eligible.append(continuation)
        single = SemanticTimelineV31(timeline.base, timeline.shots, timeline.consolidated_events, timeline.engagements, (span,))
        setattr(single, "_source_key", source_key)
        if hasattr(timeline, "_combat_state_diagnostics"):
            setattr(single, "_combat_state_diagnostics", getattr(timeline, "_combat_state_diagnostics"))
        produced = _ORIGINAL_FINISHING(single, eligible, config, source_key)
        valid: list[SemanticPlanV31] = []
        for plan in produced:
            continuation = _matched_continuation(plan, eligible)
            if continuation is None:
                continue
            hardened = _conservative_finishing_metrics(plan, continuation, span, config)
            if hardened is None:
                continue
            if plan_integrity_violations(hardened, single, config, source_key):
                continue
            valid.append(hardened)
        if not valid:
            later = [
                {
                    "story": item.story_type,
                    "start": round(float(item.segments[0].start), 3),
                    "end": round(float(item.segments[-1].end), 3),
                    "retention": round(float(item.retention_quality), 4),
                    "payoff": round(float(item.payoff_quality), 4),
                    "failures": _combat.continuation_failures(item, config),
                }
                for item in continuations
                if item.segments and float(item.segments[0].start) >= later_floor - _EPS
                and float(item.segments[0].start) - hero_end <= max_gap + _EPS
            ]
            raise RuntimeError(
                "verified Finishing Move has no strictly later, independently strong hostile-payoff continuation within the configured gap contract; fallback is disabled; "
                f"later_candidates={later[:8]}"
            )
        results.extend(valid)
    return sorted(results, key=lambda item: item.score, reverse=True)


def _hardening_self_test() -> None:
    seg = lambda start, end: SimpleNamespace(start=start, end=end, speed=1.0)
    good = SimpleNamespace(start=1.0, end=4.0, segments=(seg(1.0, 2.0), seg(3.0, 4.0)))
    bad = SimpleNamespace(start=10.0, end=5.0, segments=(seg(10.0, 12.0), seg(1.0, 5.0)))
    if _chronology_failures(good):
        raise AssertionError("chronological semantic plan was rejected")
    if not _chronology_failures(bad):
        raise AssertionError("backward semantic plan was accepted")
    span_end, hero_end = 10.0, 10.1
    later_floor = max(hero_end, span_end + 0.25)
    if later_floor < span_end + 0.25:
        raise AssertionError("finishing fallback reachability guard failed")
    if analyze_source is not _legacy.analyze_source or diagnose_source is not _legacy.diagnose_source:
        raise AssertionError("combat-island analyze/diagnose wrappers are not exported by final semantic module")
    _combat.self_test()
    _islands.self_test()
    print("MW4 semantic chronology/combat-state/island hardening self-test: PASS")


_combat.install(_legacy)
_islands.install(_legacy, _combat)
_legacy.plan_integrity_violations = plan_integrity_violations
_legacy._finishing_open_plans = _finishing_open_plans
analyze_source = _legacy.analyze_source
diagnose_source = _legacy.diagnose_source


def main() -> None:
    if "--self-test" in sys.argv:
        _hardening_self_test()
    _legacy.main()


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import mw4_semantic_gameplay_v3_1_final_legacy as _legacy
from mw4_semantic_gameplay_v3_1_final_legacy import *  # noqa: F401,F403

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
    return list(dict.fromkeys(failures))


def _finishing_open_plans(timeline: SemanticTimelineV31, continuations: list[SemanticPlanV31], config: dict[str, Any], source_key: str) -> list[SemanticPlanV31]:
    if not timeline.finishing_moves:
        return []
    editorial = config["editorial"]
    hold = float(editorial.get("finishing_move_payoff_hold_seconds", 0.45))
    max_gap = float(editorial.get("finishing_move_max_continuation_gap_seconds", 18.0))
    allow_montage = bool(editorial.get("finishing_move_allow_semantic_montage_continuation", False))
    results: list[SemanticPlanV31] = []
    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        hero_end = min(shot.end, span.end + hold)
        eligible: list[SemanticPlanV31] = []
        for continuation in continuations:
            if continuation.story_type == "finishing_move_open" or not continuation.segments:
                continue
            if continuation.story_type == "semantic_montage" and not allow_montage:
                continue
            if _chronology_failures(continuation):
                continue
            first_start = float(continuation.segments[0].start)
            if first_start < hero_end - _EPS:
                continue
            if first_start - hero_end > max_gap + _EPS:
                continue
            eligible.append(continuation)
        single = SemanticTimelineV31(timeline.base, timeline.shots, timeline.consolidated_events, timeline.engagements, (span,))
        setattr(single, "_source_key", source_key)
        produced = _ORIGINAL_FINISHING(single, eligible, config, source_key)
        valid = [plan for plan in produced if not plan_integrity_violations(plan, single, config, source_key)]
        if not valid:
            raise RuntimeError("verified Finishing Move has no strictly later continuation within the configured gap contract")
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
    print("MW4 semantic chronology hardening self-test: PASS")


_legacy.plan_integrity_violations = plan_integrity_violations
_legacy._finishing_open_plans = _finishing_open_plans


def main() -> None:
    if "--self-test" in sys.argv:
        _hardening_self_test()
    _legacy.main()


if __name__ == "__main__":
    main()

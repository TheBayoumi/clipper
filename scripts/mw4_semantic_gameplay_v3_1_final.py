from __future__ import annotations

# Canonical planner entrypoint: verified evidence -> support component -> clip proposal.
import sys
from typing import Any

import mw4_semantic_gameplay_v3_1_final_base as _base
from mw4_semantic_gameplay_v3_1_final_base import *  # noqa: F401,F403
import mw4_semantic_gameplay_v3_1_proposals as _proposals

SemanticPlanV31 = _base.SemanticPlanV31
SemanticTimelineV31 = _base.SemanticTimelineV31


def _normal_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    return _proposals.normal_plans(
        _base, timeline, config, excluded, source_key
    )


def _finishing_open_plans(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    return _proposals.finishing_open_plans(
        _base, timeline, config, excluded, source_key
    )


def build_plans_for_source(
    timeline: SemanticTimelineV31,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    return _proposals.build_plans_for_source(
        _base, timeline, config, excluded, source_key
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


_base._normal_plans = _normal_plans
_base._finishing_open_plans = _finishing_open_plans
_base.build_plans_for_source = build_plans_for_source
_base.build_plans = build_plans

select_plans = _base.select_plans
diagnose_source = _base.diagnose_source
analyze_source = _base.analyze_source
validate_configuration = _base.validate_configuration
plan_integrity_violations = _base.plan_integrity_violations


def _self_test() -> None:
    _base._self_test()
    hero = _base.EditSegment(148.07, 150.50, 1.0, "finishing_move_open_hero")
    remaining = _base._required_body_duration(hero, 10.0)
    if abs(float(remaining) - 7.57) > 1e-3:
        raise AssertionError("proposal planner changed hero-aware duration contract")
    print("MW4 verified-anchor proposal architecture self-test: PASS")


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test()
        return
    _base.main()


if __name__ == "__main__":
    main()

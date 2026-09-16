from __future__ import annotations

# Canonical planner entrypoint: every verified payoff anchor -> verified-source-region quality-gated span -> clip proposal.
# Automatic scene cuts are proposal evidence only; manually verified stringout cuts remain hard integrity walls.
import sys
from typing import Any

import mw4_semantic_gameplay_v3_1_final_base as _base
from mw4_semantic_gameplay_v3_1_final_base import *  # noqa: F401,F403
import mw4_semantic_gameplay_v3_1_payoff_complete as _proposals

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
_base.plan_integrity_violations = _proposals.plan_integrity_violations

select_plans = _base.select_plans
diagnose_source = _base.diagnose_source
analyze_source = _base.analyze_source
validate_configuration = _base.validate_configuration
plan_integrity_violations = _proposals.plan_integrity_violations


def _self_test() -> None:
    _base._self_test()
    _proposals.self_test(_base)
    print("MW4 payoff-complete verified-source-region proposal architecture: PASS")


def __getattr__(name: str) -> Any:
    return getattr(_base, name)


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test()
        return
    _base.main()


if __name__ == "__main__":
    main()

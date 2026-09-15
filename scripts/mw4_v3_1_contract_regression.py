from __future__ import annotations

from typing import Any


def run(contract: Any, legacy: Any) -> None:
    hidden_payoff = {
        "effect_events": [{"kind": "combat_burst", "time": 4.0}],
        "engagements": [{"events": [{"kinds": ["outcome_like"], "time": 5.0}]}],
    }
    if contract._semantic_anchor_times(hidden_payoff) != [5.0]:
        raise AssertionError("payoff anchor hidden behind non-payoff effect event")
    if contract._semantic_anchor_times({"effect_events": [{"kind": "combat_burst", "time": 4.0}]}) != []:
        raise AssertionError("non-payoff fallback anchor path is active")

    forbidden = {
        "story_type": "semantic_montage",
        "effect_profile": "semantic_montage",
        "segments": [{"start": 10.0, "end": 14.0, "reason": "semantic_montage_moment"}],
    }
    forbidden_failures = contract._forbidden_plan_failures("test", 1, forbidden)
    if len(forbidden_failures) != 3:
        raise AssertionError(f"forbidden fallback plan was not rejected comprehensively: {forbidden_failures}")

    bad = {"start": 10.0, "end": 5.0, "segments": [{"start": 10.0, "end": 12.0}, {"start": 1.0, "end": 5.0}]}
    if not contract._ordering_failures("test", 1, bad):
        raise AssertionError("backward source chronology was accepted")

    config = {
        "fallback_policy": {"enabled": False},
        "batch_selection": {"maximum_per_source": 8, "require_verified_finishing_move_when_available": True},
        "duplicate_policy": {
            "semantic_anchor_tolerance_seconds": 0.35,
            "max_shared_context_seconds": 2.5,
            "substantial_overlap_seconds": 6.0,
            "substantial_overlap_fraction_shorter": 0.60,
            "finishing_move_exclusive": True,
        },
        "editorial": {
            "finishing_move_allow_semantic_montage_continuation": False,
        },
        "source_integrity": {"allow_planned_transition_for_semantic_montage": False},
        "finishing_move_detector": {"allow_unverified_automatic": False},
        "combat_state_verifier": {
            "finishing_continuation": {"require_verified_payoff": True},
        },
        "semantic_editor": {
            "semantic_montage": {"enabled": False},
            "selection": {"story_diversity_bonus": 0.06, "finishing_move_bonus": 0.14},
        },
    }
    if contract._policy_failures(config):
        raise AssertionError(f"strict no-fallback test config was rejected: {contract._policy_failures(config)}")

    bad_config = {
        **config,
        "semantic_editor": {
            **config["semantic_editor"],
            "semantic_montage": {"enabled": True},
        },
    }
    if not contract._policy_failures(bad_config):
        raise AssertionError("fallback-enabled allocation config did not fail closed")

    high = {"plan_key": "high", "score": 1.0, "story_type": "impact_payoff", "segments": [{"start": 0.0, "end": 10.0}]}
    low_a = {"plan_key": "low-a", "score": 0.6, "story_type": "impact_payoff", "segments": [{"start": 0.0, "end": 4.0}]}
    low_b = {"plan_key": "low-b", "score": 0.6, "story_type": "impact_payoff", "segments": [{"start": 6.0, "end": 10.0}]}
    selected = legacy._adaptive_source_selection("test", [low_a, low_b, high], 0, config)
    if [plan["plan_key"] for plan in selected] != ["high"]:
        raise AssertionError("allocator is still cardinality-first instead of quality-first")
    print("MW4 zero-fallback semantic allocation hardening self-test: PASS")

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

    left = {"segments": [{"start": 10.0, "end": 14.0, "reason": "semantic_montage_moment"}]}
    right = {"segments": [{"start": 10.2, "end": 13.8, "reason": "semantic_montage_moment"}]}
    config = {
        "batch_selection": {"maximum_per_source": 8, "require_verified_finishing_move_when_available": True},
        "duplicate_policy": {
            "semantic_anchor_tolerance_seconds": 0.35,
            "max_shared_context_seconds": 2.5,
            "substantial_overlap_seconds": 6.0,
            "substantial_overlap_fraction_shorter": 0.60,
            "finishing_move_exclusive": True,
        },
        "semantic_editor": {"selection": {"story_diversity_bonus": 0.06, "finishing_move_bonus": 0.14}},
    }
    if not contract._plans_conflict(left, right, config):
        raise AssertionError("reused semantic montage moment was accepted")

    bad = {"start": 10.0, "end": 5.0, "segments": [{"start": 10.0, "end": 12.0}, {"start": 1.0, "end": 5.0}]}
    if not contract._ordering_failures("test", 1, bad):
        raise AssertionError("backward source chronology was accepted")

    high = {"plan_key": "high", "score": 1.0, "story_type": "impact_payoff", "segments": [{"start": 0.0, "end": 10.0}]}
    low_a = {"plan_key": "low-a", "score": 0.6, "story_type": "impact_payoff", "segments": [{"start": 0.0, "end": 4.0}]}
    low_b = {"plan_key": "low-b", "score": 0.6, "story_type": "impact_payoff", "segments": [{"start": 6.0, "end": 10.0}]}
    selected = legacy._adaptive_source_selection("test", [low_a, low_b, high], 0, config)
    if [plan["plan_key"] for plan in selected] != ["high"]:
        raise AssertionError("allocator is still cardinality-first instead of quality-first")
    print("MW4 semantic allocation hardening self-test: PASS")

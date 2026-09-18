import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from clipper.cli import main
from clipper_engine.gameplay import allocation, contract
from clipper_engine.profiles import load_profile


def _config() -> dict:
    return {
        "batch_selection": {"require_verified_finishing_move_when_available": False},
        "editorial": {
            "finishing_move_max_continuation_gap_seconds": 18.0,
            "finishing_move_body_hard_cut_max_source_gap_seconds": 18.0,
        },
        "finishing_move_detector": {},
        "source_integrity": {},
        "semantic_editor": {
            "minimum_output_seconds": 10.0,
            "maximum_output_seconds": 12.0,
            "preferred_output_seconds": 11.0,
            "selection": {"finishing_move_bonus": 0.14},
        },
    }


def _candidate(key: str, start: float, end: float, anchors: list[float], score: float) -> dict:
    return {
        "plan_key": key,
        "score": score,
        "start": start,
        "end": end,
        "raw_duration": end - start,
        "output_duration": end - start,
        "opening_quality": 0.5,
        "ending_quality": 0.5,
        "retention_quality": 0.5,
        "payoff_quality": 0.5,
        "story_coherence": 0.5,
        "weakest_quarter_interest": 0.3,
        "low_interest_fraction": 0.2,
        "max_unexplained_low_interest_run_seconds": 0.4,
        "story_type": "engagement_chain",
        "effect_profile": "clean_pressure",
        "segments": [{"start": start, "end": end, "speed": 1.0, "reason": "story"}],
        "effect_events": [
            {"kind": "outcome_like", "time": anchor, "confidence": 0.9, "evidence": {}}
            for anchor in anchors
        ],
        "engagements": [],
        "finishing_move": None,
        "editorial_reasons": [],
        "covered_outcome_anchor_times": anchors,
        "covered_payoff_anchor_times": anchors,
    }


def test_mw4_command_routes_to_clipper_engine() -> None:
    with patch("clipper.cli.run_mw4", return_value=0) as run_mw4:
        assert main(["mw4", "self-test"]) == 0
    args = run_mw4.call_args.args[0]
    assert args.command == "mw4"
    assert args.mw4_command == "self-test"


def test_mw4_profile_is_declarative() -> None:
    profile = load_profile("mw4")
    assert profile.name == "mw4"
    assert profile.expected_source_count == 4
    assert profile.capability("direct_interaction_detection")["enabled"] is True
    assert profile.capability("player_death_detection")["enabled"] is True
    assert all(not callable(value) for value in profile.config.values())


def test_overlap_conflict_uses_source_spans_not_scene_ids() -> None:
    config = {
        "duplicate_policy": {
            "semantic_anchor_tolerance_seconds": 0.35,
            "max_shared_context_seconds": 2.5,
            "substantial_overlap_seconds": 6.0,
            "substantial_overlap_fraction_shorter": 0.60,
            "finishing_move_exclusive": True,
        }
    }
    left = _candidate("a", 10.0, 22.0, [15.0], 0.8)
    right = _candidate("b", 10.05, 22.05, [21.0], 0.9)
    left["combat_scene_id"] = "old-scene-a"
    right["combat_scene_id"] = "old-scene-b"
    assert contract.plans_conflict(left, right, config)


def test_exact_allocator_prefers_one_candidate_covering_multiple_outcomes() -> None:
    config = _config()
    config["duplicate_policy"] = {
        "semantic_anchor_tolerance_seconds": 0.35,
        "max_shared_context_seconds": 2.5,
        "substantial_overlap_seconds": 6.0,
        "substantial_overlap_fraction_shorter": 0.60,
        "finishing_move_exclusive": True,
    }
    candidates = [
        _candidate("multi", 10.0, 22.0, [15.0, 20.0], 0.85),
        _candidate("first", 8.0, 18.0, [15.0], 0.95),
        _candidate("second", 15.0, 25.0, [20.0], 0.95),
    ]
    selected = allocation._solve_source(
        "test", candidates, {15.0, 20.0}, False, config
    )
    assert [item["plan_key"] for item in selected] == ["multi"]


def test_allocator_fails_when_conflicts_make_qualified_coverage_impossible() -> None:
    config = _config()
    config["duplicate_policy"] = {
        "semantic_anchor_tolerance_seconds": 0.35,
        "max_shared_context_seconds": 0.0,
        "substantial_overlap_seconds": 1.0,
        "substantial_overlap_fraction_shorter": 0.10,
        "finishing_move_exclusive": True,
    }
    candidates = [
        _candidate("first", 10.0, 20.0, [12.0], 0.8),
        _candidate("second", 10.5, 20.5, [18.0], 0.8),
    ]
    with pytest.raises(AssertionError, match="no conflict-free allocation"):
        allocation._solve_source(
            "test", candidates, {12.0, 18.0}, False, config
        )


def test_profile_override_must_still_declare_mw4(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"profile":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected 'mw4'"):
        load_profile("mw4", path)

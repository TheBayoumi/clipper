"""Fail-closed coverage, independent payoff search and source qualification tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from clipper_engine import build_identity
from clipper_engine.gameplay import allocation, contract
from clipper_engine.profiles import load_profile


def _config() -> dict[str, Any]:
    return load_profile("mw4").config


def _plan(
    *,
    anchors: list[float] | None = None,
    semantic: list[str] | None = None,
    start: float = 10.0,
    end: float = 21.0,
    score: float = 0.9,
) -> dict[str, Any]:
    anchors = anchors or []
    semantic = semantic or []
    candidate = {
        "score": score,
        "start": start,
        "end": end,
        "raw_duration": end - start,
        "output_duration": end - start,
        "opening_quality": 0.8,
        "ending_quality": 0.8,
        "retention_quality": 0.8,
        "payoff_quality": 0.8,
        "story_coherence": 0.8,
        "weakest_quarter_interest": 0.7,
        "low_interest_fraction": 0.0,
        "max_unexplained_low_interest_run_seconds": 0.0,
        "story_type": "cinematic_moment",
        "effect_profile": "source_native_full_frame",
        "segments": [{"start": start, "end": end, "speed": 1.0, "reason": "source_native"}],
        "effect_events": [{"kind": "outcome_like", "time": x} for x in anchors],
        "engagements": [],
        "finishing_move": None,
        "editorial_reasons": [],
        "covered_outcome_anchor_times": anchors,
        "covered_semantic_anchor_ids": semantic,
    }
    candidate["plan_key"] = contract.plan_key(candidate)
    return candidate


def _manifest(
    plans: list[dict[str, Any]],
    *,
    assignments: list[dict[str, Any]] | None = None,
    semantic: list[dict[str, Any]] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "source_key": "test_source",
        "execution_sha": build_identity.current_execution_sha(),
        "semantic_engine": contract.EXPECTED_ENGINE,
        "editorial_planner": contract.EXPECTED_EDITOR,
        "candidate_mode": contract.EXPECTED_MODE,
        "candidate_pool": plans,
        "required_semantic_anchors": semantic or [],
        "verified_finishing_move_count": 0,
        "failure": None,
        "diagnostics": {
            "local_interaction_verifier": {"verified_hostile_event_assignment": assignments or []},
            "proposal_diagnostics": {"payoff_anchor_diagnostics": diagnostics or []},
        },
    }


def _allocate(
    tmp_path: Path, manifest: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    (tmp_path / "test_source_analysis.json").write_text(json.dumps(manifest), encoding="utf-8")
    return allocation.allocate(tmp_path, config or _config())


def _outcome_assignment(time: float) -> dict[str, Any]:
    return {"time": time, "hostile_verified": True, "kinds": ["outcome_like"]}


def test_mixed_cinematic_and_verified_outcome_has_both_disposition_types(tmp_path: Path) -> None:
    cinematic = "cinematic:test_source:faceoff"
    p = _plan(anchors=[15.0], semantic=[cinematic])
    result = _allocate(
        tmp_path,
        _manifest(
            [p],
            assignments=[_outcome_assignment(15.0)],
            semantic=[
                {"id": cinematic, "disposition": "search_required", "legal_candidate_count": 1}
            ],
            diagnostics=[{"anchor_time": 15.0, "qualified_variant_count": 1}],
        ),
    )
    src = result["source_allocations"]["test_source"]
    assert src["required_anchor_count"] == 2
    assert src["qualified_anchor_count"] == 2
    assert src["outcome_dispositions"] == [
        {
            "anchor": 15.0,
            "disposition": "covered_by_selected_clip",
            "selected_plan_keys": [p["plan_key"]],
        }
    ]
    assert src["semantic_anchor_dispositions"][0]["anchor"] == cinematic
    assert src["selected_pair_conflict_count"] == 0


def test_false_effect_event_cannot_create_verified_coverage(tmp_path: Path) -> None:
    p = _plan(anchors=[])
    p["effect_events"] = [{"kind": "outcome_like", "time": 15.0, "confidence": 0.99}]
    p["plan_key"] = contract.plan_key(p)
    assert allocation._candidate_coverage(p) == frozenset()
    with pytest.raises(AssertionError, match="legal independent proposals"):
        _allocate(
            tmp_path,
            _manifest(
                [p],
                assignments=[_outcome_assignment(15.0)],
                diagnostics=[{"anchor_time": 15.0, "qualified_variant_count": 1}],
            ),
        )


def test_declared_coverage_cannot_invent_hostile_outcome() -> None:
    p = _plan(anchors=[15.0])
    assert allocation._validate_declared_coverage(p, set()) == [
        "unverified_or_foreign_anchor_coverage"
    ]


def test_declared_outcome_must_be_in_actual_source_segments() -> None:
    p = _plan(anchors=[50.0])
    assert allocation._validate_declared_coverage(p, {50.0}) == [
        "outcome_anchor_outside_actual_source_segments"
    ]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, "15.0"])
def test_invalid_outcome_times_are_structurally_rejected(bad: Any) -> None:
    p = _plan(anchors=[15.0])
    p["covered_outcome_anchor_times"] = [bad]
    assert (
        "explicit_anchor_coverage_invalid" in allocation._validate_declared_coverage(p, {15.0})[0]
    )


def test_missing_coverage_cannot_be_reconstructed_from_effect_events() -> None:
    p = _plan(anchors=[15.0])
    del p["covered_outcome_anchor_times"]
    with pytest.raises(AssertionError, match="missing explicit"):
        allocation._candidate_coverage(p)


def test_spare_rejected_candidate_has_machine_readable_reason(tmp_path: Path) -> None:
    p = _plan(anchors=[15.0])
    tampered = copy.deepcopy(p)
    tampered["plan_key"] = "forged"
    result = _allocate(tmp_path, _manifest([tampered, p], assignments=[_outcome_assignment(15.0)]))
    src = result["source_allocations"]["test_source"]
    assert src["rejected_by_contract"] == 1
    assert src["rejected_candidates"][0]["reasons"] == ["candidate_plan_key_identity_mismatch"]


def test_uncovered_outcome_needs_independent_search_disposition(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="lack proposal disposition"):
        _allocate(tmp_path, _manifest([], assignments=[_outcome_assignment(15.0)]))


def test_structured_no_admissible_disposition_is_preserved(tmp_path: Path) -> None:
    result = _allocate(
        tmp_path,
        _manifest(
            [],
            assignments=[_outcome_assignment(15.0)],
            diagnostics=[
                {
                    "anchor_time": 15.0,
                    "attempted_variant_count": 0,
                    "qualified_variant_count": 0,
                    "rejections": {"inside_verified_cut_or_no_region": 1},
                }
            ],
        ),
    )
    src = result["source_allocations"]["test_source"]
    assert src["count"] == 0
    assert src["outcome_dispositions"][0]["disposition"] == "no_admissible_candidate"
    assert src["outcome_dispositions"][0]["rejections"] == {"inside_verified_cut_or_no_region": 1}


def test_incomplete_six_source_manifest_set_is_a_contract_failure(tmp_path: Path) -> None:
    config = _config()
    config["source_profile"] = {
        "assets": [{"source_key": "test_source"}, {"source_key": "missing_source"}]
    }
    with pytest.raises(AssertionError, match="incomplete configured-source qualification"):
        _allocate(tmp_path, _manifest([]), config)


def test_cinematic_anchor_with_legal_candidate_cannot_be_rejected(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="incorrectly marked no_admissible_candidate"):
        _allocate(
            tmp_path,
            _manifest(
                [],
                semantic=[
                    {
                        "id": "cinematic:test_source:faceoff",
                        "disposition": "no_admissible_candidate",
                        "legal_candidate_count": 2,
                    }
                ],
            ),
        )

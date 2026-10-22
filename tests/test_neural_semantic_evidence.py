"""Visual-language claims cannot bypass Clipper's canonical semantic verification."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from clipper_engine.gameplay import neural_evidence


def _observation(timestamp: float, labels: list[str]) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "event_labels": labels,
        "summary": "The vision model claims a possible outcome",
        "confidence": 0.99,
        "verification": "PROPOSED",
    }


def _report() -> dict[str, Any]:
    return {
        "status": "PROPOSALS_ONLY",
        "source_identity": {
            "source_key": "ghost_fight",
            "analysis_sha256": "a" * 64,
        },
        "model": {"model_id": "Qwen/Qwen3-VL-8B-Instruct"},
        "coverage": {"expected_samples": 3, "observed_samples": 3},
        "observations": [
            _observation(4.0, ["confirmed_kill"]),
            _observation(8.0, ["character_confrontation"]),
            _observation(12.0, []),
        ],
    }


def _timeline() -> Any:
    return SimpleNamespace(
        duration=20.0,
        consolidated_events=(SimpleNamespace(time=4.1, kinds=("outcome_like",)),),
    )


def test_neural_labels_remain_proposals_even_near_direct_verified_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        neural_evidence.interaction,
        "hostile_decision",
        lambda event, config: SimpleNamespace(hostile=True, reason="directly_confirmed_outcome"),
    )
    result = neural_evidence.reconcile(_report(), _timeline(), "ghost_fight", {})
    first, second, third = result["proposals"]
    assert first["verification"] == "PROPOSED"
    assert first["disposition"] == "CORROBORATES_EXISTING_ANCHOR"
    assert first["may_create_required_anchor"] is False
    assert first["existing_verified_evidence"][0]["origin"] == "clipper_direct_interaction_verifier"
    assert second["disposition"] == "REQUIRES_INDEPENDENT_SOURCE_REVIEW"
    assert second["existing_verified_evidence"] == []
    assert third["disposition"] == "NO_CLAIM"
    assert result["coverage"] == {
        "observations": 3,
        "review_required": 1,
        "corroborated_existing": 1,
    }


def test_manual_cinematic_corrobation_still_cannot_add_new_verified_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        neural_evidence.interaction,
        "hostile_decision",
        lambda event, config: SimpleNamespace(hostile=False, reason="no_direct_evidence"),
    )
    config = {
        "cinematic_moments": {
            "verified_anchors": {
                "ghost_fight": [
                    {"id": "verified_faceoff", "time": 8.1, "verification": "source_playback"}
                ]
            }
        }
    }
    output = neural_evidence.reconcile(_report(), _timeline(), "ghost_fight", config)
    match = output["proposals"][1]
    assert match["disposition"] == "CORROBORATES_EXISTING_ANCHOR"
    assert match["existing_verified_evidence"][0]["verified_anchor_id"] == (
        "cinematic:ghost_fight:verified_faceoff"
    )
    assert match["may_create_required_anchor"] is False


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda v: v.update(status="VERIFIED"), "proposals-only"),
        (lambda v: v["source_identity"].update(source_key="different"), "source identity"),
        (lambda v: v["coverage"].update(observed_samples=0), "complete source-wide"),
        (lambda v: v["observations"][0].update(verification="VERIFIED"), "quarantine"),
        (lambda v: v["observations"][1].update(timestamp=4.0), "duplicate source clock"),
        (lambda v: v["observations"][0].update(timestamp=40.0), "source clock"),
        (lambda v: v["observations"][0].update(timestamp=float("nan")), "source clock"),
        (lambda v: v["observations"][0].update(event_labels=[900]), "labels"),
    ],
)
def test_reconciliation_refuses_model_certification_and_corrupt_coverage(
    monkeypatch: pytest.MonkeyPatch, mutation: Any, reason: str
) -> None:
    import copy

    report = copy.deepcopy(_report())
    mutation(report)
    monkeypatch.setattr(
        neural_evidence.interaction,
        "hostile_decision",
        lambda event, config: SimpleNamespace(hostile=False, reason="no_direct_evidence"),
    )
    with pytest.raises(RuntimeError, match=reason):
        neural_evidence.reconcile(report, _timeline(), "ghost_fight", {})

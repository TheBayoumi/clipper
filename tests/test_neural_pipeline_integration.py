"""Neural proposals use normal Clipper analysis, never an alternate planner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clipper_engine.gameplay import candidates
from clipper_engine.inference import vision


def test_neural_visual_goes_through_canonical_clip_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "ghost_fight.mp4"
    source.write_bytes(b"proxy")
    output = tmp_path / "analysis"
    seen: list[str] = []
    expected_timeline = SimpleNamespace(
        duration=20.0,
        consolidated_events=(),
        finishing_moves=(),
    )

    def scout(
        _source: Path,
        manifest: Path,
        _config: dict[str, Any],
        _output: Path,
        *,
        source_key: str,
        timeline: Any,
    ) -> dict[str, Any]:
        assert manifest == tmp_path / "ghost_fight.source_qa.json"
        assert source_key == "ghost_fight"
        assert timeline is expected_timeline
        seen.append("scout")
        return {
            "status": "PROPOSALS_ONLY",
            "source_identity": {"source_key": "ghost_fight", "analysis_sha256": "a" * 64},
            "model": {"model_id": vision.VISION_MODEL_ID},
            "coverage": {"expected_samples": 1, "observed_samples": 1},
            "observations": [
                {
                    "timestamp": 3.0,
                    "event_labels": ["possible_confrontation"],
                    "confidence": 0.9,
                    "verification": "PROPOSED",
                    "summary": "visible action",
                }
            ],
        }

    monkeypatch.setattr(vision, "analyze_source", scout)

    def canonical_analysis(*_args: object) -> Any:
        seen.append("analysis")
        return expected_timeline

    monkeypatch.setattr(candidates.analysis, "analyze_source", canonical_analysis)
    monkeypatch.setattr(candidates.analysis, "diagnose_source", lambda *_: {"canonical": True})
    monkeypatch.setattr(candidates.analysis, "build_plans_for_source", lambda *_: [])
    monkeypatch.setattr(candidates.cinematic, "analyze", lambda *_: ([], []))
    monkeypatch.setattr(candidates, "_automatic_finishing_candidates", lambda *_: [])
    payload = candidates.analyze_source_file("ghost_fight", source, {}, output)
    assert seen == ["analysis", "scout"]
    assert payload["semantic_engine"] == candidates.ENGINE
    assert payload["editorial_planner"] == candidates.EDITOR
    assert payload["candidate_count_after_semantic_gates"] == 0
    assert payload["neural_visual_evidence"]["status"] == "PROPOSALS_ONLY"
    assert payload["neural_review_queue"]["status"] == "REVIEW_QUEUE_ONLY"
    queue = json.loads((output / "neural_review_queue.json").read_text())
    assert queue["proposals"][0]["may_create_required_anchor"] is False
    assert queue["proposals"][0]["disposition"] == "REQUIRES_INDEPENDENT_SOURCE_REVIEW"
    assert payload["required_semantic_anchors"] == []
    assert payload["verified_finishing_move_count"] == 0


def test_disabled_neural_path_keeps_canonical_analysis_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    timeline = SimpleNamespace(duration=20.0, consolidated_events=(), finishing_moves=())
    monkeypatch.setattr(vision, "analyze_source", lambda *_args, **_kw: None)
    monkeypatch.setattr(candidates.analysis, "analyze_source", lambda *_: timeline)
    monkeypatch.setattr(candidates.analysis, "diagnose_source", lambda *_: {})
    monkeypatch.setattr(candidates.analysis, "build_plans_for_source", lambda *_: [])
    monkeypatch.setattr(candidates.cinematic, "analyze", lambda *_: ([], []))
    monkeypatch.setattr(candidates, "_automatic_finishing_candidates", lambda *_: [])
    output = tmp_path / "output"
    result = candidates.analyze_source_file("source", source, {}, output)
    assert result["neural_visual_evidence"] is None
    assert result["neural_review_queue"] is None
    assert not (output / "neural_review_queue.json").exists()

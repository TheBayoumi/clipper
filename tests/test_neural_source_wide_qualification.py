"""Validate that Modal source-wide GPU reuses the ONE Clipper analysis engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clipper_engine.inference import modal_worker, vision


@pytest.fixture(autouse=True)
def _exact_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "a" * 40
    monkeypatch.setattr(modal_worker, "DEPLOYED_GIT_SHA", sha)
    monkeypatch.setenv("GITHUB_SHA", sha)
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)


def _qualified_stub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    called: list[str] = []

    def certify(_source: Path, _qa: Path, *, require_original: bool) -> dict[str, Any]:
        assert require_original is False
        called.append("certify")
        return {
            "source_key": "ghost_fight",
            "derivative_type": "proxy",
            "sha256": "f" * 64,
        }

    def ready_spawn() -> SimpleNamespace:
        called.append("gpu_ready")
        return SimpleNamespace(
            get=lambda timeout: {
                "deployed_git_sha": "a" * 40,
                "model_id": vision.VISION_MODEL_ID,
                "revision": vision.VISION_REVISION,
                "cuda_available": True,
                "gpu_count": 2,
            }
        )

    class FakeWorker:
        def __init__(self) -> None:
            self.ready = SimpleNamespace(spawn=ready_spawn)
            self.inspect = SimpleNamespace(spawn=lambda *_: None)

    timeline = SimpleNamespace(duration=12.0, consolidated_events=())

    def analyze(_source: Path, _config: dict[str, Any]) -> Any:
        called.append("canonical_cpu_analysis")
        return timeline

    def neural(
        _source: Path,
        _qa: Path,
        config: dict[str, Any],
        _out: Path,
        *,
        provider: Any,
        source_key: str,
        timeline: Any,
    ) -> dict[str, Any]:
        called.append("gpu_vision")
        assert source_key == "ghost_fight"
        assert timeline is not None
        assert provider.worker is not None
        assert config["capabilities"]["neural_visual_events"]["enabled"] is True
        assert config["capabilities"]["neural_visual_events"]["max_sample_frames"] == 300
        assert config["capabilities"]["neural_visual_events"]["max_refinement_anchors"] == 30
        # Regression: six-source qualification cannot silently reset the normal
        # MW4 480s deadline to the previously hardcoded 180s.
        assert config["capabilities"]["neural_visual_events"]["deadline_seconds"] == 480.0
        return {
            "status": "PROPOSALS_ONLY",
            "model": {"model_id": vision.VISION_MODEL_ID},
            "source_clock": {"nominal_rate": "60000/1001"},
            "coverage": {"expected_samples": 2, "observed_samples": 2},
            "usage": {"gpu_seconds": 16.0, "estimated_cost_usd": 0.01},
            "observations": [
                {"timestamp": 0.0, "event_labels": [], "verification": "PROPOSED"},
                {"timestamp": 4.0, "event_labels": ["visible_combat"], "verification": "PROPOSED"},
            ],
            "source_identity": {"source_key": "ghost_fight", "sha256": "f" * 64},
        }

    monkeypatch.setattr(
        modal_worker.source_qa
        if hasattr(modal_worker, "source_qa")
        else __import__("clipper_engine.sources.qa", fromlist=["verify"]),
        "verify",
        certify,
    )
    monkeypatch.setattr(modal_worker, "VisionModel", FakeWorker)
    from clipper_engine.gameplay import analysis

    monkeypatch.setattr(analysis, "analyze_source", analyze)
    monkeypatch.setattr(vision, "analyze_source", neural)
    return called


def test_ephemeral_source_wide_entrypoint_reuses_canonical_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = _qualified_stub(monkeypatch)
    video = tmp_path / "ghost_fight.mp4"
    video.write_bytes(b"certified proxy stub")
    qa = tmp_path / "ghost_fight.source_qa.json"
    qa.write_text("{}", encoding="utf-8")
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[1]
    out = tmp_path / "qualified"
    modal_worker.source_scout(
        str(video), str(qa), str(repo / "campaigns/campaign-moments-mw4.json"), str(out)
    )
    report = json.loads((out / "source_wide_gpu_qualification.json").read_text())
    queue = json.loads((out / "neural_review_queue.json").read_text())
    assert called == ["certify", "canonical_cpu_analysis", "gpu_ready", "gpu_vision"]
    assert report["status"] == "GPU_SOURCE_WIDE_PROPOSALS_ONLY"
    assert report["execution_sha"] == "a" * 40
    assert report["coverage"] == {"expected_samples": 2, "observed_samples": 2}
    assert queue["status"] == "REVIEW_QUEUE_ONLY"
    assert report["may_create_verified_outcomes"] is False


def test_source_wide_refuses_out_of_budget_without_gpu_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = _qualified_stub(monkeypatch)
    with pytest.raises(ValueError, match="budget must be finite and bounded"):
        modal_worker.source_scout(
            str(tmp_path / "x.mp4"),
            str(tmp_path / "qa.json"),
            "campaigns/campaign-moments-mw4.json",
            str(tmp_path),
            max_estimated_usd=40.0,
        )
    assert called == []


def test_source_wide_refuses_stale_deployed_identity_before_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = _qualified_stub(monkeypatch)
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)
    with pytest.raises(RuntimeError, match="worker SHA differs"):
        modal_worker.source_scout(
            str(tmp_path / "x.mp4"),
            str(tmp_path / "qa.json"),
            "campaigns/campaign-moments-mw4.json",
            str(tmp_path),
        )
    assert called == []


def test_source_wide_readiness_failure_retains_source_bound_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = _qualified_stub(monkeypatch)
    from clipper_engine.inference import modal

    def readiness_failure(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Modal source_wide readiness exceeded 900s")

    monkeypatch.setattr(modal, "bounded_modal_readiness", readiness_failure)
    source = tmp_path / "ghost_fight.mp4"
    source.write_bytes(b"certified proxy stub")
    qa = tmp_path / "ghost_fight.source_qa.json"
    qa.write_text("{}", encoding="utf-8")
    output = tmp_path / "qualification"
    with pytest.raises(RuntimeError, match="readiness exceeded"):
        modal_worker.source_scout(
            str(source),
            str(qa),
            "campaigns/campaign-moments-mw4.json",
            str(output),
        )
    failure = json.loads((output / "neural_readiness_failure.json").read_text())
    assert failure["status"] == "GPU_SOURCE_WIDE_READINESS_REJECTED"
    assert failure["stage"] == "gpu_model_readiness"
    assert failure["source_key"] == "ghost_fight"
    assert failure["deployed_git_sha"] == "a" * 40
    assert failure["may_create_verified_outcomes"] is False
    assert called == ["certify", "canonical_cpu_analysis"]
    assert not (output / "source_wide_gpu_qualification.json").exists()

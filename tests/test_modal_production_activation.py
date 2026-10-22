"""Production MW4 enables GPU inference without making the model a verifier."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clipper_engine.inference import modal, vision
from clipper_engine.profiles import load_profile

SHA = "a" * 40


@pytest.fixture(autouse=True)
def _controlled_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setenv("CLIPPER_DEPLOYED_GIT_SHA", SHA)
    monkeypatch.setenv("CLIPPER_NEURAL_MODAL_APP", "clipper-mw4-vision-run")
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)


def test_packaged_mw4_requires_exact_sha_pinned_gpu_by_default() -> None:
    cfg = load_profile("mw4").config
    settings = vision.VisionSettings.from_profile(cfg)
    assert settings.enabled is True
    assert settings.app_name == "clipper-mw4-vision-run"
    assert settings.deployed_sha == SHA
    assert settings.max_sample_frames <= 300
    assert settings.max_gpu_seconds <= 1800
    assert settings.max_estimated_usd <= 2


def test_packaged_mw4_fails_closed_without_running_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("CLIPPER_ACCEPTANCE_SHA", raising=False)
    with pytest.raises(ValueError, match="explicit deployed 40-character"):
        vision.VisionSettings.from_profile(load_profile("mw4").config)


def test_ephemeral_gpu_scope_reuses_one_verified_worker_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = vision.VisionSettings.from_profile(load_profile("mw4").config)
    history: list[str] = []

    class AppScope:
        def __enter__(self) -> None:
            history.append("start_one_ephemeral_app")

        def __exit__(self, *_: Any) -> None:
            history.append("close_ephemeral_app")

    class Worker:
        def __init__(self) -> None:
            history.append("initialize_worker")
            self.ready = SimpleNamespace(
                remote=lambda: {
                    "deployed_git_sha": SHA,
                    "model_id": vision.VISION_MODEL_ID,
                    "cuda_available": True,
                    "gpu_count": 2,
                }
            )

    fake_module = SimpleNamespace(
        DEPLOYED_GIT_SHA=SHA,
        APP_NAME=settings.app_name,
        app=SimpleNamespace(run=lambda: AppScope()),
        VisionModel=Worker,
    )
    monkeypatch.setitem(sys.modules, "clipper_engine.inference.modal_worker", fake_module)
    with modal.ModalVisionProvider(settings) as provider:
        assert provider.worker is not None
        assert history == ["start_one_ephemeral_app", "initialize_worker"]
    assert provider.worker is None
    assert history[-1] == "close_ephemeral_app"


def test_mismatched_worker_identity_is_rejected_before_modal_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = vision.VisionSettings.from_profile(load_profile("mw4").config)
    monkeypatch.setitem(
        sys.modules,
        "clipper_engine.inference.modal_worker",
        SimpleNamespace(
            DEPLOYED_GIT_SHA="b" * 40,
            APP_NAME=settings.app_name,
            app=SimpleNamespace(run=lambda: pytest.fail("stale Modal app must not start")),
        ),
    )
    with (
        pytest.raises(RuntimeError, match="SHA or app identity mismatch"),
        modal.ModalVisionProvider(settings),
    ):
        pytest.fail("stale worker reached active inference")


def test_invalid_cuda_readiness_never_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = vision.VisionSettings.from_profile(load_profile("mw4").config)
    closed: list[bool] = []

    class AppScope:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *_: Any) -> None:
            closed.append(True)

    worker = SimpleNamespace(
        ready=SimpleNamespace(
            remote=lambda: {
                "deployed_git_sha": SHA,
                "model_id": vision.VISION_MODEL_ID,
                "cuda_available": False,
                "gpu_count": 0,
            }
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "clipper_engine.inference.modal_worker",
        SimpleNamespace(
            DEPLOYED_GIT_SHA=SHA,
            APP_NAME=settings.app_name,
            app=SimpleNamespace(run=lambda: AppScope()),
            VisionModel=lambda: worker,
        ),
    )
    with (
        pytest.raises(RuntimeError, match="model/CUDA/SHA readiness failed"),
        modal.ModalVisionProvider(settings),
    ):
        pytest.fail("missing GPU was silently accepted")
    assert closed == [True]


def test_public_mw4_analyze_invokes_gpu_and_preserves_proposals_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from clipper_engine.gameplay import candidates

    config = load_profile("mw4").config
    source = tmp_path / "ghost_fight.mp4"
    source.write_bytes(b"certified synthetic source stub")
    out = tmp_path / "analysis"
    steps: list[str] = []
    timeline = SimpleNamespace(duration=20.0, consolidated_events=(), finishing_moves=())

    class Provider:
        def __init__(self, settings: vision.VisionSettings) -> None:
            assert settings.enabled is True
            assert settings.deployed_sha == SHA
            assert settings.app_name == "clipper-mw4-vision-run"

        def __enter__(self) -> Provider:
            steps.append("gpu_session_started")
            return self

        def __exit__(self, *_: Any) -> None:
            steps.append("gpu_session_closed")

    def scout(
        _source: Path,
        _manifest: Path,
        _cfg: dict[str, Any],
        _out: Path,
        *,
        provider: Any,
        source_key: str,
        timeline: Any,
    ) -> dict[str, Any]:
        assert isinstance(provider, Provider)
        assert source_key == "ghost_fight"
        steps.append("gpu_source_inference")
        return {
            "status": "PROPOSALS_ONLY",
            "source_identity": {"source_key": "ghost_fight", "analysis_sha256": "b" * 64},
            "model": {"model_id": vision.VISION_MODEL_ID},
            "coverage": {"expected_samples": 1, "observed_samples": 1},
            "usage": {"gpu_seconds": 3.5},
            "observations": [
                {
                    "timestamp": 3.0,
                    "event_labels": ["possible_confrontation"],
                    "confidence": 0.83,
                    "verification": "PROPOSED",
                    "summary": "visible action only",
                }
            ],
        }

    monkeypatch.setattr(modal, "ModalVisionProvider", Provider)
    monkeypatch.setattr(vision, "analyze_source", scout)
    monkeypatch.setattr(candidates.analysis, "analyze_source", lambda *_: timeline)
    monkeypatch.setattr(candidates.analysis, "diagnose_source", lambda *_: {})
    monkeypatch.setattr(candidates.analysis, "build_plans_for_source", lambda *_: [])
    monkeypatch.setattr(candidates.cinematic, "analyze", lambda *_: ([], []))
    monkeypatch.setattr(candidates, "_automatic_finishing_candidates", lambda *_: [])
    result = candidates.analyze_source_file("ghost_fight", source, config, out)
    assert steps == ["gpu_session_started", "gpu_source_inference", "gpu_session_closed"]
    assert result["analysis_compute"]["neural_visual"] == (
        "modal_gpu_inference_or_provenance_bound_cache"
    )
    assert result["analysis_compute"]["neural_gpu_seconds_this_run"] == 3.5
    assert result["neural_visual_evidence"]["status"] == "PROPOSALS_ONLY"
    assert result["neural_review_queue"]["status"] == "REVIEW_QUEUE_ONLY"
    assert result["required_semantic_anchors"] == []
    assert result["candidate_count_after_semantic_gates"] == 0

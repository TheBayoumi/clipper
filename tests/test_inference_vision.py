"""Clipper inference safety contracts: GPU claims cannot become verified outcomes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from clipper_engine.inference import vision

SHA = "a" * 40
ANALYSIS_SHA = "b" * 64
SOURCE = {
    "source_key": "ghost_fight",
    "provider_asset_id": "media-silo-asset-1",
    "provider_review_url": "https://app.mediasilo.com/review/test",
    "analysis_sha256": ANALYSIS_SHA,
    "derivative_type": "proxy",
}
MODEL = {
    "model_id": vision.VISION_MODEL_ID,
    "revision": vision.VISION_REVISION,
    "prompt_version": vision.VISION_PROMPT_VERSION,
    "schema_version": vision.VISION_SCHEMA_VERSION,
}


def _config(**overrides: object) -> dict[str, Any]:
    return {
        "capabilities": {
            "neural_visual_events": {
                "enabled": True,
                "backend": "modal",
                "deployed_sha": SHA,
                "max_gpu_seconds": 10,
                "max_estimated_usd": 0.1,
                **overrides,
            }
        }
    }


class FakeVision:
    def __init__(self) -> None:
        self.calls = 0
        self.transform = lambda x: x

    def inspect(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        response = {
            "deployed_git_sha": SHA,
            "source_identity": request["source_identity"],
            "model": MODEL,
            "observations": [
                {
                    "timestamp": t,
                    "summary": "Possible hostile motion, not a verified kill",
                    "event_labels": ["possible_action"],
                    "confidence": 0.85,
                }
                for t in request["frame_timestamps"]
            ],
            "usage": {"gpu_type": "L4:2", "gpu_seconds": 2.0, "estimated_cost_usd": 0.002},
        }
        return self.transform(response)


def _source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    file = tmp_path / "ghost_fight.mp4"
    file.write_bytes(b"certified proxy stub")
    manifest = tmp_path / "ghost_fight.source_qa.json"
    manifest.write_text(json.dumps({"duration_seconds": 8.0}), encoding="utf-8")

    def verify(_file: Path, _manifest: Path, *, require_original: bool) -> dict[str, Any]:
        assert not require_original
        return {
            "source_key": SOURCE["source_key"],
            "provider_asset_id": SOURCE["provider_asset_id"],
            "provider_review_url": SOURCE["provider_review_url"],
            "sha256": SOURCE["analysis_sha256"],
            "derivative_type": SOURCE["derivative_type"],
        }

    def extract(
        _source: Path, times: tuple[float, ...], edge: int, directory: Path
    ) -> tuple[Path, ...]:
        assert edge == 640
        results = []
        for index, _ in enumerate(times):
            path = directory / f"mock{index}.jpg"
            path.write_bytes(b"fake-jpg")
            results.append(path)
        return tuple(results)

    monkeypatch.setattr(vision.source_qa, "verify", verify)
    monkeypatch.setattr(vision, "_extract_samples", extract)
    return file, manifest


def test_disabled_neural_profile_does_not_require_modal_or_media() -> None:
    assert (
        vision.analyze_source(
            Path("absent.mp4"), Path("absent.source_qa.json"), {}, Path("not-created")
        )
        is None
    )


@pytest.mark.parametrize(
    "setting, value",
    [
        ("backend", "cpu"),
        ("app_name", "clipper-open-editor"),
        ("app_name", "clipper-production-pipeline"),
        ("app_name", "external-service"),
        ("deployed_sha", ""),
        ("deployed_sha", "short"),
        ("sample_interval_seconds", 0.1),
        ("sample_interval_seconds", float("inf")),
        ("batch_frames", 0),
        ("batch_frames", 17),
        ("deadline_seconds", 0),
        ("max_gpu_seconds", -1),
        ("max_estimated_usd", 0),
        ("max_estimated_usd", float("nan")),
        ("batch_frames", 1.2),
        ("batch_frames", True),
        ("model_id", "unreviewed-model"),
        ("model_revision", "unreviewed-revision"),
        ("unsupported_field", 5),
        ("deadline_seconds", "120"),
    ],
)
def test_neural_profile_rejects_unsafe_settings(setting: str, value: object) -> None:
    with pytest.raises(ValueError):
        vision.VisionSettings.from_profile(_config(**{setting: value}))


def test_neural_source_scan_emits_unverified_timestamped_proposals_and_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, qa = _source(monkeypatch, tmp_path)
    provider = FakeVision()
    output = tmp_path / "analysis"
    report = vision.analyze_source(source, qa, _config(batch_frames=1), output, provider=provider)
    assert report is not None
    assert report["status"] == "PROPOSALS_ONLY"
    assert report["coverage"]["expected_samples"] == 2
    assert report["coverage"]["observed_samples"] == 2
    assert provider.calls == 2
    assert [item["timestamp"] for item in report["observations"]] == [0.0, 4.0]
    assert all(item["verification"] == "PROPOSED" for item in report["observations"])
    assert all(item["review_required"] for item in report["observations"])
    assert report["usage"]["gpu_seconds"] == 4.0
    replay = vision.analyze_source(source, qa, _config(batch_frames=1), output, provider=provider)
    assert replay is not None
    assert replay["coverage"]["checkpoint_hits"] == 2
    assert provider.calls == 2  # TJR-style warm replay: no GPU work.


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(deployed_git_sha="c" * 40), "deployed SHA"),
        (lambda r: r["source_identity"].update(provider_asset_id="different"), "identity"),
        (lambda r: r["model"].update(revision="different"), "identity"),
        (lambda r: r.update(observations=[]), "cover every"),
        (lambda r: r["observations"][0].update(timestamp=999), "timestamp"),
        (lambda r: r["observations"][0].update(confidence=float("nan")), "confidence"),
        (lambda r: r["observations"][0].update(event_labels=[42]), "event_labels"),
        (lambda r: r.update(usage={}), "GPU accounting"),
        (lambda r: r["usage"].update(estimated_cost_usd=-1), "GPU accounting"),
    ],
)
def test_neural_inference_fails_closed_on_invalid_remote_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    source, qa = _source(monkeypatch, tmp_path)
    provider = FakeVision()

    def corrupt(response: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(response)
        mutation(value)
        return value

    provider.transform = corrupt
    with pytest.raises((ValueError, RuntimeError), match=match):
        vision.analyze_source(source, qa, _config(), tmp_path / "results", provider=provider)
    assert not (tmp_path / "results" / "neural_visual_evidence.json").exists()


def test_neural_budget_excess_rejects_unqualified_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, qa = _source(monkeypatch, tmp_path)
    provider = FakeVision()
    with pytest.raises(RuntimeError, match="budget exceeded"):
        vision.analyze_source(
            source, qa, _config(max_gpu_seconds=1), tmp_path / "results", provider=provider
        )
    assert not (tmp_path / "results" / "neural_visual_evidence.json").exists()


def test_certified_source_identity_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, qa = _source(monkeypatch, tmp_path)
    monkeypatch.setattr(
        vision.source_qa, "verify", lambda *_args, **_kwargs: {"sha256": ANALYSIS_SHA}
    )
    with pytest.raises(ValueError, match="certified source provenance"):
        vision.analyze_source(source, qa, _config(), tmp_path / "results", provider=FakeVision())


def test_neural_cache_refuses_changed_frame_timestamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, qa = _source(monkeypatch, tmp_path)
    output = tmp_path / "result"
    vision.analyze_source(source, qa, _config(), output, provider=FakeVision())
    checkpoint = next((output / "neural_checkpoints").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["observations"][0]["timestamp"] = 999
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid source timestamps"):
        vision.analyze_source(source, qa, _config(), output, provider=FakeVision())


def test_neural_cache_refuses_tampered_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, qa = _source(monkeypatch, tmp_path)
    output = tmp_path / "result"
    vision.analyze_source(source, qa, _config(), output, provider=FakeVision())
    checkpoint = next((output / "neural_checkpoints").glob("*.json"))
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["observations"][0]["summary"] = "unverified false victory claim"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint integrity mismatch"):
        vision.analyze_source(source, qa, _config(), output, provider=FakeVision())


def test_neural_source_key_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, qa = _source(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="source key does not match"):
        vision.analyze_source(
            source,
            qa,
            _config(),
            tmp_path / "result",
            provider=FakeVision(),
            source_key="different_source",
        )


def test_grounded_vision_prompt_contains_exact_array_schema() -> None:
    text = vision.source_visual_prompt([0.0, 4.0])
    assert "top-level key MUST be 'observations'" in text
    assert '"timestamp":0.0' in text
    assert '"timestamp":4.0' in text
    assert text.count('"event_labels":[]') == 2
    retry = vision.source_visual_prompt([0.0, 4.0], recovery_attempt=1)
    assert "Your previous reply did not match this schema" in retry
    assert len(retry) > len(text)


@pytest.mark.parametrize("times", [[], [float("nan")], [-1.0], [True]])
def test_grounded_vision_prompt_refuses_invalid_clock(times: list[float]) -> None:
    with pytest.raises(ValueError, match="finite source-clock"):
        vision.source_visual_prompt(times)

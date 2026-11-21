"""CPU-only validation and manifest recording for the Wan Modal entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

WORKER = Path(__file__).resolve().parents[1] / "scripts" / "modal_wan_s2v.py"
REFERENCE = b"approved-ref" * 30
AUDIO = b"original-audio" * 4000


class _FakeImage:
    @classmethod
    def from_registry(cls, *_args: object, **_kwargs: object) -> _FakeImage:
        return cls()

    def __getattr__(self, _name: str) -> Any:
        return lambda *_args, **_kwargs: self


class _FakeVolume:
    @classmethod
    def from_name(cls, *_args: object, **_kwargs: object) -> _FakeVolume:
        return cls()


class _FakeSecret:
    @staticmethod
    def from_name(*_args: object, **_kwargs: object) -> object:
        return object()


class _FakeFunction:
    def __init__(self, function: Any) -> None:
        self.function = function
        self.remote = Mock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


class _FakeApp:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def function(self, **_kwargs: object) -> Any:
        return _FakeFunction

    def local_entrypoint(self) -> Any:
        return lambda function: function


def _worker(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Image=_FakeImage, Volume=_FakeVolume, Secret=_FakeSecret, App=_FakeApp),
    )
    spec = importlib.util.spec_from_file_location("testable_wan_s2v_worker", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference.png"
    audio = tmp_path / "narration.wav"
    reference.write_bytes(REFERENCE)
    audio.write_bytes(AUDIO)
    return reference, audio


def test_worker_request_accepts_complete_input_hash_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _worker(monkeypatch)
    module.check_request(
        REFERENCE,
        AUDIO,
        hashlib.sha256(REFERENCE).hexdigest(),
        hashlib.sha256(AUDIO).hexdigest(),
    )


@pytest.mark.parametrize("failure", ["empty_reference", "short_audio", "image_hash", "audio_hash"])
def test_worker_request_rejects_invalid_input_before_gpu(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    module = _worker(monkeypatch)
    ref = b"" if failure == "empty_reference" else REFERENCE
    audio = b"x" if failure == "short_audio" else AUDIO
    image_hash = "0" * 64 if failure == "image_hash" else hashlib.sha256(ref).hexdigest()
    audio_hash = "0" * 64 if failure == "audio_hash" else hashlib.sha256(audio).hexdigest()
    with pytest.raises(ValueError):
        module.check_request(ref, audio, image_hash, audio_hash)


def test_remote_failure_is_persisted_without_accepted_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _worker(monkeypatch)
    reference, audio = _inputs(tmp_path)
    module.generate_s2v.remote.side_effect = RuntimeError("fake remote error")
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="fake remote error"):
        module.main(reference=str(reference), audio=str(audio), output=str(output))
    evidence = json.loads((output / "model-evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "GENERATION_FAILED"
    assert evidence["publish_eligible"] is False
    assert evidence["error_type"] == "RuntimeError"
    assert not (output / "wan-s2v-candidate.mp4").exists()


def test_remote_hash_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _worker(monkeypatch)
    reference, audio = _inputs(tmp_path)
    module.generate_s2v.remote.return_value = {
        "video": b"candidate-video",
        "video_sha256": "0" * 64,
    }
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="video hash"):
        module.main(reference=str(reference), audio=str(audio), output=str(output))
    evidence = json.loads((output / "model-evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "GENERATION_FAILED"


def test_matching_remote_response_stays_human_review_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _worker(monkeypatch)
    reference, audio = _inputs(tmp_path)
    video = b"candidate-video"
    module.generate_s2v.remote.return_value = {
        "video": video,
        "video_sha256": hashlib.sha256(video).hexdigest(),
        "model_id": module.MODEL_ID,
        "model_revision": module.MODEL_REVISION,
    }
    output = tmp_path / "output"
    module.main(reference=str(reference), audio=str(audio), output=str(output))
    evidence = json.loads((output / "model-evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED"
    assert evidence["publish_eligible"] is False
    assert evidence["video_sha256"] == hashlib.sha256(video).hexdigest()

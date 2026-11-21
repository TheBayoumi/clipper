"""Kokoro golden WAV, request fingerprint, failure and legacy delegation parity."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import narration, scripting

ROOT = Path(__file__).resolve().parents[1]


def _valid_wav(*, channels: int = 1, rate: int = 24000, frames: int = 24000) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x00\x01" * frames * channels)
    return stream.getvalue()


def _legacy() -> object:
    spec = importlib.util.spec_from_file_location(
        "niche_video_narration_parity", ROOT / "scripts/niche_video.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kokoro_golden_request_fingerprint_and_legacy_cache_bytes(tmp_path: Path) -> None:
    config = scripting.NicheConfig.load(ROOT / "niches/book-lessons.yaml")
    scene = scripting.Scene("An original title", "Start with just one small step today.")
    request = narration.tts_request(scene, config)
    assert request == {
        "contract": "kokoro-24khz-wav-v1",
        "model": "hexgrad/Kokoro-82M",
        "model_revision": "v1.0",
        "voice": config.voice,
        "language": "en-us",
        "sample_rate": 24000,
        "speed": 1.0,
        "text": scene.narration,
    }
    digest = hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    original = _legacy()
    audio = _valid_wav()
    canonical = Mock(return_value=audio)
    legacy_producer = Mock(return_value=audio)
    original._kokoro_wav = legacy_producer
    package_root, legacy_root = tmp_path / "canonical", tmp_path / "legacy"
    canonical_path = narration.synthesize_scene(scene, config, package_root, producer=canonical)
    legacy_path = original.synthesize_scene(scene, config, legacy_root)
    assert canonical_path == narration.tts_cache_path(scene, config, package_root)
    assert canonical_path.name == legacy_path.name == f"{digest}.wav"
    assert canonical_path.read_bytes() == legacy_path.read_bytes() == audio
    assert narration._wav_metadata(audio) == (1, 24000, 24000)
    assert narration.synthesize_scene(scene, config, package_root, producer=canonical) == canonical_path
    assert original.synthesize_scene(scene, config, legacy_root) == legacy_path
    canonical.assert_called_once_with(scene.narration, config.voice)
    legacy_producer.assert_called_once_with(scene.narration, config.voice)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (b"invalid", "invalid WAV audio"),
        (_valid_wav(channels=2), "mono 24 kHz"),
        (_valid_wav(rate=22050), "mono 24 kHz"),
        (_valid_wav(frames=7999), "mono 24 kHz"),
    ],
)
def test_invalid_generated_audio_rejected_identically(
    tmp_path: Path, audio: bytes, message: str
) -> None:
    config = scripting.NicheConfig.load(ROOT / "niches/book-lessons.yaml")
    scene = scripting.Scene("Chapter", "A distinct narrated scene.")
    legacy = _legacy()
    legacy._kokoro_wav = Mock(return_value=audio)
    with pytest.raises(scripting.NicheVideoError, match=message) as canonical_exc:
        narration.synthesize_scene(scene, config, tmp_path / "canonical", producer=lambda *_: audio)
    with pytest.raises(scripting.NicheVideoError, match=message) as legacy_exc:
        legacy.synthesize_scene(scene, config, tmp_path / "legacy")
    assert str(canonical_exc.value) == str(legacy_exc.value)
    assert not list(tmp_path.rglob("*.wav"))


def test_corrupt_existing_cache_never_regenerated(tmp_path: Path) -> None:
    config = scripting.NicheConfig.load(ROOT / "niches/book-lessons.yaml")
    scene = scripting.Scene("Chapter", "Another narrated scene.")
    path = narration.tts_cache_path(scene, config, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupted")
    producer = Mock(side_effect=AssertionError("producer should never run"))
    with pytest.raises(scripting.NicheVideoError, match="invalid WAV audio"):
        narration.synthesize_scene(scene, config, tmp_path, producer=producer)
    producer.assert_not_called()
    path.write_bytes(_valid_wav(channels=2))
    with pytest.raises(scripting.NicheVideoError, match="Cached TTS WAV has invalid geometry"):
        narration.synthesize_scene(scene, config, tmp_path, producer=producer)
    producer.assert_not_called()


def test_optional_kokoro_import_missing_raises_legacy_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise ImportError("synthetic optional dependency")

    monkeypatch.setattr(narration.importlib, "import_module", missing)
    narration._kokoro_pipeline.cache_clear()
    try:
        with pytest.raises(scripting.NicheVideoError, match="niche-models"):
            narration._kokoro_pipeline()
        with pytest.raises(scripting.NicheVideoError, match="soundfile and numpy"):
            narration._kokoro_wav("Test", "af_heart")
    finally:
        narration._kokoro_pipeline.cache_clear()

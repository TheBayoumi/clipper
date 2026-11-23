"""Kokoro golden WAV, request fingerprint, failure and legacy delegation parity."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
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
    assert (
        narration.synthesize_scene(scene, config, package_root, producer=canonical)
        == canonical_path
    )
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


def test_optional_kokoro_import_missing_raises_legacy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_synthetic_kokoro_pipeline_and_wave_conversion_without_model_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise optional module boundaries with synthetic arrays and PCM writing."""

    class FakeArray(list[float]):
        def reshape(self, _length: int) -> FakeArray:
            return self

    voice_pipeline = Mock(return_value=[(None, None, [0.25] * 9000)])
    loader = Mock(return_value=voice_pipeline)

    def write_pcm(
        target: io.BytesIO,
        audio: FakeArray,
        rate: int,
        *,
        format: str,
        subtype: str,
    ) -> None:
        assert (rate, format, subtype) == (24000, "WAV", "PCM_16")
        with wave.open(target, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(bytes((1, 0)) * len(audio))

    numpy = SimpleNamespace(
        float32=object(),
        asarray=lambda audio, dtype: FakeArray(audio),
        concatenate=lambda arrays: FakeArray(value for array in arrays for value in array),
        isfinite=lambda _array: SimpleNamespace(all=lambda: True),
        clip=lambda array, _minimum, _maximum: array,
    )
    modules = {
        "kokoro": SimpleNamespace(KPipeline=loader),
        "numpy": numpy,
        "soundfile": SimpleNamespace(write=write_pcm),
    }
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(narration.importlib, "import_module", fake_import)
    narration._kokoro_pipeline.cache_clear()
    try:
        wav = narration._kokoro_wav("Original narration", "af_heart")
        assert narration._wav_metadata(wav) == (1, 24000, 9000)
        loader.assert_called_once_with(lang_code="a", repo_id=narration.VOICE_MODEL)
        voice_pipeline.assert_called_once_with("Original narration", voice="af_heart", speed=1.0)
        narration._kokoro_wav("Another narration", "af_heart")
        loader.assert_called_once()
        assert imported.count("kokoro") == 1

        with pytest.raises(scripting.NicheVideoError, match="no audio segments"):
            narration._kokoro_wav(
                "Empty", "af_heart", pipeline=lambda: lambda *_args, **_kwargs: []
            )
        with pytest.raises(scripting.NicheVideoError, match="empty or non-finite audio"):
            narration._kokoro_wav(
                "Short",
                "af_heart",
                pipeline=lambda: lambda *_args, **_kwargs: [(None, None, [0.2] * 30)],
            )
    finally:
        narration._kokoro_pipeline.cache_clear()



@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("mutation", ["data-claims-more", "missing-pcm", "wrong-sample-width", "trailing-data"])
def test_wav_validation_uses_real_pcm_frames_not_forged_metadata(
    tmp_path: Path, cached: bool, mutation: str
) -> None:
    original = _valid_wav(frames=8000)
    corrupted = bytearray(original)
    if mutation == "data-claims-more":
        corrupted[40:44] = (384000).to_bytes(4, "little")
        corrupted[4:8] = (384036).to_bytes(4, "little")
    elif mutation == "missing-pcm":
        corrupted = corrupted[:-2]
        corrupted[4:8] = (len(corrupted) - 8).to_bytes(4, "little")
    elif mutation == "wrong-sample-width":
        corrupted[34:36] = (1).to_bytes(2, "little")
    elif mutation == "trailing-data":
        corrupted.extend(b"\0\0")
    audio = bytes(corrupted)
    config = scripting.NicheConfig.load(ROOT / "niches/book-lessons.yaml")
    scene = scripting.Scene("synthetic", "synthetic sentence")
    root = tmp_path / ("cache" if cached else "new")
    path = narration.tts_cache_path(scene, config, root)
    if cached:
        path.parent.mkdir(parents=True)
        path.write_bytes(audio)
    supplied = Mock(return_value=audio)
    with pytest.raises(scripting.NicheVideoError, match="invalid WAV audio"):
        narration.synthesize_scene(scene, config, root, producer=supplied)
    if cached:
        supplied.assert_not_called()
    else:
        supplied.assert_called_once()
        assert not path.exists()

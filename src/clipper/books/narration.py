"""Pinned Kokoro narration and byte-compatible, content-addressed WAV cache."""

from __future__ import annotations

import importlib
import io
import wave
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import acquisition
from .scripting import (
    VOICE_CONTRACT,
    VOICE_MODEL,
    VOICE_REVISION,
    NicheConfig,
    NicheVideoError,
    Scene,
    _fingerprint,
)


@lru_cache(maxsize=1)
def _kokoro_pipeline() -> Any:
    try:
        kokoro = importlib.import_module("kokoro")
    except ImportError as exc:
        raise NicheVideoError("Install the [niche-models] runtime for Kokoro TTS") from exc
    return kokoro.KPipeline(lang_code="a", repo_id=VOICE_MODEL)


MAX_WAV_BYTES = 40_000_000


def _wav_metadata(payload: bytes) -> tuple[int, int, int]:
    """Validate RIFF structure and actual decoded PCM, never just declared frame count."""
    try:
        if (
            not isinstance(payload, bytes)
            or not 44 <= len(payload) <= MAX_WAV_BYTES
            or payload[:4] != b"RIFF"
            or payload[8:12] != b"WAVE"
            or int.from_bytes(payload[4:8], "little") + 8 != len(payload)
        ):
            raise ValueError("invalid RIFF envelope")
        offset = 12
        data_sizes: list[int] = []
        while offset < len(payload):
            if offset + 8 > len(payload):
                raise ValueError("incomplete RIFF chunk header")
            size = int.from_bytes(payload[offset + 4 : offset + 8], "little")
            end = offset + 8 + size
            if end + (size & 1) > len(payload):
                raise ValueError("truncated RIFF chunk")
            if payload[offset : offset + 4] == b"data":
                data_sizes.append(size)
            offset = end + (size & 1)
        if offset != len(payload) or len(data_sizes) != 1:
            raise ValueError("missing or duplicate PCM data chunk")
        with wave.open(io.BytesIO(payload), "rb") as wav:
            channels, rate, frames = wav.getnchannels(), wav.getframerate(), wav.getnframes()
            expected = frames * channels * wav.getsampwidth()
            if (
                wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
                or expected != data_sizes[0]
                or len(wav.readframes(frames + 1)) != expected
            ):
                raise ValueError("truncated or unsupported PCM geometry")
        return channels, rate, frames
    except (wave.Error, EOFError, ValueError, TypeError, OverflowError) as exc:
        raise NicheVideoError("Kokoro produced invalid WAV audio") from exc


def _kokoro_wav(narration: str, voice: str, *, pipeline: Callable[[], Any] | None = None) -> bytes:
    try:
        np = importlib.import_module("numpy")
        sf = importlib.import_module("soundfile")
    except ImportError as exc:
        raise NicheVideoError("Install soundfile and numpy for open-source TTS") from exc
    voice_pipeline = (pipeline or _kokoro_pipeline)()
    chunks = [
        np.asarray(audio, dtype=np.float32).reshape(-1)
        for _, _, audio in voice_pipeline(narration, voice=voice, speed=1.0)
    ]
    if not chunks:
        raise NicheVideoError("Kokoro returned no audio segments")
    speech = np.concatenate(chunks)
    if len(speech) < 8000 or not np.isfinite(speech).all():
        raise NicheVideoError("Kokoro returned empty or non-finite audio")
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(speech, -1.0, 1.0), 24000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def validated_wav(payload: bytes, *, cached: bool = False) -> tuple[int, int, int]:
    """Retain the legacy cached-versus-generated error messages exactly."""
    nch, rate, frames = _wav_metadata(payload)
    if nch != 1 or rate != 24000 or frames < 8000:
        if cached:
            raise NicheVideoError("Cached TTS WAV has invalid geometry or duration")
        raise NicheVideoError("Kokoro WAV must be mono 24 kHz with audible duration")
    return nch, rate, frames


def tts_request(scene: Scene, config: NicheConfig) -> dict[str, object]:
    """Keep the exact original mapping and hash inputs for existing WAV caches."""
    return {
        "contract": VOICE_CONTRACT,
        "model": VOICE_MODEL,
        "model_revision": VOICE_REVISION,
        "voice": config.voice,
        "language": "en-us",
        "sample_rate": 24000,
        "speed": 1.0,
        "text": scene.narration,
    }


def tts_cache_path(scene: Scene, config: NicheConfig, cache_root: Path) -> Path:
    return cache_root / "tts" / f"{_fingerprint(tts_request(scene, config))}.wav"


def synthesize_scene(
    scene: Scene,
    config: NicheConfig,
    cache_root: Path,
    *,
    producer: Callable[[str, str], bytes] | None = None,
) -> Path:
    path = tts_cache_path(scene, config, cache_root)
    if path.is_file():
        validated_wav(path.read_bytes(), cached=True)
        return path
    audio = (producer or _kokoro_wav)(scene.narration, config.voice)
    validated_wav(audio)
    acquisition._private_write(path, audio, private_root=cache_root)
    return path

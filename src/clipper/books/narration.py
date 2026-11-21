"""Pinned Kokoro narration and byte-compatible, content-addressed WAV cache."""

from __future__ import annotations

import importlib
import io
import wave
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from .scripting import (
    VOICE_CONTRACT,
    VOICE_MODEL,
    VOICE_REVISION,
    NicheConfig,
    NicheVideoError,
    Scene,
    _fingerprint,
    _write_atomic,
)


@lru_cache(maxsize=1)
def _kokoro_pipeline() -> Any:
    try:
        kokoro = importlib.import_module("kokoro")
    except ImportError as exc:
        raise NicheVideoError("Install the [niche-models] runtime for Kokoro TTS") from exc
    return kokoro.KPipeline(lang_code="a", repo_id=VOICE_MODEL)


def _wav_metadata(payload: bytes) -> tuple[int, int, int]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            return wav.getnchannels(), wav.getframerate(), wav.getnframes()
    except (wave.Error, EOFError) as exc:
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
        nch, rate, frames = _wav_metadata(path.read_bytes())
        if nch == 1 and rate == 24000 and frames >= 8000:
            return path
        raise NicheVideoError("Cached TTS WAV has invalid geometry or duration")
    audio = (producer or _kokoro_wav)(scene.narration, config.voice)
    nch, rate, frames = _wav_metadata(audio)
    if nch != 1 or rate != 24000 or frames < 8000:
        raise NicheVideoError("Kokoro WAV must be mono 24 kHz with audible duration")
    _write_atomic(path, audio)
    return path

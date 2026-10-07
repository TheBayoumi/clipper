from __future__ import annotations

import gc
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str
    probability: float


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...]


@dataclass(frozen=True)
class Transcript:
    language: str
    duration: float
    model: str
    segments: tuple[TranscriptSegment, ...]
    settings: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "duration": self.duration,
            "model": self.model,
            "segments": [asdict(item) for item in self.segments],
            "settings": dict(self.settings),
        }


def _duration(source: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(source),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    value = float(completed.stdout.strip())
    if value <= 0.0:
        raise RuntimeError(f"source duration is invalid: {source}")
    return value


def _decode_audio(source: Path, start: float, duration: float, sample_rate: int) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"decoded audio chunk is empty at {start:.3f}s")
    return audio


def transcribe(source: Path, config: dict[str, Any]) -> Transcript:
    cfg = dict(config.get("spoken_content", {}).get("asr") or {})
    model_name = str(cfg.get("model", "base.en"))
    device = str(cfg.get("device", "cpu"))
    compute_type = str(cfg.get("compute_type", "int8"))
    cpu_threads = int(cfg.get("cpu_threads", 2))
    beam_size = int(cfg.get("beam_size", 3))
    language = str(cfg.get("language", "en"))
    sample_rate = int(cfg.get("sample_rate", 16000))
    chunk_seconds = float(cfg.get("chunk_seconds", 180.0))
    if chunk_seconds < 30.0 or chunk_seconds > 600.0:
        raise RuntimeError("spoken_content.asr.chunk_seconds must be between 30 and 600")
    if sample_rate != 16000:
        raise RuntimeError("faster-whisper waveform input requires 16000 Hz analysis audio")

    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "spoken-content analysis requires faster-whisper; install the mw4/asr runtime"
        ) from exc

    duration = _duration(source)
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
    )
    output: list[TranscriptSegment] = []
    detected_language = language
    chunks = math.ceil(duration / chunk_seconds)
    for chunk_index in range(chunks):
        offset = chunk_index * chunk_seconds
        chunk_duration = min(chunk_seconds, duration - offset)
        if chunk_duration <= 0.0:
            continue
        waveform = _decode_audio(source, offset, chunk_duration, sample_rate)
        raw_segments, info = model.transcribe(
            waveform,
            language=language,
            beam_size=beam_size,
            vad_filter=bool(cfg.get("vad_filter", True)),
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        detected_language = str(getattr(info, "language", language) or language)
        for segment in raw_segments:
            words = tuple(
                TranscriptWord(
                    start=round(offset + float(word.start or segment.start), 3),
                    end=round(offset + float(word.end or segment.end), 3),
                    text=str(word.word),
                    probability=round(float(word.probability or 0.0), 4),
                )
                for word in (segment.words or [])
            )
            output.append(
                TranscriptSegment(
                    start=round(offset + float(segment.start), 3),
                    end=round(offset + float(segment.end), 3),
                    text=str(segment.text).strip(),
                    words=words,
                )
            )
        del waveform, raw_segments
        gc.collect()

    del model
    gc.collect()
    settings = {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "beam_size": beam_size,
        "language": language,
        "sample_rate": sample_rate,
        "chunk_seconds": chunk_seconds,
        "vad_filter": bool(cfg.get("vad_filter", True)),
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    return Transcript(
        language=detected_language,
        duration=round(duration, 6),
        model=model_name,
        segments=tuple(output),
        settings=settings,
    )


def write_transcript(path: Path, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript.to_json(), indent=2), encoding="utf-8")

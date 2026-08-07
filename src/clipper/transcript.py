from __future__ import annotations

import html
import re
from collections.abc import Iterable
from pathlib import Path

from .models import TranscriptSegment

_TIMESTAMP_RE = re.compile(
    r"(?P<h1>\d{2}):(?P<m1>\d{2}):(?P<s1>\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}[.,]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _seconds(hours: str, minutes: str, seconds: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))


def _clean_caption(lines: Iterable[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = html.unescape(_TAG_RE.sub("", text))
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt(text: str) -> list[TranscriptSegment]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized)
    segments: list[TranscriptSegment] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        timestamp_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timestamp_index is None:
            continue
        match = _TIMESTAMP_RE.search(lines[timestamp_index])
        if not match:
            continue
        caption = _clean_caption(lines[timestamp_index + 1 :])
        if not caption:
            continue
        start = _seconds(match["h1"], match["m1"], match["s1"])
        end = _seconds(match["h2"], match["m2"], match["s2"])
        if end <= start:
            continue
        if segments and segments[-1].text == caption and start <= segments[-1].end + 0.05:
            previous = segments.pop()
            segments.append(TranscriptSegment(previous.start, max(previous.end, end), caption))
        else:
            segments.append(TranscriptSegment(start, end, caption))
    return segments


def load_vtt(path: str | Path) -> list[TranscriptSegment]:
    return parse_vtt(Path(path).read_text(encoding="utf-8-sig"))


def transcribe_with_faster_whisper(
    media_path: str | Path,
    *,
    model_name: str = "small",
    device: str = "auto",
    compute_type: str = "int8",
    language: str | None = None,
) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "no subtitles were available and faster-whisper is not installed; "
            "install with `pip install -e '.[asr]'`"
        ) from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    raw_segments, _ = model.transcribe(
        str(media_path),
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=False,
    )
    return [
        TranscriptSegment(float(segment.start), float(segment.end), segment.text.strip())
        for segment in raw_segments
        if segment.text.strip() and float(segment.end) > float(segment.start)
    ]

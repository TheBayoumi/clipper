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


def _clean_caption_line(line: str) -> str:
    text = html.unescape(_TAG_RE.sub("", line))
    return re.sub(r"\s+", " ", text).strip()


def _clean_caption(lines: Iterable[str]) -> str:
    return " ".join(
        cleaned
        for line in lines
        if (cleaned := _clean_caption_line(line))
    ).strip()


def parse_vtt(text: str) -> list[TranscriptSegment]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    timestamp_indexes = [index for index, line in enumerate(lines) if _TIMESTAMP_RE.search(line)]
    segments: list[TranscriptSegment] = []
    last_display_line = ""

    for position, timestamp_index in enumerate(timestamp_indexes):
        match = _TIMESTAMP_RE.search(lines[timestamp_index])
        if not match:
            continue
        next_timestamp = (
            timestamp_indexes[position + 1] if position + 1 < len(timestamp_indexes) else len(lines)
        )
        body_lines = lines[timestamp_index + 1 : next_timestamp]
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        if body_lines and re.fullmatch(r"\d+", body_lines[-1].strip()):
            body_lines.pop()

        cleaned_lines = [
            cleaned for line in body_lines if (cleaned := _clean_caption_line(line))
        ]
        start = _seconds(match["h1"], match["m1"], match["s1"])
        end = _seconds(match["h2"], match["m2"], match["s2"])
        if end <= start:
            continue

        display_last_line = cleaned_lines[-1] if cleaned_lines else last_display_line
        if end - start <= 0.05:
            if display_last_line:
                last_display_line = display_last_line
            continue
        if not cleaned_lines:
            continue

        if len(cleaned_lines) == 1 and cleaned_lines[0] == last_display_line:
            if (
                segments
                and segments[-1].text == cleaned_lines[0]
                and start <= segments[-1].end + 0.05
            ):
                previous = segments.pop()
                segments.append(
                    TranscriptSegment(previous.start, max(previous.end, end), previous.text)
                )
            last_display_line = display_last_line
            continue

        if last_display_line and cleaned_lines[0] == last_display_line:
            cleaned_lines.pop(0)
        elif last_display_line and cleaned_lines[0].startswith(last_display_line + " "):
            cleaned_lines[0] = cleaned_lines[0][len(last_display_line) :].strip()

        caption = _clean_caption(cleaned_lines)
        if caption:
            segments.append(TranscriptSegment(start, end, caption))
        last_display_line = display_last_line
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

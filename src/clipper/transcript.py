from __future__ import annotations

import html
import re
from collections.abc import Iterable
from pathlib import Path

from .models import TranscriptSegment, TranscriptWord

_TIMESTAMP_RE = re.compile(
    r"(?P<h1>\d{2}):(?P<m1>\d{2}):(?P<s1>\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<h2>\d{2}):(?P<m2>\d{2}):(?P<s2>\d{2}[.,]\d{3})"
)
_INLINE_TIMESTAMP_RE = re.compile(r"<(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}[.,]\d{3})>")
_TAG_RE = re.compile(r"<[^>]+>")


def _seconds(hours: str, minutes: str, seconds: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))


def _clean_caption_line(line: str) -> str:
    text = html.unescape(_TAG_RE.sub("", line))
    text = re.sub(r"^>>\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_caption(lines: Iterable[str]) -> str:
    return " ".join(cleaned for line in lines if (cleaned := _clean_caption_line(line))).strip()


def _inline_word_spans(line: str, cue_start: float, cue_end: float) -> tuple[TranscriptWord, ...]:
    decoded = html.unescape(line)
    timestamps = list(_INLINE_TIMESTAMP_RE.finditer(decoded))
    if not timestamps:
        return ()

    pieces: list[tuple[float, float, str]] = []
    prefix = _clean_caption_line(decoded[: timestamps[0].start()])
    first_time = _seconds(timestamps[0]["h"], timestamps[0]["m"], timestamps[0]["s"])
    if prefix and first_time > cue_start:
        pieces.append((cue_start, first_time, prefix))

    for index, match in enumerate(timestamps):
        start = _seconds(match["h"], match["m"], match["s"])
        end = (
            _seconds(
                timestamps[index + 1]["h"],
                timestamps[index + 1]["m"],
                timestamps[index + 1]["s"],
            )
            if index + 1 < len(timestamps)
            else cue_end
        )
        raw_end = timestamps[index + 1].start() if index + 1 < len(timestamps) else len(decoded)
        raw = decoded[match.end() : raw_end]
        cleaned = _clean_caption_line(raw)
        if cleaned and end > start:
            pieces.append((start, end, cleaned))

    words: list[TranscriptWord] = []
    for start, end, piece in pieces:
        tokens = piece.split()
        if not tokens:
            continue
        step = (end - start) / len(tokens)
        for index, token in enumerate(tokens):
            word_start = start + index * step
            word_end = end if index + 1 == len(tokens) else start + (index + 1) * step
            if word_end > word_start:
                words.append(TranscriptWord(word_start, word_end, token))
    return tuple(words)


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
        body_lines: list[str] = []
        for line in lines[timestamp_index + 1 : next_timestamp]:
            if not line.strip():
                if body_lines:
                    break
                continue
            if "-->" in line:
                break
            body_lines.append(line)

        start = _seconds(match["h1"], match["m1"], match["s1"])
        end = _seconds(match["h2"], match["m2"], match["s2"])
        if end <= start:
            continue

        line_items = [
            (_clean_caption_line(line), _inline_word_spans(line, start, end)) for line in body_lines
        ]
        line_items = [item for item in line_items if item[0]]
        display_last_line = line_items[-1][0] if line_items else last_display_line
        if end - start <= 0.05:
            if display_last_line:
                last_display_line = display_last_line
            continue
        if not line_items:
            continue

        if len(line_items) == 1 and line_items[0][0] == last_display_line:
            if (
                segments
                and segments[-1].text == line_items[0][0]
                and start <= segments[-1].end + 0.05
            ):
                previous = segments.pop()
                merged_words = previous.words or line_items[0][1]
                segments.append(
                    TranscriptSegment(
                        previous.start, max(previous.end, end), previous.text, merged_words
                    )
                )
            last_display_line = display_last_line
            continue

        if last_display_line and line_items[0][0] == last_display_line:
            line_items.pop(0)
        elif last_display_line and line_items[0][0].startswith(last_display_line + " "):
            cleaned, words = line_items[0]
            prefix_words = len(last_display_line.split())
            line_items[0] = (
                cleaned[len(last_display_line) :].strip(),
                words[prefix_words:] if len(words) >= prefix_words else (),
            )

        caption = _clean_caption(item[0] for item in line_items)
        words = tuple(word for _, line_words in line_items for word in line_words)
        if caption:
            segments.append(TranscriptSegment(start, end, caption, words))
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
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
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
        word_timestamps=True,
    )
    segments: list[TranscriptSegment] = []
    for segment in raw_segments:
        text = segment.text.strip()
        start = float(segment.start)
        end = float(segment.end)
        if not text or end <= start:
            continue
        words: list[TranscriptWord] = []
        for raw_word in getattr(segment, "words", None) or ():
            word_text = str(raw_word.word).strip()
            word_start = float(raw_word.start)
            word_end = float(raw_word.end)
            if word_text and word_start >= 0 and word_end > word_start:
                words.append(TranscriptWord(word_start, word_end, word_text))
        segments.append(TranscriptSegment(start, end, text, tuple(words)))
    return segments

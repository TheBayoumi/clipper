from __future__ import annotations

import html
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import TranscriptSegment, WordTiming

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
    return " ".join(cleaned for line in lines if (cleaned := _clean_caption_line(line))).strip()


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

        cleaned_lines = [cleaned for line in body_lines if (cleaned := _clean_caption_line(line))]
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


def _word_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _segments_from_whisper(
    raw_segments: Iterable[Any], *, word_timestamps: bool
) -> list[TranscriptSegment]:
    if word_timestamps:
        aligned: list[TranscriptSegment] = []
        for sentence in raw_segments:
            group: list[WordTiming] = []
            for raw_word in getattr(sentence, "words", None) or []:
                raw_start = getattr(raw_word, "start", None)
                raw_end = getattr(raw_word, "end", None)
                raw_text = _word_text(getattr(raw_word, "word", ""))
                if (
                    raw_start is None
                    or raw_end is None
                    or float(raw_end) <= float(raw_start)
                    or not raw_text
                ):
                    continue
                start = float(raw_start)
                end = float(raw_end)
                if group and start < group[-1].end - 0.005:
                    continue
                word = WordTiming(start, end, raw_text)
                if group and (len(group) >= 6 or end - group[0].start > 2.6):
                    aligned.append(
                        TranscriptSegment(
                            group[0].start,
                            group[-1].end,
                            " ".join(item.text for item in group),
                            tuple(group),
                        )
                    )
                    group = []
                group.append(word)
                if raw_text.rstrip().endswith((".", "!", "?")) and len(group) >= 3:
                    aligned.append(
                        TranscriptSegment(
                            group[0].start,
                            group[-1].end,
                            " ".join(item.text for item in group),
                            tuple(group),
                        )
                    )
                    group = []
            if group:
                aligned.append(
                    TranscriptSegment(
                        group[0].start,
                        group[-1].end,
                        " ".join(item.text for item in group),
                        tuple(group),
                    )
                )
            elif (
                not getattr(sentence, "words", None)
                and sentence.text.strip()
                and sentence.end > sentence.start
            ):
                aligned.append(
                    TranscriptSegment(
                        float(sentence.start), float(sentence.end), sentence.text.strip()
                    )
                )
        return aligned
    return [
        TranscriptSegment(float(segment.start), float(segment.end), segment.text.strip())
        for segment in raw_segments
        if segment.text.strip() and float(segment.end) > float(segment.start)
    ]


class FasterWhisperTranscriber:
    """Reusable local ASR model for a complete source.

    Production uses distil-large-v3: a larger English model designed for
    faster-whisper. The model is loaded once, then reused for every audio chunk.
    """

    def __init__(
        self,
        *,
        model_name: str = "distil-large-v3",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
        word_timestamps: bool = False,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "no subtitles were available and faster-whisper is not installed; "
                "install with `pip install -e '.[asr]'`"
            ) from exc
        self.model_name = model_name
        self.language = language
        self.word_timestamps = word_timestamps
        self._model: Any = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, media_path: str | Path) -> list[TranscriptSegment]:
        raw_segments, _ = self._model.transcribe(
            str(media_path),
            language=self.language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
            word_timestamps=self.word_timestamps,
            condition_on_previous_text=False,
        )
        return _segments_from_whisper(raw_segments, word_timestamps=self.word_timestamps)


def transcribe_with_faster_whisper(
    media_path: str | Path,
    *,
    model_name: str = "distil-large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str | None = None,
    word_timestamps: bool = False,
) -> list[TranscriptSegment]:
    """Compatibility wrapper for one-off callers.

    Multi-chunk production should instantiate FasterWhisperTranscriber once.
    """
    return FasterWhisperTranscriber(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        language=language,
        word_timestamps=word_timestamps,
    ).transcribe(media_path)

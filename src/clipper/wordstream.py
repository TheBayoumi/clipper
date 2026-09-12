from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .models import SourceSpan, TranscriptSegment

_SPEAKER_RE = re.compile(r"^>>\s*")
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True, slots=True)
class SourceWord:
    source_start: float
    source_end: float
    text: str
    exact: bool


@dataclass(frozen=True, slots=True)
class ClipLocalWord:
    source_start: float
    source_end: float
    local_start: float
    local_end: float
    text: str
    exact: bool


def clean_word(text: str) -> str:
    cleaned = _SPEAKER_RE.sub("", text.strip())
    return cleaned.replace("{", "(").replace("}", ")").replace("\\", "")


def normalized_tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def segment_source_words(segment: TranscriptSegment) -> list[SourceWord]:
    if segment.words:
        return [
            SourceWord(word.start, word.end, clean_word(word.text), True)
            for word in segment.words
            if clean_word(word.text)
        ]
    tokens = [clean_word(token) for token in segment.text.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return []
    step = segment.duration / len(tokens)
    return [
        SourceWord(
            segment.start + index * step,
            segment.end if index + 1 == len(tokens) else segment.start + (index + 1) * step,
            token,
            False,
        )
        for index, token in enumerate(tokens)
    ]


def flatten_source_words(segments: Sequence[TranscriptSegment]) -> list[SourceWord]:
    flattened: list[SourceWord] = []
    seen: set[tuple[int, int, str]] = set()
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        for word in segment_source_words(segment):
            key = (
                round(word.source_start * 1000),
                round(word.source_end * 1000),
                word.text.lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            flattened.append(word)
    return sorted(flattened, key=lambda item: (item.source_start, item.source_end))


def first_complete_word(
    segments: Sequence[TranscriptSegment], start: float, end: float
) -> SourceWord | None:
    for word in flatten_source_words(segments):
        if word.source_start >= start - 1e-6 and word.source_end <= end + 1e-6:
            return word
    return None


def find_phrase_anchor(
    segments: Sequence[TranscriptSegment], start: float, end: float, phrase: str
) -> SourceWord | None:
    needle = normalized_tokens(phrase)[:5]
    if not needle:
        return None
    words = [
        word
        for word in flatten_source_words(segments)
        if word.source_end > start and word.source_start < end
    ]
    flattened = [(normalized_tokens(word.text) or [""])[0] for word in words]
    for index in range(0, max(0, len(flattened) - len(needle) + 1)):
        if flattened[index : index + len(needle)] == needle:
            return words[index]
    return None


def build_clip_word_stream(
    source_spans: Sequence[SourceSpan],
    segments: Sequence[TranscriptSegment],
    *,
    caption_start_source_time: float | None = None,
) -> tuple[list[ClipLocalWord], int]:
    source_words = flatten_source_words(segments)
    output: list[ClipLocalWord] = []
    partial_dropped = 0
    local_base = 0.0
    for span_index, span in enumerate(source_spans):
        anchor = caption_start_source_time if span_index == 0 else None
        effective_start = max(span.start, anchor) if anchor is not None else span.start
        for word in source_words:
            if word.source_end <= span.start or word.source_start >= span.end:
                continue
            if word.source_start < effective_start - 1e-6 or word.source_end > span.end + 1e-6:
                partial_dropped += 1
                continue
            local_start = local_base + (word.source_start - span.start)
            local_end = local_base + (word.source_end - span.start)
            output.append(
                ClipLocalWord(
                    word.source_start,
                    word.source_end,
                    max(0.0, local_start),
                    max(0.0, local_end),
                    word.text,
                    word.exact,
                )
            )
        local_base += span.duration
    return output, partial_dropped

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .models import TranscriptSegment, TranscriptWord

TimingMode = Literal["word_exact", "aligned", "cue_interpolated"]


@dataclass(frozen=True, slots=True)
class CanonicalWord:
    word_id: str
    text: str
    source_start: float
    source_end: float
    speaker_id: str | None
    confidence: float | None
    timing_mode: TimingMode
    transcript_source: str

    def __post_init__(self) -> None:
        if not self.word_id.strip():
            raise ValueError("canonical word_id cannot be empty")
        if not self.text.strip():
            raise ValueError("canonical word text cannot be empty")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("canonical word timing is invalid")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("canonical word confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CanonicalTimeline:
    video_id: str
    source_hash: str
    words: tuple[CanonicalWord, ...]
    schema_version: str = "canonical-timeline-v1"

    def __post_init__(self) -> None:
        if not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("canonical timeline requires video_id and source_hash")
        identifiers = [word.word_id for word in self.words]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("canonical timeline contains duplicate word IDs")
        if any(
            a.source_start > b.source_start
            for a, b in zip(self.words, self.words[1:], strict=False)
        ):
            raise ValueError("canonical timeline words must be source ordered")

    @property
    def start(self) -> float:
        return self.words[0].source_start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].source_end if self.words else 0.0

    def word(self, word_id: str) -> CanonicalWord:
        for item in self.words:
            if item.word_id == word_id:
                return item
        raise KeyError(word_id)

    def require_word_ids(self, word_ids: tuple[str, ...] | list[str]) -> tuple[CanonicalWord, ...]:
        index = {word.word_id: word for word in self.words}
        missing = [word_id for word_id in word_ids if word_id not in index]
        if missing:
            raise ValueError(f"unknown canonical word IDs: {missing[:5]}")
        return tuple(index[word_id] for word_id in word_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "source_hash": self.source_hash,
            "words": [word.to_dict() for word in self.words],
        }


def _word_id(video_id: str, source_hash: str, index: int, word: TranscriptWord) -> str:
    payload = f"{source_hash}:{word.start:.6f}:{word.end:.6f}:{word.text}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{video_id}:w{index:07d}:{digest}"


def _interpolated_words(segment: TranscriptSegment) -> tuple[TranscriptWord, ...]:
    tokens = segment.text.split()
    if not tokens:
        return ()
    width = segment.duration / len(tokens)
    return tuple(
        TranscriptWord(segment.start + index * width, segment.start + (index + 1) * width, token)
        for index, token in enumerate(tokens)
    )


def canonical_timeline_from_segments(
    video_id: str,
    source_hash: str,
    segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...],
    *,
    transcript_source: str,
) -> CanonicalTimeline:
    canonical: list[CanonicalWord] = []
    index = 0
    for segment in segments:
        words = segment.words or _interpolated_words(segment)
        timing_mode: TimingMode = "word_exact" if segment.words else "cue_interpolated"
        for word in words:
            canonical.append(
                CanonicalWord(
                    word_id=_word_id(video_id, source_hash, index, word),
                    text=word.text,
                    source_start=word.start,
                    source_end=word.end,
                    speaker_id=None,
                    confidence=None,
                    timing_mode=timing_mode,
                    transcript_source=transcript_source,
                )
            )
            index += 1
    return CanonicalTimeline(video_id=video_id, source_hash=source_hash, words=tuple(canonical))

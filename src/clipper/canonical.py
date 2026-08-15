from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

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

    def word_ref(self, word_id: str) -> str:
        word = self.word(word_id)
        prefix = f"{self.video_id}:"
        if word.word_id.startswith(prefix):
            tail = word.word_id[len(prefix) :]
            compact, separator, digest = tail.partition(":")
            if separator and digest and compact.startswith("w") and compact[1:].isdigit():
                return compact
        return word.word_id

    def resolve_word_ref(self, ref: str) -> str:
        clean = ref.strip()
        if not clean:
            raise ValueError("canonical word reference cannot be empty")
        exact = [word.word_id for word in self.words if word.word_id == clean]
        if exact:
            return exact[0]

        compact = clean
        video_prefix = f"{self.video_id}:"
        if compact.startswith(video_prefix):
            compact = compact[len(video_prefix) :]
        base, separator, _suffix = compact.partition(":")
        if separator and base.startswith("w") and base[1:].isdigit():
            compact = base

        candidates = [word.word_id for word in self.words if self.word_ref(word.word_id) == compact]
        if not candidates:
            raise ValueError(f"unknown canonical word reference: {clean}")
        if len(candidates) != 1:
            raise ValueError(f"ambiguous canonical word reference: {clean}")
        return candidates[0]

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CanonicalTimeline:
        raw_words = payload.get("words")
        if not isinstance(raw_words, list):
            raise ValueError("canonical timeline words must be a list")
        words: list[CanonicalWord] = []
        for raw in raw_words:
            if not isinstance(raw, dict):
                raise ValueError("canonical timeline word must be an object")
            mode = str(raw.get("timing_mode") or "")
            if mode not in {"word_exact", "aligned", "cue_interpolated"}:
                raise ValueError(f"unsupported canonical timing mode: {mode}")
            confidence_raw = raw.get("confidence")
            words.append(
                CanonicalWord(
                    word_id=str(raw.get("word_id") or ""),
                    text=str(raw.get("text") or ""),
                    source_start=float(raw.get("source_start") or 0.0),
                    source_end=float(raw.get("source_end") or 0.0),
                    speaker_id=(
                        str(raw["speaker_id"]) if raw.get("speaker_id") is not None else None
                    ),
                    confidence=(float(confidence_raw) if confidence_raw is not None else None),
                    timing_mode=cast(TimingMode, mode),
                    transcript_source=str(raw.get("transcript_source") or "unknown"),
                )
            )
        return cls(
            video_id=str(payload.get("video_id") or ""),
            source_hash=str(payload.get("source_hash") or ""),
            words=tuple(words),
            schema_version=str(payload.get("schema_version") or "canonical-timeline-v1"),
        )


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


def canonical_timeline_from_word_payloads(
    video_id: str,
    source_hash: str,
    words: list[dict[str, Any]],
    *,
    transcript_source: str,
) -> CanonicalTimeline:
    canonical: list[CanonicalWord] = []
    for index, raw in enumerate(words):
        text = str(raw.get("text") or "").strip()
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end") or 0.0)
        if not text or end <= start:
            continue
        source_word = TranscriptWord(start, end, text)
        confidence_raw = raw.get("confidence")
        canonical.append(
            CanonicalWord(
                word_id=_word_id(video_id, source_hash, index, source_word),
                text=text,
                source_start=start,
                source_end=end,
                speaker_id=None,
                confidence=float(confidence_raw) if confidence_raw is not None else None,
                timing_mode="word_exact",
                transcript_source=transcript_source,
            )
        )
    if not canonical:
        raise ValueError("word payloads produced no canonical words")
    return CanonicalTimeline(video_id=video_id, source_hash=source_hash, words=tuple(canonical))


def transcript_segments_from_canonical(
    timeline: CanonicalTimeline,
    *,
    max_gap_seconds: float = 1.0,
    max_words: int = 28,
) -> list[TranscriptSegment]:
    if max_gap_seconds < 0 or max_words <= 0:
        raise ValueError("canonical transcript grouping settings are invalid")
    if not timeline.words:
        return []
    groups: list[list[CanonicalWord]] = []
    current: list[CanonicalWord] = []
    previous: CanonicalWord | None = None
    for word in timeline.words:
        split = bool(
            current
            and previous is not None
            and (
                word.source_start - previous.source_end > max_gap_seconds
                or len(current) >= max_words
                or (
                    word.speaker_id is not None
                    and previous.speaker_id is not None
                    and word.speaker_id != previous.speaker_id
                )
            )
        )
        if split:
            groups.append(current)
            current = []
        current.append(word)
        previous = word
    if current:
        groups.append(current)
    return [
        TranscriptSegment(
            start=group[0].source_start,
            end=group[-1].source_end,
            text=" ".join(word.text for word in group),
            words=tuple(
                TranscriptWord(word.source_start, word.source_end, word.text) for word in group
            ),
            speaker_id=(
                group[0].speaker_id
                if group[0].speaker_id is not None
                and all(word.speaker_id == group[0].speaker_id for word in group)
                else None
            ),
        )
        for group in groups
    ]

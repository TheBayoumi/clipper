from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .canonical import CanonicalTimeline


@dataclass(frozen=True, slots=True)
class SemanticCore:
    core_id: str
    video_id: str
    source_hash: str
    source_start: float
    source_end: float
    source_word_ids: tuple[str, ...]
    semantic_summary: str
    editorial_reason: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.core_id.strip() or not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("semantic core requires stable source identity")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("semantic core timestamps are invalid")
        if not self.source_word_ids:
            raise ValueError("semantic core requires source word provenance")
        if not self.semantic_summary.strip() or not self.editorial_reason.strip():
            raise ValueError("semantic core requires summary and editorial reason")
        if not 0 <= self.confidence <= 1:
            raise ValueError("semantic core confidence must be between 0 and 1")

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_word_ids"] = list(self.source_word_ids)
        return payload

    @classmethod
    def from_word_ids(
        cls,
        timeline: CanonicalTimeline,
        *,
        core_id: str,
        source_word_ids: tuple[str, ...],
        semantic_summary: str,
        editorial_reason: str,
        confidence: float,
    ) -> SemanticCore:
        words = timeline.require_word_ids(source_word_ids)
        if not words:
            raise ValueError("semantic core requires source words")
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        indexes = [positions[word.word_id] for word in words]
        if indexes != list(range(min(indexes), max(indexes) + 1)):
            raise ValueError("semantic core source words must be contiguous and chronological")
        return cls(
            core_id=core_id,
            video_id=timeline.video_id,
            source_hash=timeline.source_hash,
            source_start=words[0].source_start,
            source_end=words[-1].source_end,
            source_word_ids=source_word_ids,
            semantic_summary=semantic_summary,
            editorial_reason=editorial_reason,
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class NarrativeEnvelope:
    envelope_id: str
    core_id: str
    video_id: str
    source_hash: str
    source_start: float
    source_end: float
    source_word_ids: tuple[str, ...]
    required_prior_context: str
    required_followup_context: str
    setup_resolved: bool
    payoff_resolved: bool
    reference_resolution: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.envelope_id.strip() or not self.core_id.strip():
            raise ValueError("narrative envelope requires envelope_id and core_id")
        if not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("narrative envelope requires stable source identity")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("narrative envelope timestamps are invalid")
        if not self.source_word_ids:
            raise ValueError("narrative envelope requires source word provenance")
        if not 0 <= self.confidence <= 1:
            raise ValueError("narrative envelope confidence must be between 0 and 1")

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def complete(self) -> bool:
        return self.setup_resolved and self.payoff_resolved

    def contains(self, core: SemanticCore) -> bool:
        return (
            self.core_id == core.core_id
            and self.video_id == core.video_id
            and self.source_hash == core.source_hash
            and self.source_start <= core.source_start
            and self.source_end >= core.source_end
            and set(core.source_word_ids).issubset(self.source_word_ids)
        )

    def require_contains(self, core: SemanticCore) -> None:
        if not self.contains(core):
            raise ValueError("narrative envelope does not contain its semantic core")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_word_ids"] = list(self.source_word_ids)
        payload["reference_resolution"] = list(self.reference_resolution)
        payload["complete"] = self.complete
        return payload

    @classmethod
    def from_word_ids(
        cls,
        timeline: CanonicalTimeline,
        core: SemanticCore,
        *,
        envelope_id: str,
        source_word_ids: tuple[str, ...],
        required_prior_context: str = "",
        required_followup_context: str = "",
        setup_resolved: bool,
        payoff_resolved: bool,
        reference_resolution: tuple[str, ...] = (),
        confidence: float,
    ) -> NarrativeEnvelope:
        words = timeline.require_word_ids(source_word_ids)
        if not words:
            raise ValueError("narrative envelope requires source words")
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        indexes = [positions[word.word_id] for word in words]
        if indexes != list(range(min(indexes), max(indexes) + 1)):
            raise ValueError("narrative envelope source words must be contiguous and chronological")
        envelope = cls(
            envelope_id=envelope_id,
            core_id=core.core_id,
            video_id=timeline.video_id,
            source_hash=timeline.source_hash,
            source_start=words[0].source_start,
            source_end=words[-1].source_end,
            source_word_ids=source_word_ids,
            required_prior_context=required_prior_context,
            required_followup_context=required_followup_context,
            setup_resolved=setup_resolved,
            payoff_resolved=payoff_resolved,
            reference_resolution=reference_resolution,
            confidence=confidence,
        )
        envelope.require_contains(core)
        return envelope

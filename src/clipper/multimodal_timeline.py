from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from .canonical import CanonicalTimeline
from .visual import VisualTimeline


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    provider: str
    model_id: str
    revision: str = "unknown"
    contract: str = "unknown"

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model_id.strip():
            raise ValueError("evidence provenance requires provider and model_id")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultimodalEvent:
    start: float
    end: float
    transcript_word_ids: tuple[str, ...] = ()
    speaker_ids: tuple[str, ...] = ()
    scene_ids: tuple[str, ...] = ()
    visible_people: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    ocr_text: tuple[str, ...] = ()
    branding: tuple[str, ...] = ()
    hazards: tuple[str, ...] = ()
    audio_events: tuple[str, ...] = ()
    visual_summaries: tuple[str, ...] = ()
    visual_salience: float = 0.0
    motion_salience: float = 0.0
    confidence: float = 0.0
    provenance: tuple[EvidenceProvenance, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("multimodal event timestamps are invalid")
        for value, label in (
            (self.visual_salience, "visual_salience"),
            (self.motion_salience, "motion_salience"),
            (self.confidence, "confidence"),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"multimodal event {label} must be between 0 and 1")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = [item.to_dict() for item in self.provenance]
        return payload


@dataclass(frozen=True, slots=True)
class MultimodalTimeline:
    video_id: str
    source_hash: str
    duration: float
    events: tuple[MultimodalEvent, ...]
    schema_version: str = "multimodal-timeline-v1"

    def __post_init__(self) -> None:
        if not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("multimodal timeline requires video_id and source_hash")
        if self.duration < 0:
            raise ValueError("multimodal timeline duration cannot be negative")
        if any(a.start > b.start for a, b in pairwise(self.events)):
            raise ValueError("multimodal timeline events must be source ordered")
        if any(event.end > self.duration + 1e-6 for event in self.events):
            raise ValueError("multimodal event exceeds source duration")

    def overlapping(self, start: float, end: float) -> tuple[MultimodalEvent, ...]:
        if start < 0 or end <= start:
            raise ValueError("multimodal query timestamps are invalid")
        return tuple(event for event in self.events if event.end > start and event.start < end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "source_hash": self.source_hash,
            "duration": self.duration,
            "events": [event.to_dict() for event in self.events],
        }


def _labels(labels: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    marker = f"{prefix}:"
    return tuple(label[len(marker) :].strip() for label in labels if label.startswith(marker))


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def build_multimodal_timeline(
    speech: CanonicalTimeline,
    visual: VisualTimeline | None,
    *,
    duration: float | None = None,
    visual_provenance: EvidenceProvenance | None = None,
) -> MultimodalTimeline:
    """Align canonical speech and visual evidence on deterministic source-time boundaries."""

    if visual is not None:
        if visual.video_id != speech.video_id:
            raise ValueError("speech and visual timelines reference different videos")
        if visual.source_hash != speech.source_hash:
            raise ValueError("speech and visual timelines reference different source hashes")

    visual_events = visual.events if visual is not None else ()
    source_end = max(
        speech.end,
        max((event.end for event in visual_events), default=0.0),
        float(duration or 0.0),
    )
    if source_end <= 0:
        return MultimodalTimeline(speech.video_id, speech.source_hash, 0.0, ())

    boundaries = {0.0, source_end}
    for word in speech.words:
        boundaries.add(word.source_start)
        boundaries.add(word.source_end)
    for event in visual_events:
        boundaries.add(event.start)
        boundaries.add(event.end)
    ordered = sorted(value for value in boundaries if 0 <= value <= source_end)

    events: list[MultimodalEvent] = []
    for start, end in pairwise(ordered):
        if end <= start:
            continue
        words = tuple(
            word for word in speech.words if word.source_end > start and word.source_start < end
        )
        visuals = tuple(event for event in visual_events if event.end > start and event.start < end)
        if not words and not visuals:
            continue

        labels = tuple(label for event in visuals for label in event.event_labels)
        visible_people = [speaker for event in visuals for speaker in event.visible_speakers]
        summaries = [event.summary for event in visuals]
        visual_confidences = [event.confidence for event in visuals]
        speech_confidences = [word.confidence for word in words if word.confidence is not None]
        confidence_values = [*visual_confidences, *speech_confidences]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        visual_salience = max(visual_confidences, default=0.0)
        motion_labels = _labels(labels, "motion") + _labels(labels, "action")
        motion_salience = visual_salience if motion_labels else 0.0
        provenance = (visual_provenance,) if visuals and visual_provenance is not None else ()

        events.append(
            MultimodalEvent(
                start=start,
                end=end,
                transcript_word_ids=tuple(word.word_id for word in words),
                speaker_ids=_unique(
                    [word.speaker_id for word in words if word.speaker_id is not None]
                ),
                scene_ids=_unique([event.scene_id for event in visuals]),
                visible_people=_unique(visible_people),
                actions=_unique(list(_labels(labels, "action"))),
                objects=_unique(list(_labels(labels, "object"))),
                ocr_text=_unique(list(_labels(labels, "ocr"))),
                branding=_unique(list(_labels(labels, "branding"))),
                hazards=_unique(list(_labels(labels, "hazard"))),
                audio_events=_unique(list(_labels(labels, "audio"))),
                visual_summaries=_unique(summaries),
                visual_salience=visual_salience,
                motion_salience=motion_salience,
                confidence=max(0.0, min(1.0, confidence)),
                provenance=provenance,
            )
        )

    return MultimodalTimeline(
        video_id=speech.video_id,
        source_hash=speech.source_hash,
        duration=source_end,
        events=tuple(events),
    )

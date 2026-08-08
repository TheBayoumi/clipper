from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class VisualEvent:
    start: float
    end: float
    scene_id: str
    summary: str
    visible_speakers: tuple[str, ...] = ()
    event_labels: tuple[str, ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("visual event timestamps are invalid")
        if not self.scene_id.strip() or not self.summary.strip():
            raise ValueError("visual event requires scene_id and summary")
        if not 0 <= self.confidence <= 1:
            raise ValueError("visual event confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["visible_speakers"] = list(self.visible_speakers)
        data["event_labels"] = list(self.event_labels)
        return data


@dataclass(frozen=True, slots=True)
class VisualTimeline:
    video_id: str
    source_hash: str
    events: tuple[VisualEvent, ...]
    schema_version: str = "visual-timeline-v1"

    def __post_init__(self) -> None:
        if not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("visual timeline requires video_id and source_hash")
        if any(a.start > b.start for a, b in zip(self.events, self.events[1:], strict=False)):
            raise ValueError("visual timeline events must be source ordered")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "source_hash": self.source_hash,
            "events": [event.to_dict() for event in self.events],
        }

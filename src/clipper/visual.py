from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .stage_contracts import structural_contract_fingerprint


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
    contract_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_fingerprint",
            structural_contract_fingerprint(
                "visual-timeline",
                VisualEvent,
                VisualTimeline,
                exclude_fields=("contract_fingerprint",),
            ),
        )
        if not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("visual timeline requires video_id and source_hash")
        if any(a.start > b.start for a, b in zip(self.events, self.events[1:], strict=False)):
            raise ValueError("visual timeline events must be source ordered")

    @property
    def schema_version(self) -> str:
        return self.contract_fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_fingerprint": self.contract_fingerprint,
            "video_id": self.video_id,
            "source_hash": self.source_hash,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VisualTimeline:
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("visual timeline events must be a list")
        events: list[VisualEvent] = []
        for raw in raw_events:
            if not isinstance(raw, dict):
                raise ValueError("visual timeline event must be an object")
            visible = raw.get("visible_speakers", [])
            labels = raw.get("event_labels", [])
            if not isinstance(visible, list) or not all(isinstance(item, str) for item in visible):
                raise ValueError("visual event visible_speakers must be strings")
            if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
                raise ValueError("visual event event_labels must be strings")
            events.append(
                VisualEvent(
                    start=float(raw.get("start") or 0.0),
                    end=float(raw.get("end") or 0.0),
                    scene_id=str(raw.get("scene_id") or ""),
                    summary=str(raw.get("summary") or ""),
                    visible_speakers=tuple(visible),
                    event_labels=tuple(labels),
                    confidence=float(raw.get("confidence") or 0.0),
                )
            )
        timeline = cls(
            video_id=str(payload.get("video_id") or ""),
            source_hash=str(payload.get("source_hash") or ""),
            events=tuple(events),
        )
        supplied = payload.get("contract_fingerprint")
        if supplied is not None and str(supplied) != timeline.contract_fingerprint:
            raise ValueError("visual timeline contract fingerprint does not match runtime contract")
        return timeline

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .multimodal_timeline import MultimodalEvent, MultimodalTimeline
from .quality_moments import QualityMoment

VisualSource = Literal[
    "original_source",
    "authorized_source_insert",
    "deterministic_graphic",
    "synthetic_illustration",
]


@dataclass(frozen=True, slots=True)
class VisualBeat:
    start: float
    end: float
    source: VisualSource
    rationale: str
    source_event_range: tuple[float, float] | None = None
    synthetic_request: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("visual beat timestamps are invalid")
        if not self.rationale.strip():
            raise ValueError("visual beat requires rationale")
        if self.source == "synthetic_illustration" and not self.synthetic_request:
            raise ValueError("synthetic visual beat requires an illustration request")
        if self.source != "synthetic_illustration" and self.synthetic_request is not None:
            raise ValueError("non-synthetic visual beat cannot carry a synthetic request")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualStrategy:
    quality_moment_id: str
    beats: tuple[VisualBeat, ...]
    source_first: bool = True
    schema_version: str = "visual-strategy-v1"

    def __post_init__(self) -> None:
        if not self.quality_moment_id.strip():
            raise ValueError("visual strategy requires quality_moment_id")
        if not self.beats:
            raise ValueError("visual strategy requires at least one beat")
        if any(a.end > b.start + 1e-6 for a, b in zip(self.beats, self.beats[1:], strict=False)):
            raise ValueError("visual strategy beats must be chronological and non-overlapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_moment_id": self.quality_moment_id,
            "source_first": self.source_first,
            "schema_version": self.schema_version,
            "beats": [beat.to_dict() for beat in self.beats],
        }


def _relevant_events(
    moment: QualityMoment,
    timeline: MultimodalTimeline,
) -> tuple[MultimodalEvent, ...]:
    window = moment.delivery_window
    return timeline.overlapping(window.source_start, window.source_end)


def derive_visual_strategy(
    moment: QualityMoment,
    timeline: MultimodalTimeline,
) -> VisualStrategy:
    """Prefer truthful original-source evidence before any enrichment mechanism."""

    window = moment.delivery_window
    if timeline.video_id != window.video_id or timeline.source_hash != window.source_hash:
        raise ValueError("quality moment and multimodal timeline reference different sources")
    relevant = _relevant_events(moment, timeline)
    visually_grounded = tuple(
        event
        for event in relevant
        if event.scene_ids
        or event.visible_people
        or event.actions
        or event.objects
        or event.ocr_text
        or event.visual_summaries
    )

    if visually_grounded:
        start = min(event.start for event in visually_grounded)
        end = max(event.end for event in visually_grounded)
        beat = VisualBeat(
            start=window.source_start,
            end=window.source_end,
            source="original_source",
            rationale="quality moment has relevant original visual evidence",
            source_event_range=(max(window.source_start, start), min(window.source_end, end)),
        )
    else:
        beat = VisualBeat(
            start=window.source_start,
            end=window.source_end,
            source="original_source",
            rationale="no truthful enrichment is needed; remain on original source",
            source_event_range=(window.source_start, window.source_end),
        )
    return VisualStrategy(moment.quality_moment_id, (beat,))

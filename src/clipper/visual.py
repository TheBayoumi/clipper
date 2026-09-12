from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any, Literal, cast

from .stage_contracts import structural_contract_fingerprint

VisualEvidenceScope = Literal["source_policy", "candidate_editorial"]


@dataclass(frozen=True, slots=True)
class VisualEvidenceSpan:
    start: float
    end: float
    sample_time: float
    scope: VisualEvidenceScope
    method: str = "representative_frame_cell"

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("visual evidence span timestamps are invalid")
        if not self.start <= self.sample_time <= self.end:
            raise ValueError("visual evidence sample time must lie inside its span")
        if self.scope not in {"source_policy", "candidate_editorial"}:
            raise ValueError("visual evidence scope is invalid")
        if not self.method.strip():
            raise ValueError("visual evidence span requires a method")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _merged_coverage_seconds(
    spans: tuple[VisualEvidenceSpan, ...],
    *,
    duration: float,
) -> float:
    clipped = sorted(
        (
            max(0.0, span.start),
            min(duration, span.end),
        )
        for span in spans
        if span.end > 0 and span.start < duration
    )
    if not clipped:
        return 0.0
    merged_start, merged_end = clipped[0]
    covered = 0.0
    for start, end in clipped[1:]:
        if end <= start:
            continue
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        covered += max(0.0, merged_end - merged_start)
        merged_start, merged_end = start, end
    covered += max(0.0, merged_end - merged_start)
    return min(duration, covered)


@dataclass(frozen=True, slots=True)
class VisualTimeline:
    video_id: str
    source_hash: str
    events: tuple[VisualEvent, ...]
    coverage_spans: tuple[VisualEvidenceSpan, ...] = ()
    source_duration: float = 0.0
    contract_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contract_fingerprint",
            structural_contract_fingerprint(
                "visual-timeline",
                VisualEvidenceSpan,
                VisualEvent,
                VisualTimeline,
                exclude_fields=("contract_fingerprint",),
            ),
        )
        if not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("visual timeline requires video_id and source_hash")
        if self.source_duration < 0:
            raise ValueError("visual timeline source duration cannot be negative")
        if any(a.start > b.start for a, b in zip(self.events, self.events[1:], strict=False)):
            raise ValueError("visual timeline events must be source ordered")
        if any(a.start > b.start for a, b in pairwise(self.coverage_spans)):
            raise ValueError("visual evidence spans must be source ordered")
        if any(span.end <= span.start for span in self.coverage_spans):
            raise ValueError("visual evidence spans must have positive duration")
        if self.source_duration > 0:
            if any(event.end > self.source_duration + 1e-6 for event in self.events):
                raise ValueError("visual event exceeds source duration")
            if any(span.end > self.source_duration + 1e-6 for span in self.coverage_spans):
                raise ValueError("visual evidence span exceeds source duration")

    @property
    def schema_version(self) -> str:
        return self.contract_fingerprint

    def coverage_summary(
        self,
        scope: VisualEvidenceScope,
        *,
        duration: float | None = None,
    ) -> dict[str, float | int | str]:
        target_duration = float(duration or self.source_duration or 0.0)
        if target_duration <= 0:
            target_duration = max(
                max((event.end for event in self.events), default=0.0),
                max((span.end for span in self.coverage_spans), default=0.0),
            )
        selected = tuple(span for span in self.coverage_spans if span.scope == scope)
        covered = (
            _merged_coverage_seconds(selected, duration=target_duration)
            if target_duration > 0
            else 0.0
        )
        samples = sorted({span.sample_time for span in selected})
        if target_duration > 0 and samples:
            gaps = [max(0.0, samples[0])]
            gaps.extend(max(0.0, right - left) for left, right in pairwise(samples))
            gaps.append(max(0.0, target_duration - samples[-1]))
            max_gap = max(gaps, default=target_duration)
        else:
            max_gap = target_duration
        return {
            "scope": scope,
            "source_duration_seconds": target_duration,
            "covered_seconds": covered,
            "coverage_fraction": covered / target_duration if target_duration > 0 else 0.0,
            "sample_count": len(samples),
            "max_sample_gap_seconds": max_gap,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_fingerprint": self.contract_fingerprint,
            "video_id": self.video_id,
            "source_hash": self.source_hash,
            "source_duration": self.source_duration,
            "coverage_spans": [span.to_dict() for span in self.coverage_spans],
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
        raw_spans = payload.get("coverage_spans", [])
        if not isinstance(raw_spans, list):
            raise ValueError("visual timeline coverage_spans must be a list")
        spans: list[VisualEvidenceSpan] = []
        for raw in raw_spans:
            if not isinstance(raw, dict):
                raise ValueError("visual evidence span must be an object")
            scope = str(raw.get("scope") or "")
            if scope not in {"source_policy", "candidate_editorial"}:
                raise ValueError("visual evidence scope is invalid")
            spans.append(
                VisualEvidenceSpan(
                    start=float(raw.get("start") or 0.0),
                    end=float(raw.get("end") or 0.0),
                    sample_time=float(raw.get("sample_time") or 0.0),
                    scope=cast(VisualEvidenceScope, scope),
                    method=str(raw.get("method") or "representative_frame_cell"),
                )
            )
        timeline = cls(
            video_id=str(payload.get("video_id") or ""),
            source_hash=str(payload.get("source_hash") or ""),
            events=tuple(events),
            coverage_spans=tuple(spans),
            source_duration=float(payload.get("source_duration") or 0.0),
        )
        supplied = payload.get("contract_fingerprint")
        if supplied is not None and str(supplied) != timeline.contract_fingerprint:
            raise ValueError("visual timeline contract fingerprint does not match runtime contract")
        return timeline

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from .multimodal_timeline import EvidenceProvenance, MultimodalEvent, MultimodalTimeline


def _json_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _nonempty_list(values: tuple[str, ...]) -> list[str] | None:
    return list(values) if values else None


def _project_event(event: MultimodalEvent, start: float, end: float) -> dict[str, object] | None:
    payload: dict[str, object] = {}

    for key, values in (
        ("scene_ids", event.scene_ids),
        ("visible_people", event.visible_people),
        ("actions", event.actions),
        ("objects", event.objects),
        ("ocr_text", event.ocr_text),
        ("branding", event.branding),
        ("hazards", event.hazards),
        ("audio_events", event.audio_events),
        ("visual_summaries", event.visual_summaries),
    ):
        projected = _nonempty_list(values)
        if projected is not None:
            payload[key] = projected

    if event.visual_salience > 0:
        payload["visual_salience"] = event.visual_salience
    if event.motion_salience > 0:
        payload["motion_salience"] = event.motion_salience

    # Speech, speaker identity, and word IDs are already represented exactly by the
    # canonical word payload. An event carrying only duplicated speech information
    # therefore contributes no independent multimodal evidence.
    if not payload:
        return None

    return {
        "start": max(start, event.start),
        "end": min(end, event.end),
        **payload,
    }


def _event_signature(payload: dict[str, object]) -> str:
    semantic = {key: value for key, value in payload.items() if key not in {"start", "end"}}
    return json.dumps(
        semantic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _provenance_payload(values: tuple[EvidenceProvenance, ...]) -> tuple[dict[str, str], ...]:
    unique: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for item in values:
        key = (item.provider, item.model_id, item.revision, item.contract)
        unique.setdefault(key, item.to_dict())
    return tuple(unique.values())


@dataclass(frozen=True, slots=True)
class EditorialEvidenceProjection:
    events: tuple[dict[str, object], ...]
    provenance: tuple[dict[str, str], ...]
    raw_event_count: int
    projected_event_count: int
    raw_serialized_bytes: int
    projected_serialized_bytes: int

    def telemetry(self, *, stage: str, start: float, end: float) -> dict[str, object]:
        return {
            "event": "editorial_evidence_projection",
            "stage": stage,
            "source_start": start,
            "source_end": end,
            "raw_event_count": self.raw_event_count,
            "projected_event_count": self.projected_event_count,
            "raw_serialized_bytes": self.raw_serialized_bytes,
            "projected_serialized_bytes": self.projected_serialized_bytes,
        }


def project_multimodal_evidence(
    multimodal: MultimodalTimeline | None,
    start: float,
    end: float,
) -> EditorialEvidenceProjection:
    """Project canonical multimodal evidence into a compact LLM-facing representation.

    The canonical timeline remains untouched. The projection removes speech duplicated by the
    word payload, omits empty/default fields, hoists repeated provenance, and coalesces adjacent
    intervals whose independent multimodal state is identical.
    """

    if multimodal is None:
        return EditorialEvidenceProjection((), (), 0, 0, 0, 0)

    raw_events = multimodal.overlapping(start, end)
    raw_payload = [event.to_dict() for event in raw_events]
    provenance = _provenance_payload(
        tuple(item for event in raw_events for item in event.provenance)
    )

    projected: list[dict[str, object]] = []
    previous_signature: str | None = None
    for event in raw_events:
        candidate = _project_event(event, start, end)
        if candidate is None:
            continue
        signature = _event_signature(candidate)
        candidate_start = cast(float, candidate["start"])
        candidate_end = cast(float, candidate["end"])
        if projected and previous_signature == signature:
            previous_end = cast(float, projected[-1]["end"])
            if candidate_start <= previous_end:
                projected[-1]["end"] = max(previous_end, candidate_end)
                continue
        projected.append(candidate)
        previous_signature = signature

    projected_payload: dict[str, Any] = {"events": projected}
    if provenance:
        projected_payload["provenance"] = provenance

    return EditorialEvidenceProjection(
        events=tuple(projected),
        provenance=provenance,
        raw_event_count=len(raw_events),
        projected_event_count=len(projected),
        raw_serialized_bytes=_json_size(raw_payload),
        projected_serialized_bytes=_json_size(projected_payload),
    )

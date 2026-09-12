from typing import Any, cast

import pytest

from clipper.visual import (
    VisualEvent,
    VisualEvidenceScope,
    VisualEvidenceSpan,
    VisualTimeline,
)


def _event(start: float = 0.0, end: float = 2.0) -> VisualEvent:
    return VisualEvent(
        start,
        end,
        "scene",
        "visible source evidence",
        visible_speakers=("speaker",),
        event_labels=("branding:example",),
        confidence=0.9,
    )


def test_visual_evidence_span_validates_identity_and_sample_cell() -> None:
    span = VisualEvidenceSpan(0.0, 4.0, 2.0, "source_policy")
    assert span.duration == pytest.approx(4.0)
    assert span.to_dict()["scope"] == "source_policy"

    with pytest.raises(ValueError, match="timestamps"):
        VisualEvidenceSpan(-1.0, 1.0, 0.0, "source_policy")
    with pytest.raises(ValueError, match="sample time"):
        VisualEvidenceSpan(0.0, 1.0, 2.0, "source_policy")
    with pytest.raises(ValueError, match="scope"):
        VisualEvidenceSpan(0.0, 1.0, 0.5, cast(VisualEvidenceScope, "invalid"))
    with pytest.raises(ValueError, match="method"):
        VisualEvidenceSpan(0.0, 1.0, 0.5, "source_policy", method=" ")


def test_visual_event_validation_remains_fail_closed() -> None:
    event = _event()
    payload = event.to_dict()
    assert payload["visible_speakers"] == ["speaker"]
    assert payload["event_labels"] == ["branding:example"]

    with pytest.raises(ValueError, match="timestamps"):
        VisualEvent(1.0, 1.0, "scene", "summary")
    with pytest.raises(ValueError, match="scene_id"):
        VisualEvent(0.0, 1.0, "", "summary")
    with pytest.raises(ValueError, match="confidence"):
        VisualEvent(0.0, 1.0, "scene", "summary", confidence=1.1)


def test_visual_timeline_coverage_is_union_of_explicit_inspection_spans() -> None:
    timeline = VisualTimeline(
        "video",
        "hash",
        (_event(0.0, 6.0),),
        coverage_spans=(
            VisualEvidenceSpan(0.0, 4.0, 2.0, "source_policy"),
            VisualEvidenceSpan(3.0, 6.0, 4.0, "source_policy"),
            VisualEvidenceSpan(8.0, 10.0, 9.0, "candidate_editorial"),
        ),
        source_duration=10.0,
    )
    policy = timeline.coverage_summary("source_policy")
    assert policy["covered_seconds"] == pytest.approx(6.0)
    assert policy["coverage_fraction"] == pytest.approx(0.6)
    assert policy["sample_count"] == 2
    assert policy["max_sample_gap_seconds"] == pytest.approx(6.0)

    candidate = timeline.coverage_summary("candidate_editorial")
    assert candidate["coverage_fraction"] == pytest.approx(0.2)
    assert timeline.schema_version == timeline.contract_fingerprint

    empty = VisualTimeline("video", "hash", ())
    empty_summary = empty.coverage_summary("source_policy")
    assert empty_summary["coverage_fraction"] == 0.0
    assert empty_summary["max_sample_gap_seconds"] == 0.0


def test_visual_timeline_round_trip_preserves_policy_coverage() -> None:
    timeline = VisualTimeline(
        "video",
        "hash",
        (_event(),),
        coverage_spans=(VisualEvidenceSpan(0.0, 2.0, 1.0, "source_policy"),),
        source_duration=2.0,
    )
    payload = timeline.to_dict()
    restored = VisualTimeline.from_dict(payload)
    assert restored == timeline
    assert restored.coverage_summary("source_policy")["coverage_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"video_id": "v", "source_hash": "h", "events": {}}, "events must be a list"),
        (
            {"video_id": "v", "source_hash": "h", "events": ["bad"]},
            "event must be an object",
        ),
        (
            {
                "video_id": "v",
                "source_hash": "h",
                "events": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "scene_id": "s",
                        "summary": "x",
                        "visible_speakers": "bad",
                    }
                ],
            },
            "visible_speakers",
        ),
        (
            {
                "video_id": "v",
                "source_hash": "h",
                "events": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "scene_id": "s",
                        "summary": "x",
                        "event_labels": "bad",
                    }
                ],
            },
            "event_labels",
        ),
        (
            {"video_id": "v", "source_hash": "h", "events": [], "coverage_spans": {}},
            "coverage_spans must be a list",
        ),
        (
            {"video_id": "v", "source_hash": "h", "events": [], "coverage_spans": ["bad"]},
            "span must be an object",
        ),
        (
            {
                "video_id": "v",
                "source_hash": "h",
                "events": [],
                "coverage_spans": [
                    {"start": 0.0, "end": 1.0, "sample_time": 0.5, "scope": "invalid"}
                ],
            },
            "scope is invalid",
        ),
    ],
)
def test_visual_timeline_from_dict_rejects_malformed_evidence(
    payload: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VisualTimeline.from_dict(payload)


def test_visual_timeline_rejects_invalid_source_bounds_and_fingerprint() -> None:
    with pytest.raises(ValueError, match="video_id"):
        VisualTimeline("", "hash", ())
    with pytest.raises(ValueError, match="duration cannot be negative"):
        VisualTimeline("video", "hash", (), source_duration=-1.0)
    with pytest.raises(ValueError, match="source ordered"):
        VisualTimeline("video", "hash", (_event(2.0, 3.0), _event(0.0, 1.0)))
    with pytest.raises(ValueError, match="visual event exceeds"):
        VisualTimeline("video", "hash", (_event(0.0, 2.0),), source_duration=1.0)
    with pytest.raises(ValueError, match="visual evidence span exceeds"):
        VisualTimeline(
            "video",
            "hash",
            (),
            coverage_spans=(VisualEvidenceSpan(0.0, 2.0, 1.0, "source_policy"),),
            source_duration=1.0,
        )

    payload = VisualTimeline("video", "hash", ()).to_dict()
    payload["contract_fingerprint"] = "stale"
    with pytest.raises(ValueError, match="fingerprint"):
        VisualTimeline.from_dict(payload)

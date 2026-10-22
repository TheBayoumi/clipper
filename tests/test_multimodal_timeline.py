import pytest

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.multimodal_timeline import (
    EvidenceProvenance,
    MultimodalEvent,
    MultimodalTimeline,
    build_multimodal_timeline,
)
from clipper.visual import VisualEvent, VisualEvidenceSpan, VisualTimeline


def _speech(source_hash: str = "source") -> CanonicalTimeline:
    return CanonicalTimeline(
        video_id="video",
        source_hash=source_hash,
        words=(
            CanonicalWord(
                "video:w0000000:a",
                "watch",
                0.0,
                0.5,
                "speaker-a",
                0.9,
                "word_exact",
                "test",
            ),
            CanonicalWord(
                "video:w0000001:b",
                "this",
                0.5,
                1.0,
                "speaker-a",
                0.8,
                "word_exact",
                "test",
            ),
        ),
    )


def _visual(source_hash: str = "source") -> VisualTimeline:
    return VisualTimeline(
        video_id="video",
        source_hash=source_hash,
        events=(
            VisualEvent(
                0.25,
                0.75,
                "scene-1",
                "a visible action",
                visible_speakers=("speaker-a",),
                event_labels=(
                    "action:turns wheel",
                    "motion:high",
                    "object:wheel",
                    "ocr:SPEED 42",
                    "branding:source-logo",
                    "hazard:foreign-logo",
                ),
                confidence=0.9,
            ),
        ),
        coverage_spans=(VisualEvidenceSpan(0.0, 1.0, 0.5, "source_policy"),),
        source_duration=1.0,
    )


def test_builder_aligns_words_and_visual_evidence_on_exact_boundaries() -> None:
    provenance = EvidenceProvenance("modal", "vlm", "rev", "visual-scout-v2")
    timeline = build_multimodal_timeline(
        _speech(),
        _visual(),
        duration=1.0,
        visual_provenance=provenance,
    )

    assert timeline.duration == 1.0
    assert timeline.visual_evidence_spans[0].scope == "source_policy"
    assert [(event.start, event.end) for event in timeline.events] == [
        (0.0, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
    ]
    middle = timeline.events[1]
    assert middle.transcript_word_ids == ("video:w0000000:a",)
    assert middle.actions == ("turns wheel",)
    assert middle.objects == ("wheel",)
    assert middle.ocr_text == ("SPEED 42",)
    assert middle.branding == ("source-logo",)
    assert middle.hazards == ("foreign-logo",)
    assert middle.visible_people == ("speaker-a",)
    assert middle.provenance == (provenance,)
    assert middle.motion_salience == 0.9


def test_builder_accepts_speech_only_and_empty_timelines() -> None:
    speech_only = build_multimodal_timeline(_speech(), None)
    assert speech_only.events
    assert all(not event.scene_ids for event in speech_only.events)

    empty = CanonicalTimeline("video", "source", ())
    result = build_multimodal_timeline(empty, None)
    assert result.duration == 0.0
    assert result.events == ()


def test_builder_rejects_mismatched_sources() -> None:
    with pytest.raises(ValueError, match="source hashes"):
        build_multimodal_timeline(_speech(), _visual("other"))

    wrong_video = VisualTimeline("other", "source", ())
    with pytest.raises(ValueError, match="different videos"):
        build_multimodal_timeline(_speech(), wrong_video)


def test_multimodal_timeline_validates_queries_and_serializes_provenance() -> None:
    event = MultimodalEvent(
        0.0,
        1.0,
        confidence=1.0,
        provenance=(EvidenceProvenance("test", "model"),),
    )
    timeline = MultimodalTimeline("video", "source", 1.0, (event,))
    assert timeline.overlapping(0.2, 0.3) == (event,)
    assert timeline.to_dict()["events"][0]["provenance"][0]["model_id"] == "model"

    with pytest.raises(ValueError, match="query"):
        timeline.overlapping(1.0, 1.0)
    with pytest.raises(ValueError, match="exceeds"):
        MultimodalTimeline("video", "source", 0.5, (event,))
    with pytest.raises(ValueError, match="confidence"):
        MultimodalEvent(0.0, 1.0, confidence=1.1)
    with pytest.raises(ValueError, match="provider"):
        EvidenceProvenance("", "model")

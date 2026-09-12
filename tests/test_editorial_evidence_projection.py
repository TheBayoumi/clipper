from __future__ import annotations

from pathlib import Path
from typing import Any

from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.editorial_evidence import project_multimodal_evidence
from clipper.multimodal_timeline import EvidenceProvenance, MultimodalEvent, MultimodalTimeline
from clipper.providers.base import ModelIdentity, ProviderResult
from clipper.source_hazards import SourceHazardClassifier


def _timeline() -> CanonicalTimeline:
    words = tuple(
        CanonicalWord(
            f"video:w{index:07d}:digest",
            f"word-{index}",
            float(index),
            float(index + 1),
            "speaker-a" if index < 2 else "speaker-b",
            0.99,
            "word_exact",
            "test",
        )
        for index in range(4)
    )
    return CanonicalTimeline("video", "source-hash", words)


def _multimodal() -> MultimodalTimeline:
    provenance = EvidenceProvenance(
        provider="modal",
        model_id="vision",
        revision="rev",
        contract="contract",
    )
    return MultimodalTimeline(
        "video",
        "source-hash",
        4.0,
        (
            MultimodalEvent(
                0.0,
                1.0,
                transcript_word_ids=("w0",),
                speaker_ids=("speaker-a",),
                scene_ids=("scene-host",),
                visible_people=("host",),
                objects=("microphone",),
                visual_summaries=("host at desk",),
                visual_salience=0.9,
                motion_salience=0.4,
                confidence=0.95,
                provenance=(provenance,),
            ),
            MultimodalEvent(
                1.0,
                2.0,
                transcript_word_ids=("w1",),
                speaker_ids=("speaker-a",),
                scene_ids=("scene-host",),
                visible_people=("host",),
                objects=("microphone",),
                visual_summaries=("host at desk",),
                visual_salience=0.9,
                motion_salience=0.4,
                confidence=0.70,
                provenance=(provenance,),
            ),
            MultimodalEvent(
                2.0,
                3.0,
                transcript_word_ids=("w2",),
                speaker_ids=("speaker-b",),
                confidence=0.42,
            ),
            MultimodalEvent(
                3.0,
                4.0,
                transcript_word_ids=("w3",),
                speaker_ids=("speaker-b",),
                hazards=("synthetic_overlay",),
                visual_summaries=("graphic overlay",),
                visual_salience=0.8,
                confidence=0.8,
                provenance=(provenance,),
            ),
        ),
    )


class _UnusedEditorial:
    identity = ModelIdentity("unused", "rev", "none", "test", "editor", "schema")

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        raise AssertionError((task, payload))


def test_projection_coalesces_redundant_word_aligned_visual_state() -> None:
    projected = project_multimodal_evidence(_multimodal(), 0.0, 4.0)

    assert projected.raw_event_count == 4
    assert projected.projected_event_count == 2
    assert projected.raw_serialized_bytes > projected.projected_serialized_bytes
    assert projected.provenance == (
        {
            "provider": "modal",
            "model_id": "vision",
            "revision": "rev",
            "contract": "contract",
        },
    )

    first, second = projected.events
    assert first["start"] == 0.0
    assert first["end"] == 2.0
    assert first["scene_ids"] == ["scene-host"]
    assert first["objects"] == ["microphone"]
    assert first["motion_salience"] == 0.4
    assert "transcript_word_ids" not in first
    assert "speaker_ids" not in first
    assert "provenance" not in first
    assert "confidence" not in first

    assert second["start"] == 3.0
    assert second["end"] == 4.0
    assert second["hazards"] == ["synthetic_overlay"]


def test_projection_drops_events_that_only_duplicate_canonical_speech() -> None:
    projected = project_multimodal_evidence(
        MultimodalTimeline(
            "video",
            "source-hash",
            1.0,
            (
                MultimodalEvent(
                    0.0,
                    1.0,
                    transcript_word_ids=("w0",),
                    speaker_ids=("speaker-a",),
                    confidence=0.9,
                ),
            ),
        ),
        0.0,
        1.0,
    )
    assert projected.raw_event_count == 1
    assert projected.projected_event_count == 0
    assert projected.events == ()


def test_source_hazard_uncached_payload_uses_projection_and_preserves_legacy_identity(
    tmp_path: Path,
) -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    multimodal = _multimodal()
    classifier = SourceHazardClassifier(_UnusedEditorial(), DagStore(tmp_path / "dag"))

    legacy = classifier._payload_for_range(brief, timeline, multimodal, 0, 4)
    projected, projection = classifier._projected_payload_for_range(
        brief,
        timeline,
        multimodal,
        0,
        4,
    )

    assert isinstance(legacy["multimodal_evidence"], list)
    assert len(legacy["multimodal_evidence"]) == 4
    assert "capacity_repartitionable" not in legacy

    assert projected["capacity_repartitionable"] is True
    assert projected["multimodal_evidence"] == list(projection.events)
    assert projected["multimodal_provenance"] == list(projection.provenance)
    assert len(projected["multimodal_evidence"]) < len(legacy["multimodal_evidence"])

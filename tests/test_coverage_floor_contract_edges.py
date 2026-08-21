from pathlib import Path

import pytest

from clipper.autonomous_quality_planner import AutonomousQualityPlanner
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.models import CampaignBrief
from clipper.multimodal_timeline import MultimodalTimeline
from clipper.providers.base import ModelIdentity, ProviderResult
from clipper.quality_batch import RecordingEditorialProvider
from clipper.source_hazards import SourceHazardClassifier
from clipper.stage_contracts import StageIdentity
from clipper.story_graph import NarrativeEnvelope, SemanticCore


class _UnusedEditorial:
    identity = ModelIdentity("unused", "rev", "none", "test", "editor", "schema")

    def complete_json(
        self, *, task: str, payload: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        raise AssertionError((task, payload))


def _timeline(count: int = 220) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:x",
                f"word-{index}",
                float(index),
                float(index + 1),
                "speaker",
                1.0,
                "word_exact",
                "test",
            )
            for index in range(count)
        ),
    )


def _acceptance_brief() -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "hazard-contract",
            "title": "Hazard contract",
            "objective": "Exercise fail-closed source policy",
            "keywords": ["policy"],
            "source_channel_ids": ["UC1"],
            "rights_confirmed": True,
            "min_clip_seconds": 8,
            "max_clip_seconds": 20,
            "clip_count": 1,
            "acceptance_policy": {"enabled": True},
        }
    )


def test_stage_identity_rejects_each_missing_required_identity_component() -> None:
    with pytest.raises(ValueError, match="stage, source, and contract hashes"):
        StageIdentity("", "source", "contract")
    with pytest.raises(ValueError, match="stage, source, and contract hashes"):
        StageIdentity("stage", "", "contract")
    with pytest.raises(ValueError, match="stage, source, and contract hashes"):
        StageIdentity("stage", "source", "")


def test_story_graph_factories_reject_empty_word_provenance() -> None:
    timeline = _timeline()
    with pytest.raises(ValueError, match="semantic core requires source words"):
        SemanticCore.from_word_ids(
            timeline,
            core_id="core",
            source_word_ids=(),
            semantic_summary="summary",
            editorial_reason="reason",
            confidence=0.9,
        )

    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=(timeline.words[100].word_id,),
        semantic_summary="summary",
        editorial_reason="reason",
        confidence=0.9,
    )
    with pytest.raises(ValueError, match="narrative envelope requires source words"):
        NarrativeEnvelope.from_word_ids(
            timeline,
            core,
            envelope_id="envelope",
            source_word_ids=(),
            setup_resolved=True,
            payoff_resolved=True,
            confidence=0.9,
        )


def test_planner_empty_chunks_and_wide_core_context_paths(tmp_path: Path) -> None:
    planner = AutonomousQualityPlanner(
        _UnusedEditorial(),
        DagStore(tmp_path / "dag"),
        max_words_per_chunk=200,
        chunk_overlap_words=20,
        envelope_context_words=100,
    )
    empty = CanonicalTimeline("empty-video", "empty-source", ())
    assert planner._chunks(empty) == ()

    timeline = _timeline()
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="wide-core",
        source_word_ids=tuple(word.word_id for word in timeline.words[50:171]),
        semantic_summary="wide complete event",
        editorial_reason="exercise wide context path",
        confidence=0.9,
    )
    assert planner._context_range(timeline, core) == (50, 171)


def test_recording_editorial_provider_reports_failures_to_progress_callback() -> None:
    progress: list[tuple[str, str]] = []
    recorder = RecordingEditorialProvider(
        _UnusedEditorial(),
        progress_callback=lambda task, state: progress.append((task, state)),
    )
    with pytest.raises(AssertionError):
        recorder.complete_json(task="semantic_cores:0", payload={})
    assert progress == [("semantic_cores:0", "running"), ("semantic_cores:0", "failed")]
    assert recorder.invocations == []


def test_source_hazard_classifier_rejects_bad_configuration_and_mismatched_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least 200"):
        SourceHazardClassifier(
            _UnusedEditorial(), DagStore(tmp_path / "small"), max_words_per_chunk=199
        )
    with pytest.raises(ValueError, match="overlap must be smaller"):
        SourceHazardClassifier(
            _UnusedEditorial(),
            DagStore(tmp_path / "overlap"),
            max_words_per_chunk=200,
            chunk_overlap_words=200,
        )

    classifier = SourceHazardClassifier(
        _UnusedEditorial(),
        DagStore(tmp_path / "valid"),
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    assert classifier._chunks(CanonicalTimeline("empty", "empty-source", ())) == ()
    mismatch = MultimodalTimeline("video", "different-source", 220.0, ())
    with pytest.raises(ValueError, match="different sources"):
        classifier.classify(_acceptance_brief(), _timeline(), multimodal=mismatch)


def test_source_hazard_classifier_fails_closed_when_model_inference_errors(tmp_path: Path) -> None:
    classifier = SourceHazardClassifier(
        _UnusedEditorial(),
        DagStore(tmp_path / "hazards"),
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    result = classifier.classify(_acceptance_brief(), _timeline(), multimodal=None)

    assert len(result.hazards) == 2
    assert len(result.rejections) == 2
    assert all(item.classification.value == "unknown" for item in result.hazards)
    assert all(item.evidence == ("source_hazard_classification_failed",) for item in result.hazards)
    assert all(item["decision"] == "ESCALATE" for item in result.rejections)
    assert all(item["reasons"] == ["policy_uncertain"] for item in result.rejections)

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from clipper.autonomous_quality_planner import AutonomousQualityPlanner
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.dag import DagStore
from clipper.modality_profile import _covered_duration
from clipper.models import CampaignBrief
from clipper.multimodal_timeline import MultimodalTimeline
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.quality_batch import RecordingEditorialProvider, plan_quality_batch
from clipper.source_hazards import SourceHazardClassifier
from clipper.stage_contracts import StageContract, StageIdentity
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.visual import VisualTimeline


class _UnusedEditorial:
    identity = ModelIdentity("unused", "rev", "none", "test", "editor", "schema")

    def complete_json(
        self, *, task: str, payload: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        raise AssertionError((task, payload))


class _StaticEditorial:
    identity = ModelIdentity("static", "rev", "none", "test", "editor", "schema")

    def __init__(self, value: Any) -> None:
        self.value = value

    def complete_json(self, *, task: str, payload: dict[str, object]) -> ProviderResult[Any]:
        del task, payload
        return ProviderResult(self.value, self.identity, InferenceUsage("test", "now", 0.0))


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
            "allowed_video_ids": ["video"],
            "rights_confirmed": True,
            "min_clip_seconds": 8,
            "max_clip_seconds": 20,
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


def test_stage_contract_serializes_fingerprint_and_rejects_empty_dependencies() -> None:
    contract = StageContract("quality", {"instruction": "rank legal windows"}, {"policy": "safe"})
    payload = contract.to_dict()
    assert payload["name"] == "quality"
    assert payload["contract_hash"] == contract.contract_hash
    with pytest.raises(ValueError, match="dependency output fingerprints cannot be empty"):
        StageIdentity(
            "quality",
            "source",
            contract.contract_hash,
            dependency_output_hashes=("",),
        )


def test_modality_coverage_requires_a_callable_predicate() -> None:
    timeline = MultimodalTimeline("video", "source", 1.0, ())
    with pytest.raises(TypeError, match="predicate must be callable"):
        _covered_duration(timeline, object())


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


@pytest.mark.parametrize("value", [[], {"segments": "not-a-list"}])
def test_source_hazard_classifier_fails_closed_on_invalid_model_shapes(
    tmp_path: Path, value: Any
) -> None:
    classifier = SourceHazardClassifier(
        _StaticEditorial(value),
        DagStore(tmp_path / f"invalid-{type(value).__name__}"),
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    result = classifier.classify(_acceptance_brief(), _timeline(10), multimodal=None)
    assert len(result.hazards) == 1
    assert result.hazards[0].classification.value == "unknown"
    assert result.rejections[0]["decision"] == "ESCALATE"


def test_source_hazard_classifier_fails_closed_when_segment_escapes_chunk(tmp_path: Path) -> None:
    classifier = SourceHazardClassifier(
        _StaticEditorial({"segments": [{}]}),
        DagStore(tmp_path / "escaped"),
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    with patch(
        "clipper.source_hazards.SourceHazardSegment.from_payload",
        return_value=SimpleNamespace(source_word_ids=("outside-chunk",)),
    ):
        result = classifier.classify(_acceptance_brief(), _timeline(10), multimodal=None)
    assert result.hazards[0].classification.value == "unknown"
    assert result.rejections[0]["decision"] == "ESCALATE"


def test_quality_batch_rejects_ineligible_quality_moment_without_quota_fill(tmp_path: Path) -> None:
    hazard_result = SimpleNamespace(
        rejections=(),
        hazards=(),
        stage_cache_hits=0,
        stage_executions=0,
        to_dict=lambda: {},
    )
    planning = SimpleNamespace(
        rejections=(),
        cores=(),
        quality_moments=(SimpleNamespace(quality_moment_id="qm-1"),),
        stage_cache_hits=0,
        stage_executions=0,
        to_dict=lambda: {},
    )
    with (
        patch("clipper.quality_batch._requires_source_visual_policy", return_value=False),
        patch("clipper.quality_batch.SourceHazardClassifier.classify", return_value=hazard_result),
        patch("clipper.quality_batch.AutonomousQualityPlanner.plan", return_value=planning),
        patch("clipper.quality_batch.adapt_quality_moment", return_value=None),
    ):
        result = plan_quality_batch(
            _acceptance_brief(),
            {"video": _timeline(10)},
            {},
            _UnusedEditorial(),
            dag_root=tmp_path / "batch",
            max_words_per_chunk=200,
            chunk_overlap_words=20,
        )

    assert result.plans == ()
    assert result.quality_moments == ()
    assert result.rejections[-1]["stage"] == "quality_moment_pre_render_eligibility"
    assert result.rejections[-1]["decision"] == "REJECT"


def test_quality_batch_fails_closed_on_insufficient_required_visual_coverage(
    tmp_path: Path,
) -> None:
    timeline = _timeline(10)
    visual = VisualTimeline("video", "source", ())
    with (
        patch("clipper.quality_batch._requires_source_visual_policy", return_value=True),
        pytest.raises(RuntimeError, match="broader source visual evidence coverage"),
    ):
        plan_quality_batch(
            _acceptance_brief(),
            {"video": timeline},
            {"video": visual},
            _UnusedEditorial(),
            dag_root=tmp_path / "visual-policy",
            max_words_per_chunk=200,
            chunk_overlap_words=20,
        )

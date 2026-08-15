from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import pytest

from clipper.ai_editorial import (
    EditorialGroundingError,
    EpisodeEditorialProfile,
    GroundedClipConcept,
    GroundedEditPlan,
    GroundedHookVariant,
    GroundedStoryMoment,
    source_spans_from_word_ids,
)
from clipper.autonomous_editor import AutonomousEditorialPlanner, OpenVideoAnalysis, _cosine
from clipper.cache import FileCache
from clipper.canonical import (
    CanonicalTimeline,
    CanonicalWord,
    canonical_timeline_from_segments,
    canonical_timeline_from_word_payloads,
    transcript_segments_from_canonical,
)
from clipper.editorial_integrity import (
    BoundaryAudit,
    BoundaryFailureReason,
    BoundaryStatus,
)
from clipper.models import (
    AcceptancePolicy,
    CampaignBrief,
    ClipConcept,
    EditorialScores,
    ProductionConfig,
    StoryMoment,
    TranscriptSegment,
    TranscriptWord,
)
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult, compute_profile
from clipper.providers.editorial_prompt import (
    EDITORIAL_PROMPT_VERSION,
    EDITORIAL_SCHEMA_VERSION,
    editorial_contract,
    editorial_output_budget,
)
from clipper.providers.factory import (
    editorial_and_embedding_providers,
    speech_providers,
    vision_provider,
)
from clipper.providers.local import (
    LocalEditorialProvider,
    LocalEmbeddingProvider,
    LocalVisionProvider,
    ProviderUnavailable,
)
from clipper.providers.modal import (
    ModalEditorialProvider,
    ModalEmbeddingProvider,
    ModalVisionProvider,
)
from clipper.providers.modal_endpoint import ModalEndpointEditorialProvider
from clipper.providers.modal_speech import (
    ModalAlignmentProvider,
    ModalDiarizationProvider,
    ModalMediaBridge,
    ModalTranscriptionProvider,
)
from clipper.providers.speech import (
    FasterWhisperTranscriptionProvider,
    PassthroughDiarizationProvider,
    PyannoteDiarizationProvider,
    WhisperXAlignmentProvider,
    _alignment_segments,
    apply_speaker_turns,
    apply_whisperx_alignment,
)
from clipper.visual import VisualEvent, VisualTimeline


def _timeline() -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source-hash",
        (
            CanonicalWord("w1", "what's", 10.0, 10.2, "A", 0.98, "aligned", "whisperx"),
            CanonicalWord("w2", "one", 10.21, 10.4, "A", 0.99, "aligned", "whisperx"),
            CanonicalWord("w3", "message", 10.41, 10.8, "A", 0.97, "aligned", "whisperx"),
            CanonicalWord("w4", "today", 10.81, 11.1, "A", 0.95, "aligned", "whisperx"),
        ),
    )


def _passing_boundary_payload() -> dict[str, object]:
    return {
        "start_status": "COMPLETE",
        "end_status": "COMPLETE",
        "standalone_status": "COMPLETE",
        "required_prior_context": "",
        "required_followup_context": "",
        "prior_context_included": True,
        "followup_context_included": True,
        "setup_resolved": True,
        "payoff_resolved": True,
        "open_questions": [],
        "open_references": [],
        "narrative_structure": "complete story",
        "boundary_confidence": 0.95,
        "failure_reasons": [],
        "repair_start_word_id": None,
        "repair_end_word_id": None,
    }


def test_canonical_timeline_exact_and_interpolated_ids_are_stable() -> None:
    segments = [
        TranscriptSegment(
            1.0,
            2.0,
            "hello world",
            (TranscriptWord(1.0, 1.4, "hello"), TranscriptWord(1.5, 2.0, "world")),
        ),
        TranscriptSegment(3.0, 5.0, "fallback cue"),
    ]
    one = canonical_timeline_from_segments("v", "hash", segments, transcript_source="youtube-vtt")
    two = canonical_timeline_from_segments("v", "hash", segments, transcript_source="youtube-vtt")
    assert one == two
    assert len(one.words) == 4
    assert [word.timing_mode for word in one.words] == [
        "word_exact",
        "word_exact",
        "cue_interpolated",
        "cue_interpolated",
    ]
    assert one.word(one.words[0].word_id).text == "hello"
    assert one.require_word_ids([one.words[-1].word_id])[0].text == "cue"
    with pytest.raises(KeyError):
        one.word("missing")
    with pytest.raises(ValueError, match="unknown canonical"):
        one.require_word_ids(["missing"])


def test_canonical_validation_is_immutable_and_source_ordered() -> None:
    with pytest.raises(ValueError, match="confidence"):
        CanonicalWord("w", "x", 0, 1, None, 2.0, "aligned", "x")
    with pytest.raises(ValueError, match="duplicate"):
        CanonicalTimeline(
            "v",
            "h",
            (
                CanonicalWord("same", "a", 0, 1, None, None, "aligned", "x"),
                CanonicalWord("same", "b", 1, 2, None, None, "aligned", "x"),
            ),
        )
    with pytest.raises(ValueError, match="source ordered"):
        CanonicalTimeline(
            "v",
            "h",
            (
                CanonicalWord("a", "a", 2, 3, None, None, "aligned", "x"),
                CanonicalWord("b", "b", 1, 2, None, None, "aligned", "x"),
            ),
        )


def test_canonical_word_refs_are_short_unique_and_resolve_hashed_ids() -> None:
    timeline = CanonicalTimeline(
        "video",
        "hash",
        (
            CanonicalWord(
                "video:w0000044:aaaaaaaaaaaa",
                "hello",
                1.0,
                1.2,
                None,
                None,
                "aligned",
                "test",
            ),
            CanonicalWord(
                "video:w0000045:bbbbbbbbbbbb",
                "world",
                1.21,
                1.4,
                None,
                None,
                "aligned",
                "test",
            ),
        ),
    )
    full = "video:w0000044:aaaaaaaaaaaa"
    assert timeline.word_ref(full) == "w0000044"
    assert timeline.resolve_word_ref("w0000044") == full
    assert timeline.resolve_word_ref("video:w0000044") == full
    assert timeline.resolve_word_ref(full) == full
    with pytest.raises(ValueError, match="unknown canonical word reference"):
        timeline.resolve_word_ref("w9999999")


def test_grounded_ai_contract_rejects_unknown_and_reordered_spoken_words() -> None:
    timeline = _timeline()
    with pytest.raises(ValueError, match="unknown canonical"):
        GroundedStoryMoment.from_payload(
            {
                "moment_id": "m",
                "supporting_word_ids": ["missing"],
                "semantic_summary": "summary",
                "narrative_structure": "answer",
                "editorial_reason": "self contained",
                "confidence": 0.9,
            },
            timeline,
        )
    with pytest.raises(EditorialGroundingError, match="chronology"):
        source_spans_from_word_ids(timeline, ("w2", "w1"))
    with pytest.raises(EditorialGroundingError, match="belong"):
        GroundedEditPlan.from_payload(
            {
                "plan_id": "p",
                "video_id": "video",
                "concept_id": "c",
                "variant_id": "v",
                "source_word_ids": ["w2", "w3", "w4"],
                "hook_source_word_ids": ["w1"],
                "strategy_label": "direct question",
                "caption_platform": "tiktok",
                "confidence": 0.9,
            },
            timeline,
        )


def test_compact_grounded_ranges_expand_to_exact_canonical_words() -> None:
    timeline = _timeline()
    moment = GroundedStoryMoment.from_payload(
        {
            "moment_id": "range-moment",
            "start_word_id": "w1",
            "end_word_id": "w4",
            "semantic_summary": "A compact grounded moment",
            "narrative_structure": "answer",
            "editorial_reason": "complete source range",
            "confidence": 0.9,
        },
        timeline,
    )
    concept = GroundedClipConcept.from_payload(
        {
            "concept_id": "range-concept",
            "story_moment_ids": ["range-moment"],
            "start_word_id": "w1",
            "end_word_id": "w4",
            "semantic_summary": "Compact concept",
            "standalone_context": "",
            "narrative_structure": "answer",
            "recommended_duration": 20,
            "visual_dependencies": [],
            "confidence": 0.85,
        },
        timeline,
    )
    hook = GroundedHookVariant.from_payload(
        {
            "variant_id": "range-hook",
            "strategy_label": "direct source hook",
            "source_start_word_id": "w1",
            "source_end_word_id": "w3",
            "overlay_text": None,
            "rationale": "grounded opening",
            "confidence": 0.8,
        },
        timeline,
    )
    plan = GroundedEditPlan.from_payload(
        {
            "plan_id": "range-plan",
            "video_id": "video",
            "concept_id": concept.concept_id,
            "variant_id": hook.variant_id,
            "source_start_word_id": "w1",
            "source_end_word_id": "w4",
            "hook_start_word_id": "w1",
            "hook_end_word_id": "w3",
            "overlay_text": None,
            "strategy_label": "direct source hook",
            "caption_platform": "tiktok",
            "confidence": 0.8,
        },
        timeline,
    )
    assert moment.supporting_word_ids == ("w1", "w2", "w3", "w4")
    assert concept.supporting_word_ids == ("w1", "w2", "w3", "w4")
    assert hook.source_word_ids == ("w1", "w2", "w3")
    assert plan.source_word_ids == ("w1", "w2", "w3", "w4")
    assert plan.hook_source_word_ids == ("w1", "w2", "w3")
    with pytest.raises(EditorialGroundingError, match="unknown canonical"):
        GroundedStoryMoment.from_payload(
            {
                "moment_id": "missing-range",
                "start_word_id": "missing",
                "end_word_id": "w4",
                "semantic_summary": "bad",
                "narrative_structure": "answer",
                "editorial_reason": "bad",
                "confidence": 0.5,
            },
            timeline,
        )
    with pytest.raises(EditorialGroundingError, match="chronology"):
        GroundedHookVariant.from_payload(
            {
                "variant_id": "reversed",
                "strategy_label": "bad",
                "source_start_word_id": "w4",
                "source_end_word_id": "w2",
                "overlay_text": None,
                "rationale": "bad",
                "confidence": 0.5,
            },
            timeline,
        )


def test_grounded_editorial_payloads_compile_without_fabricating_spoken_text() -> None:
    timeline = _timeline()
    profile = EpisodeEditorialProfile.from_payload(
        {
            "summary": "A reflective interview",
            "valuable_moment_characteristics": ["self-contained insight"],
            "avoid_characteristics": ["housekeeping"],
            "confidence": 0.9,
        }
    )
    moment = GroundedStoryMoment.from_payload(
        {
            "moment_id": "m1",
            "supporting_word_ids": ["w1", "w2", "w3", "w4"],
            "semantic_summary": "A direct question",
            "narrative_structure": "question",
            "editorial_reason": "clear standalone opening",
            "confidence": 0.9,
        },
        timeline,
    )
    concept = GroundedClipConcept.from_payload(
        {
            "concept_id": "c1",
            "story_moment_ids": ["m1"],
            "supporting_word_ids": ["w1", "w2", "w3", "w4"],
            "semantic_summary": "Question clip",
            "standalone_context": "none",
            "narrative_structure": "question",
            "recommended_duration": 20,
            "visual_dependencies": [],
            "confidence": 0.8,
        },
        timeline,
    )
    hook = GroundedHookVariant.from_payload(
        {
            "variant_id": "v1",
            "strategy_label": "start on source question",
            "source_word_ids": ["w1", "w2", "w3"],
            "overlay_text": "ONE QUESTION CHANGED THE CONVERSATION",
            "rationale": "adds curiosity without pretending it was spoken",
            "confidence": 0.8,
        },
        timeline,
    )
    grounded = GroundedEditPlan.from_payload(
        {
            "plan_id": "p1",
            "video_id": "video",
            "concept_id": concept.concept_id,
            "variant_id": hook.variant_id,
            "source_word_ids": ["w1", "w2", "w3", "w4"],
            "hook_source_word_ids": ["w1", "w2", "w3"],
            "overlay_text": hook.overlay_text,
            "strategy_label": hook.strategy_label,
            "caption_platform": "tiktok",
            "confidence": 0.8,
        },
        timeline,
    )
    plan = grounded.compile(timeline, "transcript-fp")
    assert profile.confidence == 0.9
    assert moment.to_dict()["supporting_word_ids"][0] == "w1"
    assert plan.caption_start_source_time == 10.0
    assert plan.caption_start_word == "what's"
    assert plan.hook_text == "ONE QUESTION CHANGED THE CONVERSATION"
    assert plan.hook_mode == "curiosity_text"
    assert plan.source_spans[0].start == 10.0
    assert plan.source_spans[0].end == 11.1


def test_model_identity_and_compute_profiles_are_reproducible() -> None:
    identity = ModelIdentity("model", "rev", "int8", "engine", "prompt", "schema")
    assert identity.cache_fingerprint(sampling={"temperature": 0}) == identity.cache_fingerprint(
        sampling={"temperature": 0}
    )
    assert identity.cache_fingerprint(sampling={"temperature": 0}) != identity.cache_fingerprint(
        sampling={"temperature": 1}
    )
    assert compute_profile("local-lite").editorial_location == "local"
    assert compute_profile("balanced").editorial_location == "modal"
    assert compute_profile("quality").allow_large_vlm_escalation is True


def test_visual_timeline_validates_order_and_serializes() -> None:
    event = VisualEvent(0, 1, "scene-1", "speaker reacts", ("A",), ("reaction",), 0.8)
    timeline = VisualTimeline("v", "hash", (event,))
    assert timeline.to_dict()["events"][0]["event_labels"] == ["reaction"]
    with pytest.raises(ValueError, match="confidence"):
        VisualEvent(0, 1, "scene", "x", confidence=2)
    with pytest.raises(ValueError, match="source ordered"):
        VisualTimeline(
            "v",
            "hash",
            (
                VisualEvent(2, 3, "b", "b"),
                VisualEvent(1, 2, "a", "a"),
            ),
        )


class _FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def numel(self) -> int:
        return len(self.values)

    def __getitem__(self, item):  # type: ignore[no-untyped-def]
        if isinstance(item, slice):
            return _FakeTensor(self.values[item])
        return self.values[item]


class _FakeInputs(dict[str, _FakeTensor]):
    def to(self, _device: str):
        return self


class _FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("return_dict"):
            return _FakeInputs({"input_ids": _FakeTensor([1, 2])})
        return "prompt"

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeInputs({"input_ids": _FakeTensor([1, 2])})

    def decode(self, _tensor, **kwargs):  # type: ignore[no-untyped-def]
        return '{"ok": true}'


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        return [_FakeTensor([1, 2, 3, 4])]


def test_local_editorial_provider_is_lazy_and_json_grounded() -> None:
    provider = LocalEditorialProvider()
    with patch.object(provider, "_load", return_value=(_FakeTokenizer(), _FakeModel())):
        result = provider.complete_json(task="story_mining", payload={"word_ids": ["w1"]})
    assert result.value == {"ok": True}
    assert result.usage.provider == "local"
    assert result.usage.input_units == 2
    assert result.usage.output_units == 2


def test_local_embedding_provider_uses_normalized_embeddings() -> None:
    provider = LocalEmbeddingProvider()
    model = Mock()
    model.encode.return_value = [[1.0, 0.0], [0.0, 1.0]]
    with patch.object(provider, "_load", return_value=model):
        result = provider.embed(["a", "b"])
    assert result.value == [[1.0, 0.0], [0.0, 1.0]]
    model.encode.assert_called_once_with(
        ["a", "b"], normalize_embeddings=True, convert_to_numpy=True
    )


def test_optional_provider_dependencies_fail_explicitly() -> None:
    provider = LocalEmbeddingProvider()
    with (
        patch("clipper.providers.local.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="embedding"),
    ):
        provider._load()


def test_local_vision_provider_json_path_without_real_model(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    provider = LocalVisionProvider()
    processor = _FakeTokenizer()
    model = _FakeModel()
    with patch.object(provider, "_load", return_value=(processor, model)):
        result = provider.inspect(task="review", frames=[frame], context={"clip": "x"})
    assert result.value == {"ok": True}
    with pytest.raises(ValueError, match="at least one frame"):
        provider.inspect(task="review", frames=[], context={})


def test_modal_adapters_validate_response_and_record_usage(tmp_path: Path) -> None:
    identity = ModelIdentity("m", "r", "none", "modal")
    function = Mock()
    function.remote.return_value = {
        "value": {"ok": True},
        "model": {"model_id": "resolved/model", "revision": "sha123"},
        "usage": {
            "started_at": "2026-08-08T00:00:00Z",
            "duration_seconds": 2.5,
            "gpu_type": "L40S",
            "gpu_seconds": 2.5,
            "peak_vram_mb": 100,
            "input_units": 10,
            "output_units": 3,
            "estimated_cost_usd": 0.01,
        },
    }
    vision = ModalVisionProvider(app_name="clipper", function_name="vision", identity=identity)
    frame = tmp_path / "a.jpg"
    frame.write_bytes(b"frame-bytes")
    with patch.object(vision, "_function", return_value=function):
        assert vision.inspect(task="review", frames=[frame], context={}).value["ok"] is True
        payload = function.remote.call_args.args[0]
        assert payload["frames_base64"]
        assert "frame_paths" not in payload


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: CanonicalWord("", "x", 0, 1, None, None, "aligned", "x"), "word_id"),
        (lambda: CanonicalWord("w", "", 0, 1, None, None, "aligned", "x"), "text"),
        (lambda: CanonicalWord("w", "x", 1, 1, None, None, "aligned", "x"), "timing"),
        (lambda: CanonicalTimeline("", "h", ()), "video_id"),
    ],
)
def test_canonical_rejects_invalid_identity_text_and_timing(factory, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=match):
        factory()


def test_canonical_empty_timeline_serializes() -> None:
    empty = CanonicalTimeline("v", "h", ())
    assert empty.start == 0.0 and empty.end == 0.0
    assert empty.to_dict()["words"] == []


def test_grounded_payload_validation_branches() -> None:
    timeline = _timeline()
    with pytest.raises(EditorialGroundingError, match="valuable"):
        EpisodeEditorialProfile.from_payload({"summary": "x", "confidence": 0.5})
    with pytest.raises(EditorialGroundingError, match="avoid"):
        EpisodeEditorialProfile.from_payload(
            {
                "summary": "x",
                "valuable_moment_characteristics": ["insight"],
                "avoid_characteristics": "bad",
                "confidence": 0.5,
            }
        )
    with pytest.raises(EditorialGroundingError, match="numeric"):
        EpisodeEditorialProfile.from_payload(
            {
                "summary": "x",
                "valuable_moment_characteristics": ["insight"],
                "confidence": None,
            }
        )
    with pytest.raises(EditorialGroundingError, match="between"):
        EpisodeEditorialProfile.from_payload(
            {
                "summary": "x",
                "valuable_moment_characteristics": ["insight"],
                "confidence": 2,
            }
        )
    with pytest.raises(EditorialGroundingError, match="overlay_text"):
        GroundedHookVariant.from_payload(
            {
                "variant_id": "v",
                "strategy_label": "s",
                "source_word_ids": ["w1"],
                "overlay_text": 123,
                "rationale": "r",
                "confidence": 0.5,
            },
            timeline,
        )
    with pytest.raises(EditorialGroundingError, match="story_moment_ids"):
        GroundedClipConcept.from_payload(
            {
                "concept_id": "c",
                "story_moment_ids": [],
                "supporting_word_ids": ["w1"],
                "semantic_summary": "x",
                "narrative_structure": "x",
                "recommended_duration": 10,
                "confidence": 0.5,
            },
            timeline,
        )
    with pytest.raises(EditorialGroundingError, match="visual_dependencies"):
        GroundedClipConcept.from_payload(
            {
                "concept_id": "c",
                "story_moment_ids": ["m"],
                "supporting_word_ids": ["w1"],
                "semantic_summary": "x",
                "narrative_structure": "x",
                "recommended_duration": 10,
                "visual_dependencies": "bad",
                "confidence": 0.5,
            },
            timeline,
        )
    with pytest.raises(EditorialGroundingError, match="recommended_duration"):
        GroundedClipConcept.from_payload(
            {
                "concept_id": "c",
                "story_moment_ids": ["m"],
                "supporting_word_ids": ["w1"],
                "semantic_summary": "x",
                "narrative_structure": "x",
                "recommended_duration": 0,
                "confidence": 0.5,
            },
            timeline,
        )
    bound_plan = GroundedEditPlan.from_payload(
        {
            "plan_id": "p",
            "video_id": "hallucinated-other-video",
            "concept_id": "c",
            "variant_id": "v",
            "source_word_ids": ["w1"],
            "hook_source_word_ids": ["w1"],
            "strategy_label": "s",
            "caption_platform": "tiktok",
            "confidence": 0.5,
        },
        timeline,
    )
    assert bound_plan.video_id == timeline.video_id
    missing_video_plan = GroundedEditPlan.from_payload(
        {
            "plan_id": "p-no-video",
            "concept_id": "c",
            "variant_id": "v",
            "source_word_ids": ["w1"],
            "hook_source_word_ids": ["w1"],
            "strategy_label": "s",
            "caption_platform": "tiktok",
            "confidence": 0.5,
        },
        timeline,
    )
    assert missing_video_plan.video_id == timeline.video_id
    with pytest.raises(EditorialGroundingError, match="overlay_text"):
        GroundedEditPlan.from_payload(
            {
                "plan_id": "p",
                "video_id": "video",
                "concept_id": "c",
                "variant_id": "v",
                "source_word_ids": ["w1", "w2"],
                "hook_source_word_ids": ["w1"],
                "overlay_text": 99,
                "strategy_label": "s",
                "caption_platform": "tiktok",
                "confidence": 0.5,
            },
            timeline,
        )


def test_grounded_continuous_plan_preserves_internal_source_silence() -> None:
    timeline = CanonicalTimeline(
        "video",
        "hash",
        (
            CanonicalWord("a", "a", 0, 0.2, None, None, "aligned", "x"),
            CanonicalWord("b", "b", 3, 3.2, None, None, "aligned", "x"),
        ),
    )
    grounded = GroundedEditPlan.from_payload(
        {
            "plan_id": "p",
            "video_id": "video",
            "concept_id": "c",
            "variant_id": "v",
            "source_word_ids": ["a", "b"],
            "hook_source_word_ids": ["a"],
            "strategy_label": "s",
            "caption_platform": "tiktok",
            "confidence": 0.8,
        },
        timeline,
    )
    plan = grounded.compile(timeline, "fp")
    assert len(plan.source_spans) == 1
    assert plan.source_spans[0].start == 0
    assert plan.source_spans[0].end == pytest.approx(3.2)


def test_grounded_plan_rejects_actual_canonical_word_skips() -> None:
    timeline = CanonicalTimeline(
        "video",
        "hash",
        (
            CanonicalWord("a", "a", 0, 0.2, None, None, "aligned", "x"),
            CanonicalWord("b", "b", 1, 1.2, None, None, "aligned", "x"),
            CanonicalWord("c", "c", 3, 3.2, None, None, "aligned", "x"),
        ),
    )
    with pytest.raises(EditorialGroundingError, match="consecutive canonical words"):
        GroundedEditPlan.from_payload(
            {
                "plan_id": "p",
                "concept_id": "c",
                "variant_id": "v",
                "source_word_ids": ["a", "c"],
                "hook_source_word_ids": ["a"],
                "strategy_label": "s",
                "caption_platform": "tiktok",
                "confidence": 0.8,
            },
            timeline,
        )


def test_compute_profile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        compute_profile("unknown")  # type: ignore[arg-type]
    assert ModelIdentity("m", "r", "q", "e").to_dict()["model_id"] == "m"


def test_visual_identity_and_timestamp_validation() -> None:
    with pytest.raises(ValueError, match="timestamps"):
        VisualEvent(1, 1, "scene", "x")
    with pytest.raises(ValueError, match="scene_id"):
        VisualEvent(0, 1, "", "x")
    with pytest.raises(ValueError, match="video_id"):
        VisualTimeline("", "hash", ())


def test_local_provider_lazy_load_success_paths() -> None:
    embedding_model = Mock()
    sentence_module = SimpleNamespace(SentenceTransformer=Mock(return_value=embedding_model))
    embedding = LocalEmbeddingProvider(device="cpu")
    with patch("clipper.providers.local.importlib.import_module", return_value=sentence_module):
        assert embedding._load() is embedding_model
        assert embedding._load() is embedding_model
    sentence_module.SentenceTransformer.assert_called_once()

    tokenizer = Mock()
    language_model = Mock()
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=language_model)),
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=Mock(return_value=language_model)),
    )
    editorial = LocalEditorialProvider()
    vision = LocalVisionProvider()
    with patch("clipper.providers.local.importlib.import_module", return_value=transformers):
        assert editorial._load() == (tokenizer, language_model)
        assert vision._load() == (tokenizer, language_model)


def test_editorial_and_vision_invalid_json_are_explicit(tmp_path: Path) -> None:
    class BadTokenizer(_FakeTokenizer):
        def decode(self, _tensor, **kwargs):  # type: ignore[no-untyped-def]
            return "not-json"

    editorial = LocalEditorialProvider()
    with (
        patch.object(editorial, "_load", return_value=(BadTokenizer(), _FakeModel())),
        pytest.raises(ValueError, match="valid JSON"),
    ):
        editorial.complete_json(task="mine", payload={})

    class ListTokenizer(_FakeTokenizer):
        def decode(self, _tensor, **kwargs):  # type: ignore[no-untyped-def]
            return "[]"

    with (
        patch.object(editorial, "_load", return_value=(ListTokenizer(), _FakeModel())),
        pytest.raises(ValueError, match="must be an object"),
    ):
        editorial.complete_json(task="mine", payload={})

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"x")
    vision = LocalVisionProvider()
    with (
        patch.object(vision, "_load", return_value=(BadTokenizer(), _FakeModel())),
        pytest.raises(ValueError, match="valid JSON"),
    ):
        vision.inspect(task="review", frames=[frame], context={})
    with (
        patch.object(vision, "_load", return_value=(ListTokenizer(), _FakeModel())),
        pytest.raises(ValueError, match="must be an object"),
    ):
        vision.inspect(task="review", frames=[frame], context={})


def test_optional_editorial_vision_and_modal_dependencies_fail_explicitly() -> None:
    with patch("clipper.providers.local.importlib.import_module", side_effect=ImportError):
        with pytest.raises(ProviderUnavailable, match="editorial"):
            LocalEditorialProvider()._load()
        with pytest.raises(ProviderUnavailable, match="vision"):
            LocalVisionProvider()._load()
    identity = ModelIdentity("m", "r", "none", "modal")
    modal = ModalEditorialProvider(app_name="a", function_name="f", identity=identity)
    with (
        patch("clipper.providers.modal.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="modal"),
    ):
        modal._function()


def test_modal_function_lookup_and_empty_usage_defaults() -> None:
    identity = ModelIdentity("m", "r", "none", "modal")
    function = Mock()
    function.remote.return_value = {"value": {"ok": True}}
    module = SimpleNamespace(Function=SimpleNamespace(from_name=Mock(return_value=function)))
    provider = ModalEditorialProvider(app_name="app", function_name="fn", identity=identity)
    with patch("clipper.providers.modal.importlib.import_module", return_value=module):
        assert provider._function() is function
    with patch.object(provider, "_function", return_value=function):
        result = provider.complete_json(task="x", payload={})
    assert result.usage.started_at == "unknown"
    assert result.usage.duration_seconds == 0.0


class _PlannerEditorial:
    identity = ModelIdentity("planner-editor", "rev", "none", "test", "p", "s")

    def __init__(self, response: dict[str, object] | None = None) -> None:
        self.response = response or {"ok": True}
        self.calls = 0

    def complete_json(
        self, *, task: str, payload: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        del task, payload
        self.calls += 1
        return ProviderResult(
            self.response,
            self.identity,
            InferenceUsage("test", "now", 0.01),
        )


class _PlannerEmbeddings:
    identity = ModelIdentity("planner-embed", "rev", "none", "test", "none", "s")

    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.calls = 0

    def embed(self, texts: list[str]) -> ProviderResult[list[list[float]]]:
        self.calls += 1
        vectors = (
            self.vectors
            if self.vectors is not None
            else [[1.0, float(i)] for i, _ in enumerate(texts)]
        )
        return ProviderResult(
            vectors,
            self.identity,
            InferenceUsage("test", "now", 0.01, input_units=len(texts)),
        )


def _open_brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="c",
        title="Any conversation",
        objective="Find source-grounded clips",
        keywords=["schema-placeholder"],
        source_channel_ids=["UC1"],
        allowed_video_ids=["video"],
        rights_confirmed=True,
        min_clip_seconds=8,
        max_clip_seconds=45,
        clip_count=1,
        production=ProductionConfig(
            candidate_pool_size=10,
            concept_count=2,
            variants_per_concept=2,
            final_render_budget=2,
            minimum_distinct_finalist_concepts=1,
        ),
    )


def _grounded_concept(
    concept_id: str, summary: str, confidence: float = 0.9
) -> GroundedClipConcept:
    return GroundedClipConcept(
        concept_id=concept_id,
        story_moment_ids=("m",),
        supporting_word_ids=("w1", "w2", "w3", "w4"),
        semantic_summary=summary,
        standalone_context="",
        narrative_structure="explanation",
        recommended_duration=20,
        visual_dependencies=(),
        confidence=confidence,
    )


def _long_grounded_timeline(word_count: int = 100) -> CanonicalTimeline:
    words = tuple(
        CanonicalWord(
            f"w{i:03d}",
            f"token{i}{'.' if i % 20 == 19 else ''}",
            i * 0.5,
            i * 0.5 + 0.42,
            "A",
            0.99,
            "aligned",
            "whisperx",
        )
        for i in range(word_count)
    )
    return CanonicalTimeline("video", "source-hash", words)


def test_edit_plan_context_exposes_grounded_words_around_concept(tmp_path: Path) -> None:
    timeline = _long_grounded_timeline()
    brief = replace(_open_brief(), min_clip_seconds=20, max_clip_seconds=45)
    concept = GroundedClipConcept(
        "c",
        ("m",),
        tuple(word.word_id for word in timeline.words[40:50]),
        "grounded story",
        "",
        "story",
        25,
        (),
        0.9,
    )
    planner = AutonomousEditorialPlanner(
        _PlannerEditorial(), _PlannerEmbeddings(), FileCache(tmp_path / "context")
    )
    evidence = planner._plan_context_words(timeline, concept, brief)
    refs = [str(item["word_ref"]) for item in evidence]
    assert len(evidence) <= 360
    assert timeline.word_ref(timeline.words[40].word_id) in refs
    assert timeline.word_ref(timeline.words[49].word_id) in refs
    assert float(evidence[0]["source_start"]) < timeline.words[40].source_start
    assert float(evidence[-1]["source_end"]) > timeline.words[49].source_end
    assert all(
        "text" in item and "source_start" in item and "source_end" in item for item in evidence
    )


def test_plan_context_preserves_oversized_grounded_concept(tmp_path: Path) -> None:
    timeline = _long_grounded_timeline(1000)
    concept = GroundedClipConcept(
        "wide",
        ("m",),
        tuple(word.word_id for word in timeline.words[800:900]),
        "wide grounded story",
        "",
        "story",
        40,
        (),
        0.9,
    )
    planner = AutonomousEditorialPlanner(
        _PlannerEditorial(), _PlannerEmbeddings(), FileCache(tmp_path / "wide-context")
    )
    evidence = planner._plan_context_words(timeline, concept, _open_brief(), max_words=50)
    refs = {str(item["word_ref"]) for item in evidence}
    assert len(evidence) == 100
    assert timeline.word_ref(timeline.words[800].word_id) in refs
    assert timeline.word_ref(timeline.words[899].word_id) in refs


def test_grounded_boundary_repair_preserves_chronology_and_spoken_hook() -> None:
    timeline = _long_grounded_timeline()
    plan = GroundedEditPlan(
        "p",
        "video",
        "c",
        "v",
        tuple(word.word_id for word in timeline.words[10:60]),
        tuple(word.word_id for word in timeline.words[20:25]),
        None,
        "direct",
        "tiktok",
        0.9,
    )

    def audit(start: str | None, end: str | None) -> BoundaryAudit:
        return BoundaryAudit(
            timeline.words[10].source_start,
            timeline.words[59].source_end,
            timeline.words[10].text,
            timeline.words[59].text,
            "prior",
            "after",
            BoundaryStatus.NEEDS_CONTEXT,
            BoundaryStatus.COMPLETE,
            BoundaryStatus.NEEDS_CONTEXT,
            "setup",
            "",
            False,
            True,
            True,
            True,
            (),
            (),
            "story",
            0.9,
            (BoundaryFailureReason.START_REQUIRES_PRIOR_CONTEXT,),
            plan.source_word_ids,
            start,
            end,
            {"model_id": "test"},
        )

    repaired = AutonomousEditorialPlanner._apply_boundary_repair(
        timeline, plan, audit("w005", "w070")
    )
    assert repaired.source_word_ids[0] == "w005"
    assert repaired.source_word_ids[-1] == "w070"
    assert set(plan.hook_source_word_ids) <= set(repaired.source_word_ids)

    with pytest.raises(EditorialGroundingError, match="chronology"):
        AutonomousEditorialPlanner._apply_boundary_repair(timeline, plan, audit("w070", "w005"))
    with pytest.raises(EditorialGroundingError, match="hook"):
        AutonomousEditorialPlanner._apply_boundary_repair(timeline, plan, audit("w000", "w015"))


def test_structured_campaign_policy_builds_grounded_source_hazard_timeline(
    tmp_path: Path,
) -> None:
    timeline = _long_grounded_timeline()
    policy = AcceptancePolicy.from_dict(
        {
            "source_segments": {
                "allow": ["editorial_content"],
                "forbid": ["advertisement", "sponsor_read"],
                "unknown": "escalate",
            }
        }
    )
    brief = replace(_open_brief(), acceptance_policy=policy)
    response = {
        "segments": [
            {
                "start_word_id": timeline.word_ref(timeline.words[0].word_id),
                "end_word_id": timeline.word_ref(timeline.words[-1].word_id),
                "classification": "editorial_content",
                "confidence": 0.98,
                "evidence": ["ordinary source conversation"],
            }
        ]
    }
    planner = AutonomousEditorialPlanner(
        _PlannerEditorial(response),
        _PlannerEmbeddings(),
        FileCache(tmp_path / "hazard-cache"),
    )
    hazards, rejections = planner._classify_source_hazards(brief, timeline, None)
    assert rejections == []
    assert len(hazards) == 1
    assert hazards[0].classification.value == "editorial_content"
    assert hazards[0].source_word_ids[0] == timeline.words[0].word_id
    assert hazards[0].source_word_ids[-1] == timeline.words[-1].word_id


def test_source_hazard_model_failure_becomes_unknown_escalation_evidence(
    tmp_path: Path,
) -> None:
    timeline = _long_grounded_timeline()
    brief = replace(
        _open_brief(),
        acceptance_policy=AcceptancePolicy.from_dict({"source_segments": {"unknown": "escalate"}}),
    )
    planner = AutonomousEditorialPlanner(
        _PlannerEditorial({"bad": []}),
        _PlannerEmbeddings(),
        FileCache(tmp_path / "hazard-failure-cache"),
    )
    hazards, rejections = planner._classify_source_hazards(brief, timeline, None)
    assert hazards[0].classification.value == "unknown"
    assert hazards[0].confidence == 0.0
    assert rejections[0]["decision"] == "ESCALATE"


def test_duration_repair_expands_short_grounded_plan_to_campaign_minimum() -> None:
    timeline = _long_grounded_timeline()
    brief = replace(_open_brief(), min_clip_seconds=20, max_clip_seconds=45)
    concept = GroundedClipConcept(
        "c",
        ("m",),
        tuple(word.word_id for word in timeline.words),
        "complete grounded story",
        "",
        "story",
        28,
        (),
        0.9,
    )
    grounded = GroundedEditPlan(
        "p-short",
        "video",
        "c",
        "v",
        tuple(word.word_id for word in timeline.words[:12]),
        tuple(word.word_id for word in timeline.words[:4]),
        None,
        "direct",
        "tiktok",
        0.9,
    )
    repaired, evidence = AutonomousEditorialPlanner._repair_grounded_plan_duration(
        brief, timeline, concept, grounded
    )
    assert repaired is not None
    plan = repaired.compile(timeline, "fp")
    assert brief.min_clip_seconds <= plan.duration <= brief.max_clip_seconds
    assert plan.duration == pytest.approx(20.42)
    assert set(grounded.hook_source_word_ids).issubset(repaired.source_word_ids)
    assert evidence["decision"] == "REPAIR"
    assert evidence["original_duration"] == pytest.approx(5.92)
    assert evidence["repaired_duration"] == pytest.approx(plan.duration)


def test_duration_repair_trims_long_plan_and_reports_unrepairable_context() -> None:
    timeline = _long_grounded_timeline()
    brief = replace(_open_brief(), min_clip_seconds=20, max_clip_seconds=45)
    all_ids = tuple(word.word_id for word in timeline.words)
    concept = GroundedClipConcept("c", ("m",), all_ids, "story", "", "story", 30, (), 0.9)
    grounded = GroundedEditPlan(
        "p-long",
        "video",
        "c",
        "v",
        all_ids,
        tuple(word.word_id for word in timeline.words[4:8]),
        None,
        "direct",
        "tiktok",
        0.9,
    )
    repaired, evidence = AutonomousEditorialPlanner._repair_grounded_plan_duration(
        brief, timeline, concept, grounded
    )
    assert repaired is not None
    plan = repaired.compile(timeline, "fp")
    assert 44 <= plan.duration <= 45
    assert evidence["decision"] == "REPAIR"
    assert evidence["original_duration"] == pytest.approx(49.92)

    short_concept = replace(concept, supporting_word_ids=all_ids[:20])
    short_plan = replace(grounded, plan_id="p-unrepairable", source_word_ids=all_ids[:10])
    missing, rejected = AutonomousEditorialPlanner._repair_grounded_plan_duration(
        brief, timeline, short_concept, short_plan
    )
    assert missing is None
    assert rejected["decision"] == "REJECT"
    assert rejected["reasons"] == ["duration_outside_campaign_bounds_no_grounded_repair"]
    assert rejected["grounded_context_duration"] == pytest.approx(9.92)


def test_duration_repair_rejects_oversized_source_spanning_hook_without_compiling() -> None:
    timeline = _long_grounded_timeline(200)
    brief = replace(_open_brief(), min_clip_seconds=45, max_clip_seconds=60)
    all_ids = tuple(word.word_id for word in timeline.words)
    concept = GroundedClipConcept("c", ("m",), all_ids, "story", "", "story", 55, (), 0.9)
    grounded = GroundedEditPlan.from_payload(
        {
            "plan_id": "p0",
            "concept_id": "c",
            "variant_id": "v0",
            "source_start_word_id": timeline.word_ref(all_ids[0]),
            "source_end_word_id": timeline.word_ref(all_ids[-1]),
            "hook_start_word_id": timeline.word_ref(all_ids[0]),
            "hook_end_word_id": timeline.word_ref(all_ids[-1]),
            "overlay_text": "A grounded overlay.",
            "strategy_label": "curiosity",
            "caption_platform": "tiktok",
            "confidence": 0.95,
        },
        timeline,
    )

    repaired, evidence = AutonomousEditorialPlanner._repair_grounded_plan_duration(
        brief, timeline, concept, grounded
    )

    assert repaired is None
    assert evidence["plan_id"] == "p0"
    assert evidence["original_duration"] == pytest.approx(99.92)
    assert evidence["reasons"] == ["duration_outside_campaign_bounds_no_grounded_repair"]


def test_autonomous_planner_validation_cosine_and_array_guards(tmp_path: Path) -> None:
    editor = _PlannerEditorial()
    embedder = _PlannerEmbeddings()
    cache = FileCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="at least 200"):
        AutonomousEditorialPlanner(editor, embedder, cache, max_words_per_chunk=199)
    with pytest.raises(ValueError, match="smaller than chunk"):
        AutonomousEditorialPlanner(
            editor, embedder, cache, max_words_per_chunk=200, chunk_overlap_words=200
        )
    with pytest.raises(ValueError, match="threshold"):
        AutonomousEditorialPlanner(editor, embedder, cache, semantic_duplicate_threshold=0.2)
    with pytest.raises(ValueError, match="equal dimensions"):
        _cosine([1.0], [1.0, 2.0])
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    with pytest.raises(EditorialGroundingError, match="must be a list"):
        AutonomousEditorialPlanner._array({"x": {}}, "x")
    with pytest.raises(EditorialGroundingError, match="contain objects"):
        AutonomousEditorialPlanner._array({"x": ["bad"]}, "x")


def test_autonomous_planner_model_and_embedding_cache_hits(tmp_path: Path) -> None:
    editor = _PlannerEditorial({"profile": "cached"})
    embedder = _PlannerEmbeddings([[0.1, 0.2]])
    planner = AutonomousEditorialPlanner(editor, embedder, FileCache(tmp_path / "cache"))
    timeline = _timeline()
    brief = _open_brief()
    payload = {"a": 1}
    first = planner._complete("stage", timeline, brief, payload)
    second = planner._complete("stage", timeline, brief, payload)
    assert first == second == {"profile": "cached"}
    assert editor.calls == 1
    vectors1 = planner._embed("embed", timeline, brief, ["text"])
    vectors2 = planner._embed("embed", timeline, brief, ["text"])
    assert vectors1 == vectors2 == [[0.1, 0.2]]
    assert embedder.calls == 1
    assert [item["cache_hit"] for item in planner.invocations] == [False, True, False, True]


def test_autonomous_planner_chunking_empty_and_long_profile_sampling(tmp_path: Path) -> None:
    planner = AutonomousEditorialPlanner(
        _PlannerEditorial(),
        _PlannerEmbeddings(),
        FileCache(tmp_path / "cache"),
        max_words_per_chunk=200,
        chunk_overlap_words=20,
    )
    assert planner._chunks(CanonicalTimeline("v", "h", ())) == []
    words = tuple(
        CanonicalWord(f"w{i}", "token", i * 0.1, i * 0.1 + 0.05, None, None, "aligned", "x")
        for i in range(1901)
    )
    timeline = CanonicalTimeline("v", "h", words)
    chunks = planner._chunks(timeline)
    assert len(chunks) > 9
    assert len(chunks[0]) == 200
    evidence = planner._profile_evidence(timeline)
    assert len(evidence) == 8 * 60
    assert evidence[0]["word_id"] == "w0"
    assert evidence[-1]["word_id"] == "w1900"


def test_learned_semantic_dedupe_keeps_best_and_rejects_duplicate(tmp_path: Path) -> None:
    planner = AutonomousEditorialPlanner(
        _PlannerEditorial(),
        _PlannerEmbeddings([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        FileCache(tmp_path / "cache"),
        semantic_duplicate_threshold=0.9,
    )
    kept, clusters, rejections = planner._semantic_dedupe(
        _open_brief(),
        _timeline(),
        [
            _grounded_concept("a", "same", 0.95),
            _grounded_concept("b", "duplicate", 0.9),
            _grounded_concept("c", "different", 0.8),
        ],
    )
    assert [item.concept_id for item in kept] == ["a", "c"]
    assert clusters["a"] == clusters["b"]
    assert rejections[0]["reasons"] == ["learned_embedding_duplicate"]
    broken = AutonomousEditorialPlanner(
        _PlannerEditorial(), _PlannerEmbeddings([[1.0]]), FileCache(tmp_path / "broken")
    )
    with pytest.raises(ValueError, match="wrong number"):
        broken._semantic_dedupe(
            _open_brief(), _timeline(), [_grounded_concept("a", "a"), _grounded_concept("b", "b")]
        )


def test_hook_embedding_dedupe_and_vector_count_validation(tmp_path: Path) -> None:
    hooks = [
        GroundedHookVariant("h1", "direct", ("w1", "w2"), None, "r1", 0.9),
        GroundedHookVariant("h2", "near duplicate", ("w1", "w2"), None, "r2", 0.8),
        GroundedHookVariant("h3", "different", ("w3", "w4"), "CONTEXT", "r3", 0.7),
    ]
    planner = AutonomousEditorialPlanner(
        _PlannerEditorial(),
        _PlannerEmbeddings([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        FileCache(tmp_path / "cache"),
    )
    rejections: list[dict[str, object]] = []
    kept = planner._dedupe_hooks(_open_brief(), _timeline(), hooks, "c", rejections)
    assert [item.variant_id for item in kept] == ["h1", "h3"]
    assert rejections[0]["reasons"] == ["learned_embedding_hook_duplicate"]
    broken = AutonomousEditorialPlanner(
        _PlannerEditorial(), _PlannerEmbeddings([[1.0]]), FileCache(tmp_path / "broken")
    )
    with pytest.raises(ValueError, match="wrong number"):
        broken._dedupe_hooks(_open_brief(), _timeline(), hooks, "c", [])


def test_open_analysis_rejects_empty_and_unknown_model_references(tmp_path: Path) -> None:
    brief = _open_brief()
    timeline = _timeline()

    class Scripted(_PlannerEditorial):
        def complete_json(
            self, *, task: str, payload: dict[str, object]
        ) -> ProviderResult[dict[str, object]]:
            del payload
            if task == "episode_editorial_profile":
                value: dict[str, object] = {
                    "summary": "x",
                    "valuable_moment_characteristics": ["x"],
                    "avoid_characteristics": [],
                    "confidence": 0.9,
                }
            elif task.startswith("story_moments:"):
                value = {"moments": []}
            else:
                raise AssertionError(task)
            return ProviderResult(value, self.identity, InferenceUsage("test", "now", 0.01))

    with pytest.raises(EditorialGroundingError, match="no grounded StoryMoments"):
        AutonomousEditorialPlanner(
            Scripted(), _PlannerEmbeddings(), FileCache(tmp_path / "empty")
        ).analyze_video(brief, timeline)

    class UnknownMoment(Scripted):
        def complete_json(
            self, *, task: str, payload: dict[str, object]
        ) -> ProviderResult[dict[str, object]]:
            if task.startswith("story_moments:"):
                value: dict[str, object] = {
                    "moments": [
                        {
                            "moment_id": "m",
                            "supporting_word_ids": ["w1", "w2", "w3", "w4"],
                            "semantic_summary": "x",
                            "narrative_structure": "x",
                            "editorial_reason": "x",
                            "confidence": 0.9,
                        }
                    ]
                }
            elif task == "clip_concepts":
                value = {
                    "concepts": [
                        {
                            "concept_id": "c",
                            "story_moment_ids": ["missing"],
                            "supporting_word_ids": ["w1", "w2", "w3", "w4"],
                            "semantic_summary": "x",
                            "narrative_structure": "x",
                            "recommended_duration": 20,
                            "confidence": 0.9,
                        }
                    ]
                }
            else:
                return super().complete_json(task=task, payload=payload)
            return ProviderResult(value, self.identity, InferenceUsage("test", "now", 0.01))

    with pytest.raises(EditorialGroundingError, match="unknown StoryMoments"):
        AutonomousEditorialPlanner(
            UnknownMoment(), _PlannerEmbeddings([[1.0, 0.0]]), FileCache(tmp_path / "unknown")
        ).analyze_video(brief, timeline)


def test_plan_batch_global_selection_and_plan_validation_errors(tmp_path: Path) -> None:
    timeline = _timeline()
    concept = ClipConcept(
        "c",
        "video",
        10.0,
        11.1,
        "what's one message today",
        "summary",
        "",
        "",
        "question",
        20.0,
        EditorialScores(*(8.0 for _ in range(12))),
        8.0,
        "sem",
        "fp",
    )
    moment = StoryMoment(
        "m",
        "video",
        10.0,
        11.1,
        concept.text,
        "question",
        "summary",
        "",
        "",
        EditorialScores(*(8.0 for _ in range(12))),
        8.0,
        "fp",
    )
    grounded = _grounded_concept("c", "summary")
    analysis = OpenVideoAnalysis(
        EpisodeEditorialProfile("x", ("x",), (), 0.9),
        [moment],
        [concept],
        {
            "m": GroundedStoryMoment(
                "m", ("w1", "w2", "w3", "w4"), "x", "question", "", "", "x", 0.9
            )
        },
        {"c": grounded},
        [],
    )

    class Global(_PlannerEditorial):
        def __init__(self, selection: object) -> None:
            super().__init__()
            self.selection = selection

        def complete_json(
            self, *, task: str, payload: dict[str, object]
        ) -> ProviderResult[dict[str, object]]:
            del payload
            if task == "global_concept_comparison":
                return ProviderResult(
                    {"concept_ids": self.selection},
                    self.identity,
                    InferenceUsage("test", "now", 0.01),
                )
            raise AssertionError(task)

    for selection, match in [
        ("bad", "concept_ids"),
        (["missing"], "unknown concept"),
        ([], "selected no concepts"),
    ]:
        with pytest.raises(EditorialGroundingError, match=match):
            AutonomousEditorialPlanner(
                Global(selection),
                _PlannerEmbeddings(),
                FileCache(tmp_path / str(match).replace(" ", "-")),
            ).plan_batch(_open_brief(), {"video": timeline}, [analysis])
    with pytest.raises(EditorialGroundingError, match="produced no concepts"):
        AutonomousEditorialPlanner(
            Global([]), _PlannerEmbeddings(), FileCache(tmp_path / "none")
        ).plan_batch(_open_brief(), {"video": timeline}, [])


def test_managed_modal_endpoint_editorial_provider_uses_proxy_auth_and_json() -> None:
    identity = ModelIdentity(
        "Qwen/Qwen3.6-27B-FP8", "modal-managed", "fp8", "modal-managed-endpoint"
    )
    provider = ModalEndpointEditorialProvider(
        endpoint_url="https://example.modal.direct",
        proxy_token_id=("wk-test", "ws-test")[0],
        proxy_token_secret=("wk-test", "ws-test")[1],
        identity=identity,
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": "works",
                                        "valuable_moment_characteristics": [],
                                        "avoid_characteristics": [],
                                        "confidence": 0.9,
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                }
            ).encode()

    with patch("clipper.providers.modal_endpoint.urlopen", return_value=Response()) as opened:
        result = provider.complete_json(task="episode_editorial_profile", payload={"source": "x"})
    request = opened.call_args.args[0]
    assert request.full_url == "https://example.modal.direct/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer wk-test.ws-test"
    body = json.loads(request.data.decode())
    assert body["model"] == "Qwen/Qwen3.6-27B-FP8"
    assert body["temperature"] == 0
    assert result.value["summary"] == "works"
    assert result.usage.provider == "modal-endpoint"
    assert result.usage.input_units == 12
    assert result.usage.output_units == 7


def test_managed_modal_endpoint_provider_rejects_missing_credentials() -> None:
    identity = ModelIdentity("m", "r", "none", "modal-managed-endpoint")
    with pytest.raises(ProviderUnavailable, match="required"):
        ModalEndpointEditorialProvider(
            endpoint_url="",
            proxy_token_id=("wk", "ws")[0],
            proxy_token_secret=("wk", "ws")[1],
            identity=identity,
        )
    with pytest.raises(ProviderUnavailable, match="proxy token"):
        ModalEndpointEditorialProvider(
            endpoint_url="https://endpoint",
            proxy_token_id="",
            proxy_token_secret="",
            identity=identity,
        )


def test_editorial_prompt_contracts_cover_all_grounded_tasks() -> None:
    assert EDITORIAL_PROMPT_VERSION == "editor-v2"
    assert EDITORIAL_SCHEMA_VERSION == "editorial-json-v2"
    assert editorial_output_budget({"task": "episode_editorial_profile"}) == 768
    assert editorial_output_budget({"task": "global_concept_comparison"}) == 768
    assert editorial_output_budget({"task": "hook_variants:c1"}) == 1024
    assert editorial_output_budget({"task": "edit_plans:c1"}) == 1536
    assert "summary" in editorial_contract("episode_editorial_profile")
    assert "moments" in editorial_contract("story_moments:0")
    assert "concepts" in editorial_contract("clip_concepts")
    assert "concept_ids" in editorial_contract("global_concept_comparison")
    assert "variants" in editorial_contract("hook_variants:c1")
    assert "plans" in editorial_contract("edit_plans:c1")
    assert "segments" in editorial_contract("source_hazards:0")
    assert "start_status" in editorial_contract("boundary_audit:p1")
    assert "ceiling, never a target" in editorial_contract("edit_plans:c1")
    assert "Follow the task payload" in editorial_contract("unknown")
    modal_source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "def _editorial_contract" not in modal_source
    assert "from clipper.providers.editorial_prompt import editorial_contract" in modal_source


def test_managed_modal_endpoint_json_validation_and_https_requirement() -> None:
    identity = ModelIdentity("m", "r", "none", "modal-managed-endpoint")
    with pytest.raises(ProviderUnavailable, match="https"):
        ModalEndpointEditorialProvider(
            endpoint_url="http://unsafe",
            proxy_token_id=("wk", "ws")[0],
            proxy_token_secret=("wk", "ws")[1],
            identity=identity,
        )
    assert ModalEndpointEditorialProvider._json_object('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(ValueError, match="JSON object"):
        ModalEndpointEditorialProvider._json_object("[1, 2]")


def test_managed_modal_endpoint_retries_transient_http_errors() -> None:
    identity = ModelIdentity("m", "r", "none", "modal-managed-endpoint")
    provider = ModalEndpointEditorialProvider(
        endpoint_url="https://example.modal.direct",
        proxy_token_id=("wk", "ws")[0],
        proxy_token_secret=("wk", "ws")[1],
        identity=identity,
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'

    transient = HTTPError(
        "https://example.modal.direct",
        503,
        "cold",
        hdrs=None,
        fp=io.BytesIO(b"cold start"),
    )
    with (
        patch(
            "clipper.providers.modal_endpoint.urlopen", side_effect=[transient, Response()]
        ) as opened,
        patch("clipper.providers.modal_endpoint.time.sleep") as slept,
    ):
        result = provider.complete_json(task="unknown", payload={})
    assert result.value == {"ok": True}
    assert opened.call_count == 2
    slept.assert_called_once()


def test_managed_modal_endpoint_surfaces_transport_and_shape_failures() -> None:
    identity = ModelIdentity("m", "r", "none", "modal-managed-endpoint")
    provider = ModalEndpointEditorialProvider(
        endpoint_url="https://example.modal.direct",
        proxy_token_id=("wk", "ws")[0],
        proxy_token_secret=("wk", "ws")[1],
        identity=identity,
    )

    class Response:
        def __init__(self, value: object) -> None:
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(self.value).encode()

    fatal = HTTPError(
        "https://example.modal.direct", 401, "unauthorized", hdrs=None, fp=io.BytesIO(b"denied")
    )
    with (
        patch("clipper.providers.modal_endpoint.urlopen", side_effect=fatal),
        pytest.raises(RuntimeError, match="HTTP 401: denied"),
    ):
        provider.complete_json(task="x", payload={})

    for value, message in [
        ([], "invalid response"),
        ({"choices": []}, "no choices"),
        ({"choices": [{}]}, "no message content"),
        ({"choices": [{"message": {"content": []}}]}, "no message content"),
    ]:
        with (
            patch("clipper.providers.modal_endpoint.urlopen", return_value=Response(value)),
            pytest.raises(ValueError, match=message),
        ):
            provider.complete_json(task="x", payload={})

    with (
        patch("clipper.providers.modal_endpoint.urlopen", side_effect=URLError("offline")),
        patch("clipper.providers.modal_endpoint.time.sleep"),
        pytest.raises(RuntimeError, match="offline"),
    ):
        provider.complete_json(task="x", payload={})


def test_provider_factory_profiles_and_modal_embedding_validation(monkeypatch) -> None:
    local_editor, local_embed = editorial_and_embedding_providers("local-lite")
    assert isinstance(local_editor, LocalEditorialProvider)
    assert isinstance(local_embed, LocalEmbeddingProvider)
    modal_editor, modal_embed = editorial_and_embedding_providers("balanced")
    assert isinstance(modal_editor, ModalEditorialProvider)
    assert modal_editor.identity.model_id == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert modal_editor.identity.quantization == "bnb-4bit-nf4"

    monkeypatch.setenv("CLIPPER_MODAL_EDITORIAL_BACKEND", "managed")
    monkeypatch.setenv("CLIPPER_MODAL_EDITORIAL_ENDPOINT_URL", "https://example.modal.direct")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    managed_editor, _ = editorial_and_embedding_providers("balanced")
    assert isinstance(managed_editor, ModalEndpointEditorialProvider)
    assert managed_editor.identity.model_id == "Qwen/Qwen3.5-4B"
    quality_editor, _ = editorial_and_embedding_providers("quality")
    assert isinstance(quality_editor, ModalEndpointEditorialProvider)
    assert quality_editor.identity.model_id == "Qwen/Qwen3.6-27B-FP8"

    monkeypatch.setenv("CLIPPER_MODAL_EDITORIAL_BACKEND", "invalid")
    with pytest.raises(ValueError, match="unsupported Modal editorial backend"):
        editorial_and_embedding_providers("balanced")
    assert isinstance(modal_embed, ModalEmbeddingProvider)
    assert isinstance(vision_provider("local-lite"), LocalVisionProvider)
    assert isinstance(vision_provider("balanced"), ModalVisionProvider)
    assert isinstance(vision_provider("quality", large=True), ModalVisionProvider)
    with pytest.raises(ValueError, match="disabled"):
        vision_provider("balanced", large=True)

    identity = ModelIdentity("m", "r", "none", "modal")
    provider = ModalEmbeddingProvider(app_name="a", function_name="f", identity=identity)
    function = Mock()
    function.remote.return_value = {
        "vectors": [[1, 2], [3.5, 4]],
        "usage": {"gpu_type": "L4", "gpu_seconds": 1.2, "estimated_cost_usd": 0.01},
    }
    with patch.object(provider, "_function", return_value=function):
        result = provider.embed(["a", "b"])
    assert result.value == [[1.0, 2.0], [3.5, 4.0]]
    assert result.usage.gpu_type == "L4"
    function.remote.return_value = {"bad": []}
    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(ValueError, match="invalid response"),
    ):
        provider.embed(["a"])
    function.remote.return_value = {"vectors": ["bad"]}
    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(ValueError, match="vector must be a list"),
    ):
        provider.embed(["a"])


def test_whisperx_alignment_preserves_word_ids_and_updates_only_matched_words() -> None:
    timeline = _timeline()
    aligned = apply_whisperx_alignment(
        timeline,
        [
            {
                "words": [
                    {"word": "what's", "start": 10.02, "end": 10.19, "score": 0.91},
                    {"word": "noise", "start": 10.2, "end": 10.3, "score": 0.2},
                    {"word": "one", "start": 10.22, "end": 10.39, "score": 0.92},
                    {"word": "message", "start": None, "end": 10.8},
                ]
            }
        ],
    )
    assert [word.word_id for word in aligned.words] == ["w1", "w2", "w3", "w4"]
    assert [word.text for word in aligned.words] == ["what's", "one", "message", "today"]
    assert aligned.words[0].source_start == pytest.approx(10.02)
    assert aligned.words[0].timing_mode == "aligned"
    assert aligned.words[0].confidence == pytest.approx(0.91)
    assert aligned.words[2].source_start == 10.41
    with pytest.raises(ValueError, match="no canonical word matches"):
        apply_whisperx_alignment(
            timeline, [{"words": [{"word": "unrelated", "start": 1, "end": 2}]}]
        )


def test_whisperx_alignment_rejects_nonmonotonic_points_without_clamping() -> None:
    timeline = CanonicalTimeline(
        "v",
        "h",
        (
            CanonicalWord("w1", "alpha", 1.0, 1.1, None, 0.8, "word_exact", "whisper"),
            CanonicalWord("w2", "beta", 1.2, 1.3, None, 0.8, "word_exact", "whisper"),
            CanonicalWord("w3", "gamma", 1.4, 1.5, None, 0.8, "word_exact", "whisper"),
        ),
    )
    aligned = apply_whisperx_alignment(
        timeline,
        [
            {
                "words": [
                    {"word": "alpha", "start": 1.25, "end": 1.35, "score": 0.20},
                    {"word": "beta", "start": 1.10, "end": 1.22, "score": 0.95},
                    {"word": "gamma", "start": 1.42, "end": 1.52, "score": 0.90},
                ]
            }
        ],
    )
    assert [word.word_id for word in aligned.words] == ["w1", "w2", "w3"]
    assert [word.source_start for word in aligned.words] == pytest.approx([1.0, 1.10, 1.42])
    assert aligned.words[0].timing_mode == "word_exact"
    assert aligned.words[1].timing_mode == "aligned"
    assert aligned.words[2].timing_mode == "aligned"


def test_whisperx_alignment_rejects_invalid_raw_timings() -> None:
    timeline = _timeline()
    aligned = apply_whisperx_alignment(
        timeline,
        [
            {
                "words": [
                    {"word": "what's", "start": 10.3, "end": 10.2, "score": 0.9},
                    {"word": "one", "start": 10.22, "end": 10.39, "score": 2.0},
                ]
            }
        ],
    )
    assert aligned.words[0].source_start == pytest.approx(10.0)
    assert aligned.words[1].source_start == pytest.approx(10.22)
    assert aligned.words[1].confidence == pytest.approx(0.99)


def test_alignment_segments_split_on_gap_and_duration() -> None:
    words = (
        CanonicalWord("a", "a", 0, 0.2, None, None, "word_exact", "x"),
        CanonicalWord("b", "b", 0.3, 0.5, None, None, "word_exact", "x"),
        CanonicalWord("c", "c", 2.0, 2.2, None, None, "word_exact", "x"),
        CanonicalWord("d", "d", 35.0, 35.2, None, None, "word_exact", "x"),
    )
    segments = _alignment_segments(CanonicalTimeline("v", "h", words), max_seconds=30)
    assert [item["text"] for item in segments] == ["a b", "c", "d"]
    assert _alignment_segments(CanonicalTimeline("v", "h", ())) == []


def test_speaker_turn_assignment_uses_maximum_temporal_overlap() -> None:
    timeline = _timeline()
    assigned = apply_speaker_turns(
        timeline,
        [(9.9, 10.35, "SPEAKER_00"), (10.3, 11.2, "SPEAKER_01")],
    )
    assert assigned.words[0].speaker_id == "SPEAKER_00"
    assert assigned.words[1].speaker_id == "SPEAKER_00"
    assert assigned.words[2].speaker_id == "SPEAKER_01"
    assert assigned.words[3].speaker_id == "SPEAKER_01"
    assert [word.word_id for word in assigned.words] == ["w1", "w2", "w3", "w4"]


class _FWWord:
    def __init__(self, start=None, end=None, word="", probability=None):  # type: ignore[no-untyped-def]
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class _FWSegment:
    def __init__(self, words):  # type: ignore[no-untyped-def]
        self.words = words


def test_faster_whisper_provider_lazy_load_and_transcription(tmp_path: Path) -> None:
    model = Mock()
    model.transcribe.return_value = (
        [
            _FWSegment(
                [
                    _FWWord(0.0, 0.2, " Hello", 0.9),
                    _FWWord(None, 0.3, "bad", 0.1),
                    _FWWord(0.3, 0.6, " world", None),
                ]
            )
        ],
        object(),
    )
    module = SimpleNamespace(WhisperModel=Mock(return_value=model))
    provider = FasterWhisperTranscriptionProvider(device="cpu", compute_type="int8")
    with patch("clipper.providers.speech.importlib.import_module", return_value=module):
        assert provider._load() is model
        assert provider._load() is model
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    with patch.object(provider, "_load", return_value=model):
        result = provider.transcribe(source, video_id="v", source_hash="h")
    assert [word.text for word in result.value.words] == ["Hello", "world"]
    assert [word.word_id for word in result.value.words] == ["v:w0000000", "v:w0000001"]
    assert result.value.words[0].confidence == pytest.approx(0.9)
    assert result.value.words[1].confidence is None
    model.transcribe.assert_called_with(str(source), word_timestamps=True, vad_filter=True)


def test_faster_whisper_provider_dependency_and_empty_result_failures(tmp_path: Path) -> None:
    provider = FasterWhisperTranscriptionProvider()
    with (
        patch("clipper.providers.speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="asr"),
    ):
        provider._load()
    model = Mock()
    model.transcribe.return_value = ([_FWSegment([])], object())
    with (
        patch.object(provider, "_load", return_value=model),
        pytest.raises(ValueError, match="no timestamped words"),
    ):
        provider.transcribe(tmp_path / "x.wav", video_id="v", source_hash="h")


def test_whisperx_provider_runtime_and_failures(tmp_path: Path) -> None:
    timeline = _timeline()
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    whisperx = SimpleNamespace(
        load_audio=Mock(return_value="audio"),
        load_align_model=Mock(return_value=("model", {"language": "en"})),
        align=Mock(
            return_value={
                "segments": [
                    {
                        "words": [
                            {
                                "word": word.text,
                                "start": word.source_start,
                                "end": word.source_end,
                                "score": 0.9,
                            }
                            for word in timeline.words
                        ]
                    }
                ]
            }
        ),
    )
    provider = WhisperXAlignmentProvider(device="cpu")
    with patch("clipper.providers.speech.importlib.import_module", return_value=whisperx):
        result = provider.align(source, timeline)
    assert all(word.timing_mode == "aligned" for word in result.value.words)
    whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    with pytest.raises(ValueError, match="empty"):
        provider.align(source, CanonicalTimeline("v", "h", ()))
    with (
        patch("clipper.providers.speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="alignment"),
    ):
        provider.align(source, timeline)
    whisperx.align.return_value = {}
    with (
        patch("clipper.providers.speech.importlib.import_module", return_value=whisperx),
        pytest.raises(ValueError, match="no aligned segments"),
    ):
        provider.align(source, timeline)


class _Turn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _Diarization:
    def itertracks(self, *, yield_label: bool):
        assert yield_label is True
        yield _Turn(9.9, 10.4), "track", "A"
        yield _Turn(10.4, 11.2), "track", "B"
        yield _Turn(12.0, 12.0), "track", "ignored"


def test_pyannote_provider_requires_gated_token_and_assigns_speakers(tmp_path: Path) -> None:
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ProviderUnavailable, match="HF_TOKEN"),
    ):
        PyannoteDiarizationProvider(token=None)._load()

    fake_hf_token = "test-hf-token"  # noqa: S105
    pipeline = Mock(return_value=SimpleNamespace(speaker_diarization=_Diarization()))
    module = SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipeline)))
    provider = PyannoteDiarizationProvider(token=fake_hf_token)
    with patch("clipper.providers.speech.importlib.import_module", return_value=module):
        assert provider._load() is pipeline
        assert provider._load() is pipeline
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    with patch.object(provider, "_load", return_value=pipeline):
        result = provider.diarize(source, _timeline())
    assert [word.speaker_id for word in result.value.words] == ["A", "A", "B", "B"]
    assert PyannoteDiarizationProvider._turns(_Diarization()) == [
        (9.9, 10.4, "A"),
        (10.4, 11.2, "B"),
    ]


def test_pyannote_dependency_device_and_invalid_output_failures() -> None:
    fake_hf_token = "test-hf-token"  # noqa: S105
    provider = PyannoteDiarizationProvider(token=fake_hf_token)
    with (
        patch("clipper.providers.speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="diarization"),
    ):
        provider._load()
    pipeline = Mock()
    module = SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipeline)))
    device_provider = PyannoteDiarizationProvider(token=fake_hf_token, device="cuda")
    with (
        patch(
            "clipper.providers.speech.importlib.import_module", side_effect=[module, ImportError()]
        ),
        pytest.raises(ProviderUnavailable, match="torch device"),
    ):
        device_provider._load()
    with pytest.raises(ValueError, match="no speaker diarization tracks"):
        PyannoteDiarizationProvider._turns(object())
    empty = SimpleNamespace(itertracks=lambda **_kwargs: iter(()))
    with pytest.raises(ValueError, match="no speaker turns"):
        PyannoteDiarizationProvider._turns(empty)


def test_canonical_roundtrip_word_payloads_and_segment_grouping() -> None:
    timeline = canonical_timeline_from_word_payloads(
        "video",
        "source-hash",
        [
            {
                "text": "hello",
                "start": 0.0,
                "end": 0.2,
                "confidence": 0.9,
            },
            {
                "text": "there",
                "start": 0.21,
                "end": 0.45,
                "confidence": 0.8,
            },
            {
                "text": "again",
                "start": 2.0,
                "end": 2.3,
                "confidence": None,
            },
        ],
        transcript_source="modal-asr",
    )
    timeline = replace(
        timeline,
        words=tuple(
            replace(word, speaker_id="SPEAKER_00" if index < 2 else "SPEAKER_01")
            for index, word in enumerate(timeline.words)
        ),
    )
    restored = CanonicalTimeline.from_dict(timeline.to_dict())
    assert restored == timeline
    segments = transcript_segments_from_canonical(restored, max_gap_seconds=0.5)
    assert [segment.text for segment in segments] == ["hello there", "again"]
    assert [segment.speaker_id for segment in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert all(segment.words for segment in segments)
    with pytest.raises(ValueError, match="grouping"):
        transcript_segments_from_canonical(restored, max_words=0)
    with pytest.raises(ValueError, match="no canonical words"):
        canonical_timeline_from_word_payloads(
            "video", "source-hash", [{"text": "", "start": 0, "end": 0}], transcript_source="x"
        )


def test_modal_media_bridge_and_speech_adapters(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    volume = Mock()
    volume.listdir.return_value = []
    upload = Mock()
    manager = Mock()
    manager.__enter__ = Mock(return_value=upload)
    manager.__exit__ = Mock(return_value=False)
    volume.batch_upload.return_value = manager
    modal_module = SimpleNamespace(Volume=SimpleNamespace(from_name=Mock(return_value=volume)))
    bridge = ModalMediaBridge("media")
    with patch("clipper.providers.modal_speech.importlib.import_module", return_value=modal_module):
        remote = bridge.ensure_uploaded(source, "abc")
        assert remote == "/media/inputs/abc.wav"
        assert bridge.ensure_uploaded(source, "abc") == remote
    upload.put_file.assert_called_once_with(str(source), "/inputs/abc.wav")

    identity = ModelIdentity("speech", "rev", "none", "modal")
    transcribe = ModalTranscriptionProvider(
        app_name="app", function_name="transcribe", identity=identity, media_bridge=bridge
    )
    function = Mock()
    function.remote.return_value = {
        "words": [{"text": "hello", "start": 0.0, "end": 0.2, "confidence": 0.9}],
        "model": {"model_id": "resolved/asr", "revision": "asr-sha"},
        "usage": {"gpu_type": "L4", "gpu_seconds": 1.0},
    }
    with (
        patch.object(bridge, "ensure_uploaded", return_value="/media/inputs/abc.wav"),
        patch.object(transcribe, "_function", return_value=function),
    ):
        result = transcribe.transcribe(source, video_id="video", source_hash="abc")
    assert result.value.words[0].text == "hello"
    assert result.model.model_id == "resolved/asr"
    assert result.model.revision == "asr-sha"
    assert result.usage.gpu_type == "L4"

    timeline = result.value
    align = ModalAlignmentProvider(
        app_name="app", function_name="align", identity=identity, media_bridge=bridge
    )
    function.remote.return_value = {
        "segments": [{"words": [{"word": "hello", "start": 0.01, "end": 0.21, "score": 0.95}]}],
        "model": {"model_id": "resolved/align", "revision": "align-sha"},
    }
    with (
        patch.object(bridge, "ensure_uploaded", return_value="/media/inputs/abc.wav"),
        patch.object(align, "_function", return_value=function),
    ):
        aligned_result = align.align(source, timeline)
    aligned = aligned_result.value
    assert aligned.words[0].timing_mode == "aligned"
    assert aligned_result.model.revision == "align-sha"

    diarize = ModalDiarizationProvider(
        app_name="app", function_name="diarize", identity=identity, media_bridge=bridge
    )
    function.remote.return_value = {
        "turns": [[0.0, 1.0, "SPEAKER_00"]],
        "model": {"model_id": "resolved/diar", "revision": "diar-sha"},
    }
    with (
        patch.object(bridge, "ensure_uploaded", return_value="/media/inputs/abc.wav"),
        patch.object(diarize, "_function", return_value=function),
    ):
        spoken_result = diarize.diarize(source, aligned)
    spoken = spoken_result.value
    assert spoken.words[0].speaker_id == "SPEAKER_00"
    assert spoken_result.model.revision == "diar-sha"

    function.remote.return_value = {
        "error": {"type": "GatedRepoError", "message": "403 gated config.yaml"}
    }
    with (
        patch.object(bridge, "ensure_uploaded", return_value="/media/inputs/abc.wav"),
        patch.object(diarize, "_function", return_value=function),
        pytest.raises(RuntimeError, match=r"GatedRepoError: 403 gated config\.yaml"),
    ):
        diarize.diarize(source, aligned)


def test_speech_provider_factory_routes_local_modal_and_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = speech_providers("local-lite")
    assert isinstance(local[0], FasterWhisperTranscriptionProvider)
    assert isinstance(local[1], WhisperXAlignmentProvider)
    assert isinstance(local[2], PyannoteDiarizationProvider)
    remote = speech_providers("balanced")
    assert isinstance(remote[0], ModalTranscriptionProvider)
    assert isinstance(remote[1], ModalAlignmentProvider)
    assert isinstance(remote[2], ModalDiarizationProvider)

    monkeypatch.setenv("CLIPPER_DIARIZATION_MODE", "passthrough")
    degraded = speech_providers("balanced")
    assert isinstance(degraded[2], PassthroughDiarizationProvider)
    timeline = _timeline()
    result = degraded[2].diarize(Path("unused.wav"), timeline)
    assert result.value is timeline
    assert result.degraded is True
    assert result.model.model_id == "none/passthrough-diarization"

    monkeypatch.setenv("CLIPPER_DIARIZATION_MODE", "invalid")
    with pytest.raises(ValueError, match="unsupported diarization mode"):
        speech_providers("balanced")


def test_plan_batch_reports_duration_rejections_when_all_model_plans_are_invalid(
    tmp_path: Path,
) -> None:
    timeline = _timeline()
    concept = ClipConcept(
        "c",
        "video",
        10.0,
        11.1,
        "what's one message today",
        "summary",
        "",
        "",
        "question",
        20.0,
        EditorialScores(*(8.0 for _ in range(12))),
        8.0,
        "sem",
        "fp",
    )
    moment = StoryMoment(
        "m",
        "video",
        10.0,
        11.1,
        concept.text,
        "question",
        "summary",
        "",
        "",
        EditorialScores(*(8.0 for _ in range(12))),
        8.0,
        "fp",
    )
    grounded = _grounded_concept("c", "summary")
    analysis = OpenVideoAnalysis(
        EpisodeEditorialProfile("x", ("x",), (), 0.9),
        [moment],
        [concept],
        {
            "m": GroundedStoryMoment(
                "m", ("w1", "w2", "w3", "w4"), "x", "question", "", "", "x", 0.9
            )
        },
        {"c": grounded},
        [],
    )

    class InvalidDuration(_PlannerEditorial):
        def complete_json(
            self, *, task: str, payload: dict[str, object]
        ) -> ProviderResult[dict[str, object]]:
            del payload
            if task == "global_concept_comparison":
                value: dict[str, object] = {"concept_ids": ["c"]}
            elif task == "hook_variants:c":
                value = {
                    "variants": [
                        {
                            "variant_id": "h",
                            "strategy_label": "direct",
                            "source_word_ids": ["w1", "w2", "w3", "w4"],
                            "overlay_text": None,
                            "rationale": "source grounded",
                            "confidence": 0.9,
                        }
                    ]
                }
            elif task == "edit_plans:c":
                value = {
                    "plans": [
                        {
                            "plan_id": "p",
                            "concept_id": "c",
                            "variant_id": "h",
                            "source_word_ids": ["w1", "w2", "w3", "w4"],
                            "hook_source_word_ids": ["w1", "w2"],
                            "overlay_text": None,
                            "strategy_label": "direct",
                            "caption_platform": "tiktok",
                            "confidence": 0.9,
                        }
                    ]
                }
            elif task.startswith("boundary_audit:"):
                value = _passing_boundary_payload()
            else:
                raise AssertionError(task)
            return ProviderResult(value, self.identity, InferenceUsage("test", "now", 0.01))

    with pytest.raises(EditorialGroundingError, match=r"duration_outside_campaign_bounds.*1\.1"):
        AutonomousEditorialPlanner(
            InvalidDuration(), _PlannerEmbeddings(), FileCache(tmp_path / "duration")
        ).plan_batch(_open_brief(), {"video": timeline}, [analysis])


def test_plan_batch_rejects_oversized_hook_and_keeps_other_valid_plan(tmp_path: Path) -> None:
    timeline = _long_grounded_timeline(200)
    brief = replace(_open_brief(), min_clip_seconds=45, max_clip_seconds=60)
    all_ids = tuple(word.word_id for word in timeline.words)
    grounded = GroundedClipConcept("c", ("m",), all_ids, "complete story", "", "story", 55, (), 0.9)
    concept = ClipConcept(
        "c",
        "video",
        timeline.start,
        timeline.end,
        "complete story",
        "summary",
        "",
        "",
        "story",
        55,
        EditorialScores(*(8.0 for _ in range(12))),
        8.0,
        "sem",
        "fp",
    )
    moment = StoryMoment(
        "m",
        "video",
        timeline.start,
        timeline.end,
        "complete story",
        "story",
        "summary",
        "",
        "",
        EditorialScores(*(8.0 for _ in range(12))),
        8.0,
        "fp",
    )
    analysis = OpenVideoAnalysis(
        EpisodeEditorialProfile("x", ("x",), (), 0.9),
        [moment],
        [concept],
        {
            "m": GroundedStoryMoment(
                "m", all_ids, "complete story", "story", "", "", "grounded", 0.9
            )
        },
        {"c": grounded},
        [],
    )

    class MixedPlans(_PlannerEditorial):
        def complete_json(
            self, *, task: str, payload: dict[str, object]
        ) -> ProviderResult[dict[str, object]]:
            del payload
            if task == "global_concept_comparison":
                value: dict[str, object] = {"concept_ids": ["c"]}
            elif task == "hook_variants:c":
                value = {
                    "variants": [
                        {
                            "variant_id": "h-long",
                            "strategy_label": "overlay",
                            "source_word_ids": list(all_ids),
                            "overlay_text": "A grounded overlay.",
                            "rationale": "source grounded",
                            "confidence": 0.95,
                        },
                        {
                            "variant_id": "h-good",
                            "strategy_label": "direct",
                            "source_word_ids": list(all_ids[:4]),
                            "overlay_text": None,
                            "rationale": "source grounded",
                            "confidence": 0.9,
                        },
                    ]
                }
            elif task == "edit_plans:c":
                value = {
                    "plans": [
                        {
                            "plan_id": "p-bad",
                            "concept_id": "c",
                            "variant_id": "h-long",
                            "source_word_ids": list(all_ids),
                            "hook_source_word_ids": list(all_ids),
                            "overlay_text": "A grounded overlay.",
                            "strategy_label": "overlay",
                            "caption_platform": "tiktok",
                            "confidence": 0.95,
                        },
                        {
                            "plan_id": "p-good",
                            "concept_id": "c",
                            "variant_id": "h-good",
                            "source_word_ids": list(all_ids[:120]),
                            "hook_source_word_ids": list(all_ids[:4]),
                            "overlay_text": None,
                            "strategy_label": "direct",
                            "caption_platform": "tiktok",
                            "confidence": 0.9,
                        },
                    ]
                }
            elif task.startswith("boundary_audit:"):
                value = _passing_boundary_payload()
            else:
                raise AssertionError(task)
            return ProviderResult(value, self.identity, InferenceUsage("test", "now", 0.01))

    batch = AutonomousEditorialPlanner(
        MixedPlans(), _PlannerEmbeddings(), FileCache(tmp_path / "mixed-duration")
    ).plan_batch(brief, {"video": timeline}, [analysis])

    assert [plan.plan_id for plan in batch.plans] == ["p-good"]
    assert batch.plans[0].duration == pytest.approx(59.92)
    bad_rejection = next(item for item in batch.rejections if item.get("plan_id") == "p-bad")
    assert bad_rejection["reasons"] == ["duration_outside_campaign_bounds_no_grounded_repair"]


def test_plan_batch_rejects_failed_model_completion_and_keeps_other_concept(
    tmp_path: Path,
) -> None:
    timeline = _long_grounded_timeline()
    all_ids = tuple(word.word_id for word in timeline.words)
    concept_ids = ("c-bad", "c-good")
    moments = [
        StoryMoment(
            f"m-{concept_id}",
            "video",
            timeline.start,
            timeline.end,
            f"story {concept_id}",
            "story",
            "summary",
            "",
            "",
            EditorialScores(*(8.0 for _ in range(12))),
            8.0,
            "fp",
        )
        for concept_id in concept_ids
    ]
    concepts = [
        ClipConcept(
            concept_id,
            "video",
            timeline.start,
            timeline.end,
            f"story {concept_id}",
            "summary",
            "",
            "",
            "story",
            30,
            EditorialScores(*(8.0 for _ in range(12))),
            8.0,
            "sem",
            "fp",
        )
        for concept_id in concept_ids
    ]
    grounded_moments = {
        f"m-{concept_id}": GroundedStoryMoment(
            f"m-{concept_id}", all_ids, f"story {concept_id}", "story", "", "", "grounded", 0.9
        )
        for concept_id in concept_ids
    }
    grounded_concepts = {
        concept_id: GroundedClipConcept(
            concept_id,
            (f"m-{concept_id}",),
            all_ids,
            f"story {concept_id}",
            "",
            "story",
            30,
            (),
            0.9,
        )
        for concept_id in concept_ids
    }
    analysis = OpenVideoAnalysis(
        EpisodeEditorialProfile("x", ("x",), (), 0.9),
        moments,
        concepts,
        grounded_moments,
        grounded_concepts,
        [],
    )

    class OneFailedPlan(_PlannerEditorial):
        def complete_json(
            self, *, task: str, payload: dict[str, object]
        ) -> ProviderResult[dict[str, object]]:
            del payload
            if task == "global_concept_comparison":
                value: dict[str, object] = {"concept_ids": list(concept_ids)}
            elif task.startswith("hook_variants:"):
                concept_id = task.removeprefix("hook_variants:")
                value = {
                    "variants": [
                        {
                            "variant_id": f"h-{concept_id}",
                            "strategy_label": "direct",
                            "source_word_ids": list(all_ids[:4]),
                            "overlay_text": None,
                            "rationale": "source grounded",
                            "confidence": 0.9,
                        }
                    ]
                }
            elif task == "edit_plans:c-bad":
                raise RuntimeError("JSONDecodeError: exhausted retries")
            elif task == "edit_plans:c-good":
                value = {
                    "plans": [
                        {
                            "plan_id": "p-good",
                            "concept_id": "c-good",
                            "variant_id": "h-c-good",
                            "source_word_ids": list(all_ids[:80]),
                            "hook_source_word_ids": list(all_ids[:4]),
                            "overlay_text": None,
                            "strategy_label": "direct",
                            "caption_platform": "tiktok",
                            "confidence": 0.9,
                        }
                    ]
                }
            elif task.startswith("boundary_audit:"):
                value = _passing_boundary_payload()
            else:
                raise AssertionError(task)
            return ProviderResult(value, self.identity, InferenceUsage("test", "now", 0.01))

    batch = AutonomousEditorialPlanner(
        OneFailedPlan(), _PlannerEmbeddings(), FileCache(tmp_path / "model-failure")
    ).plan_batch(_open_brief(), {"video": timeline}, [analysis])

    assert [plan.plan_id for plan in batch.plans] == ["p-good"]
    rejection = next(item for item in batch.rejections if item.get("concept_id") == "c-bad")
    assert rejection["stage"] == "edit_plan"
    assert rejection["reasons"] == ["model_completion_failed"]
    assert rejection["error_type"] == "RuntimeError"
    assert "JSONDecodeError" in str(rejection["error"])


def test_visual_timeline_roundtrip_and_payload_validation() -> None:
    timeline = VisualTimeline(
        "v",
        "hash",
        (VisualEvent(0.2, 1.0, "s", "reaction", ("A",), ("laugh",), 0.7),),
    )
    assert VisualTimeline.from_dict(timeline.to_dict()) == timeline
    with pytest.raises(ValueError, match="events must be a list"):
        VisualTimeline.from_dict({"video_id": "v", "source_hash": "h", "events": {}})
    with pytest.raises(ValueError, match="event must be an object"):
        VisualTimeline.from_dict({"video_id": "v", "source_hash": "h", "events": ["bad"]})
    with pytest.raises(ValueError, match="visible_speakers"):
        VisualTimeline.from_dict(
            {
                "video_id": "v",
                "source_hash": "h",
                "events": [
                    {"start": 0, "end": 1, "scene_id": "s", "summary": "x", "visible_speakers": "A"}
                ],
            }
        )
    with pytest.raises(ValueError, match="event_labels"):
        VisualTimeline.from_dict(
            {
                "video_id": "v",
                "source_hash": "h",
                "events": [
                    {"start": 0, "end": 1, "scene_id": "s", "summary": "x", "event_labels": "x"}
                ],
            }
        )


def test_canonical_word_refs_accept_unique_model_truncated_digest_suffix() -> None:
    timeline = _timeline()
    assert timeline.resolve_word_ref("w1:partial") == "w1"
    assert timeline.resolve_word_ref("video:w1:partial") == "w1"
    with pytest.raises(ValueError, match="unknown canonical"):
        timeline.resolve_word_ref("w9999999:partial")


def test_open_analysis_rejects_bad_proposal_without_discarding_valid_moment(
    tmp_path: Path,
) -> None:
    brief = _open_brief()
    timeline = _timeline()

    class Mixed(_PlannerEditorial):
        def complete_json(
            self, *, task: str, payload: dict[str, object]
        ) -> ProviderResult[dict[str, object]]:
            del payload
            if task == "episode_editorial_profile":
                value: dict[str, object] = {
                    "summary": "grounded test",
                    "valuable_moment_characteristics": ["self contained"],
                    "avoid_characteristics": [],
                    "confidence": 0.9,
                }
            elif task == "story_moments:0":
                value = {
                    "moments": [
                        {
                            "moment_id": "bad",
                            "start_word_id": "w9999999:invented",
                            "end_word_id": "w4",
                            "semantic_summary": "bad",
                            "narrative_structure": "bad",
                            "editorial_reason": "bad",
                            "confidence": 0.9,
                        },
                        {
                            "moment_id": "good",
                            "start_word_id": "w1:partial",
                            "end_word_id": "w4",
                            "semantic_summary": "valid grounded moment",
                            "narrative_structure": "answer",
                            "editorial_reason": "self contained",
                            "confidence": 0.9,
                        },
                    ]
                }
            elif task == "clip_concepts":
                value = {
                    "concepts": [
                        {
                            "concept_id": "c1",
                            "story_moment_ids": ["chunk-0:good"],
                            "start_word_id": "w1",
                            "end_word_id": "w4",
                            "semantic_summary": "valid concept",
                            "standalone_context": "",
                            "narrative_structure": "answer",
                            "recommended_duration": 20,
                            "visual_dependencies": [],
                            "confidence": 0.9,
                        }
                    ]
                }
            else:
                raise AssertionError(task)
            return ProviderResult(value, self.identity, InferenceUsage("test", "now", 0.01))

    analysis = AutonomousEditorialPlanner(
        Mixed(), _PlannerEmbeddings([[1.0, 0.0]]), FileCache(tmp_path / "mixed")
    ).analyze_video(brief, timeline)
    assert list(analysis.grounded_moments) == ["chunk-0:good"]
    assert [concept.concept_id for concept in analysis.concepts] == ["c1"]
    assert any(
        item.get("reasons") == ["invalid_grounded_story_moment"] for item in analysis.rejections
    )


def test_story_moment_alias_disambiguates_with_grounded_word_overlap() -> None:
    moments = {
        "chunk-0:m1": GroundedStoryMoment(
            "chunk-0:m1",
            ("w1", "w2", "w3"),
            "first",
            "story",
            "",
            "",
            "grounded",
            0.9,
        ),
        "chunk-1:m1": GroundedStoryMoment(
            "chunk-1:m1",
            ("w5", "w6", "w7"),
            "second",
            "story",
            "",
            "",
            "grounded",
            0.9,
        ),
    }
    concept = GroundedClipConcept(
        "c1",
        ("m1",),
        ("w5", "w6", "w7", "w8"),
        "second concept",
        "",
        "story",
        20.0,
        (),
        0.9,
    )
    resolved = AutonomousEditorialPlanner._resolve_story_moment_id("m1", concept, moments)
    assert resolved == "chunk-1:m1"
    assert (
        AutonomousEditorialPlanner._resolve_story_moment_id("chunk-0:m1", concept, moments)
        == "chunk-0:m1"
    )


def test_story_moment_alias_keeps_true_overlap_ties_ambiguous() -> None:
    moments = {
        "chunk-0:m1": GroundedStoryMoment(
            "chunk-0:m1",
            ("w1", "w2"),
            "first",
            "story",
            "",
            "",
            "grounded",
            0.9,
        ),
        "chunk-1:m1": GroundedStoryMoment(
            "chunk-1:m1",
            ("w3", "w4"),
            "second",
            "story",
            "",
            "",
            "grounded",
            0.9,
        ),
    }
    concept = GroundedClipConcept(
        "c1",
        ("m1",),
        ("w1", "w3"),
        "ambiguous concept",
        "",
        "story",
        20.0,
        (),
        0.9,
    )
    assert AutonomousEditorialPlanner._resolve_story_moment_id("m1", concept, moments) is None


def test_story_moment_chunk_inference_failure_preserves_other_grounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    words = tuple(
        CanonicalWord(
            f"w{i}",
            f"token{i}",
            i * 0.1,
            i * 0.1 + 0.08,
            "A",
            0.99,
            "aligned",
            "test",
        )
        for i in range(400)
    )
    timeline = CanonicalTimeline("video", "source-hash", words)
    identity = ModelIdentity("test", "rev", None, "test", "p", "s")
    editorial = Mock(identity=identity)
    embeddings = Mock(identity=identity)
    planner = AutonomousEditorialPlanner(
        editorial,
        embeddings,
        FileCache(tmp_path / "cache"),
        max_words_per_chunk=200,
        chunk_overlap_words=0,
    )

    def complete(stage: str, *_args: object, **_kwargs: object) -> dict[str, object]:
        if stage == "episode_editorial_profile":
            return {
                "summary": "episode",
                "valuable_moment_characteristics": ["self contained"],
                "avoid_characteristics": [],
                "confidence": 0.9,
            }
        if stage == "story_moments:0":
            raise RuntimeError("transient chunk failure")
        if stage == "story_moments:1":
            return {
                "moments": [
                    {
                        "moment_id": "m1",
                        "start_word_id": timeline.word_ref("w220"),
                        "end_word_id": timeline.word_ref("w320"),
                        "semantic_summary": "grounded surviving moment",
                        "narrative_structure": "story",
                        "required_prior_context": "",
                        "required_followup_context": "",
                        "editorial_reason": "self contained",
                        "confidence": 0.9,
                    }
                ]
            }
        if stage == "clip_concepts":
            return {
                "concepts": [
                    {
                        "concept_id": "c1",
                        "story_moment_ids": ["chunk-1:m1"],
                        "start_word_id": timeline.word_ref("w220"),
                        "end_word_id": timeline.word_ref("w320"),
                        "semantic_summary": "surviving concept",
                        "standalone_context": "",
                        "narrative_structure": "story",
                        "recommended_duration": 10.0,
                        "visual_dependencies": [],
                        "confidence": 0.9,
                    }
                ]
            }
        raise AssertionError(stage)

    monkeypatch.setattr(planner, "_complete", complete)
    monkeypatch.setattr(
        planner,
        "_semantic_dedupe",
        lambda _brief, _timeline, concepts: (
            concepts,
            {concept.concept_id: "sem-test" for concept in concepts},
            [],
        ),
    )
    brief = CampaignBrief(
        campaign_id="test",
        title="Test",
        objective="Test grounded editorial resilience.",
        keywords=["test"],
        source_channel_ids=["channel"],
        clip_count=1,
        min_clip_seconds=5.0,
        max_clip_seconds=60.0,
        watermark_text="TEST",
    )
    analysis = planner.analyze_video(brief, timeline)
    assert len(analysis.moments) == 1
    assert len(analysis.concepts) == 1
    assert any(item.get("reasons") == ["chunk_inference_failed"] for item in analysis.rejections)

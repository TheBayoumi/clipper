from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from clipper.canonical import CanonicalTimeline, CanonicalWord, canonical_timeline_from_segments
from clipper.models import TranscriptSegment, TranscriptWord
from clipper.providers.base import ModelIdentity, compute_profile
from clipper.providers.local import (
    LocalEditorialProvider,
    LocalEmbeddingProvider,
    LocalVisionProvider,
    ProviderUnavailable,
)
from clipper.providers.modal import ModalEditorialProvider, ModalVisionProvider
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


def test_modal_adapters_validate_response_and_record_usage() -> None:
    identity = ModelIdentity("m", "r", "none", "modal")
    function = Mock()
    function.remote.return_value = {
        "value": {"ok": True},
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
    editorial = ModalEditorialProvider(
        app_name="clipper", function_name="editorial", identity=identity
    )
    vision = ModalVisionProvider(app_name="clipper", function_name="vision", identity=identity)
    with patch.object(editorial, "_function", return_value=function):
        result = editorial.complete_json(task="mine", payload={})
    assert result.value == {"ok": True}
    assert result.usage.gpu_type == "L40S"
    with patch.object(vision, "_function", return_value=function):
        assert vision.inspect(task="review", frames=[Path("a.jpg")], context={}).value["ok"] is True
    function.remote.return_value = {"value": []}
    with (
        patch.object(editorial, "_function", return_value=function),
        pytest.raises(ValueError, match="invalid response"),
    ):
        editorial.complete_json(task="mine", payload={})


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
    with pytest.raises(EditorialGroundingError, match="wrong source"):
        GroundedEditPlan.from_payload(
            {
                "plan_id": "p",
                "video_id": "other",
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


def test_grounded_noncontiguous_plan_is_rejected_at_compile() -> None:
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
    with pytest.raises(EditorialGroundingError, match="contiguous"):
        grounded.compile(timeline, "fp")


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

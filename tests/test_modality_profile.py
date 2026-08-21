import pytest

from clipper.modality_profile import (
    SourceModalityProfile,
    assert_required_modalities_available,
    infer_source_modality_profile,
)
from clipper.multimodal_timeline import MultimodalEvent, MultimodalTimeline


def _timeline(events: tuple[MultimodalEvent, ...], duration: float = 10.0) -> MultimodalTimeline:
    return MultimodalTimeline("video", "source", duration, events)


def test_speech_dominant_evidence_does_not_require_visual_perception() -> None:
    profile = infer_source_modality_profile(
        _timeline(
            (
                MultimodalEvent(
                    0.0,
                    10.0,
                    transcript_word_ids=("w1",),
                    speaker_ids=("s1",),
                    confidence=0.9,
                ),
            )
        )
    )
    assert profile.speech_dependency > profile.visual_dependency
    assert profile.requires_visual_evidence is False
    assert_required_modalities_available(profile)


def test_action_rich_evidence_requires_visual_perception_without_source_type_labels() -> None:
    profile = infer_source_modality_profile(
        _timeline(
            (
                MultimodalEvent(
                    0.0,
                    10.0,
                    scene_ids=("scene",),
                    actions=("moves",),
                    objects=("object",),
                    visual_summaries=("visible event",),
                    visual_salience=0.95,
                    motion_salience=0.95,
                    confidence=0.9,
                ),
            )
        )
    )
    assert profile.visual_dependency > profile.speech_dependency
    assert profile.action_dependency > 0.5
    assert profile.requires_visual_evidence is True
    assert profile.visual_evidence_coverage == 1.0
    assert_required_modalities_available(profile)


def test_screen_text_signal_is_first_class_visual_dependency() -> None:
    profile = infer_source_modality_profile(
        _timeline(
            (
                MultimodalEvent(
                    0.0,
                    10.0,
                    scene_ids=("scene",),
                    ocr_text=("visible text",),
                    visual_summaries=("screen evidence",),
                    visual_salience=0.8,
                    confidence=0.8,
                ),
            )
        )
    )
    assert profile.screen_text_dependency > 0.5
    assert profile.requires_visual_evidence


def test_speaker_identity_dependency_uses_cross_modal_evidence() -> None:
    profile = infer_source_modality_profile(
        _timeline(
            (
                MultimodalEvent(
                    0.0,
                    10.0,
                    transcript_word_ids=("w1",),
                    speaker_ids=("speaker-a",),
                    scene_ids=("scene",),
                    visible_people=("speaker-a",),
                    visual_summaries=("speaker visible",),
                    visual_salience=0.8,
                    confidence=0.8,
                ),
            )
        )
    )
    assert profile.requires_speaker_identity


def test_required_visual_evidence_fails_closed_when_coverage_is_insufficient() -> None:
    profile = SourceModalityProfile(
        speech_dependency=0.2,
        visual_dependency=0.9,
        motion_dependency=0.8,
        screen_text_dependency=0.0,
        speaker_identity_dependency=0.0,
        action_dependency=0.9,
        visual_evidence_coverage=0.2,
        confidence=0.9,
    )
    with pytest.raises(RuntimeError, match="coverage is insufficient"):
        assert_required_modalities_available(profile)


def test_empty_evidence_has_zero_dependencies_and_profile_validation_is_strict() -> None:
    profile = infer_source_modality_profile(_timeline((), duration=0.0))
    assert profile.to_dict()["confidence"] == 0.0
    assert not profile.requires_visual_evidence

    with pytest.raises(ValueError, match="between 0 and 1"):
        SourceModalityProfile(1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

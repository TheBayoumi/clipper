from __future__ import annotations

from clipper.brief import load_brief
from clipper.editorial_integrity import HazardClassification, SourceHazardSegment
from clipper.modality_profile import (
    assert_required_modalities_available,
    infer_source_modality_profile,
)
from clipper.multimodal_timeline import MultimodalEvent, MultimodalTimeline
from clipper.quality_moments import choose_quality_moments
from clipper.quality_pipeline import forbidden_spans_for_campaign, source_branding_evidence


def _event(
    start: float,
    end: float,
    *,
    speech: bool = False,
    people: tuple[str, ...] = (),
    motion: bool = False,
    action: bool = False,
    ocr: bool = False,
    branding: tuple[str, ...] = (),
) -> MultimodalEvent:
    return MultimodalEvent(
        start=start,
        end=end,
        transcript_word_ids=((f"w-{start}",) if speech else ()),
        speaker_ids=(("speaker",) if speech else ()),
        scene_ids=(f"scene-{start}",),
        visible_people=people,
        actions=(("visible-action",) if action else ()),
        objects=(("relevant-object",) if action else ()),
        ocr_text=(("screen text",) if ocr else ()),
        branding=branding,
        visual_summaries=("grounded visual evidence",),
        visual_salience=0.95,
        motion_salience=0.95 if motion else 0.0,
        confidence=0.95,
    )


def _profile(*events: MultimodalEvent, duration: float = 60.0):
    return infer_source_modality_profile(MultimodalTimeline("video", "source", duration, events))


def test_phase_k_two_person_podcast_is_speech_and_speaker_identity_aware() -> None:
    profile = _profile(
        _event(0.0, 30.0, speech=True, people=("speaker-a", "speaker-b")),
        _event(30.0, 60.0, speech=True, people=("speaker-a", "speaker-b")),
    )
    assert profile.speech_dependency > profile.visual_dependency
    assert profile.requires_speaker_identity
    assert_required_modalities_available(profile)


def test_phase_k_single_person_talking_head_remains_speech_led() -> None:
    profile = _profile(
        _event(0.0, 60.0, speech=True, people=("speaker-a",)),
    )
    assert profile.speech_dependency > profile.visual_dependency
    assert profile.speaker_identity_dependency >= 0.45


def test_phase_k_screen_tutorial_requires_screen_text_visual_evidence() -> None:
    profile = _profile(_event(0.0, 60.0, speech=True, ocr=True))
    assert profile.screen_text_dependency >= 0.7
    assert profile.requires_visual_evidence
    assert_required_modalities_available(profile)


def test_phase_k_gameplay_requires_motion_and_action_evidence() -> None:
    profile = _profile(_event(0.0, 60.0, motion=True, action=True))
    assert profile.motion_dependency >= 0.7
    assert profile.action_dependency >= 0.7
    assert profile.requires_visual_evidence


def test_phase_k_sports_action_requires_motion_and_action_evidence() -> None:
    profile = _profile(
        _event(0.0, 20.0, motion=True, action=True, people=("athlete-a",)),
        _event(20.0, 40.0, motion=True, action=True, people=("athlete-b",)),
        _event(40.0, 60.0, motion=True, action=True, people=("athlete-a", "athlete-b")),
    )
    assert profile.motion_dependency >= 0.7
    assert profile.visual_dependency >= 0.45
    assert profile.requires_visual_evidence


def test_phase_k_visual_demonstration_can_be_visual_first_without_speech() -> None:
    profile = _profile(_event(0.0, 60.0, motion=True, action=True))
    assert profile.speech_dependency == 0.0
    assert profile.visual_dependency >= 0.45
    assert profile.requires_visual_evidence


def test_phase_k_low_speech_source_does_not_silently_become_transcript_first() -> None:
    profile = _profile(
        _event(0.0, 10.0, speech=True, motion=True, action=True),
        _event(10.0, 60.0, motion=True, action=True),
    )
    assert profile.speech_dependency < 0.5
    assert profile.visual_dependency > profile.speech_dependency
    assert profile.requires_visual_evidence


def test_phase_k_sponsor_region_is_excluded_before_window_ranking() -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    sponsor = SourceHazardSegment(
        10.0,
        20.0,
        HazardClassification.SPONSOR_READ,
        0.99,
        ("spoken sponsor evidence",),
        {"model": "test"},
    )
    spans = forbidden_spans_for_campaign(brief, (sponsor,), ())
    assert spans
    assert spans[0].start <= sponsor.start
    assert spans[0].end >= sponsor.end


def test_phase_k_source_logo_becomes_a_forbidden_campaign_span() -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = MultimodalTimeline(
        "video",
        "source",
        60.0,
        (_event(5.0, 15.0, branding=("foreign source logo",)),),
    )
    branding = source_branding_evidence(timeline)
    spans = forbidden_spans_for_campaign(brief, (), branding)
    assert len(branding) == 1
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (5.0, 15.0)


def test_phase_k_source_with_no_worthwhile_moments_finishes_with_zero_quality_yield() -> None:
    assert choose_quality_moments((), (), (), ()) == ()

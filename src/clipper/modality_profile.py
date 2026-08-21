from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .multimodal_timeline import MultimodalTimeline


@dataclass(frozen=True, slots=True)
class SourceModalityProfile:
    speech_dependency: float
    visual_dependency: float
    motion_dependency: float
    screen_text_dependency: float
    speaker_identity_dependency: float
    action_dependency: float
    visual_evidence_coverage: float
    confidence: float
    schema_version: str = "source-modality-profile-v1"

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if field == "schema_version":
                continue
            if not isinstance(value, int | float) or not 0 <= float(value) <= 1:
                raise ValueError(f"source modality {field} must be between 0 and 1")

    @property
    def requires_visual_evidence(self) -> bool:
        visual_need = max(
            self.visual_dependency,
            self.motion_dependency,
            self.screen_text_dependency,
            self.action_dependency,
        )
        return visual_need >= 0.45

    @property
    def requires_speaker_identity(self) -> bool:
        return self.speaker_identity_dependency >= 0.45

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _covered_duration(timeline: MultimodalTimeline, predicate: object) -> float:
    check = predicate
    if not callable(check):
        raise TypeError("modality coverage predicate must be callable")
    return sum(event.duration for event in timeline.events if check(event))


def infer_source_modality_profile(timeline: MultimodalTimeline) -> SourceModalityProfile:
    """Infer modality dependence from observed evidence, never from source-type labels."""

    duration = timeline.duration
    if duration <= 0:
        return SourceModalityProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    speech = _covered_duration(timeline, lambda event: bool(event.transcript_word_ids)) / duration
    visual = _covered_duration(
        timeline,
        lambda event: bool(event.scene_ids or event.visual_summaries or event.visible_people),
    ) / duration
    motion = _covered_duration(timeline, lambda event: event.motion_salience > 0) / duration
    screen_text = _covered_duration(timeline, lambda event: bool(event.ocr_text)) / duration
    speakers = _covered_duration(
        timeline,
        lambda event: bool(event.speaker_ids and event.visible_people),
    ) / duration
    actions = _covered_duration(timeline, lambda event: bool(event.actions)) / duration

    visual_richness = min(1.0, 0.45 * visual + 0.25 * motion + 0.15 * screen_text + 0.15 * actions)
    speech_dependency = min(1.0, speech * (1.0 - 0.35 * visual_richness))
    visual_dependency = min(1.0, visual_richness * (1.0 - 0.25 * speech))
    motion_dependency = min(1.0, motion * (0.7 + 0.3 * visual))
    screen_dependency = min(1.0, screen_text * (0.7 + 0.3 * visual))
    speaker_dependency = min(1.0, speakers * (0.6 + 0.4 * speech))
    action_dependency = min(1.0, actions * (0.7 + 0.3 * motion))

    signal_count = sum(
        1
        for value in (speech, visual, motion, screen_text, speakers, actions)
        if value > 0
    )
    confidence = min(1.0, 0.25 + 0.1 * signal_count + 0.45 * max(speech, visual))

    return SourceModalityProfile(
        speech_dependency=max(0.0, speech_dependency),
        visual_dependency=max(0.0, visual_dependency),
        motion_dependency=max(0.0, motion_dependency),
        screen_text_dependency=max(0.0, screen_dependency),
        speaker_identity_dependency=max(0.0, speaker_dependency),
        action_dependency=max(0.0, action_dependency),
        visual_evidence_coverage=max(0.0, min(1.0, visual)),
        confidence=max(0.0, min(1.0, confidence)),
    )


def assert_required_modalities_available(profile: SourceModalityProfile) -> None:
    if profile.requires_visual_evidence and profile.visual_evidence_coverage < 0.5:
        raise RuntimeError(
            "source modality profile requires visual evidence but visual perception coverage is insufficient"
        )

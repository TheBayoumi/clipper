"""Automated, rights-gated short-form production pipeline."""

from .models import (
    CampaignBrief,
    ClipCandidate,
    ClipConcept,
    EditPlan,
    HookVariant,
    StoryMoment,
    TranscriptSegment,
    TranscriptWord,
    VideoCandidate,
)

__all__ = [
    "CampaignBrief",
    "ClipCandidate",
    "ClipConcept",
    "EditPlan",
    "HookVariant",
    "PipelineSettings",
    "StoryMoment",
    "TranscriptSegment",
    "TranscriptWord",
    "VideoCandidate",
    "run_pipeline",
]

__version__ = "0.1.0"


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in {"PipelineSettings", "run_pipeline"}:
        from .pipeline import PipelineSettings, run_pipeline

        return {"PipelineSettings": PipelineSettings, "run_pipeline": run_pipeline}[name]
    raise AttributeError(name)

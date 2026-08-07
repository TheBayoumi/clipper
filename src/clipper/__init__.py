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
from .pipeline import PipelineSettings, run_pipeline

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

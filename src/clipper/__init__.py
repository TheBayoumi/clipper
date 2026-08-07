"""Automated, rights-gated campaign clipping pipeline."""

from .models import CampaignBrief, ClipCandidate, TranscriptSegment, TranscriptWord, VideoCandidate
from .pipeline import PipelineSettings, run_pipeline

__all__ = [
    "CampaignBrief",
    "ClipCandidate",
    "PipelineSettings",
    "TranscriptSegment",
    "TranscriptWord",
    "VideoCandidate",
    "run_pipeline",
]

__version__ = "0.1.0"

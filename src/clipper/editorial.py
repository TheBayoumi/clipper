from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from .models import (
    CampaignBrief,
    ClipConcept,
    EditPlan,
    HookVariant,
    StoryMoment,
    TranscriptSegment,
)


class LegacyEditorialRemovedError(RuntimeError):
    """Raised when obsolete deterministic editorial code is invoked."""


def _removed() -> NoReturn:
    raise LegacyEditorialRemovedError(
        "the deterministic lexical editorial engine has been removed; "
        "editorial quality, openings, narrative structure, and semantic value must be "
        "inferred from source evidence by the autonomous quality graph"
    )


def discover_story_moments(
    brief: CampaignBrief,
    video_id: str,
    segments: Sequence[TranscriptSegment],
) -> list[StoryMoment]:
    del brief, video_id, segments
    _removed()


def mine_clip_concepts(
    brief: CampaignBrief,
    video_id: str,
    segments: Sequence[TranscriptSegment],
    moments: Sequence[StoryMoment],
    *,
    rejections: list[dict[str, object]] | None = None,
    stats: dict[str, int] | None = None,
) -> list[ClipConcept]:
    del brief, video_id, segments, moments, rejections, stats
    _removed()


def select_distinct_concepts(
    brief: CampaignBrief,
    concepts: Sequence[ClipConcept],
    *,
    rejections: list[dict[str, object]] | None = None,
) -> list[ClipConcept]:
    del brief, concepts, rejections
    _removed()


def generate_hook_variants(
    brief: CampaignBrief,
    concept: ClipConcept,
    segments: Sequence[TranscriptSegment],
) -> list[HookVariant]:
    del brief, concept, segments
    _removed()


def build_edit_plan(
    brief: CampaignBrief,
    concept: ClipConcept,
    variant: HookVariant,
    segments: Sequence[TranscriptSegment],
) -> EditPlan:
    del brief, concept, variant, segments
    _removed()

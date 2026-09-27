from itertools import pairwise

import pytest

from clipper.models import CampaignBrief, ClipCandidate, TranscriptSegment
from clipper.scoring import score_transcript, select_diverse_clips


def brief() -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "c",
            "title": "Automation",
            "objective": "Explain business AI",
            "keywords": ["automation", "business"],
            "negative_keywords": ["giveaway"],
            "required_phrases": ["save time"],
            "source_channel_ids": ["UC1"],
            "rights_confirmed": True,
            "min_clip_seconds": 8,
            "max_clip_seconds": 20,
        }
    )


def test_score_transcript_ranks_relevant_complete_windows() -> None:
    segments = [
        TranscriptSegment(0, 4, "Here is the problem with manual work"),
        TranscriptSegment(4, 9, "automation can save time for every business."),
        TranscriptSegment(9, 13, "Never repeat the same task again."),
        TranscriptSegment(20, 29, "This giveaway is unrelated."),
    ]
    candidates = score_transcript(brief(), "v1", segments)
    assert candidates
    assert "save time" in candidates[0].text
    assert candidates[0].duration >= 8
    assert candidates[0].score > 0


def test_score_empty_and_select_diverse() -> None:
    assert score_transcript(brief(), "v", []) == []
    candidates = [
        ClipCandidate("a", 0, 10, "x", 10),
        ClipCandidate("a", 20, 30, "y", 9),
        ClipCandidate("b", 0, 10, "z", 8),
    ]
    selected = select_diverse_clips(candidates, clip_count=2, max_per_source=1)
    assert [item.video_id for item in selected] == ["a", "b"]


def test_overlapping_windows_are_deduplicated() -> None:
    segments = [
        TranscriptSegment(0, 8, "How automation helps business."),
        TranscriptSegment(8, 16, "Automation can save time."),
        TranscriptSegment(16, 24, "Another business result."),
    ]
    candidates = score_transcript(brief(), "v", segments, limit=10)
    for left, right in pairwise(candidates):
        intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
        union = max(left.end, right.end) - min(left.start, right.start)
        assert intersection / union < 0.55


def tjr_brief() -> CampaignBrief:
    return CampaignBrief.from_dict(
        {
            "campaign_id": "tjr-test",
            "title": "TJR",
            "objective": "trading",
            "keywords": ["risk", "trade"],
            "source_channel_ids": ["UC1"],
            "rights_confirmed": True,
            "min_clip_seconds": 20,
            "max_clip_seconds": 42,
        }
    )


def test_story_boundaries_keep_complete_openings_and_endings() -> None:
    segments = [
        TranscriptSegment(0, 10, "Why would the trade fail when the price changed"),
        TranscriptSegment(10, 23, "because we did not manage the stop loss."),
        TranscriptSegment(23, 34, "We need to plan before the next trade"),
        TranscriptSegment(34, 46, "because every decision carries market risk."),
    ]
    result = score_transcript(tjr_brief(), "v1", segments, limit=100, sentence_boundaries=True)
    assert result
    assert all(clip.start in (0, 23) for clip in result)
    assert all(clip.text.endswith((".", "!", "?")) for clip in result)
    assert all(
        any(reason.startswith("start_boundary=") for reason in clip.reasons)
        and any(reason.startswith("end_boundary=") for reason in clip.reasons)
        for clip in result
    )


def test_story_mode_rejects_mid_sentence_end_and_disconnected_speech() -> None:
    continuous = [
        TranscriptSegment(0, 11, "Why is the market risk so high"),
        TranscriptSegment(11, 23, "and why do we need to manage the trade"),
        TranscriptSegment(23, 34, "because our stop loss can be hit."),
    ]
    candidates = score_transcript(tjr_brief(), "v1", continuous, sentence_boundaries=True)
    assert candidates
    assert all(c.end >= 34 for c in candidates)

    disconnected = [
        TranscriptSegment(0, 12, "Why is the market risk so high?"),
        TranscriptSegment(17, 35, "We explain the trade plan before entry."),
    ]
    assert score_transcript(tjr_brief(), "v1", disconnected, sentence_boundaries=True) == []


def test_story_mode_respects_duration_after_timestamp_rounding() -> None:
    segments = [TranscriptSegment(0.09, 42.08, "Why is market risk high?")]
    assert score_transcript(tjr_brief(), "v1", segments, sentence_boundaries=True) == []


def test_relaxed_pause_uses_real_aligned_boundaries_not_arbitrary_mid_sentence() -> None:
    brief = tjr_brief()
    segments = [
        TranscriptSegment(0, 15, "We will explain why this market move matters"),
        TranscriptSegment(15.4, 30, "because the trade setup was unexpected"),
        TranscriptSegment(30.4, 44, "and we must manage the risk before entry"),
        TranscriptSegment(44.4, 66, "so we exit the position and protect the profit."),
    ]
    strict = score_transcript(brief, "v", segments, sentence_boundaries=True)
    relaxed = score_transcript(brief, "v", segments, sentence_boundaries=True, pause_threshold=0.35)
    assert strict == []
    assert relaxed
    assert relaxed[0].duration <= 42
    assert "pause_threshold=0.35" in relaxed[0].reasons
    assert any(reason.startswith("end_boundary=") for reason in relaxed[0].reasons)
    with pytest.raises(ValueError, match="pause_threshold"):
        score_transcript(brief, "v", segments, pause_threshold=0.1)



def test_editorial_mode_keeps_overlapping_windows_until_after_gate() -> None:
    segments = [
        TranscriptSegment(i * 5, i * 5 + 4.6, "why this trade reversed the market")
        for i in range(12)
    ]
    regular = score_transcript(brief(), "v1", segments, limit=100)
    editorial = score_transcript(
        brief(), "v1", segments, limit=100, diversify=False
    )
    assert len(editorial) > len(regular)
    assert editorial[0].score >= editorial[-1].score

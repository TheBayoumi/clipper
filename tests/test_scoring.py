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
    for left, right in zip(candidates, candidates[1:], strict=False):
        intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
        union = max(left.end, right.end) - min(left.start, right.start)
        assert intersection / union < 0.55

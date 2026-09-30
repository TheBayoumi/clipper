from clipper.models import CampaignBrief, TranscriptSegment
from scripts.tjr_semantic_editor import build_semantic_editorial_candidates


def _brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="tjr-test",
        title="TJR",
        objective="Find complete trading moments",
        keywords=["trading"],
        source_channel_ids=["UCGHBUXjDCeiIXNdKR0HUZnA"],
        min_clip_seconds=20,
        max_clip_seconds=42,
        rights_confirmed=True,
    )


def _fake_embedder(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        if "concrete trading setup" in lowered or "trade" in lowered or "market" in lowered:
            vectors.append([1.0, 0.0, 0.0])
        else:
            vectors.append([0.0, 1.0, 0.0])
    return vectors


def test_semantic_editor_builds_complete_story_without_sliding_window_quota() -> None:
    segments = [
        TranscriptSegment(0, 6, "Why I waited for this trade before entering."),
        TranscriptSegment(6, 12, "The market pushed higher but the setup was incomplete."),
        TranscriptSegment(12, 18, "My stop would have been too wide for the risk."),
        TranscriptSegment(18, 24, "So I waited for price to return before taking the trade."),
    ]
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", segments, embedder=_fake_embedder
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert 20 <= candidate.duration <= 42
    assert "candidate_origin=source_level_semantic_event" in candidate.reasons
    assert any(reason.startswith("semantic_event=") for reason in candidate.reasons)
    hook = next(
        reason.removeprefix("semantic_hook=")
        for reason in candidate.reasons
        if reason.startswith("semantic_hook=")
    )
    assert hook == "WHY I WAITED FOR THIS TRADE BEFORE ENTERING"
    assert audit["fixed_candidate_or_output_quota"] is False
    assert audit["architecture"] == "source_level_semantic_event_segmentation_v1"


def test_semantic_editor_returns_zero_when_source_has_no_units() -> None:
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", [], embedder=_fake_embedder
    )
    assert candidates == []
    assert audit["candidate_count"] == 0

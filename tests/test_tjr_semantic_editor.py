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
        if (
            "concrete trading setup" in lowered
            or "trading" in lowered
            or "trade" in lowered
            or "market" in lowered
            or "risk" in lowered
        ):
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
    assert audit["architecture"] == "source_level_semantic_campaign_event_segmentation_v2"
    assert audit["campaign_relevant_unit_count"] > 0
    assert audit["out_of_domain_unit_count"] == 0
    assert any(reason.startswith("campaign_relevance=") for reason in candidate.reasons)
    assert any(reason.startswith("relevance_margin=") for reason in candidate.reasons)


def test_semantic_editor_returns_zero_when_source_has_no_units() -> None:
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", [], embedder=_fake_embedder
    )
    assert candidates == []
    assert audit["candidate_count"] == 0


def test_semantic_editor_rejects_out_of_domain_luxury_content_before_event_labeling() -> None:
    segments = [
        TranscriptSegment(0, 6, "This Chrome Hearts vest is one of my favorite pieces."),
        TranscriptSegment(6, 12, "The bracelet is twenty two carat gold and custom made."),
        TranscriptSegment(12, 18, "This Rolex is another watch from the collection."),
        TranscriptSegment(18, 24, "The leopard bag is the last thing I wanted to show."),
    ]
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", segments, embedder=_fake_embedder
    )
    assert candidates == []
    assert audit["campaign_relevant_unit_count"] == 0
    assert audit["out_of_domain_unit_count"] == audit["semantic_unit_count"]
    assert audit["event_anchor_count"] == 0
    assert audit["fixed_candidate_or_output_quota"] is False


def test_semantic_editor_rejects_non_trading_lifestyle_source_before_regex_gate() -> None:
    segments = [
        TranscriptSegment(0, 7, "Look at this chrome vest and how much it costs."),
        TranscriptSegment(7, 14, "These pants have real gold hardware and custom patches."),
        TranscriptSegment(14, 21, "I can bring out the Rolex and all of the silver jewelry."),
        TranscriptSegment(21, 28, "My girlfriend also has this leopard bag with zebra handles."),
    ]
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", segments, embedder=_fake_embedder
    )
    assert candidates == []
    assert audit["event_anchor_count"] == 0
    assert audit["campaign_relevant_unit_count"] == 0
    assert audit["campaign_domain_gate"]["minimum_margin_over_non_campaign"] == 0.02

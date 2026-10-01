from clipper.models import CampaignBrief, TranscriptSegment
from scripts.tjr_semantic_editor import build_semantic_editorial_candidates


def _brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="double-coverage-test",
        title="Double Coverage Podcast",
        objective="Find complete funny, surprising, memorable podcast moments",
        keywords=["podcast", "story", "reaction", "guest"],
        source_channel_ids=["UCf1q6dhccWr6eQEcFFnJSbA"],
        min_clip_seconds=20,
        max_clip_seconds=45,
        rights_confirmed=True,
    )


def _fake_embedder(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lowered = text.lower()
        if any(
            term in lowered
            for term in (
                "filler",
                "housekeeping",
                "sponsor boilerplate",
                "navigation chatter",
                "chrome hearts",
                "bracelet",
                "rolex",
                "leopard bag",
            )
        ):
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([1.0, 0.0, 0.0])
    return vectors


def test_semantic_editor_builds_complete_story_without_sliding_window_quota() -> None:
    segments = [
        TranscriptSegment(0, 6, "The guest told us the wildest story from his first big show."),
        TranscriptSegment(6, 12, "Zach asked what happened when security suddenly stopped him."),
        TranscriptSegment(12, 18, "He admitted he had forgotten the one pass everyone needed."),
        TranscriptSegment(
            18,
            24,
            "Then the whole room laughed when he revealed how he got inside.",
        ),
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
    assert hook.upper() in candidate.text.upper()
    assert not hook.startswith("THE MOMENT:")
    assert audit["fixed_candidate_or_output_quota"] is False
    assert audit["architecture"] == "source_level_semantic_campaign_event_segmentation_v3"
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


def test_question_window_keeps_setup_and_stops_before_new_question() -> None:
    from scripts.tjr_semantic_editor import SemanticUnit, _anchor_window

    units = [
        SemanticUnit(0, 6, "Yeah, that reminds me of something else."),
        SemanticUnit(6, 12, "How did your first show pay you?"),
        SemanticUnit(12, 20, "They promised a percentage of the ticket revenue."),
        SemanticUnit(20, 28, "The room sold out, but I received nothing."),
        SemanticUnit(28, 36, "What happened at your next show?"),
    ]
    assert _anchor_window(2, units, [[1.0]] * 5, [0] * 5, min_seconds=20, max_seconds=45) == (1, 3)
    assert _anchor_window(2, units, [[1.0]] * 5, [0] * 5, min_seconds=30, max_seconds=45) is None

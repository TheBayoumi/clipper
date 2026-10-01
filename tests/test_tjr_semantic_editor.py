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
    from scripts.tjr_semantic_editor import CONTEXT_RUBRIC, OFF_TOPIC_DESCRIPTIONS

    negative_references = {*OFF_TOPIC_DESCRIPTIONS, *(pair[1] for pair in CONTEXT_RUBRIC.values())}
    return [[0.0, 1.0, 0.0] if text in negative_references else [1.0, 0.0, 0.0] for text in texts]


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
    assert "candidate_origin=podcast_contextual_editor" in candidate.reasons
    assert any(reason.startswith("semantic_event=") for reason in candidate.reasons)
    hook = next(
        reason.removeprefix("semantic_hook=")
        for reason in candidate.reasons
        if reason.startswith("semantic_hook=")
    )
    assert hook.upper() in candidate.text.upper()
    assert not hook.startswith("THE MOMENT:")
    assert audit["fixed_candidate_or_output_quota"] is False
    assert audit["architecture"] == "podcast_contextual_editor_v4"
    assert audit["campaign_keyword_gate"] is False
    assert any(reason.startswith("context_story_margin=") for reason in candidate.reasons)


def test_semantic_editor_returns_zero_when_source_has_no_units() -> None:
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", [], embedder=_fake_embedder
    )
    assert candidates == []
    assert audit["candidate_count"] == 0


def test_semantic_editor_does_not_reject_topics_absent_from_campaign_keywords() -> None:
    segments = [
        TranscriptSegment(0, 6, "This Chrome Hearts vest is one of my favorite pieces."),
        TranscriptSegment(6, 12, "The bracelet is twenty two carat gold and custom made."),
        TranscriptSegment(12, 18, "This Rolex is another watch from the collection."),
        TranscriptSegment(18, 24, "The leopard bag is the last thing I wanted to show."),
    ]
    candidates, audit = build_semantic_editorial_candidates(
        _brief(), "976-d0RlyfQ", segments, embedder=_fake_embedder
    )
    assert candidates
    assert audit["campaign_keyword_gate"] is False
    assert audit["event_anchor_count"] > 0
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


def test_campaign_title_objective_and_keywords_never_enter_editor_model() -> None:
    from dataclasses import replace

    segments = [
        TranscriptSegment(i * 6, (i + 1) * 6, text)
        for i, text in enumerate(
            [
                "My first attempt reached an unexpected result.",
                "The measurement contradicted our explanation.",
                "We repeated the experiment with a better control.",
                "The result showed why the first assumption failed.",
            ]
        )
    ]
    seen = []

    def embed(texts):
        seen.extend(texts)
        return _fake_embedder(texts)

    brief = _brief()
    first, _ = build_semantic_editorial_candidates(brief, "v", segments, embedder=embed)
    second, _ = build_semantic_editorial_candidates(
        replace(
            brief,
            title="UNRELATED_CAMPAIGN_TITLE",
            objective="UNRELATED_CAMPAIGN_OBJECTIVE",
            keywords=["UNRELATED_CAMPAIGN_KEYWORD"],
        ),
        "v",
        segments,
        embedder=embed,
    )
    assert first == second and first
    assert all("UNRELATED_CAMPAIGN" not in text for text in seen)


def test_boundary_repair_preserves_anchor_and_removes_unresolved_tail() -> None:
    segments = [
        TranscriptSegment(i * 8, (i + 1) * 8, text)
        for i, text in enumerate(
            [
                "Our first measurement changed the entire explanation.",
                "The repeat measurement produced the same result.",
                "That result settled the question we had started with.",
                "Another unrelated question begins without its answer.",
            ]
        )
    ]
    whole = " ".join(segment.text for segment in segments)

    def embed(texts):
        return [
            [0.0, 1.0, 0.0] if text == whole else vector
            for text, vector in zip(texts, _fake_embedder(texts), strict=True)
        ]

    candidates, audit = build_semantic_editorial_candidates(_brief(), "v", segments, embedder=embed)
    assert candidates
    assert all(candidate.text != whole for candidate in candidates)
    assert audit["context_rejected_variants"] > 0
    assert audit["boundary_repaired_candidate_count"] > 0
    assert all(20 <= candidate.duration <= 45 for candidate in candidates)


def test_incomplete_context_batch_raises_instead_of_using_keyword_fallback() -> None:
    import pytest

    calls = 0

    def embed(texts):
        nonlocal calls
        calls += 1
        return _fake_embedder(texts) if calls == 1 else []

    segments = [
        TranscriptSegment(i * 6, (i + 1) * 6, text)
        for i, text in enumerate(
            [
                "The first result contradicted our initial explanation.",
                "A second measurement confirmed the unexpected finding.",
                "We tested the proposed explanation using a control.",
                "The control resolved why the original result was different.",
            ]
        )
    ]
    with pytest.raises(RuntimeError, match=r"context assessment.*incomplete batch"):
        build_semantic_editorial_candidates(_brief(), "v", segments, embedder=embed)


def test_invalid_embedding_vectors_fail_closed() -> None:
    import pytest

    from scripts.tjr_semantic_editor import _embedding_batch

    for vector in ([], [float("nan")], [float("inf")], [0.0]):
        with pytest.raises(RuntimeError, match="invalid embedding vectors"):
            _embedding_batch(
                lambda texts, item=vector: [item for _ in texts], ["source"], "context assessment"
            )

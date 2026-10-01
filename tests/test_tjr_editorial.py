"""Pure quality controls for scalable campaign editorial selection."""

import pytest

from clipper.models import ClipCandidate, TranscriptSegment, WordTiming
from scripts.tjr_editorial import (
    RUBRIC_VERSION,
    WEIGHTS,
    EditorialReview,
    evaluate_candidate,
    select_editorial_moments,
)


def clip(start: float, text: str, score: float = 8) -> ClipCandidate:
    return ClipCandidate(
        "p2LU37eat70",
        start,
        start + 31,
        text,
        score,
        reasons=(
            "context_story_margin=0.25",
            "context_ending_margin=0.2",
            "context_opening_margin=0.1",
        ),
    )


def test_rejects_filler_and_weak_openers() -> None:
    filler = clip(
        0,
        "okay so like we should be doing the thing and then we wait "
        "and it gets there i guess now we can go and look at it tomorrow "
        "and wait to see what happens",
    )
    unassessed = ClipCandidate(filler.video_id, filler.start, filler.end, filler.text, filler.score)
    assert evaluate_candidate(unassessed) is None


def test_render_safety_bounds_are_explicit() -> None:
    with pytest.raises(ValueError):
        select_editorial_moments([], render_safety_limit=0)


def test_no_creative_trading_claim_for_unrelated_stream_banter() -> None:
    clip = ClipCandidate(
        "video",
        0,
        31,
        "bro that is the truth done never done cocaine and never done ketamine "
        "never done any of those things people ask all the time what about weed",
        12,
    )
    assert evaluate_candidate(clip) is None


STRONG_SEGMENT = (
    "Why did security stop me at my own show? I had planned the entire evening "
    "with the team. But I forgot the one pass everyone needed at the door. "
    "So the manager called me and fixed the problem before the whole room started laughing."
)


def test_weighted_provisional_score_tracks_unverified_visuals() -> None:
    result = evaluate_candidate(clip(0, STRONG_SEGMENT))
    assert result is not None
    assert RUBRIC_VERSION == "podcast-contextual-v4-review-drafts"
    assert sum(WEIGHTS.values()) == 100
    assert result.score_coverage == 75
    assert result.criteria["visuals"].score is None
    assert result.criteria["visuals"].basis == "manual_required"
    assert result.criteria["emotion"].basis == "manual_required"
    assert result.criteria["emotion"].score is None
    assert 0 <= result.editorial_score <= 100
    audit = result.to_dict()
    assert audit["publish_approved"] is False
    assert audit["human_review_required"] is True
    assert audit["integrity_gate"]["status"] == "unverified"
    assert audit["rubric_version"] == RUBRIC_VERSION
    assert audit["criterion_scores"]["opening"]["evidence"]


def test_manual_review_supplies_visual_rating_and_integrity_verdict() -> None:
    review = EditorialReview(
        visual_score=4.5,
        visual_notes=(
            "Source and final portrait crop manually inspected; host and important scene visible."
        ),
        integrity_passed=True,
        integrity_notes=(
            "Creative hook matches source audio and actual payoff after full-context review."
        ),
    )
    result = evaluate_candidate(clip(0, STRONG_SEGMENT), review=review)
    assert result is not None
    assert result.score_coverage == 90
    assert result.criteria["visuals"].score == 4.5
    assert result.criteria["visuals"].basis == "manual_verified"
    assert result.integrity_status == "pass"
    assert result.to_dict()["human_review_required"] is True
    # Explicit completion of these two rubric gates does not authorize publishing.
    assert result.to_dict()["publish_approved"] is False


def test_integrity_rejection_overrides_a_high_candidate_score() -> None:
    failed = EditorialReview(
        visual_score=5,
        visual_notes="Clearly framed source footage.",
        integrity_passed=False,
        integrity_notes="The hook implies an outcome absent from the source.",
    )
    assert evaluate_candidate(clip(0, STRONG_SEGMENT, 1000), review=failed) is None


def test_unsupported_numeric_hook_claim_is_rejected() -> None:
    assert (
        evaluate_candidate(
            clip(0, STRONG_SEGMENT),
            hook_override="THIS ONE TRADE MADE $9000000",
        )
        is None
    )


def test_reaction_words_do_not_override_contextual_opening_evidence() -> None:
    source = clip(0, STRONG_SEGMENT)
    source = ClipCandidate(
        source.video_id,
        source.start,
        source.end,
        source.text,
        source.score,
        reasons=(
            "context_story_margin=0.25",
            "context_ending_margin=0.2",
            "context_opening_margin=-0.2",
        ),
    )
    aligned = [
        TranscriptSegment(
            0,
            31,
            STRONG_SEGMENT,
            words=(WordTiming(0.1, 0.4, "WOW"), WordTiming(0.5, 0.8, "INSANE")),
        )
    ]
    assert evaluate_candidate(source, segments=aligned) is None
    selected, rejected = select_editorial_moments([source], segments=aligned)
    assert selected == []
    assert "CONTEXT_OPENING_NEEDS_REVIEW" in rejected[0]["failed_gates"]
    result = evaluate_candidate(source, segments=aligned, allow_review_only_opening=True)
    assert result is not None and result.to_dict()["human_review_required"] is True


@pytest.mark.parametrize(
    "review",
    [
        EditorialReview(visual_score=None),
        EditorialReview(integrity_passed=None),
    ],
)
def test_incomplete_review_does_not_claim_publish_approval(
    review: EditorialReview,
) -> None:
    result = evaluate_candidate(clip(0, STRONG_SEGMENT), review=review)
    assert result is not None
    assert result.to_dict()["publish_approved"] is False


def test_review_requires_real_notes_for_manual_assertions() -> None:
    with pytest.raises(ValueError, match="visual rating"):
        EditorialReview(visual_score=5)
    with pytest.raises(ValueError, match="integrity decision"):
        EditorialReview(integrity_passed=True)
    with pytest.raises(ValueError, match="0-5"):
        EditorialReview(visual_score=5.5, visual_notes="looks fine")


def test_provisional_score_is_deterministic_and_serializable() -> None:
    import json

    first = evaluate_candidate(clip(0, STRONG_SEGMENT))
    second = evaluate_candidate(clip(0, STRONG_SEGMENT))
    assert first == second
    assert first is not None
    json.dumps(first.to_dict())


def test_rejection_audit_identifies_each_individual_gate() -> None:
    from scripts.tjr_editorial import candidate_gate_failures

    item = ClipCandidate("v", 0, 31, "the chat is typing all day but nothing new happened", 1)
    reasons = candidate_gate_failures(item)
    assert "CONTEXTUAL_ASSESSMENT_REQUIRED" in reasons
    assert "WORD_COUNT_OUT_OF_RANGE" not in reasons


def test_hook_led_fallback_never_accepts_unfinished_trade_story() -> None:
    item = clip(
        0,
        "i wanted a retrace on the nasdaq before taking another trade but the "
        "market already moved too far up and i knew chasing the position would "
        "increase the risk for no reason so i decided to wait for the next "
        "entry instead of risking another loss and then",
    )
    item = ClipCandidate(
        item.video_id,
        item.start,
        item.end,
        item.text,
        item.score,
        reasons=(
            "context_story_margin=0.2",
            "context_ending_margin=-0.15",
            "context_opening_margin=0.1",
        ),
    )
    assert evaluate_candidate(item, allow_review_only_opening=True) is None


@pytest.mark.parametrize("subject", ["security", "the referee", "the scientist"])
def test_complete_stories_are_not_restricted_to_trading(subject: str) -> None:
    text = STRONG_SEGMENT.replace("security", subject)
    result = evaluate_candidate(clip(0, text))
    assert result is not None
    assert result.hook in text.upper()
    assert result.to_dict()["publish_approved"] is False


def test_distinct_source_stories_and_render_limit() -> None:
    texts = [
        STRONG_SEGMENT,
        "Why did the referee cancel that goal? We had planned a final attack against "
        "the other team. But the ball touched my hand before the shot went in. So "
        "the replay showed the mistake and we accepted the decision before starting "
        "again.",
        "How did the scientist save the experiment? She wanted to measure a rare "
        "reaction with the equipment. But the temperature suddenly increased during "
        "the night. So she changed the cooling system and solved the problem before "
        "losing the entire sample.",
    ]
    inputs = [clip(i * 45, text) for i, text in enumerate(texts)]
    picks, rejected = select_editorial_moments(inputs, render_safety_limit=10)
    assert len(picks) == 3
    assert len({pick.hook for pick in picks}) == 3
    picks, rejected = select_editorial_moments(inputs, render_safety_limit=1)
    assert len(picks) == 1
    assert sum(x["reason"] == "RENDER_SAFETY_LIMIT" for x in rejected) == 2


def test_repeated_identical_headlines_do_not_fill_a_quota() -> None:
    picks, rejected = select_editorial_moments([clip(i * 45, STRONG_SEGMENT) for i in range(5)])
    assert len({pick.hook for pick in picks}) == len(picks)
    assert len(picks) + len(rejected) == 5


def test_complete_story_with_weak_audio_opening_requires_review() -> None:
    text = STRONG_SEGMENT.replace(
        "Why did security stop me at my own show?", "I had a problem at the door."
    )
    item = clip(0, text)
    item = ClipCandidate(
        item.video_id,
        item.start,
        item.end,
        item.text,
        item.score,
        reasons=(
            "context_story_margin=0.2",
            "context_ending_margin=0.2",
            "context_opening_margin=-0.1",
        ),
    )
    assert evaluate_candidate(item) is None
    result = evaluate_candidate(item, allow_review_only_opening=True)
    assert result is not None
    assert result.to_dict()["publish_approved"] is False


def test_override_cannot_invent_a_non_numeric_claim() -> None:
    assert (
        evaluate_candidate(clip(0, STRONG_SEGMENT), hook_override="SECURITY RUINED THE ENTIRE SHOW")
        is None
    )


def test_semantic_headline_cannot_bypass_source_grounding() -> None:
    source = clip(0, STRONG_SEGMENT)
    fabricated = ClipCandidate(
        source.video_id,
        source.start,
        source.end,
        source.text,
        source.score,
        reasons=(
            "semantic_hook=SECURITY RUINED THE ENTIRE SHOW",
            "event_similarity=0.65",
            "campaign_relevance=0.8",
            "relevance_margin=0.4",
        ),
    )
    picks, rejected = select_editorial_moments([fabricated])
    assert picks == []
    assert "UNSUPPORTED_SOURCE_HOOK" in rejected[0]["failed_gates"]


def test_low_similarity_rank_is_not_a_creator_quality_veto() -> None:
    item = clip(0, STRONG_SEGMENT, score=0.01)
    item = ClipCandidate(
        item.video_id,
        item.start,
        item.end,
        item.text,
        item.score,
        reasons=(
            "context_story_margin=0.001",
            "context_ending_margin=0.001",
            "context_opening_margin=0.001",
        ),
    )
    selected, rejected = select_editorial_moments([item])
    assert len(selected) == 1 and rejected == []
    assert selected[0].editorial_score < 74
    assert selected[0].to_dict()["publish_approved"] is False


@pytest.mark.parametrize("value", ["nan", "inf", "invalid", "3"])
def test_invalid_context_evidence_fails_closed(value: str) -> None:
    item = clip(0, STRONG_SEGMENT)
    item = ClipCandidate(
        item.video_id,
        item.start,
        item.end,
        item.text,
        item.score,
        reasons=(
            f"context_story_margin={value}",
            "context_ending_margin=0.2",
            "context_opening_margin=0.1",
        ),
    )
    assert evaluate_candidate(item) is None


def test_word_quota_does_not_reject_assessed_longer_exchange() -> None:
    item = clip(0, STRONG_SEGMENT + " " + "A substantive explanation continues. " * 35)
    assert len(item.text.split()) > 155
    assert evaluate_candidate(item) is not None

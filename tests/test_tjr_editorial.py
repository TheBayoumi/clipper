"""Pure quality controls for scalable TJR editorial selection."""

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
    return ClipCandidate("p2LU37eat70", start, start + 31, text, score)


def test_rejects_filler_and_weak_openers() -> None:
    filler = clip(
        0,
        "okay so like we should be doing the thing and then we wait "
        "and it gets there i guess now we can go and look at it tomorrow "
        "and wait to see what happens",
    )
    assert evaluate_candidate(filler) is None


def test_preserves_authentic_reaction_hook() -> None:
    moment = clip(
        0,
        "damn wait what the hell just happened to the price i was "
        "short and the market moved much faster than i expected "
        "so i have to manage my position before getting stopped out again",
    )
    pick = evaluate_candidate(moment)
    assert pick is not None
    assert pick.hook_score >= 2
    assert pick.to_dict()["publish_approved"] is False


def test_selects_distinct_moments_without_two_clip_cap() -> None:
    text = "what just happened i never expected to see a move like that "
    text += "the risk on this trade is important and the position needs "
    text += "proper management before the price moves against us right now "
    text += "and we avoid another risky mistake"
    inputs = [clip(i * 45, text + f" number {i}", 20 - i) for i in range(5)]
    chosen, rejected = select_editorial_moments(inputs, batch_limit=10)
    # Repetitive scripts are deduplicated even at separate timestamps.
    assert len(chosen) == 1
    assert any(x["reason"] == "DUPLICATE_OR_OVERLAP" for x in rejected)


def test_real_distinct_hooks_can_yield_more_than_two_clips() -> None:
    originals = [
        "damn look at that move i did not expect the market to reverse "
        "i will manage the stop before entering this next trade tomorrow",
        "why do traders keep risking too much money on positions "
        "because they do not have a trading plan and chase the price every time",
        "what is the biggest mistake in a losing trade people move their "
        "stop loss and turn a small loss into a huge financial problem",
    ]
    fuller = [
        text + " my plan sets clear exits before taking the position so there "
        "is never any need to panic or chase the next move"
        for text in originals
    ]
    picks, _ = select_editorial_moments(
        [clip(i * 45, text, 10) for i, text in enumerate(fuller)], batch_limit=10
    )
    assert len(picks) == 3


def test_batch_size_bounds_are_explicit() -> None:
    with pytest.raises(ValueError):
        select_editorial_moments([], batch_limit=0)


def test_headline_from_full_context_and_campaign_topic() -> None:
    clip = ClipCandidate(
        "video",
        0,
        31,
        "what is happening in this market here so i wonder why he stopped using order blocks "
        "because there's no reason to use them anymore since equilibrium gets hit "
        "and that is the trade",
        12,
    )
    p = evaluate_candidate(clip)
    assert p is not None
    assert p.hook == "WHY HE STOPPED USING ORDER BLOCKS"


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


def test_distinct_trade_moments_still_scored() -> None:
    a = ClipCandidate(
        "video",
        0,
        31,
        "why do traders keep risking too much money on positions because they do "
        "not have a trading plan "
        "and chase the price every time the market moves and i never want that mistake again",
        10,
    )
    b = ClipCandidate(
        "video",
        45,
        76,
        "what is the biggest mistake in a losing trade people move their stop loss "
        "and turn a small loss "
        "into a huge financial problem in this market and that changes the setup "
        "for the next position",
        10,
    )
    selected, _ = select_editorial_moments([a, b], batch_limit=12)
    assert len(selected) == 2


STRONG_SEGMENT = (
    "damn what happened to this price i entered the trade but the market reversed "
    "and the position moved against me i had to manage risk and exit my position "
    "before the loss became even bigger so the lesson is to plan the stop loss "
    "before entering another trade tomorrow"
)


def test_weighted_provisional_score_tracks_unverified_visuals() -> None:
    result = evaluate_candidate(clip(0, STRONG_SEGMENT))
    assert result is not None
    assert RUBRIC_VERSION == "tjr-editorial-v1"
    assert sum(WEIGHTS.values()) == 100
    assert result.score_coverage == 85
    assert result.criteria["visuals"].score is None
    assert result.criteria["visuals"].basis == "manual_required"
    assert result.criteria["emotion"].basis == "transcript_proxy"
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
        visual_notes="Source and final portrait crop manually inspected; TJR and chart visible.",
        integrity_passed=True,
        integrity_notes="Creative hook matches source audio and actual payoff after full-context review.",
    )
    result = evaluate_candidate(clip(0, STRONG_SEGMENT), review=review)
    assert result is not None
    assert result.score_coverage == 100
    assert result.criteria["visuals"].score == 4.5
    assert result.criteria["visuals"].basis == "manual_verified"
    assert result.integrity_status == "pass"
    assert result.to_dict()["human_review_required"] is False
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


def test_actual_first_two_seconds_override_misleading_transcript_opening() -> None:
    source = clip(0, STRONG_SEGMENT)
    first_words = (
        WordTiming(0.1, 0.4, "okay"),
        WordTiming(0.5, 0.8, "so"),
        WordTiming(0.9, 1.2, "now"),
        WordTiming(3.0, 3.2, "damn"),
    )
    aligned = [TranscriptSegment(0, 31, STRONG_SEGMENT, words=first_words)]
    # Candidate text contains "damn", but the actual first 2 seconds are filler.
    assert evaluate_candidate(source, segments=aligned) is None
    selected, rejected = select_editorial_moments(
        [source], segments=aligned, batch_limit=1
    )
    assert selected == []
    assert rejected[0]["reason"] == "TOPIC_LENGTH_DENSITY_OPENING_OR_INTEGRITY_GATE"


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

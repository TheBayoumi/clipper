"""Pure quality controls for scalable TJR editorial selection."""

import pytest

from clipper.models import ClipCandidate
from scripts.tjr_editorial import evaluate_candidate, select_editorial_moments


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

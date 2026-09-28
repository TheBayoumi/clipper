"""Style B v2 actual word timing, safe portrait layout, and full-duration hook."""

from pathlib import Path

from clipper.models import ClipCandidate, TranscriptSegment, WordTiming
from clipper.tiktok import (
    _ass_time,
    _fit_lines,
    _safe,
    create_tiktok_ass,
    creative_hook_from_text,
)


def test_creative_headline_from_full_context_not_spoken_intro() -> None:
    assert (
        creative_hook_from_text(
            "What's up guys. We will explain how to start trading meme coins as beginners."
        )
        == "MEMECOIN TRADING: WHERE DO YOU START?"
    )
    assert (
        creative_hook_from_text(
            "Hey man, why did you stop using order blocks? There's no reason to use them."
        )
        == "WHY HE STOPPED USING ORDER BLOCKS"
    )
    assert (
        creative_hook_from_text(
            "He made over ten million, I'm up over a million in the last two months."
        )
        == "A TRADER CLAIMS MILLIONS: HOW?"
    )
    assert creative_hook_from_text("") == ""
    assert _safe(r"{\pos(0,0)}show \N next") == "pos(0,0) show N next"


def test_hook_bounds_measured_with_real_font() -> None:
    for headline in (
        "MEMECOIN TRADING: WHERE DO YOU START?",
        "A TRADER CLAIMS MILLIONS: HOW?",
        "WHY HE STOPPED USING ORDER BLOCKS",
        "WHY HE'S WAITING TO ENTER THIS TRADE",
    ):
        size, lines, width = _fit_lines(headline, max_width=800, max_size=70)
        assert 42 <= size <= 70
        assert 1 <= len(lines) <= 2
        assert width * 1.12 + 40 <= 800


def test_creative_hook_persists_and_active_word_follows_original_audio(tmp_path: Path) -> None:
    clip = ClipCandidate("p2LU37eat70", 10, 36, "Let's talk about entry", 12)
    words = (
        WordTiming(10.0, 10.35, "we"),
        WordTiming(10.35, 10.67, "have"),
        WordTiming(10.67, 11.10, "to"),
        WordTiming(11.10, 11.70, "wait"),
        WordTiming(11.70, 12.40, "here"),
    )
    segments = [
        TranscriptSegment(10, 12.4, "we have to wait here", words),
        TranscriptSegment(15.0, 17.0, "fallback words don't have timestamps"),
    ]
    path = create_tiktok_ass(
        clip, segments, tmp_path / "v2.ass", hook_text="WHY HE'S WAITING TO ENTER THIS TRADE"
    )
    ass = path.read_text()
    assert "Style: Hook,DejaVu Sans" in ass
    assert "Style: Caption,DejaVu Sans" in ass
    assert "&H54131620" in ass and "&H58131620" in ass
    assert "-1,0,0,0,100,100,0,0,3,18,0" in ass
    assert f"{_ass_time(0)},{_ass_time(26)},Hook" in ass
    assert r"\pos(540,185)" in ass
    assert r"\pos(540,1540)" in ass
    # Real word timings: each spoken word changes its own highlight interval.
    assert f"{_ass_time(0)},{_ass_time(0.35)},Caption" in ass
    assert f"{_ass_time(0.35)},{_ass_time(0.67)},Caption" in ass
    assert "{\\c&H0059DEFF&" in ass
    # Unaligned fallback is static; it receives no invented word transitions.
    assert f"{_ass_time(5)},{_ass_time(7)},Caption" in ass
    assert "TJR Style B v2" in ass
    assert r"{\bord9}" not in ass


def test_no_caption_or_hook_for_tiny_or_outside_segments(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 0.4, "", 1)
    path = create_tiktok_ass(
        clip,
        [TranscriptSegment(4, 5, "outside"), TranscriptSegment(0, 0.04, "tiny")],
        tmp_path / "empty.ass",
        hook_text="",
    )
    assert "Dialogue:" not in path.read_text()


def test_long_unbreakable_hook_rejected_instead_of_rendering_offscreen(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 10, "", 1)
    from pytest import raises

    with raises(ValueError, match="safe area"):
        create_tiktok_ass(clip, [], tmp_path / "oversize.ass", hook_text="W" * 90)


def test_active_word_not_random_keyword_highlight(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 3, "risk", 1)
    words = (
        WordTiming(0.2, 0.55, "risk"),
        WordTiming(0.55, 0.9, "profit"),
        WordTiming(0.9, 1.3, "analysis"),
    )
    result = create_tiktok_ass(
        clip,
        [TranscriptSegment(0.2, 1.3, "risk profit analysis", words)],
        tmp_path / "sync.ass",
        hook_text="WHAT IS THE RISK HERE?",
    )
    caption_events = [
        row for row in result.read_text().splitlines() if row.startswith("Dialogue: 2")
    ]
    assert len(caption_events) == 3
    assert all(event.count("\\c&H0059DEFF&") == 1 for event in caption_events)
    assert "RISK" in caption_events[0]
    assert "PROFIT" in caption_events[1]


def test_libass_hook_stays_in_safe_area_first_middle_and_last_frame(tmp_path: Path) -> None:
    import shutil
    import subprocess

    import pytest
    from PIL import Image, ImageChops

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is unavailable on this test runner")
    clip = ClipCandidate("x", 0, 3, "original", 1)
    words = (
        WordTiming(0.1, 0.4, "Wait"),
        WordTiming(0.4, 0.9, "for"),
        WordTiming(0.9, 1.3, "this"),
        WordTiming(1.3, 2.2, "move"),
    )
    ass = create_tiktok_ass(
        clip,
        [TranscriptSegment(0.1, 2.2, "Wait for this move", words)],
        tmp_path / "overlay.ass",
        hook_text="WHY HE'S WAITING TO ENTER THIS TRADE",
    )
    for label, ts in [("first", 0), ("middle", 1.2), ("last", 2.9)]:
        target = tmp_path / f"{label}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x25354f:s=1080x1920:r=30:d=3",
                "-ss",
                str(ts),
                "-vf",
                f"ass={ass}",
                "-frames:v",
                "1",
                "-y",
                str(target),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        image = Image.open(target).convert("RGB")
        background = Image.new("RGB", image.size, image.getpixel((0, 0)))
        # A real rendered libass frame, not only estimated text width.
        bbox = ImageChops.difference(image, background).crop((0, 100, 1080, 420)).getbbox()
        assert bbox is not None
        left, top, right, bottom = bbox
        assert 110 <= left < right <= 970
        assert 30 <= top < bottom <= 300


def test_v2_ass_auditor_rejects_missing_hook_and_missing_timing(tmp_path: Path) -> None:
    import pytest

    from clipper.tiktok import audit_tiktok_ass

    clip = ClipCandidate("x", 0, 3, "", 1)
    word = WordTiming(0.2, 0.4, "hello")
    path = create_tiktok_ass(
        clip,
        [TranscriptSegment(0.2, 0.4, "hello", (word,))],
        tmp_path / "valid.ass",
        hook_text="HOW HE READS THE SETUP",
    )
    proof = audit_tiktok_ass(path, clip_duration=clip.duration)
    assert proof["style"] == "B2" and proof["spoken_word_highlight_events"] == 1
    corrupted = tmp_path / "invalid.ass"
    corrupted.write_text(path.read_text().replace("Dialogue: 5,", "Dialogue: 3,"))
    with pytest.raises(ValueError, match="full-duration"):
        audit_tiktok_ass(corrupted, clip_duration=clip.duration)


def test_editorial_hook_variants_are_supported_by_their_transcripts() -> None:
    """Exercise every truthful headline branch, including neutral fallbacks."""
    pairs = (
        ("Never copy trade blindly when the price is moving.", "WHY HE WARNS ABOUT COPY TRADING"),
        ("The stop loss got hit before the reversal.", "WHAT HAPPENS WHEN THE STOP GETS HIT?"),
        ("I will wait before I enter the market.", "WHY HE'S WAITING TO ENTER THIS TRADE"),
        (
            "There was a massive sell-off but I would not short.",
            "WHY HE'S NOT SHORTING THE SELLOFF",
        ),
        ("We reviewed this meme coin chart yesterday.", "WHAT MATTERS IN A MEMECOIN TRADE?"),
        ("The order block is still a useful concept.", "DO ORDER BLOCKS REALLY MATTER HERE?"),
        ("We looked at the screenshot together.", "WHAT'S THE REAL TAKEAWAY HERE?"),
    )
    for transcript, expected_hook in pairs:
        assert creative_hook_from_text(transcript) == expected_hook


def test_empty_headline_and_corrupted_ass_fail_closed(tmp_path: Path) -> None:
    import pytest

    from clipper.tiktok import audit_tiktok_ass

    with pytest.raises(ValueError, match="empty headline"):
        _fit_lines("  ", max_width=800, max_size=70)

    clip = ClipCandidate("x", 0, 3, "How to plan a trade", 5)
    word = WordTiming(0.25, 0.70, "risk")
    original = create_tiktok_ass(
        clip,
        [TranscriptSegment(0.25, 0.70, "risk", (word,))],
        tmp_path / "original.ass",
        hook_text="HOW TO PLAN THIS TRADE",
    ).read_text(encoding="utf-8")

    mutations = (
        (
            original.replace("0:00:03.00,Hook", "0:00:02.00,Hook"),
            "entire clip",
        ),
        (
            original.replace(r"\an8\pos(540,185)", r"\an8\pos(999,185)"),
            "safe-area",
        ),
        (
            original.replace("Style: Caption,", "Style: Removed,"),
            "missing opaque",
        ),
        (
            original.replace("Dialogue: 2,", "Dialogue: 1,"),
            "no genuinely",
        ),
    )
    for index, (body, expected_error) in enumerate(mutations):
        path = tmp_path / f"mutated-{index}.ass"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError, match=expected_error):
            audit_tiktok_ass(path, clip_duration=clip.duration)


def test_distinct_hook_excludes_generic_and_repeated_templates() -> None:
    from clipper.tiktok import distinct_hook_from_text

    text = (
        "why did the stop loss hit when the market started reversing our position "
        "we decided to exit the trade because the risk was growing every minute"
    )
    first = distinct_hook_from_text(text)
    second = distinct_hook_from_text(text, {first})
    assert first == "WHAT HAPPENS WHEN THE STOP GETS HIT?"
    assert second == ""
    quoted = distinct_hook_from_text(text, {first}, allow_quote_fallback=True)
    assert quoted and quoted != first
    assert quoted.startswith('THE MOMENT: "')
    assert (
        distinct_hook_from_text("What if our market entry was late and the trading plan changed?")
        != "WHAT'S THE REAL TAKEAWAY HERE?"
    )
    assert distinct_hook_from_text("") == ""


def test_waiting_hook_requires_affirmative_trading_plan() -> None:
    assert (
        creative_hook_from_text("I am waiting to enter this trade after the market pulls back")
        == "WHY HE'S WAITING TO ENTER THIS TRADE"
    )
    assert (
        creative_hook_from_text("I entered early because I did not wait for the trade")
        != "WHY HE'S WAITING TO ENTER THIS TRADE"
    )
    assert (
        creative_hook_from_text("I am not waiting to enter this trade")
        != "WHY HE'S WAITING TO ENTER THIS TRADE"
    )


def test_waiting_hook_rejects_modal_negations() -> None:
    for sentence in (
        "I cannot wait to enter this trade",
        "I can't wait to enter this trade",
        "I won't wait to enter this trade",
        "I couldn't wait to enter this trade",
        "I have no reason to wait to enter this trade",
    ):
        assert creative_hook_from_text(sentence) != "WHY HE'S WAITING TO ENTER THIS TRADE"
    assert (
        creative_hook_from_text("I will wait to enter after the pullback")
        == "WHY HE'S WAITING TO ENTER THIS TRADE"
    )


def test_short_selloff_hooks_never_reverse_the_spoken_trade() -> None:
    positive = (
        "There was a massive sell-off but I would not short.",
        "I won't short this massive selloff because the bounce is too risky.",
        "I avoided shorting the massive sell off and waited.",
    )
    for transcript in positive:
        assert creative_hook_from_text(transcript) == "WHY HE'S NOT SHORTING THE SELLOFF"
    for transcript in (
        "I shorted the massive selloff and made a profit.",
        "I am shorting this massive sell-off.",
        "We discussed whether to short during this massive sell off.",
    ):
        assert creative_hook_from_text(transcript) != "WHY HE'S NOT SHORTING THE SELLOFF"


def test_market_cap_hook_is_source_specific_not_a_fragment_quote() -> None:
    transcript = "Late at $200K? The coin migrates at $50K market caps, then people buy."
    assert creative_hook_from_text(transcript) == "WHEN IS THE MARKET-CAP ENTRY TOO LATE?"


def test_contextual_questions_avoid_baseline_fragment_hooks() -> None:
    from clipper.tiktok import distinct_hook_from_text

    examples = (
        (
            "Late at $200K? The coin migrates at $50K market caps.",
            "WHEN IS THE MARKET-CAP ENTRY TOO LATE?",
        ),
        (
            "I don't see any socials. We have some wallets buying right now.",
            "WHY ARE WALLETS BUYING WITHOUT SOCIALS?",
        ),
        ("Should we just sell 50% of this position?", "WOULD YOU SELL HALF HERE?"),
        (
            "This coin has been at $80K market cap; I think it is bullshit.",
            "WHAT MAKES THIS MARKET-CAP SETUP SUSPICIOUS?",
        ),
        (
            "I can see on chain buys starting as volume pushes higher.",
            "CAN ON-CHAIN BUYS CONFIRM THE MOVE?",
        ),
        ("Why are people buying? This is FOMO.", "IS FOMO DRIVING THESE BUYS?"),
        ("Elon could tweet about a coin any day.", "CAN A SINGLE TWEET MOVE A COIN?"),
        (
            "The coin looks interesting but is getting sold off crazy.",
            "WHY IS THIS COIN GETTING SOLD OFF?",
        ),
        (
            "These community coins: how do you tell good coins from bad?",
            "GOOD COIN OR BAD COIN: HOW DO YOU TELL?",
        ),
        ("The fees should filter the rug coins.", "CAN FEES FILTER OUT RUG COINS?"),
        ("He says the trade is risk-free now.", "IS THIS 'RISK-FREE' TRADE REALLY SAFE?"),
    )
    for source, expected in examples:
        headline = distinct_hook_from_text(source)
        assert headline == expected
        assert not headline.startswith("THE MOMENT:")


def test_contextual_questions_reject_negated_wallet_buying() -> None:
    from clipper.tiktok import creative_hook_from_text

    text = "No socials linked, but no wallets buying either."
    assert creative_hook_from_text(text) != "WHY ARE WALLETS BUYING WITHOUT SOCIALS?"


def test_sell_half_hook_requires_the_amount_to_modify_the_sale() -> None:
    assert creative_hook_from_text("Should we just sell 50% of this position?") == (
        "WOULD YOU SELL HALF HERE?"
    )
    assert creative_hook_from_text("Could we sell now? Half the viewers said no.") != (
        "WOULD YOU SELL HALF HERE?"
    )


def test_short_decline_must_negate_the_shorting_action_itself() -> None:
    assert creative_hook_from_text("I won't short this massive selloff.") == (
        "WHY HE'S NOT SHORTING THE SELLOFF"
    )
    assert (
        creative_hook_from_text("I am not worried about shorting this massive selloff.")
        != "WHY HE'S NOT SHORTING THE SELLOFF"
    )


def test_excess_risk_hook_requires_affirmative_risk_taking() -> None:
    assert (
        creative_hook_from_text("Why do traders keep risking too much money on positions?")
        == "WHY TRADERS RISK TOO MUCH"
    )
    for source in (
        "Traders are not risking too much on this trade.",
        "I am not worried about risking too much on this trade.",
        "They never risk too much money because the plan caps the position.",
    ):
        assert creative_hook_from_text(source) != "WHY TRADERS RISK TOO MUCH"

def test_generic_risk_talk_is_rejected_instead_of_rendered_with_a_weak_hook() -> None:
    source = (
        "Risk matters in every trade and position, so you need a plan before trading. "
        "The market can move either way and the position has to be managed."
    )
    assert creative_hook_from_text(source) == "WHAT'S THE REAL TAKEAWAY HERE?"
    assert distinct_hook_from_text(source) == ""


def test_real_failed_batch_patterns_now_get_specific_semantic_hooks() -> None:
    examples = (
        (
            "Late at $200K? the coin migrates at $50K market caps and people are buying.",
            "WHEN IS THE MARKET-CAP ENTRY TOO LATE?",
        ),
        (
            "I don't see any socials linked. We have some wallets buying this token.",
            "WHY ARE WALLETS BUYING WITHOUT SOCIALS?",
        ),
        ("So should we just sell like 50%?", "WOULD YOU SELL HALF HERE?"),
        (
            "This has been sitting at $80,000 market cap and it is clearly some bullshit.",
            "WHAT MAKES THIS MARKET-CAP SETUP SUSPICIOUS?",
        ),
        (
            "We don't want to copy trade this person blindly after one winning trade.",
            "WHY HE WARNS ABOUT COPY TRADING",
        ),
        (
            "On chain, buys are starting to come through as volume starts pushing up.",
            "CAN ON-CHAIN BUYS CONFIRM THE MOVE?",
        ),
        ("Why are people buying this? This is the beauty of FOMO.", "IS FOMO DRIVING THESE BUYS?"),
        (
            "There was a coin today that ran because Trump tweeted about it.",
            "CAN A SINGLE TWEET MOVE A COIN?",
        ),
        (
            "I can see why people like this coin, but it is getting sold off crazy.",
            "WHY IS THIS COIN GETTING SOLD OFF?",
        ),
        (
            "You are kind of risk-free in the trade now after covering the initial.",
            "IS THIS 'RISK-FREE' TRADE REALLY SAFE?",
        ),
    )
    for source, expected in examples:
        assert distinct_hook_from_text(source) == expected


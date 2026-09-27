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

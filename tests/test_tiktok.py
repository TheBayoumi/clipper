"""Style B v2 actual word timing, safe portrait layout, and full-duration hook."""

from pathlib import Path

from clipper.models import ClipCandidate, TranscriptSegment, WordTiming
from clipper.tiktok import (
    _ass_time,
    _fit_lines,
    _safe,
    create_tiktok_ass,
    creative_hook_from_text,
    distinct_hook_from_text,
)


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
    assert "-1,0,0,0,100,100,0,0,1,0,0" in ass
    assert ",CaptionPlate," in ass and r"\p1" in ass
    assert f"{_ass_time(0)},{_ass_time(26)},Hook" in ass
    assert r"\pos(540,260)" in ass
    assert r"\pos(540,1460)" in ass
    # Real word timings: each spoken word changes its own highlight interval.
    assert f"{_ass_time(0)},{_ass_time(0.35)},Caption" in ass
    assert f"{_ass_time(0.35)},{_ass_time(0.67)},Caption" in ass
    assert "{\\c&H0059DEFF&" in ass
    # Unaligned fallback is static; it receives no invented word transitions.
    assert f"{_ass_time(5)},{_ass_time(7)},Caption" in ass
    assert "Source-grounded Style B v2" in ass
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
            original.replace(r"\an5\pos(540,260)", r"\an5\pos(999,260)"),
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


def test_active_word_highlight_ends_at_measured_word_end(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 3, "wait now", 1)
    words = (
        WordTiming(0.10, 0.40, "wait"),
        WordTiming(1.00, 1.30, "now"),
    )
    path = create_tiktok_ass(
        clip,
        [TranscriptSegment(0.10, 1.30, "wait now", words)],
        tmp_path / "timed.ass",
        hook_text="WHY HE'S WAITING TO ENTER THIS TRADE",
    )
    events = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Dialogue: 2,")
    ]
    assert "0:00:00.10,0:00:00.40,Caption" in events[0]
    assert "0:00:01.00,0:00:01.30,Caption" in events[1]


def test_headlines_preserve_source_subject_negation_and_amounts() -> None:
    from scripts.tjr_feedback_gate import hook_grounded_in_transcript

    sources = [
        "I never made a million dollars.",
        "Security stopped me at my own show.",
        "We found water beneath the frozen surface.",
        "Why did the referee cancel that goal?",
        "I gave away 300 computers last year.",
        "I would not short this selloff.",
    ]
    for source in sources:
        headline = creative_hook_from_text(source)
        assert headline == source.upper()
        assert hook_grounded_in_transcript(headline, source)
    assert not hook_grounded_in_transcript("I MADE A MILLION DOLLARS.", sources[0])
    assert not hook_grounded_in_transcript("I GAVE AWAY 3000 COMPUTERS LAST YEAR.", sources[4])


def test_headline_selection_uses_entire_clip_and_preserves_distinct_sentences() -> None:
    assert creative_hook_from_text("And I knew it.") == ""
    assert (
        creative_hook_from_text("And I knew it. Security stopped me at my own show.")
        == "SECURITY STOPPED ME AT MY OWN SHOW."
    )
    text = (
        "Welcome back. Why did the referee cancel that goal? Nobody expected that final decision."
    )
    first = distinct_hook_from_text(text)
    second = distinct_hook_from_text(text, [first])
    assert first == "WHY DID THE REFEREE CANCEL THAT GOAL?"
    assert second == "NOBODY EXPECTED THAT FINAL DECISION."
    assert distinct_hook_from_text(text, [first, second]) == ""
    assert distinct_hook_from_text("") == ""


def test_oversized_sentence_is_rejected_without_cutting_off_its_negation() -> None:
    text = (
        "The entire audience assumed the presenter had earned the promised prize "
        "even though the final decision showed that he had never won it."
    )
    assert creative_hook_from_text(text) == ""
    assert distinct_hook_from_text(text, allow_quote_fallback=True) == ""


def test_generic_headline_layout_has_bounded_height_and_width() -> None:
    text = "The guest told us the wildest story from his first big show."
    headline = creative_hook_from_text(text)
    size, lines, width = _fit_lines(headline, max_width=800, max_size=70, max_lines=3)
    assert 1 <= len(lines) <= 3
    assert len(lines) * size <= 210
    assert width * 1.12 + 40 <= 800
    assert _safe(r"{\pos(0,0)}show \N next") == "pos(0,0) show N next"


def test_decimal_amount_is_preserved_in_source_headline() -> None:
    from scripts.tjr_feedback_gate import hook_grounded_in_transcript

    text = "I did not earn $3.5 million from that show."
    hook = creative_hook_from_text(text)
    assert hook == text.upper()
    assert hook_grounded_in_transcript(hook, text)


def test_stable_background_does_not_scale_with_active_word(tmp_path: Path) -> None:
    import shutil
    import subprocess

    import pytest
    from PIL import Image

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    words = (
        WordTiming(0.1, 0.6, "yeah,"),
        WordTiming(0.6, 1.1, "for"),
        WordTiming(1.1, 1.6, "sure."),
    )
    ass = create_tiktok_ass(
        ClipCandidate("v", 0, 2, "Yeah, for sure.", 1),
        [TranscriptSegment(0.1, 1.6, "Yeah, for sure.", words)],
        tmp_path / "background.ass",
        hook_text="THIS IS THE WHOLE STORY",
    )
    bboxes = []
    for index, timestamp in enumerate((0.2, 0.8, 1.3)):
        frame = tmp_path / f"plate-{index}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=white:s=1080x1920:r=30:d=2",
                "-ss",
                str(timestamp),
                "-vf",
                f"ass={ass}",
                "-frames:v",
                "1",
                "-y",
                str(frame),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        crop = Image.open(frame).convert("RGB").crop((0, 1300, 1080, 1650))
        # The rectangle is independent of glyph colors and active-word scale.
        mask = crop.point(lambda value: 255 if value < 200 else 0).convert("L")
        bboxes.append(mask.getbbox())
        # Inspect painted glyphs inside the black plate, rather than trusting
        # ASS anchors or font estimates. Both axes must have balanced padding.
        pixels = crop.load()
        black = Image.new("L", crop.size)
        black.putdata([255 if max(pixel) < 15 else 0 for pixel in crop.getdata()])
        bounds = black.getbbox()
        assert bounds is not None
        left, top, right, bottom = bounds
        inner = crop.crop((left + 2, top + 2, right - 2, bottom - 2))
        ink = Image.new("L", inner.size)
        ink.putdata([255 if max(pixel) > 150 else 0 for pixel in inner.getdata()])
        painted = ink.getbbox()
        assert painted is not None
        x0, y0, x1, y1 = painted
        assert abs(x0 - (inner.width - x1)) <= 8
        assert abs(y0 - (inner.height - y1)) <= 8
        assert pixels[left + 3, top + 3] == (0, 0, 0)
        # The hook uses the same centered block contract.
        hook_crop = Image.open(frame).convert("RGB").crop((0, 100, 1080, 420))
        hook_black = Image.new("L", hook_crop.size)
        hook_black.putdata([255 if max(pixel) < 15 else 0 for pixel in hook_crop.getdata()])
        hook_bounds = hook_black.getbbox()
        assert hook_bounds is not None
        hx0, hy0, hx1, hy1 = hook_bounds
        hook_inner = hook_crop.crop((hx0 + 2, hy0 + 2, hx1 - 2, hy1 - 2))
        hook_ink = Image.new("L", hook_inner.size)
        hook_ink.putdata([255 if max(pixel) > 150 else 0 for pixel in hook_inner.getdata()])
        hook_painted = hook_ink.getbbox()
        assert hook_painted is not None
        ix0, iy0, ix1, iy1 = hook_painted
        assert abs(ix0 - (hook_inner.width - ix1)) <= 8
        assert abs(iy0 - (hook_inner.height - iy1)) <= 8
    assert bboxes[0] is not None and len(set(bboxes)) == 1
    text = ass.read_text()
    plates = [row for row in text.splitlines() if ",CaptionPlate," in row]
    assert len(plates) == 1 and r"\t(" not in plates[0]


def test_phrase_background_survives_pause_without_invented_highlight(tmp_path: Path) -> None:
    words = (WordTiming(0.1, 0.4, "yeah"), WordTiming(0.7, 1.0, "exactly"))
    ass = create_tiktok_ass(
        ClipCandidate("v", 0, 2, "", 1),
        [TranscriptSegment(0.1, 1, "yeah exactly", words)],
        tmp_path / "pause.ass",
        hook_text="HOW THE STORY REALLY ENDED",
    ).read_text()
    assert "0:00:00.10,0:00:01.00,CaptionPlate" in ass
    gap = next(row for row in ass.splitlines() if "0:00:00.40,0:00:00.70,Caption," in row)
    assert r"\c&H" not in gap


def test_caption_text_does_not_inconsistently_mask_profanity(tmp_path):
    words = (
        WordTiming(0.1, 0.5, "fucking"),
        WordTiming(0.5, 0.9, "shit"),
        WordTiming(0.9, 1.3, "f***ing"),
    )
    ass = create_tiktok_ass(
        ClipCandidate("v", 0, 2, "fucking shit f***ing", 1),
        [TranscriptSegment(0.1, 1.3, "fucking shit f***ing", words)],
        tmp_path / "verbatim.ass",
        hook_text="THE COMPLETE SOURCE STORY IS HERE",
    ).read_text()
    assert "FUCKING" in ass and "SHIT" in ass and "F***ING" in ass

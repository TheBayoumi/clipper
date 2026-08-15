from pathlib import Path

import pytest

from clipper.captions import CaptionLayout, create_word_reveal_ass, platform_caption_layout
from clipper.models import (
    ClipCandidate,
    EditPlan,
    SourceSpan,
    TranscriptSegment,
    TranscriptWord,
)
from clipper.wordstream import segment_source_words


def test_word_reveal_ass_uses_real_word_timestamps(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 10, 15, "text", 1)
    segment = TranscriptSegment(
        10,
        12,
        "Hello world now",
        (
            TranscriptWord(10.0, 10.4, "Hello"),
            TranscriptWord(10.7, 11.1, "world"),
            TranscriptWord(11.3, 11.8, "now"),
        ),
    )
    path = create_word_reveal_ass(clip, [segment], tmp_path / "captions.ass")
    text = path.read_text(encoding="utf-8")
    assert "SecondaryColour" in text
    assert "&HFFFFFFFF" in text
    assert r"{\ko70}Hello" in text
    assert r"{\ko60}world" in text
    assert r"{\ko50}now" in text
    assert "Dialogue: 0,0:00:00.00,0:00:01.80" in text


def test_word_reveal_ass_synthesizes_missing_word_timing_and_groups(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 6, "text", 1)
    segment = TranscriptSegment(
        0,
        6,
        ">> one two three four five six seven eight nine ten eleven twelve",
    )
    path = create_word_reveal_ass(clip, [segment], tmp_path / "synthetic.ass")
    text = path.read_text(encoding="utf-8")
    assert ">>" not in text
    assert text.count("Dialogue: 0,") >= 2
    assert text.count(r"{\ko") == 12


def test_word_reveal_ass_drops_partial_boundary_words(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 5, 7, "text", 1)
    segment = TranscriptSegment(
        4.5,
        7.5,
        "before inside after",
        (
            TranscriptWord(4.5, 5.2, "before"),
            TranscriptWord(5.4, 6.2, "inside"),
            TranscriptWord(6.4, 7.5, "after"),
        ),
    )
    path = create_word_reveal_ass(clip, [segment], tmp_path / "clamped.ass")
    text = path.read_text()
    audit = __import__("json").loads(path.with_suffix(".caption-audit.json").read_text())
    assert "before" not in text
    assert "inside" in text
    assert "after" not in text
    assert audit["partial_words_dropped"] == 2
    assert audit["first_audio_word"] == "inside"


def test_platform_caption_safe_zone_moves_tiktok_captions_up() -> None:
    tiktok = platform_caption_layout("tiktok")
    generic = platform_caption_layout("generic_vertical")
    assert tiktok.bottom_margin_px(1920) > 400
    assert generic.bottom_margin_px(1920) > 300
    assert tiktok.top_limit_px(1920) < 1100
    assert tiktok.hook_margin_px(1920) < 300
    with pytest.raises(ValueError, match="unsupported"):
        platform_caption_layout("unknown")


def test_caption_layout_validates_bounds() -> None:
    with pytest.raises(ValueError, match="fractions"):
        CaptionLayout("bad", 0.8, 0.7, 0.1).bottom_margin_px(1920)
    with pytest.raises(ValueError, match="hook"):
        CaptionLayout("bad", 0.4, 0.8, 0.5).validate()
    with pytest.raises(ValueError, match="max_lines"):
        CaptionLayout("bad", 0.4, 0.8, 0.1, 3).validate()


def test_hook_overlay_uses_source_derived_text_and_safe_style(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 3, "text", 1)
    segment = TranscriptSegment(2, 3, "I made five million dollars.")
    path = create_word_reveal_ass(
        clip,
        [segment],
        tmp_path / "hook.ass",
        platform="tiktok",
        hook_text="THE NUMBER THAT CHANGED EVERYTHING",
    )
    text = path.read_text()
    audit = __import__("json").loads(path.with_suffix(".caption-audit.json").read_text())
    assert "Style: Hook" in text
    assert "Dialogue: 1,0:00:00.00,0:00:01.80,Hook" in text
    assert "THE NUMBER THAT CHANGED EVERYTHING" in text
    assert ",461,1" in text
    assert audit["hook_overlay_rendered"] is True
    assert audit["simultaneous_narrative_layers_max"] == 1


def test_hook_overlay_is_suppressed_while_spoken_captions_are_active(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 3, "text", 1)
    segment = TranscriptSegment(0, 3, "I made five million dollars.")
    path = create_word_reveal_ass(
        clip,
        [segment],
        tmp_path / "single-lane.ass",
        hook_text="THE NUMBER THAT CHANGED EVERYTHING",
    )
    text = path.read_text()
    audit = __import__("json").loads(path.with_suffix(".caption-audit.json").read_text())
    assert "Dialogue: 1," not in text
    assert "THE NUMBER THAT CHANGED EVERYTHING" not in text
    assert audit["hook_overlay_suppression_reason"] == "caption_timing_overlap"
    assert audit["potential_hook_caption_overlap_seconds"] > 0
    assert audit["simultaneous_narrative_layers_max"] == 1


def _plan(start: float, end: float, *, anchor: float | None, word: str | None) -> EditPlan:
    return EditPlan(
        "plan",
        "v",
        "concept",
        "variant",
        "question",
        (SourceSpan(start, end),),
        None,
        (),
        "tiktok",
        9.0,
        "fingerprint",
        anchor,
        word,
    )


def test_first_caption_uses_trimmed_hook_word_across_vtt_cues(tmp_path: Path) -> None:
    segments = [
        TranscriptSegment(9.0, 9.3, "Dude,", (TranscriptWord(9.0, 9.3, "Dude"),)),
        TranscriptSegment(
            9.3,
            10.0,
            "before we head out,",
            (
                TranscriptWord(9.3, 9.48, "before"),
                TranscriptWord(9.49, 9.62, "we"),
                TranscriptWord(9.63, 9.82, "head"),
                TranscriptWord(9.83, 10.0, "out"),
            ),
        ),
        TranscriptSegment(
            10.05,
            12.4,
            "what's one message for esports fans?",
            (
                TranscriptWord(10.05, 10.32, "what's"),
                TranscriptWord(10.33, 10.50, "one"),
                TranscriptWord(10.51, 10.78, "message"),
                TranscriptWord(10.79, 10.92, "for"),
                TranscriptWord(10.93, 11.35, "esports"),
                TranscriptWord(11.36, 11.70, "fans?"),
            ),
        ),
    ]
    clip = ClipCandidate("v", 10.05, 12.4, "text", 1)
    path = create_word_reveal_ass(
        clip,
        segments,
        tmp_path / "trimmed.ass",
        edit_plan=_plan(10.05, 12.4, anchor=10.05, word="what's"),
    )
    text = path.read_text()
    audit = __import__("json").loads(path.with_suffix(".caption-audit.json").read_text())
    assert "Dude" not in text and "before" not in text and "head" not in text
    assert "what's" in text
    assert audit["first_audio_word"] == "what's"
    assert audit["first_caption_text"].lower().startswith("what's one message")
    assert audit["alignment"] == "PASS"


def test_first_caption_alignment_with_cue_interpolated_fallback(tmp_path: Path) -> None:
    segment = TranscriptSegment(
        9.0, 13.0, "Dude before we head out what's one message for esports fans?"
    )
    source_words = segment_source_words(segment)
    anchor_word = next(word for word in source_words if word.text.lower() == "what's")
    clip = ClipCandidate("v", anchor_word.source_start, 13.0, "text", 1)
    path = create_word_reveal_ass(
        clip,
        [segment],
        tmp_path / "fallback.ass",
        edit_plan=_plan(
            anchor_word.source_start, 13.0, anchor=anchor_word.source_start, word="what's"
        ),
    )
    audit = __import__("json").loads(path.with_suffix(".caption-audit.json").read_text())
    assert audit["timing_mode"] == "cue_interpolated"
    assert audit["first_audio_word"].lower() == "what's"
    assert audit["alignment"] == "PASS"


def test_caption_grouping_crosses_original_vtt_cue_boundaries(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 2.0, "text", 1)
    segments = [
        TranscriptSegment(
            0, 0.8, "one two", (TranscriptWord(0, 0.35, "one"), TranscriptWord(0.36, 0.7, "two"))
        ),
        TranscriptSegment(
            0.72,
            1.5,
            "three four",
            (TranscriptWord(0.72, 1.05, "three"), TranscriptWord(1.06, 1.4, "four")),
        ),
    ]
    text = create_word_reveal_ass(clip, segments, tmp_path / "cross-cue.ass").read_text()
    assert text.count("Dialogue: 0,") == 1
    assert "one" in text and "four" in text


def test_duplicate_hook_overlay_is_suppressed(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 3, "text", 1)
    segment = TranscriptSegment(0, 3, "I made five million dollars.")
    path = create_word_reveal_ass(
        clip, [segment], tmp_path / "duplicate.ass", hook_text="I MADE FIVE MILLION DOLLARS"
    )
    text = path.read_text()
    audit = __import__("json").loads(path.with_suffix(".caption-audit.json").read_text())
    assert "Dialogue: 1," not in text
    assert audit["hook_overlay_suppressed_duplicate"] is True
    assert audit["hook_overlay_suppression_reason"] == "duplicate_caption"

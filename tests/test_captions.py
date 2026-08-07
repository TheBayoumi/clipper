from pathlib import Path

from clipper.captions import create_word_reveal_ass
from clipper.models import ClipCandidate, TranscriptSegment, TranscriptWord


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


def test_word_reveal_ass_clamps_words_to_clip(tmp_path: Path) -> None:
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
    text = create_word_reveal_ass(clip, [segment], tmp_path / "clamped.ass").read_text()
    assert "0:00:00.00" in text
    assert "0:00:02.00" in text

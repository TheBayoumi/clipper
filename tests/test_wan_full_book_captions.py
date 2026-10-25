"""Subtitle contract for complete book episode with actual word timestamps."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wan_full_book_captions import ass_time, make_ass  # noqa: E402


def test_word_by_word_caption_emphasis_and_style_b_background() -> None:
    words = [{"word": f"word{i}", "start": i * 0.6, "end": i * 0.6 + 0.45} for i in range(60)]
    transcript = make_ass(words, 38.9)
    assert "PlayResX: 1080" in transcript
    assert "PlayResY: 1920" in transcript
    assert "BorderStyle" in transcript
    assert "Dialogue: 0,0:00:00.00,0:00:03.00,Book" in transcript
    assert transcript.count("Words,,0,0,0,,") == 60
    assert r"{\c&H009EFF&\b1}" in transcript
    assert "word59" in transcript
    assert ass_time(38.9) == "0:00:38.90"


def test_caption_timeline_must_reject_incomplete_episode() -> None:
    words = [{"word": "a", "start": i * 0.6, "end": i * 0.6 + 0.45} for i in range(20)]
    with pytest.raises(ValueError, match="full valid episode"):
        make_ass(words, 38.9)
    with pytest.raises(ValueError, match="full valid episode"):
        make_ass(words * 3, 10.0)

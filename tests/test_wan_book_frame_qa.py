"""Book title and subtitle-panel regressions stay tied to original Qwen shot."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wan_book_frame_qa import contains_old_subtitle_panel  # noqa: E402
from wan_keyframe_contract import SCENE_INSTRUCTIONS  # noqa: E402


def test_detects_historical_dark_subtitle_box(tmp_path: Path) -> None:
    frame = Image.new("RGB", (832, 1472), (168, 103, 58))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((200, 1240, 700, 1380), fill=(12, 14, 17))
    path = tmp_path / "broken.png"
    frame.save(path)
    assert contains_old_subtitle_panel(path)
    clean = tmp_path / "clean.png"
    Image.new("RGB", (832, 1472), (168, 103, 58)).save(clean)
    assert not contains_old_subtitle_panel(clean)


def test_book_cover_must_have_correct_physical_text_and_no_embedded_captions() -> None:
    text = SCENE_INSTRUCTIONS["book_intro"].upper()
    assert "ATOMIC HABITS" in text
    assert "JAMES CLEAR" in text
    assert "NO EMBEDDED SUBTITLES" in text
    assert "GIBBERISH" in text
    assert "ON THE FRONT COVER" in text

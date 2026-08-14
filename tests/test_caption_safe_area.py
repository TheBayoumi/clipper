from pathlib import Path

from clipper.captions import create_word_reveal_ass
from clipper.models import ClipCandidate, TranscriptSegment


def test_long_top_hook_wraps_inside_portrait_safe_width(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 3, "text", 1)
    segment = TranscriptSegment(0, 3, "unrelated spoken caption text")
    hook = (
        "THIS LONG TOP CAPTION MUST STAY INSIDE THE PORTRAIT FRAME "
        "WITHOUT BEING CUT OFF AT EITHER SIDE"
    )
    text = create_word_reveal_ass(
        clip,
        [segment],
        tmp_path / "hook-wrap.ass",
        platform="tiktok",
        hook_text=hook,
    ).read_text(encoding="utf-8")

    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "WrapStyle: 0" in text

    hook_style = next(line for line in text.splitlines() if line.startswith("Style: Hook,"))
    hook_font_size = int(hook_style.split(",")[2])
    assert 34 <= hook_font_size <= 54

    hook_event = next(line for line in text.splitlines() if line.startswith("Dialogue: 1,"))
    rendered_hook = hook_event.rsplit(",,", 1)[1]
    lines = rendered_hook.split(r"\N")
    assert len(lines) == 2
    assert all(line.strip() for line in lines)

    usable_width = 1080 - (2 * 90)
    assert all(len(line) * hook_font_size * 0.56 <= usable_width for line in lines)

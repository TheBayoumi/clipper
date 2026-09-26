"""TikTok overlays preserve genuine speech and 9:16 text-safe placement."""
from pathlib import Path

from clipper.models import ClipCandidate, TranscriptSegment
from clipper.tiktok import _ass_time, _safe, create_tiktok_ass, hook_from_quote


def test_hook_uses_only_contiguous_original_words() -> None:
    hook = hook_from_quote("And how much have you made across your whole career?")
    assert hook.startswith("HOW MUCH HAVE YOU MADE")
    assert len(hook) <= 44
    assert hook_from_quote("put whatever like $10,000 into a coin") == "$10,000 INTO A COIN"
    assert hook_from_quote("") == ""
    assert _safe(r"{\pos(0,0)}show \N next") == "pos(0,0) show N next"


def test_ass_overlay_uses_safe_animated_hook_and_short_pop_captions(tmp_path: Path) -> None:
    clip = ClipCandidate("p2LU37eat70", 10, 36, "Why did I lose this trade?", 12.0)
    words = [
        TranscriptSegment(9.5, 10.7, "why did"),
        TranscriptSegment(10.7, 11.4, "I lose"),
        TranscriptSegment(11.4, 12.2, "this trade?"),
        TranscriptSegment(12.2, 12.3, "tiny"),
        TranscriptSegment(50.0, 55.0, "outside"),
        TranscriptSegment(13.0, 15.0, r"RISK 100% {\bord9}"),
        TranscriptSegment(16.0, 18.0, "a long fallback sentence with no word alignment"),
    ]
    result = create_tiktok_ass(clip, words, tmp_path / "captions.ass", hook_text=clip.text)
    text = result.read_text()
    assert text.count("Dialogue:") == 6
    assert r"\pos(540,190)" in text
    assert r"\pos(540,1510)" in text
    assert r"\t(0,140" in text
    assert "WHY DID I LOSE" in text
    assert r"{\c&H0059DEFF&}" in text
    assert r"{\bord9}" not in text
    assert "outside" not in text
    assert _ass_time(61.23) == "0:01:01.23"


def test_short_clip_or_empty_caption_does_not_create_hook_or_dialogue(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 0.6, "What?", 1)
    path = create_tiktok_ass(
        clip,
        [
            TranscriptSegment(0, 0.03, "tiny"),
            TranscriptSegment(0.2, 0.5, ""),
        ],
        tmp_path / "empty.ass",
        hook_text="",
    )
    assert "Dialogue:" not in path.read_text()

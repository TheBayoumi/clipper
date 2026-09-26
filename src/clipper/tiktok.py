"""TikTok editorial overlays using the original speaker's actual words.

Write a separate editable ASS sidecar; publication remains subject to visual
review for logos, context and caption accuracy. No fabricated hook claims.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from .models import ClipCandidate, TranscriptSegment

_ACCENT = frozenset(
    {"WAIT", "WHAT", "WHY", "HOW", "NEVER", "NO", "STOP", "RISK", "LOSS", "PROFIT",
     "MONEY", "DAMN", "CRAZY", "MISTAKE", "WIN", "LOST", "TRUTH"}
)
_WORD = re.compile(r"\S+")
_HOOK_ANCHOR = re.compile(
    r"\b(?:why|how|what|wait|damn|never|no way|stop|secret)\b|\$[0-9]",
    re.IGNORECASE,
)


def _safe(text: str) -> str:
    """Remove ASS controls and line breaks from untrusted ASR text."""
    return re.sub(r"\s+", " ", re.sub(r"[{}\\\r\n]", " ", text)).strip()


def hook_from_quote(quote: str) -> str:
    """Find a short contiguous real quote, never invent a financial claim."""
    cleaned = _safe(quote)
    if not cleaned:
        return ""
    anchor = _HOOK_ANCHOR.search(cleaned[:125])
    if anchor is not None:
        cleaned = cleaned[anchor.start():]
    words = _WORD.findall(cleaned)
    if not words:
        return ""
    take: list[str] = []
    for word in words[:9]:
        if take and len(" ".join([*take, word])) > 42:
            break
        take.append(word)
        if word.endswith(("?", "!")) and len(take) >= 3:
            break
    return " ".join(take).rstrip(",. ").upper()


def _ass_time(seconds: float) -> str:
    ticks = max(0, round(seconds * 100))
    hours, ticks = divmod(ticks, 360000)
    minutes, ticks = divmod(ticks, 6000)
    whole, centiseconds = divmod(ticks, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def _two_lines(text: str, *, max_chars: int = 20) -> str:
    words = text.split()
    if not words:
        return ""
    first: list[str] = []
    second: list[str] = []
    for word in words:
        if not second and (not first or len(" ".join([*first, word])) <= max_chars):
            first.append(word)
        else:
            second.append(word)
    return " ".join(first) + (r"\N" + " ".join(second) if second else "")


def _highlight(text: str) -> str:
    """A single authentic reaction or number gets a contrasting yellow accent."""
    parts = text.split()
    for i, word in enumerate(parts):
        if word.strip(".,!?$").upper() in _ACCENT or any(x.isdigit() for x in word):
            parts[i] = (
                r"{\c&H0059DEFF&}" + word + r"{\c&H00FFFFFF&}"
            )
            break
    return " ".join(parts)


def create_tiktok_ass(
    clip: ClipCandidate,
    segments: Sequence[TranscriptSegment],
    output_path: str | Path,
    *,
    hook_text: str,
) -> Path:
    """Burn-in-ready 9:16 ASS: genuine on-screen hook + punchy pop captions.

    ASR segments must already use measured word timestamps. We never guess a
    word's timing to manufacture a karaoke effect.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "[Script Info]\n"
        "Title: TJR TikTok word-aligned review overlay\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Hook,DejaVu Sans,86,&H00FFFFFF,&H00FFFFFF,&H00000000,"
        "&H50000000,-1,0,0,0,100,100,1,0,1,6,3,8,85,85,185,1\n"
        "Style: Caption,DejaVu Sans,79,&H00FFFFFF,&H00FFFFFF,&H00000000,"
        "&H50000000,-1,0,0,0,100,100,1,0,1,7,3,5,95,95,390,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    events: list[str] = []
    hook = hook_from_quote(hook_text)
    if hook and clip.duration > 1:
        hook_lines = _two_lines(hook, max_chars=19)
        animated = (
            r"{\an8\pos(540,190)\fscx86\fscy86"
            r"\t(0,170,\fscx100\fscy100)\fad(70,220)}"
        )
        events.append(
            "Dialogue: 5,"
            f"{_ass_time(0.05)},{_ass_time(min(2.8, clip.duration - 0.1))},"
            f"Hook,,0,0,0,,{animated}{hook_lines}"
        )
    for segment in segments:
        start = max(segment.start, clip.start) - clip.start
        end = min(segment.end, clip.end) - clip.start
        if end - start < 0.12:
            continue
        original = _safe(segment.text)
        if not original:
            continue
        # Measured Whisper word groups are normally 2-4 words. For a rare
        # unaligned fallback, show two lines at its actual sentence time.
        # Show no more than six measured words per pop caption; fallback
        # sentence transcripts remain in the complete editable SRT sidecar.
        caption = _two_lines(" ".join(original.upper().split()[:6]), max_chars=19)
        caption = _highlight(caption.replace(r"\N", " __LINE__ ")).replace(
            "__LINE__", r"\N"
        )
        animation = (
            r"{\an5\pos(540,1510)\fscx86\fscy86"
            r"\t(0,140,\fscx100\fscy100)\fad(45,65)}"
        )
        events.append(
            "Dialogue: 2,"
            f"{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
            f"{animation}{caption}"
        )
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return output

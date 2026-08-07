from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from .models import ClipCandidate, TranscriptSegment, TranscriptWord

_SPEAKER_RE = re.compile(r"^>>\s*")


def _ass_timestamp(seconds: float) -> str:
    centiseconds = round(max(0.0, seconds) * 100)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _clean_word(text: str) -> str:
    cleaned = _SPEAKER_RE.sub("", text.strip())
    return cleaned.replace("{", "(").replace("}", ")").replace("\\", "")


def _synthetic_words(segment: TranscriptSegment) -> tuple[TranscriptWord, ...]:
    tokens = [_clean_word(token) for token in segment.text.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return ()
    step = segment.duration / len(tokens)
    return tuple(
        TranscriptWord(
            segment.start + index * step,
            segment.end if index + 1 == len(tokens) else segment.start + (index + 1) * step,
            token,
        )
        for index, token in enumerate(tokens)
    )


def _clip_words(clip: ClipCandidate, segment: TranscriptSegment) -> list[TranscriptWord]:
    source_words = segment.words or _synthetic_words(segment)
    clipped: list[TranscriptWord] = []
    for word in source_words:
        start = max(word.start, clip.start)
        end = min(word.end, clip.end)
        text = _clean_word(word.text)
        if text and end > start:
            clipped.append(TranscriptWord(start, end, text))
    return clipped


def _group_words(words: Sequence[TranscriptWord]) -> list[list[TranscriptWord]]:
    groups: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []
    chars = 0
    for word in words:
        projected_chars = chars + len(word.text) + (1 if current else 0)
        gap = word.start - current[-1].end if current else 0.0
        duration = word.end - current[0].start if current else word.duration
        if current and (len(current) >= 5 or projected_chars > 32 or gap > 0.65 or duration > 3.0):
            groups.append(current)
            current = []
            chars = 0
        current.append(word)
        chars += len(word.text) + (1 if len(current) > 1 else 0)
    if current:
        groups.append(current)
    return groups


def _karaoke_text(words: Sequence[TranscriptWord]) -> str:
    parts: list[str] = []
    for index, word in enumerate(words):
        next_start = words[index + 1].start if index + 1 < len(words) else word.end
        duration_cs = max(1, round((max(word.end, next_start) - word.start) * 100))
        parts.append(f"{{\\ko{duration_cs}}}{word.text}")
    return " ".join(parts)


def create_word_reveal_ass(
    clip: ClipCandidate,
    segments: Sequence[TranscriptSegment],
    output_path: str | Path,
    *,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    output = Path(output_path)
    events: list[str] = []
    for segment in segments:
        if segment.end <= clip.start or segment.start >= clip.end:
            continue
        for group in _group_words(_clip_words(clip, segment)):
            if not group:
                continue
            start = group[0].start - clip.start
            end = group[-1].end - clip.start
            if end <= start:
                continue
            events.append(
                "Dialogue: 0,"
                f"{_ass_timestamp(start)},{_ass_timestamp(end)},Default,,0,0,0,,"
                f"{_karaoke_text(group)}"
            )

    style_format = (
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"
    )
    style = (
        "Style: Default,DejaVu Sans,58,&H00FFFFFF,&HFFFFFFFF,&H00000000,&H80000000,"
        "1,0,0,0,100,100,0,0,1,4,0,2,80,80,230,1"
    )
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        f"{style_format}\n"
        f"{style}\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return output

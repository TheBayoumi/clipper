from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import ClipCandidate, TranscriptSegment, TranscriptWord

_SPEAKER_RE = re.compile(r"^>>\s*")


@dataclass(frozen=True, slots=True)
class CaptionLayout:
    platform: str
    top_fraction: float
    bottom_fraction: float
    hook_top_fraction: float
    max_lines: int = 2

    def validate(self) -> None:
        if not 0.0 < self.top_fraction < self.bottom_fraction < 1.0:
            raise ValueError("caption safe-zone fractions are invalid")
        if not 0.0 < self.hook_top_fraction < 0.4:
            raise ValueError("hook top fraction is invalid")
        if self.max_lines not in {1, 2}:
            raise ValueError("caption max_lines must be 1 or 2")

    def bottom_margin_px(self, height: int) -> int:
        self.validate()
        return round(height * (1.0 - self.bottom_fraction))

    def top_limit_px(self, height: int) -> int:
        self.validate()
        return round(height * self.top_fraction)

    def hook_margin_px(self, height: int) -> int:
        self.validate()
        return round(height * self.hook_top_fraction)


_PLATFORM_LAYOUTS = {
    "tiktok": CaptionLayout("tiktok", 0.50, 0.76, 0.12),
    "instagram_reels": CaptionLayout("instagram_reels", 0.50, 0.79, 0.12),
    "youtube_shorts": CaptionLayout("youtube_shorts", 0.52, 0.81, 0.11),
    "generic_vertical": CaptionLayout("generic_vertical", 0.52, 0.82, 0.11),
}


def platform_caption_layout(platform: str, *, max_lines: int = 2) -> CaptionLayout:
    key = platform.strip().lower()
    if key not in _PLATFORM_LAYOUTS:
        raise ValueError(f"unsupported caption platform: {platform}")
    base = _PLATFORM_LAYOUTS[key]
    return CaptionLayout(
        base.platform, base.top_fraction, base.bottom_fraction, base.hook_top_fraction, max_lines
    )


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
    platform: str = "tiktok",
    max_lines: int = 2,
    hook_text: str | None = None,
) -> Path:
    output = Path(output_path)
    layout = platform_caption_layout(platform, max_lines=max_lines)
    bottom_margin = layout.bottom_margin_px(height)
    hook_margin = layout.hook_margin_px(height)
    relevant_segments = [
        segment for segment in segments if segment.end > clip.start and segment.start < clip.end
    ]
    timing_mode = (
        "word_exact"
        if relevant_segments and all(segment.words for segment in relevant_segments)
        else "cue_interpolated"
    )
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

    if hook_text and hook_text.strip():
        clean_hook = _clean_word(hook_text)[:90]
        hook_end = min(1.8, clip.duration)
        if clean_hook and hook_end > 0.2:
            events.insert(
                0,
                "Dialogue: 1,"
                f"{_ass_timestamp(0.0)},{_ass_timestamp(hook_end)},Hook,,0,0,0,,{clean_hook}",
            )

    style_format = (
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"
    )
    style = (
        "Style: Default,DejaVu Sans,58,&H00FFFFFF,&HFFFFFFFF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,4,0,2,80,80,{bottom_margin},1"
    )
    hook_style = (
        "Style: Hook,DejaVu Sans,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,4,0,8,90,90,{hook_margin},1"
    )
    header = (
        "[Script Info]\n"
        f"; TimingMode: {timing_mode}\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        f"{style_format}\n"
        f"{style}\n"
        f"{hook_style}\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return output

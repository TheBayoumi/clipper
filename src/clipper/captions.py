from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .models import ClipCandidate, EditPlan, SourceSpan, TranscriptSegment
from .wordstream import ClipLocalWord, build_clip_word_stream, clean_word, normalized_tokens


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

_HOOK_HORIZONTAL_MARGIN_PX = 90
_HOOK_MAX_FONT_SIZE = 54
_HOOK_MIN_FONT_SIZE = 34
_HOOK_AVERAGE_GLYPH_WIDTH_EM = 0.56


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


def _group_words(words: Sequence[ClipLocalWord]) -> list[list[ClipLocalWord]]:
    groups: list[list[ClipLocalWord]] = []
    current: list[ClipLocalWord] = []
    chars = 0
    for word in words:
        projected_chars = chars + len(word.text) + (1 if current else 0)
        gap = word.local_start - current[-1].local_end if current else 0.0
        duration = (
            word.local_end - current[0].local_start
            if current
            else word.local_end - word.local_start
        )
        punctuation_break = bool(current and current[-1].text.rstrip().endswith((".", "?", "!")))
        if current and (
            len(current) >= 5
            or projected_chars > 32
            or gap > 0.65
            or duration > 3.0
            or punctuation_break
        ):
            groups.append(current)
            current = []
            chars = 0
        current.append(word)
        chars += len(word.text) + (1 if len(current) > 1 else 0)
    if current:
        groups.append(current)
    return groups


def _karaoke_text(words: Sequence[ClipLocalWord]) -> str:
    parts: list[str] = []
    for index, word in enumerate(words):
        next_start = words[index + 1].local_start if index + 1 < len(words) else word.local_end
        duration_cs = max(1, round((max(word.local_end, next_start) - word.local_start) * 100))
        parts.append(f"{{\\ko{duration_cs}}}{word.text}")
    return " ".join(parts)


def _near_duplicate(left: str, right: str) -> bool:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    left_set, right_set = set(left_tokens), set(right_tokens)
    containment = len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))
    prefix_equal = (
        left_tokens[: min(5, len(left_tokens))] == right_tokens[: min(5, len(right_tokens))]
    )
    return containment >= 0.8 or prefix_equal


def _fit_top_hook(
    text: str,
    width: int,
    *,
    max_lines: int = 2,
) -> tuple[str, int]:
    normalized = " ".join(text.split())
    if not normalized:
        return "", _HOOK_MAX_FONT_SIZE

    usable_width = max(320, width - 2 * _HOOK_HORIZONTAL_MARGIN_PX)
    for font_size in range(_HOOK_MAX_FONT_SIZE, _HOOK_MIN_FONT_SIZE - 1, -2):
        max_chars = max(
            12,
            int(usable_width / (font_size * _HOOK_AVERAGE_GLYPH_WIDTH_EM)),
        )
        lines = textwrap.wrap(
            normalized,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if len(lines) <= max_lines:
            return r"\N".join(lines), font_size

    max_chars = max(
        12,
        int(usable_width / (_HOOK_MIN_FONT_SIZE * _HOOK_AVERAGE_GLYPH_WIDTH_EM)),
    )
    lines = textwrap.wrap(
        normalized,
        width=max_chars,
        break_long_words=True,
        break_on_hyphens=False,
    )
    fitted = lines[:max_lines]
    if len(lines) > max_lines and fitted:
        suffix = "..."
        last = fitted[-1]
        if len(last) + len(suffix) > max_chars:
            last = last[: max(1, max_chars - len(suffix))].rstrip()
        fitted[-1] = last.rstrip(" .") + suffix
    return r"\N".join(fitted), _HOOK_MIN_FONT_SIZE


def _caption_audit(
    stream: Sequence[ClipLocalWord],
    groups: Sequence[Sequence[ClipLocalWord]],
    *,
    timing_mode: str,
    partial_words_dropped: int,
    hook_text: str | None,
    hook_suppressed: bool,
) -> dict[str, object]:
    first_group = list(groups[0]) if groups else []
    caption_text = " ".join(word.text for word in first_group)
    audible = list(stream[: len(first_group)])
    audible_text = " ".join(word.text for word in audible)
    first_audio_time = stream[0].local_start if stream else None
    first_caption_time = first_group[0].local_start if first_group else None
    delta = (
        abs(float(first_caption_time) - float(first_audio_time))
        if first_audio_time is not None and first_caption_time is not None
        else None
    )
    aligned = bool(
        first_group
        and audible
        and normalized_tokens(caption_text) == normalized_tokens(audible_text)
        and delta is not None
        and delta <= 0.08
    )
    return {
        "timing_mode": timing_mode,
        "first_audio_word": stream[0].text if stream else None,
        "first_audio_word_time": first_audio_time,
        "first_audio_words": audible_text,
        "first_caption_text": caption_text,
        "first_caption_time": first_caption_time,
        "first_caption_timing_delta_seconds": delta,
        "alignment": "PASS" if aligned else "FAIL",
        "partial_words_dropped": partial_words_dropped,
        "first_words": [
            {
                "text": word.text,
                "source_start": word.source_start,
                "source_end": word.source_end,
                "local_start": word.local_start,
                "local_end": word.local_end,
                "exact": word.exact,
            }
            for word in stream[:8]
        ],
        "hook_overlay_text": hook_text,
        "hook_overlay_suppressed_duplicate": hook_suppressed,
    }


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
    edit_plan: EditPlan | None = None,
    audit_path: str | Path | None = None,
) -> Path:
    output = Path(output_path)
    layout = platform_caption_layout(platform, max_lines=max_lines)
    bottom_margin = layout.bottom_margin_px(height)
    hook_margin = layout.hook_margin_px(height)
    source_spans = (
        edit_plan.source_spans if edit_plan is not None else (SourceSpan(clip.start, clip.end),)
    )
    caption_anchor = edit_plan.caption_start_source_time if edit_plan is not None else None
    stream, partial_words_dropped = build_clip_word_stream(
        source_spans, segments, caption_start_source_time=caption_anchor
    )
    groups = _group_words(stream)
    timing_mode = (
        "word_exact" if stream and all(word.exact for word in stream) else "cue_interpolated"
    )
    events: list[str] = []
    for group in groups:
        if not group:
            continue
        start = group[0].local_start
        end = group[-1].local_end
        if end <= start:
            continue
        events.append(
            "Dialogue: 0,"
            f"{_ass_timestamp(start)},{_ass_timestamp(end)},Default,,0,0,0,,"
            f"{_karaoke_text(group)}"
        )

    clean_hook = clean_word(hook_text)[:90] if hook_text and hook_text.strip() else ""
    fitted_hook, hook_font_size = _fit_top_hook(clean_hook, width, max_lines=max_lines)
    first_caption = " ".join(word.text for word in groups[0]) if groups else ""
    suppress_hook = bool(
        clean_hook and first_caption and _near_duplicate(clean_hook, first_caption)
    )
    if fitted_hook and not suppress_hook:
        hook_end = min(1.8, clip.duration)
        if hook_end > 0.2:
            events.insert(
                0,
                "Dialogue: 1,"
                f"{_ass_timestamp(0.0)},{_ass_timestamp(hook_end)},Hook,,0,0,0,,{fitted_hook}",
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
        f"Style: Hook,DejaVu Sans,{hook_font_size},"
        "&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,4,0,8,{_HOOK_HORIZONTAL_MARGIN_PX},"
        f"{_HOOK_HORIZONTAL_MARGIN_PX},{hook_margin},1"
    )
    header = (
        "[Script Info]\n"
        f"; TimingMode: {timing_mode}\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        f"{style_format}\n"
        f"{style}\n"
        f"{hook_style}\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    audit = _caption_audit(
        stream,
        groups,
        timing_mode=timing_mode,
        partial_words_dropped=partial_words_dropped,
        hook_text=clean_hook or None,
        hook_suppressed=suppress_hook,
    )
    target_audit = (
        Path(audit_path) if audit_path is not None else output.with_suffix(".caption-audit.json")
    )
    target_audit.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return output

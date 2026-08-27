from __future__ import annotations

from .canonical import CanonicalTimeline

_TERMINAL_SUFFIXES = (".", "!", "?", "…", "。")


def natural_split_index(
    timeline: CanonicalTimeline,
    start: int,
    end: int,
) -> int | None:
    """Choose a source-derived split boundary inside [start, end)."""

    if end - start <= 1:
        return None
    midpoint = start + (end - start) // 2

    sentence_boundaries = [
        index
        for index in range(start + 1, end)
        if timeline.words[index - 1].text.rstrip().endswith(_TERMINAL_SUFFIXES)
    ]
    if sentence_boundaries:
        return min(sentence_boundaries, key=lambda index: (abs(index - midpoint), index))

    speaker_boundaries = [
        index
        for index in range(start + 1, end)
        if timeline.words[index - 1].speaker_id is not None
        and timeline.words[index].speaker_id is not None
        and timeline.words[index - 1].speaker_id != timeline.words[index].speaker_id
    ]
    if speaker_boundaries:
        return min(speaker_boundaries, key=lambda index: (abs(index - midpoint), index))

    gaps = [
        (timeline.words[index].source_start - timeline.words[index - 1].source_end, index)
        for index in range(start + 1, end)
        if timeline.words[index].source_start > timeline.words[index - 1].source_end
    ]
    if gaps:
        largest_gap = max(gap for gap, _index in gaps)
        candidates = [index for gap, index in gaps if gap == largest_gap]
        return min(candidates, key=lambda index: (abs(index - midpoint), index))

    return midpoint


def stable_range_stage(
    prefix: str,
    timeline: CanonicalTimeline,
    start: int,
    end: int,
) -> str:
    if start < 0 or end > len(timeline.words) or end <= start:
        raise ValueError("editorial range is invalid")
    first = timeline.word_ref(timeline.words[start].word_id)
    last = timeline.word_ref(timeline.words[end - 1].word_id)
    return f"{prefix}:{first}-{last}"


def shrink_context_around_interval(
    timeline: CanonicalTimeline,
    start: int,
    end: int,
    required_start: int,
    required_end: int,
) -> tuple[int, int] | None:
    """Shrink context while preserving [required_start, required_end)."""

    if not (start <= required_start < required_end <= end):
        raise ValueError("required editorial interval is outside context")
    if start == required_start and end == required_end:
        return None

    left_extra = required_start - start
    right_extra = end - required_end
    if left_extra >= right_extra and left_extra > 0:
        split = natural_split_index(timeline, start, required_start + 1)
        return (split if split is not None else required_start, end)
    if right_extra > 0:
        split = natural_split_index(timeline, required_end - 1, end)
        return (start, split if split is not None else required_end)
    return (required_start, required_end)

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .canonical import CanonicalTimeline

_TERMINAL_SUFFIXES = (".", "!", "?", "…", "。")


def natural_boundary_near(
    timeline: CanonicalTimeline,
    start: int,
    end: int,
    target: int,
) -> int | None:
    """Choose the nearest source-derived boundary to a requested split target."""

    if end - start <= 1:
        return None
    target = min(max(target, start + 1), end - 1)

    sentence_boundaries = [
        index
        for index in range(start + 1, end)
        if timeline.words[index - 1].text.rstrip().endswith(_TERMINAL_SUFFIXES)
    ]
    if sentence_boundaries:
        return min(sentence_boundaries, key=lambda index: (abs(index - target), index))

    speaker_boundaries = [
        index
        for index in range(start + 1, end)
        if timeline.words[index - 1].speaker_id is not None
        and timeline.words[index].speaker_id is not None
        and timeline.words[index - 1].speaker_id != timeline.words[index].speaker_id
    ]
    if speaker_boundaries:
        return min(speaker_boundaries, key=lambda index: (abs(index - target), index))

    positive_gaps = [
        (
            timeline.words[index].source_start - timeline.words[index - 1].source_end,
            index,
        )
        for index in range(start + 1, end)
        if timeline.words[index].source_start > timeline.words[index - 1].source_end
    ]
    if positive_gaps:
        return min(
            positive_gaps,
            key=lambda item: (-item[0], abs(item[1] - target), item[1]),
        )[1]

    return target


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def capacity_target_input_tokens(details: dict[str, Any]) -> int | None:
    """Derive a safe next input target from observed model/runtime capacity evidence."""

    observed = _positive_int(details.get("input_tokens"))
    context = _positive_int(details.get("context_limit_tokens"))
    output_reserve = next(
        (
            value
            for value in (
                _positive_int(details.get("requested_output_tokens")),
                _positive_int(details.get("generation_budget_tokens")),
                _positive_int(details.get("structural_output_tokens")),
            )
            if value is not None
        ),
        None,
    )

    candidates: list[int] = []
    if context is not None:
        candidates.append(max(1, context - (output_reserve or 0)))

    largest_good = _positive_int(details.get("largest_good_input_tokens"))
    if largest_good is not None:
        candidates.append(largest_good)

    smallest_bad = _positive_int(details.get("smallest_bad_input_tokens"))
    if smallest_bad is not None:
        candidates.append(max(1, smallest_bad - 1))

    runtime_safe = _positive_int(details.get("runtime_safe_input_tokens"))
    if runtime_safe is not None:
        candidates.append(runtime_safe)

    reason = str(details.get("reason") or "")
    if observed is not None and reason in {
        "cuda_oom_after_offloaded_cache",
        "cuda_oom_dynamic_cache",
    }:
        # With no learned lower working-set boundary yet, binary subdivision is the
        # source-independent bootstrap. Once a successful child is observed, the
        # persisted largest-good boundary replaces this estimate.
        candidates.append(max(1, observed // 2))

    if not candidates:
        return None
    return min(candidates)


@dataclass(frozen=True, slots=True)
class TokenAwareRepartitionPlan:
    reason: str
    observed_input_tokens: int | None
    target_input_tokens: int | None
    partition_count: int
    ranges: tuple[tuple[int, int], ...]

    def telemetry(self, *, stage: str) -> dict[str, object]:
        return {
            "event": "editorial_repartition",
            "execution_id": os.getenv("CLIPPER_EXECUTION_ID", ""),
            "stage": stage,
            "reason": self.reason,
            "observed_input_tokens": self.observed_input_tokens,
            "target_input_tokens": self.target_input_tokens,
            "partition_count": self.partition_count,
            "ranges": [list(item) for item in self.ranges],
        }


def token_aware_repartition(
    timeline: CanonicalTimeline,
    start: int,
    end: int,
    details: dict[str, Any],
) -> TokenAwareRepartitionPlan | None:
    """Convert measured model capacity into source-grounded multi-way partitions."""

    word_count = end - start
    if word_count <= 1:
        return None

    observed = _positive_int(details.get("input_tokens"))
    target_tokens = capacity_target_input_tokens(details)
    if observed is not None and target_tokens is not None and target_tokens < observed:
        partition_count = math.ceil(observed / target_tokens)
    else:
        partition_count = 2

    partition_count = min(word_count, max(2, partition_count))
    boundaries = [start]
    previous = start
    for ordinal in range(1, partition_count):
        remaining_partitions = partition_count - ordinal
        ideal = start + round(word_count * ordinal / partition_count)
        lower = previous + 1
        upper = end - remaining_partitions
        if lower > upper:
            break
        ideal = min(max(ideal, lower), upper)
        boundary = natural_boundary_near(timeline, lower - 1, upper + 1, ideal)
        if boundary is None:
            boundary = ideal
        boundary = min(max(boundary, lower), upper)
        boundaries.append(boundary)
        previous = boundary
    boundaries.append(end)

    ranges = tuple((left, right) for left, right in pairwise(boundaries) if right > left)
    if len(ranges) <= 1:
        return None

    return TokenAwareRepartitionPlan(
        reason=str(details.get("reason") or "capacity_rejected"),
        observed_input_tokens=observed,
        target_input_tokens=target_tokens,
        partition_count=len(ranges),
        ranges=ranges,
    )


def natural_split_index(
    timeline: CanonicalTimeline,
    start: int,
    end: int,
) -> int | None:
    """Choose a source-derived split boundary inside [start, end)."""

    if end - start <= 1:
        return None
    midpoint = start + (end - start) // 2
    return natural_boundary_near(timeline, start, end, midpoint)


def token_aware_context_range(
    timeline: CanonicalTimeline,
    start: int,
    end: int,
    required_start: int,
    required_end: int,
    details: dict[str, Any],
) -> tuple[int, int] | None:
    """Shrink context from measured token density while preserving a required span."""

    if not (start <= required_start < required_end <= end):
        raise ValueError("required editorial interval is outside context")
    if start == required_start and end == required_end:
        return None

    observed = _positive_int(details.get("input_tokens"))
    target_tokens = capacity_target_input_tokens(details)
    total_words = end - start
    required_words = required_end - required_start

    if observed is None or target_tokens is None or target_tokens >= observed:
        return shrink_context_around_interval(
            timeline,
            start,
            end,
            required_start,
            required_end,
        )

    target_words = max(
        required_words,
        min(total_words - 1, math.floor(total_words * target_tokens / observed)),
    )
    if target_words <= required_words:
        return (required_start, required_end)

    extras = target_words - required_words
    left_available = required_start - start
    right_available = end - required_end
    total_available = left_available + right_available
    if total_available <= 0:
        return None

    left_keep = min(left_available, round(extras * left_available / total_available))
    right_keep = min(right_available, extras - left_keep)
    remaining = extras - left_keep - right_keep
    if remaining > 0:
        additional_left = min(left_available - left_keep, remaining)
        left_keep += additional_left
        remaining -= additional_left
    if remaining > 0:
        right_keep += min(right_available - right_keep, remaining)

    proposed_start = required_start - left_keep
    proposed_end = required_end + right_keep

    if proposed_start > start:
        boundary = natural_boundary_near(
            timeline,
            start,
            required_start + 1,
            proposed_start,
        )
        if boundary is not None:
            proposed_start = min(boundary, required_start)
    if proposed_end < end:
        boundary = natural_boundary_near(
            timeline,
            required_end - 1,
            end,
            proposed_end,
        )
        if boundary is not None:
            proposed_end = max(boundary, required_end)

    candidate = (max(start, proposed_start), min(end, proposed_end))
    if candidate == (start, end):
        return shrink_context_around_interval(
            timeline,
            start,
            end,
            required_start,
            required_end,
        )
    return candidate


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

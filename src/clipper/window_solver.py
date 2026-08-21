from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import CanonicalTimeline
from .models import SourceSpan
from .story_graph import NarrativeEnvelope, SemanticCore


@dataclass(frozen=True, slots=True)
class FeasibleDeliveryWindow:
    window_id: str
    core_id: str
    envelope_id: str
    video_id: str
    source_hash: str
    source_start: float
    source_end: float
    source_word_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.window_id.strip() or not self.core_id.strip() or not self.envelope_id.strip():
            raise ValueError("feasible window requires stable graph identity")
        if not self.video_id.strip() or not self.source_hash.strip():
            raise ValueError("feasible window requires stable source identity")
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("feasible window timestamps are invalid")
        if not self.source_word_ids:
            raise ValueError("feasible window requires source word provenance")

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start

    def overlaps(self, span: SourceSpan) -> bool:
        return self.source_end > span.start and self.source_start < span.end

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_word_ids"] = list(self.source_word_ids)
        payload["duration"] = self.duration
        return payload


def _window_id(
    source_hash: str,
    core_id: str,
    envelope_id: str,
    first_word_id: str,
    last_word_id: str,
) -> str:
    payload = ":".join((source_hash, core_id, envelope_id, first_word_id, last_word_id))
    return f"window-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _build_window(
    timeline: CanonicalTimeline,
    core: SemanticCore,
    envelope: NarrativeEnvelope,
    *,
    start_index: int,
    end_index: int,
) -> FeasibleDeliveryWindow:
    source_word_ids = tuple(
        word.word_id for word in timeline.words[start_index : end_index + 1]
    )
    first = timeline.words[start_index]
    last = timeline.words[end_index]
    return FeasibleDeliveryWindow(
        window_id=_window_id(
            timeline.source_hash,
            core.core_id,
            envelope.envelope_id,
            source_word_ids[0],
            source_word_ids[-1],
        ),
        core_id=core.core_id,
        envelope_id=envelope.envelope_id,
        video_id=timeline.video_id,
        source_hash=timeline.source_hash,
        source_start=first.source_start,
        source_end=last.source_end,
        source_word_ids=source_word_ids,
    )


def enumerate_feasible_windows(
    timeline: CanonicalTimeline,
    core: SemanticCore,
    envelope: NarrativeEnvelope,
    *,
    min_seconds: float,
    max_seconds: float,
    forbidden_spans: tuple[SourceSpan, ...] = (),
) -> tuple[FeasibleDeliveryWindow, ...]:
    """Enumerate legal word-aligned spans that preserve the complete narrative envelope.

    The NarrativeEnvelope is the minimum complete story, so a final delivery window may never
    be a subrange of it. If the complete envelope already satisfies campaign duration, it is the
    canonical delivery window. If it is shorter than the campaign minimum, deterministic padding
    may extend before/after the envelope while preserving the whole envelope. If the envelope is
    longer than the campaign maximum, no legal delivery window exists.
    """

    if min_seconds <= 0 or max_seconds < min_seconds:
        raise ValueError("campaign duration bounds are invalid")
    if timeline.video_id != core.video_id or timeline.source_hash != core.source_hash:
        raise ValueError("semantic core does not belong to the canonical timeline")
    if timeline.video_id != envelope.video_id or timeline.source_hash != envelope.source_hash:
        raise ValueError("narrative envelope does not belong to the canonical timeline")
    envelope.require_contains(core)
    if not envelope.complete:
        return ()
    if envelope.duration > max_seconds + 1e-6:
        return ()

    positions = {word.word_id: index for index, word in enumerate(timeline.words)}
    try:
        envelope_start = positions[envelope.source_word_ids[0]]
        envelope_end = positions[envelope.source_word_ids[-1]]
    except KeyError as exc:
        raise ValueError("story graph references a word outside the canonical timeline") from exc

    if min_seconds - 1e-6 <= envelope.duration <= max_seconds + 1e-6:
        canonical = _build_window(
            timeline,
            core,
            envelope,
            start_index=envelope_start,
            end_index=envelope_end,
        )
        if any(canonical.overlaps(span) for span in forbidden_spans):
            return ()
        validate_feasible_window(
            canonical,
            core,
            envelope,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            forbidden_spans=forbidden_spans,
        )
        return (canonical,)

    windows: list[FeasibleDeliveryWindow] = []
    for start_index in range(envelope_start, -1, -1):
        first = timeline.words[start_index]
        minimum_possible_duration = timeline.words[envelope_end].source_end - first.source_start
        if minimum_possible_duration > max_seconds + 1e-6:
            break
        for end_index in range(envelope_end, len(timeline.words)):
            last = timeline.words[end_index]
            duration = last.source_end - first.source_start
            if duration < min_seconds - 1e-6:
                continue
            if duration > max_seconds + 1e-6:
                break
            window = _build_window(
                timeline,
                core,
                envelope,
                start_index=start_index,
                end_index=end_index,
            )
            if any(window.overlaps(span) for span in forbidden_spans):
                continue
            validate_feasible_window(
                window,
                core,
                envelope,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                forbidden_spans=forbidden_spans,
            )
            windows.append(window)

    windows.sort(
        key=lambda item: (item.duration, item.source_start, item.source_end, item.window_id)
    )
    return tuple(windows)


def validate_feasible_window(
    window: FeasibleDeliveryWindow,
    core: SemanticCore,
    envelope: NarrativeEnvelope,
    *,
    min_seconds: float,
    max_seconds: float,
    forbidden_spans: tuple[SourceSpan, ...] = (),
) -> None:
    if window.core_id != core.core_id or window.envelope_id != envelope.envelope_id:
        raise ValueError("feasible window graph identity is inconsistent")
    if window.video_id != core.video_id or window.source_hash != core.source_hash:
        raise ValueError("feasible window source identity is inconsistent")
    if window.video_id != envelope.video_id or window.source_hash != envelope.source_hash:
        raise ValueError("feasible window narrative source identity is inconsistent")
    if not min_seconds - 1e-6 <= window.duration <= max_seconds + 1e-6:
        raise ValueError("feasible window violates campaign duration bounds")
    if window.source_start > envelope.source_start + 1e-6:
        raise ValueError("feasible window amputates the start of the complete narrative envelope")
    if window.source_end < envelope.source_end - 1e-6:
        raise ValueError("feasible window amputates the end of the complete narrative envelope")
    if not set(envelope.source_word_ids).issubset(window.source_word_ids):
        raise ValueError("feasible window does not contain the complete narrative envelope")
    if not set(core.source_word_ids).issubset(window.source_word_ids):
        raise ValueError("feasible window does not contain semantic core")
    if any(window.overlaps(span) for span in forbidden_spans):
        raise ValueError("feasible window overlaps forbidden source evidence")

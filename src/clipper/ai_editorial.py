from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from .canonical import CanonicalTimeline
from .models import EditPlan, HookMode, SourceSpan


class EditorialGroundingError(ValueError):
    """Raised when generative editorial output cannot be proven from the source timeline."""


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EditorialGroundingError(f"{field} must be a non-empty string")
    return value.strip()


def _word_ids(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise EditorialGroundingError(f"{field} must be a non-empty list of word IDs")
    return tuple(item.strip() for item in value if item.strip())


def _confidence(value: object) -> float:
    if not isinstance(value, int | float | str):
        raise EditorialGroundingError("confidence must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise EditorialGroundingError("confidence must be between 0 and 1")
    return result


def _grounded_word_range(
    payload: dict[str, Any],
    timeline: CanonicalTimeline,
    *,
    list_field: str,
    start_field: str,
    end_field: str,
) -> tuple[str, ...]:
    raw_ids = payload.get(list_field)
    if raw_ids is not None:
        return _word_ids(raw_ids, list_field)
    start_ref = _nonempty(payload.get(start_field), start_field)
    end_ref = _nonempty(payload.get(end_field), end_field)
    try:
        start_id = timeline.resolve_word_ref(start_ref)
        end_id = timeline.resolve_word_ref(end_ref)
    except ValueError as exc:
        raise EditorialGroundingError(str(exc)) from exc
    positions = {word.word_id: index for index, word in enumerate(timeline.words)}
    start_index = positions[start_id]
    end_index = positions[end_id]
    if end_index < start_index:
        raise EditorialGroundingError(f"{start_field}/{end_field} must preserve chronology")
    return tuple(word.word_id for word in timeline.words[start_index : end_index + 1])


def source_spans_from_word_ids(
    timeline: CanonicalTimeline, word_ids: tuple[str, ...], *, allow_reorder: bool = False
) -> tuple[SourceSpan, ...]:
    words = timeline.require_word_ids(word_ids)
    if not allow_reorder and any(a.source_start > b.source_start for a, b in pairwise(words)):
        raise EditorialGroundingError("source word IDs must preserve chronology")
    ordered = words if allow_reorder else tuple(sorted(words, key=lambda item: item.source_start))
    spans: list[SourceSpan] = []
    start = ordered[0].source_start
    end = ordered[0].source_end
    for word in ordered[1:]:
        if word.source_start <= end + 0.8:
            end = max(end, word.source_end)
            continue
        spans.append(SourceSpan(start, end))
        start, end = word.source_start, word.source_end
    spans.append(SourceSpan(start, end))
    return tuple(spans)


def continuous_source_span_from_word_ids(
    timeline: CanonicalTimeline, word_ids: tuple[str, ...]
) -> SourceSpan:
    """Resolve one continuous source interval without treating ordinary pauses as cuts."""
    if not word_ids:
        raise EditorialGroundingError("source word IDs must not be empty")
    words = timeline.require_word_ids(word_ids)
    positions = {word.word_id: index for index, word in enumerate(timeline.words)}
    selected_positions = tuple(positions[word_id] for word_id in word_ids)
    expected_positions = tuple(range(selected_positions[0], selected_positions[-1] + 1))
    if selected_positions != expected_positions:
        raise EditorialGroundingError(
            "source word IDs must be consecutive canonical words in source order"
        )
    return SourceSpan(words[0].source_start, words[-1].source_end)


@dataclass(frozen=True, slots=True)
class EpisodeEditorialProfile:
    summary: str
    valuable_moment_characteristics: tuple[str, ...]
    avoid_characteristics: tuple[str, ...]
    confidence: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EpisodeEditorialProfile:
        valuable = payload.get("valuable_moment_characteristics")
        avoid = payload.get("avoid_characteristics", [])
        if (
            not isinstance(valuable, list)
            or not valuable
            or not all(isinstance(x, str) for x in valuable)
        ):
            raise EditorialGroundingError("valuable_moment_characteristics must be strings")
        if not isinstance(avoid, list) or not all(isinstance(x, str) for x in avoid):
            raise EditorialGroundingError("avoid_characteristics must be strings")
        return cls(
            _nonempty(payload.get("summary"), "summary"),
            tuple(x.strip() for x in valuable if x.strip()),
            tuple(x.strip() for x in avoid if x.strip()),
            _confidence(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class GroundedStoryMoment:
    moment_id: str
    supporting_word_ids: tuple[str, ...]
    semantic_summary: str
    narrative_structure: str
    required_prior_context: str
    required_followup_context: str
    editorial_reason: str
    confidence: float

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], timeline: CanonicalTimeline
    ) -> GroundedStoryMoment:
        ids = _grounded_word_range(
            payload,
            timeline,
            list_field="supporting_word_ids",
            start_field="start_word_id",
            end_field="end_word_id",
        )
        source_spans_from_word_ids(timeline, ids)
        return cls(
            _nonempty(payload.get("moment_id"), "moment_id"),
            ids,
            _nonempty(payload.get("semantic_summary"), "semantic_summary"),
            _nonempty(payload.get("narrative_structure"), "narrative_structure"),
            str(payload.get("required_prior_context") or "").strip(),
            str(payload.get("required_followup_context") or "").strip(),
            _nonempty(payload.get("editorial_reason"), "editorial_reason"),
            _confidence(payload.get("confidence", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["supporting_word_ids"] = list(self.supporting_word_ids)
        return data


@dataclass(frozen=True, slots=True)
class GroundedHookVariant:
    variant_id: str
    strategy_label: str
    source_word_ids: tuple[str, ...]
    overlay_text: str | None
    rationale: str
    confidence: float

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], timeline: CanonicalTimeline
    ) -> GroundedHookVariant:
        ids = _grounded_word_range(
            payload,
            timeline,
            list_field="source_word_ids",
            start_field="source_start_word_id",
            end_field="source_end_word_id",
        )
        source_spans_from_word_ids(timeline, ids)
        overlay = payload.get("overlay_text")
        if overlay is not None and not isinstance(overlay, str):
            raise EditorialGroundingError("overlay_text must be a string or null")
        return cls(
            _nonempty(payload.get("variant_id"), "variant_id"),
            _nonempty(payload.get("strategy_label"), "strategy_label"),
            ids,
            overlay.strip() if isinstance(overlay, str) and overlay.strip() else None,
            _nonempty(payload.get("rationale"), "rationale"),
            _confidence(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class GroundedClipConcept:
    concept_id: str
    story_moment_ids: tuple[str, ...]
    supporting_word_ids: tuple[str, ...]
    semantic_summary: str
    standalone_context: str
    narrative_structure: str
    recommended_duration: float
    visual_dependencies: tuple[str, ...]
    confidence: float
    required_prior_context: str = ""
    required_followup_context: str = ""

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], timeline: CanonicalTimeline
    ) -> GroundedClipConcept:
        moment_ids = payload.get("story_moment_ids")
        visual = payload.get("visual_dependencies", [])
        if (
            not isinstance(moment_ids, list)
            or not moment_ids
            or not all(isinstance(x, str) for x in moment_ids)
        ):
            raise EditorialGroundingError("story_moment_ids must be strings")
        if not isinstance(visual, list) or not all(isinstance(x, str) for x in visual):
            raise EditorialGroundingError("visual_dependencies must be strings")
        ids = _grounded_word_range(
            payload,
            timeline,
            list_field="supporting_word_ids",
            start_field="start_word_id",
            end_field="end_word_id",
        )
        source_spans_from_word_ids(timeline, ids)
        duration = float(payload.get("recommended_duration", 0.0))
        if duration <= 0:
            raise EditorialGroundingError("recommended_duration must be positive")
        return cls(
            _nonempty(payload.get("concept_id"), "concept_id"),
            tuple(x.strip() for x in moment_ids if x.strip()),
            ids,
            _nonempty(payload.get("semantic_summary"), "semantic_summary"),
            str(payload.get("standalone_context") or "").strip(),
            _nonempty(payload.get("narrative_structure"), "narrative_structure"),
            duration,
            tuple(x.strip() for x in visual if x.strip()),
            _confidence(payload.get("confidence", 0.0)),
            str(payload.get("required_prior_context") or "").strip(),
            str(payload.get("required_followup_context") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class GroundedEditPlan:
    plan_id: str
    video_id: str
    concept_id: str
    variant_id: str
    source_word_ids: tuple[str, ...]
    hook_source_word_ids: tuple[str, ...]
    overlay_text: str | None
    strategy_label: str
    caption_platform: str
    confidence: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any], timeline: CanonicalTimeline) -> GroundedEditPlan:
        # Source identity is provenance owned by the canonical timeline, never by model output.
        video_id = timeline.video_id
        source_ids = _grounded_word_range(
            payload,
            timeline,
            list_field="source_word_ids",
            start_field="source_start_word_id",
            end_field="source_end_word_id",
        )
        hook_ids = _grounded_word_range(
            payload,
            timeline,
            list_field="hook_source_word_ids",
            start_field="hook_start_word_id",
            end_field="hook_end_word_id",
        )
        continuous_source_span_from_word_ids(timeline, source_ids)
        continuous_source_span_from_word_ids(timeline, hook_ids)
        source_words = timeline.require_word_ids(source_ids)
        hook_words = timeline.require_word_ids(hook_ids)
        source_id_set = set(source_ids)
        if any(word.word_id not in source_id_set for word in hook_words):
            raise EditorialGroundingError(
                "spoken hook word IDs must belong to the edit source words"
            )
        overlay = payload.get("overlay_text")
        if overlay is not None and not isinstance(overlay, str):
            raise EditorialGroundingError("overlay_text must be a string or null")
        if source_words[0].source_start > hook_words[0].source_start:
            raise EditorialGroundingError("hook cannot begin before the selected source edit")
        return cls(
            _nonempty(payload.get("plan_id"), "plan_id"),
            video_id,
            _nonempty(payload.get("concept_id"), "concept_id"),
            _nonempty(payload.get("variant_id"), "variant_id"),
            source_ids,
            hook_ids,
            overlay.strip() if isinstance(overlay, str) and overlay.strip() else None,
            _nonempty(payload.get("strategy_label"), "strategy_label"),
            _nonempty(payload.get("caption_platform"), "caption_platform"),
            _confidence(payload.get("confidence", 0.0)),
        )

    def compile(self, timeline: CanonicalTimeline, transcript_fingerprint: str) -> EditPlan:
        spans = (continuous_source_span_from_word_ids(timeline, self.source_word_ids),)
        hook_words = timeline.require_word_ids(self.hook_source_word_ids)
        first_hook = hook_words[0]
        structural_mode: HookMode = "curiosity_text" if self.overlay_text else "direct"
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "video_id": self.video_id,
                    "word_ids": self.source_word_ids,
                    "hook_word_ids": self.hook_source_word_ids,
                    "overlay_text": self.overlay_text,
                    "strategy": self.strategy_label,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return EditPlan(
            plan_id=self.plan_id,
            video_id=self.video_id,
            concept_id=self.concept_id,
            variant_id=self.variant_id,
            hook_mode=structural_mode,
            source_spans=spans,
            hook_text=self.overlay_text,
            beats=(),
            caption_platform=self.caption_platform,
            score=self.confidence * 10,
            transcript_fingerprint=transcript_fingerprint or fingerprint,
            caption_start_source_time=first_hook.source_start,
            caption_start_word=first_hook.text,
        )

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .ai_editorial import (
    EditorialGroundingError,
    EpisodeEditorialProfile,
    GroundedClipConcept,
    GroundedEditPlan,
    GroundedHookVariant,
    GroundedStoryMoment,
    continuous_source_span_from_word_ids,
    source_spans_from_word_ids,
)
from .cache import FileCache, model_stage_cache_key, stable_hash
from .canonical import CanonicalTimeline
from .editorial_integrity import (
    BoundaryAudit,
    GateDecision,
    HazardClassification,
    SourceHazardSegment,
    evaluate_campaign_policy,
    evaluate_pre_render_eligibility,
)
from .models import (
    CampaignBrief,
    ClipConcept,
    EditorialScores,
    EditPlan,
    HookVariant,
    StoryMoment,
)
from .providers.base import EditorialProvider, EmbeddingProvider, InferenceUsage, ProviderResult
from .visual import VisualTimeline


@dataclass(slots=True)
class OpenVideoAnalysis:
    profile: EpisodeEditorialProfile
    moments: list[StoryMoment]
    concepts: list[ClipConcept]
    grounded_moments: dict[str, GroundedStoryMoment]
    grounded_concepts: dict[str, GroundedClipConcept]
    rejections: list[dict[str, object]]
    source_hazards: list[SourceHazardSegment] = field(default_factory=list)


@dataclass(slots=True)
class OpenEditorialBatch:
    discovered_moments: list[StoryMoment]
    discovered_concepts: list[ClipConcept]
    selected_concepts: list[ClipConcept]
    variants: list[HookVariant]
    plans: list[EditPlan]
    rejections: list[dict[str, object]]
    model_invocations: list[dict[str, object]]
    boundary_audits: list[dict[str, object]] = field(default_factory=list)
    campaign_policy_audits: list[dict[str, object]] = field(default_factory=list)
    source_hazards: list[dict[str, object]] = field(default_factory=list)


def _compat_scores(confidence: float) -> EditorialScores:
    value = round(max(0.0, min(1.0, confidence)) * 10, 4)
    return EditorialScores(
        hook_strength=value,
        curiosity=value,
        payoff_strength=value,
        standalone_clarity=value,
        emotional_energy=value,
        information_value=value,
        controversy_or_tension=value,
        quoteability=value,
        specificity=value,
        campaign_relevance=value,
        story_completeness=value,
        retention_potential=value,
    )


def _source_text(timeline: CanonicalTimeline, word_ids: tuple[str, ...]) -> str:
    return " ".join(word.text for word in timeline.require_word_ids(word_ids)).strip()


def _compact_campaign(brief: CampaignBrief) -> dict[str, object]:
    # Keep the historical cache material while the external campaign schema migrates away
    # from output quotas. These compatibility values are not production yield targets.
    return {
        "campaign_id": brief.campaign_id,
        "title": brief.title,
        "objective": brief.objective,
        "language": brief.language,
        "min_clip_seconds": brief.min_clip_seconds,
        "max_clip_seconds": brief.max_clip_seconds,
        "posting_requirements": list(brief.posting_requirements),
        "clip_count": brief.clip_count,
        "planning_concept_budget": brief.production.concept_count,
        "final_render_budget": brief.production.final_render_budget,
        "acceptance_policy": asdict(brief.acceptance_policy),
    }


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("embedding vectors must be non-empty and have equal dimensions")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class AutonomousEditorialPlanner:
    """Model-driven editorial planner. No lexical/domain heuristics are used here."""

    def __init__(
        self,
        editorial: EditorialProvider,
        embeddings: EmbeddingProvider,
        cache: FileCache,
        *,
        max_words_per_chunk: int = 900,
        chunk_overlap_words: int = 120,
        semantic_duplicate_threshold: float = 0.9,
        hook_duplicate_threshold: float = 0.94,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        if max_words_per_chunk < 200:
            raise ValueError("max_words_per_chunk must be at least 200")
        if not 0 <= chunk_overlap_words < max_words_per_chunk:
            raise ValueError("chunk_overlap_words must be smaller than chunk size")
        if not 0.5 <= semantic_duplicate_threshold <= 0.999:
            raise ValueError("semantic duplicate threshold is invalid")
        self.editorial = editorial
        self.embeddings = embeddings
        self.cache = cache
        self.max_words_per_chunk = max_words_per_chunk
        self.chunk_overlap_words = chunk_overlap_words
        self.semantic_duplicate_threshold = semantic_duplicate_threshold
        self.hook_duplicate_threshold = hook_duplicate_threshold
        self.progress_callback = progress_callback
        self.invocations: list[dict[str, object]] = []

    def _campaign(self, brief: CampaignBrief) -> dict[str, object]:
        return _compact_campaign(brief)

    def _complete(
        self,
        stage: str,
        timeline: CanonicalTimeline,
        brief: CampaignBrief,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        key = model_stage_cache_key(
            stage,
            source_hash=timeline.source_hash,
            campaign=self._campaign(brief),
            model=self.editorial.identity,
            payload=payload,
            sampling={"do_sample": False},
        )
        cached = self.cache.read(key, "open-model-output")
        if isinstance(cached, dict):
            if self.progress_callback is not None:
                self.progress_callback(stage, "cache_hit")
            self.invocations.append(
                {
                    "stage": stage,
                    "cache_key": key,
                    "cache_hit": True,
                    "model": self.editorial.identity.to_dict(),
                }
            )
            return {str(k): v for k, v in cached.items()}
        if self.progress_callback is not None:
            self.progress_callback(stage, "running")
        try:
            result = self.editorial.complete_json(task=stage, payload=payload)
        except Exception:
            if self.progress_callback is not None:
                self.progress_callback(stage, "failed")
            raise
        if self.progress_callback is not None:
            self.progress_callback(stage, "success")
        self.cache.write(key, "open-model-output", result.value)
        self._record(stage, key, result, cache_hit=False)
        return result.value

    def _embed(
        self,
        stage: str,
        timeline: CanonicalTimeline,
        brief: CampaignBrief,
        texts: list[str],
    ) -> list[list[float]]:
        payload = {"texts": texts}
        key = model_stage_cache_key(
            stage,
            source_hash=timeline.source_hash,
            campaign=self._campaign(brief),
            model=self.embeddings.identity,
            payload=payload,
            sampling=None,
        )
        cached = self.cache.read(key, "embeddings")
        if isinstance(cached, list) and all(isinstance(row, list) for row in cached):
            if self.progress_callback is not None:
                self.progress_callback(stage, "cache_hit")
            vectors = [[float(value) for value in row] for row in cached]
            self.invocations.append(
                {
                    "stage": stage,
                    "cache_key": key,
                    "cache_hit": True,
                    "model": self.embeddings.identity.to_dict(),
                }
            )
            return vectors
        if self.progress_callback is not None:
            self.progress_callback(stage, "running")
        try:
            result = self.embeddings.embed(texts)
        except Exception:
            if self.progress_callback is not None:
                self.progress_callback(stage, "failed")
            raise
        if self.progress_callback is not None:
            self.progress_callback(stage, "success")
        self.cache.write(key, "embeddings", result.value)
        self._record(stage, key, result, cache_hit=False)
        return result.value

    @staticmethod
    def _repair_grounded_plan_duration(
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        concept: GroundedClipConcept,
        grounded_plan: GroundedEditPlan,
    ) -> tuple[GroundedEditPlan | None, dict[str, object]]:
        original_duration = continuous_source_span_from_word_ids(
            timeline, grounded_plan.source_word_ids
        ).duration
        if brief.min_clip_seconds <= original_duration <= brief.max_clip_seconds:
            return grounded_plan, {}

        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        concept_positions = [positions[word_id] for word_id in concept.supporting_word_ids]
        hook_positions = [positions[word_id] for word_id in grounded_plan.hook_source_word_ids]
        source_positions = [positions[word_id] for word_id in grounded_plan.source_word_ids]
        concept_start = min(concept_positions)
        concept_end = max(concept_positions)
        hook_start = min(hook_positions)
        hook_end = max(hook_positions)
        original_start = min(source_positions)
        target = min(
            brief.max_clip_seconds,
            max(brief.min_clip_seconds, original_duration),
        )

        # A duration repair may only select source words from the grounded concept.
        # Semantic expansion outside that evidence requires another model decision.
        candidate_starts = list(range(concept_start, hook_start + 1))
        candidates: list[tuple[float, int, int, float]] = []
        for start_index in candidate_starts:
            start_word = timeline.words[start_index]
            for end_index in range(max(hook_end, start_index), concept_end + 1):
                end_word = timeline.words[end_index]
                duration = end_word.source_end - start_word.source_start
                if duration < brief.min_clip_seconds:
                    continue
                if duration > brief.max_clip_seconds:
                    break
                boundary_bonus = 0.0
                if end_word.text.rstrip().endswith((".", "?", "!")):
                    boundary_bonus -= 1.0
                if end_index < len(timeline.words) - 1:
                    pause = timeline.words[end_index + 1].source_start - end_word.source_end
                    if pause >= 0.35:
                        boundary_bonus -= min(0.75, pause)
                start_shift = abs(
                    start_word.source_start - timeline.words[original_start].source_start
                )
                hook_delay = max(
                    0.0, timeline.words[hook_start].source_start - start_word.source_start
                )
                score = (
                    abs(duration - target) + 0.12 * start_shift + 0.08 * hook_delay + boundary_bonus
                )
                candidates.append((score, start_index, end_index, duration))

        if not candidates:
            return None, {
                "concept_id": concept.concept_id,
                "stage": "edit_plan",
                "decision": "REJECT",
                "reasons": ["duration_outside_campaign_bounds_no_grounded_repair"],
                "plan_id": grounded_plan.plan_id,
                "original_duration": round(original_duration, 6),
                "duration_seconds": round(original_duration, 6),
                "minimum_seconds": brief.min_clip_seconds,
                "maximum_seconds": brief.max_clip_seconds,
                "campaign_min_seconds": brief.min_clip_seconds,
                "campaign_max_seconds": brief.max_clip_seconds,
                "grounded_context_duration": round(
                    timeline.words[concept_end].source_end
                    - timeline.words[concept_start].source_start,
                    6,
                ),
            }

        _, start_index, end_index, repaired_duration = min(candidates, key=lambda item: item[0])
        repaired_ids = tuple(word.word_id for word in timeline.words[start_index : end_index + 1])
        hook_id_set = set(grounded_plan.hook_source_word_ids)
        if not hook_id_set.issubset(repaired_ids):
            raise EditorialGroundingError("duration repair dropped grounded spoken hook words")
        repaired = replace(grounded_plan, source_word_ids=repaired_ids)
        evidence: dict[str, object] = {
            "concept_id": concept.concept_id,
            "stage": "edit_plan",
            "decision": "REPAIR",
            "reasons": ["duration_repaired_to_campaign_bounds"],
            "plan_id": grounded_plan.plan_id,
            "original_duration": round(original_duration, 6),
            "duration_seconds": round(original_duration, 6),
            "repaired_duration": round(repaired_duration, 6),
            "minimum_seconds": brief.min_clip_seconds,
            "maximum_seconds": brief.max_clip_seconds,
            "campaign_min_seconds": brief.min_clip_seconds,
            "campaign_max_seconds": brief.max_clip_seconds,
            "source_start_word_id": timeline.word_ref(repaired_ids[0]),
            "source_end_word_id": timeline.word_ref(repaired_ids[-1]),
        }
        return repaired, evidence

    def _record(
        self,
        stage: str,
        key: str,
        result: ProviderResult[Any],
        *,
        cache_hit: bool,
    ) -> None:
        usage: InferenceUsage = result.usage
        self.invocations.append(
            {
                "stage": stage,
                "cache_key": key,
                "cache_hit": cache_hit,
                "model": result.model.to_dict(),
                "usage": asdict(usage),
                "degraded": result.degraded,
            }
        )

    @staticmethod
    def _word_payload(timeline: CanonicalTimeline, start: int, end: int) -> list[dict[str, object]]:
        return [
            {
                "word_id": word.word_id,
                "word_ref": timeline.word_ref(word.word_id),
                "text": word.text,
                "source_start": word.source_start,
                "source_end": word.source_end,
                "speaker_id": word.speaker_id,
            }
            for word in timeline.words[start:end]
        ]

    def _plan_context_words(
        self,
        timeline: CanonicalTimeline,
        concept: GroundedClipConcept,
        brief: CampaignBrief,
        *,
        max_words: int = 360,
    ) -> list[dict[str, object]]:
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        concept_positions = [positions[word_id] for word_id in concept.supporting_word_ids]
        first_index = min(concept_positions)
        last_index = max(concept_positions)
        context_start_time = max(
            timeline.start, timeline.words[first_index].source_start - brief.min_clip_seconds
        )
        context_end_time = min(
            timeline.end, timeline.words[last_index].source_end + brief.max_clip_seconds
        )
        start_index = first_index
        while start_index > 0 and timeline.words[start_index - 1].source_end >= context_start_time:
            start_index -= 1
        end_index = last_index + 1
        while (
            end_index < len(timeline.words)
            and timeline.words[end_index].source_start <= context_end_time
        ):
            end_index += 1
        if end_index - start_index > max_words:
            concept_width = last_index - first_index + 1
            if concept_width >= max_words:
                # The cap applies to surrounding context, never to the grounded concept itself.
                # Truncating either edge can amputate the setup or resolution before planning.
                start_index = first_index
                end_index = last_index + 1
            else:
                before = max(0, first_index - start_index)
                remaining = max_words - concept_width
                keep_before = min(before, remaining // 3)
                start_index = max(0, first_index - keep_before)
                end_index = min(len(timeline.words), start_index + max_words)
                if end_index <= last_index:
                    end_index = last_index + 1
                    start_index = max(0, end_index - max_words)
        return self._word_payload(timeline, start_index, end_index)

    def _chunks(self, timeline: CanonicalTimeline) -> list[list[dict[str, object]]]:
        if not timeline.words:
            return []
        step = self.max_words_per_chunk - self.chunk_overlap_words
        chunks: list[list[dict[str, object]]] = []
        for start in range(0, len(timeline.words), step):
            end = min(len(timeline.words), start + self.max_words_per_chunk)
            chunks.append(self._word_payload(timeline, start, end))
            if end == len(timeline.words):
                break
        return chunks

    def _classify_source_hazards(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        visual_timeline: VisualTimeline | None,
    ) -> tuple[list[SourceHazardSegment], list[dict[str, object]]]:
        if not brief.acceptance_policy.enabled:
            return [], []
        segments: list[SourceHazardSegment] = []
        rejections: list[dict[str, object]] = []
        model_identity: dict[str, object] = dict(self.editorial.identity.to_dict())
        for chunk_index, words in enumerate(self._chunks(timeline)):
            raw_chunk_start = words[0]["source_start"]
            raw_chunk_end = words[-1]["source_end"]
            if not isinstance(raw_chunk_start, int | float | str) or not isinstance(
                raw_chunk_end, int | float | str
            ):
                raise EditorialGroundingError("source hazard chunk timestamps are invalid")
            chunk_start = float(raw_chunk_start)
            chunk_end = float(raw_chunk_end)
            stage = f"source_hazards:{chunk_index}"
            try:
                payload = self._complete(
                    stage,
                    timeline,
                    brief,
                    {
                        "campaign": self._campaign(brief),
                        "instruction": (
                            "Classify the entire supplied source-word interval into exhaustive, "
                            "chronological source-content segments. Fuse transcript semantics with "
                            "the supplied visual evidence. Do not omit ordinary editorial content. "
                            "Use UNKNOWN when evidence is insufficient; uncertainty is not PASS."
                        ),
                        "words": words,
                        "visual_evidence": self._visual_evidence(
                            visual_timeline, chunk_start, chunk_end
                        ),
                    },
                )
                for raw in self._array(payload, "segments"):
                    segments.append(
                        SourceHazardSegment.from_payload(
                            raw,
                            timeline,
                            model_identity=model_identity,
                        )
                    )
            except Exception as exc:
                source_ids = tuple(str(item["word_id"]) for item in words)
                segments.append(
                    SourceHazardSegment(
                        start=chunk_start,
                        end=chunk_end,
                        classification=HazardClassification.UNKNOWN,
                        confidence=0.0,
                        evidence=("source_hazard_classification_failed",),
                        model_identity=model_identity,
                        source_word_ids=source_ids,
                    )
                )
                rejections.append(
                    {
                        "stage": "source_hazard_classification",
                        "model_stage": stage,
                        "decision": "ESCALATE",
                        "reasons": ["policy_uncertain"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        segments.sort(key=lambda item: (item.start, item.end, item.classification.value))
        return segments, rejections

    def _boundary_audit(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        concept: GroundedClipConcept,
        plan: GroundedEditPlan,
        *,
        stage_suffix: str = "",
    ) -> BoundaryAudit:
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        start_index = positions[plan.source_word_ids[0]]
        end_index = positions[plan.source_word_ids[-1]]
        selected = timeline.words[start_index : end_index + 1]
        before = timeline.words[max(0, start_index - 36) : start_index]
        after = timeline.words[end_index + 1 : min(len(timeline.words), end_index + 37)]
        stage = f"boundary_audit:{plan.plan_id}{stage_suffix}"
        payload = self._complete(
            stage,
            timeline,
            brief,
            {
                "campaign": self._campaign(brief),
                "instruction": (
                    "Evaluate the exact proposed clip as a minimal sufficient story. The maximum "
                    "duration is a ceiling, never a target. Judge semantics, not punctuation or "
                    "pauses alone. First audible content must be understandable without hidden "
                    "context; the ending must resolve every obligation created by the clip. "
                    "Report uncertainty rather than optimistic PASS. Suggest grounded repair word "
                    "references only when a localized chronological repair can preserve truth."
                ),
                "concept": {
                    "concept_id": concept.concept_id,
                    "semantic_summary": concept.semantic_summary,
                    "narrative_structure": concept.narrative_structure,
                    "required_prior_context": concept.required_prior_context,
                    "required_followup_context": concept.required_followup_context,
                },
                "plan": {
                    "plan_id": plan.plan_id,
                    "source_start_word_id": timeline.word_ref(plan.source_word_ids[0]),
                    "source_end_word_id": timeline.word_ref(plan.source_word_ids[-1]),
                    "duration_seconds": selected[-1].source_end - selected[0].source_start,
                },
                "pre_start_words": self._word_payload(
                    timeline, max(0, start_index - 36), start_index
                ),
                "clip_words": self._word_payload(timeline, start_index, end_index + 1),
                "post_end_words": self._word_payload(
                    timeline, end_index + 1, min(len(timeline.words), end_index + 37)
                ),
            },
        )
        return BoundaryAudit.from_payload(
            payload,
            source_start=selected[0].source_start,
            source_end=selected[-1].source_end,
            first_source_word=selected[0].text,
            last_source_word=selected[-1].text,
            pre_start_context=" ".join(word.text for word in before),
            post_end_context=" ".join(word.text for word in after),
            source_word_evidence=tuple(word.word_id for word in selected),
            model_identity=dict(self.editorial.identity.to_dict()),
            prompt_version=self.editorial.identity.prompt_version,
            schema_version="boundary-audit-v1",
            required_prior_context=concept.required_prior_context,
            required_followup_context=concept.required_followup_context,
        )

    @staticmethod
    def _apply_boundary_repair(
        timeline: CanonicalTimeline,
        plan: GroundedEditPlan,
        audit: BoundaryAudit,
    ) -> GroundedEditPlan:
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        start_id = (
            timeline.resolve_word_ref(audit.repair_start_word_id)
            if audit.repair_start_word_id
            else plan.source_word_ids[0]
        )
        end_id = (
            timeline.resolve_word_ref(audit.repair_end_word_id)
            if audit.repair_end_word_id
            else plan.source_word_ids[-1]
        )
        start_index = positions[start_id]
        end_index = positions[end_id]
        if end_index < start_index:
            raise EditorialGroundingError("boundary repair must preserve source chronology")
        repaired_ids = tuple(word.word_id for word in timeline.words[start_index : end_index + 1])
        if not set(plan.hook_source_word_ids).issubset(repaired_ids):
            raise EditorialGroundingError("boundary repair dropped grounded spoken hook words")
        return replace(plan, source_word_ids=repaired_ids)

    @staticmethod
    def _resolve_story_moment_id(
        requested_id: str,
        concept: GroundedClipConcept,
        grounded_moments: dict[str, GroundedStoryMoment],
    ) -> str | None:
        """Resolve model-local moment aliases using grounded source-word evidence."""
        if requested_id in grounded_moments:
            return requested_id
        candidates = sorted(
            moment_id for moment_id in grounded_moments if moment_id.endswith(f":{requested_id}")
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            return None

        concept_words = set(concept.supporting_word_ids)
        scored: list[tuple[tuple[float, float, int], str]] = []
        for candidate_id in candidates:
            moment_words = set(grounded_moments[candidate_id].supporting_word_ids)
            overlap = len(concept_words & moment_words)
            if overlap == 0:
                continue
            score = (
                overlap / len(moment_words),
                overlap / len(concept_words),
                overlap,
            )
            scored.append((score, candidate_id))
        if not scored:
            return None
        best_score = max(score for score, _ in scored)
        winners = [candidate_id for score, candidate_id in scored if score == best_score]
        return winners[0] if len(winners) == 1 else None

    def _profile_evidence(self, timeline: CanonicalTimeline) -> list[dict[str, object]]:
        if len(timeline.words) <= 1800:
            return self._word_payload(timeline, 0, len(timeline.words))
        window = 60
        windows = 8
        maximum_start = max(0, len(timeline.words) - window)
        starts = [round(index * maximum_start / (windows - 1)) for index in range(windows)]
        evidence: list[dict[str, object]] = []
        for start in starts:
            evidence.extend(
                self._word_payload(timeline, start, min(len(timeline.words), start + window))
            )
        return evidence

    @staticmethod
    def _array(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise EditorialGroundingError(f"model output {key} must be a list")
        if not all(isinstance(item, dict) for item in value):
            raise EditorialGroundingError(f"model output {key} must contain objects")
        return [{str(k): v for k, v in item.items()} for item in value]

    @staticmethod
    def _visual_evidence(
        timeline: VisualTimeline | None, start: float | None = None, end: float | None = None
    ) -> list[dict[str, object]]:
        if timeline is None:
            return []
        events = timeline.events
        if start is not None and end is not None:
            events = tuple(event for event in events if event.end > start and event.start < end)
        return [
            {
                "start": event.start,
                "end": event.end,
                "scene_id": event.scene_id,
                "summary": event.summary,
                "visible_speakers": list(event.visible_speakers),
                "event_labels": list(event.event_labels),
                "confidence": event.confidence,
            }
            for event in events
        ]

    def _empty_video_analysis(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        visual_timeline: VisualTimeline | None,
        profile: EpisodeEditorialProfile,
        grounded_moments: dict[str, GroundedStoryMoment],
        rejections: list[dict[str, object]],
    ) -> OpenVideoAnalysis:
        source_hazards, hazard_rejections = self._classify_source_hazards(
            brief, timeline, visual_timeline
        )
        rejections.extend(hazard_rejections)
        return OpenVideoAnalysis(
            profile=profile,
            moments=[self._compat_moment(timeline, item) for item in grounded_moments.values()],
            concepts=[],
            grounded_moments=grounded_moments,
            grounded_concepts={},
            rejections=rejections,
            source_hazards=source_hazards,
        )

    def analyze_video(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        visual_timeline: VisualTimeline | None = None,
    ) -> OpenVideoAnalysis:
        profile_payload = self._complete(
            "episode_editorial_profile",
            timeline,
            brief,
            {
                "campaign": self._campaign(brief),
                "instruction": (
                    "Infer what is editorially valuable for this episode itself. "
                    "Do not use a fixed domain ontology."
                ),
                "timeline_evidence": self._profile_evidence(timeline),
                "visual_evidence": self._visual_evidence(visual_timeline),
            },
        )
        profile = EpisodeEditorialProfile.from_payload(profile_payload)
        grounded_moments: dict[str, GroundedStoryMoment] = {}
        rejections: list[dict[str, object]] = []
        successful_story_chunks = 0
        chunks = self._chunks(timeline)
        for chunk_index, words in enumerate(chunks):
            first_start = words[0].get("source_start") if words else None
            last_end = words[-1].get("source_end") if words else None
            chunk_start = float(first_start) if isinstance(first_start, int | float | str) else 0.0
            chunk_end = float(last_end) if isinstance(last_end, int | float | str) else 0.0
            stage = f"story_moments:{chunk_index}"
            try:
                payload = self._complete(
                    stage,
                    timeline,
                    brief,
                    {
                        "campaign": self._campaign(brief),
                        "editorial_profile": asdict(profile),
                        "instruction": (
                            "Return every independently meaningful moment supported by this chunk. "
                            "Optimize recall. "
                            "Reference canonical word IDs only; do not invent transcript text."
                        ),
                        "words": words,
                        "visual_evidence": self._visual_evidence(
                            visual_timeline, chunk_start, chunk_end
                        ),
                    },
                )
                successful_story_chunks += 1
            except Exception as exc:
                rejections.append(
                    {
                        "stage": "story_moment_inference",
                        "model_stage": stage,
                        "decision": "REJECT",
                        "reasons": ["chunk_inference_failed"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            for proposal_index, raw in enumerate(self._array(payload, "moments")):
                try:
                    moment = GroundedStoryMoment.from_payload(raw, timeline)
                except ValueError as exc:
                    rejections.append(
                        {
                            "stage": "story_moment_grounding",
                            "model_stage": stage,
                            "proposal_index": proposal_index,
                            "moment_id": str(raw.get("moment_id") or ""),
                            "decision": "REJECT",
                            "reasons": ["invalid_grounded_story_moment"],
                            "error": str(exc),
                        }
                    )
                    continue
                namespaced_id = f"chunk-{chunk_index}:{moment.moment_id}"
                moment = replace(moment, moment_id=namespaced_id)
                grounded_moments[namespaced_id] = moment
        if not grounded_moments:
            if chunks and successful_story_chunks == 0:
                diagnostics = [item.get("error") for item in rejections[-8:] if item.get("error")]
                raise EditorialGroundingError(
                    "open editorial model failed every StoryMoment chunk; "
                    f"rejected={len(rejections)}; diagnostics={diagnostics}"
                )
            return self._empty_video_analysis(
                brief,
                timeline,
                visual_timeline,
                profile,
                grounded_moments,
                rejections,
            )

        moment_payloads = [
            {
                "moment_id": moment.moment_id,
                "start_word_id": timeline.word_ref(moment.supporting_word_ids[0]),
                "end_word_id": timeline.word_ref(moment.supporting_word_ids[-1]),
                "semantic_summary": moment.semantic_summary,
                "narrative_structure": moment.narrative_structure,
                "required_prior_context": moment.required_prior_context,
                "required_followup_context": moment.required_followup_context,
                "editorial_reason": moment.editorial_reason,
                "confidence": moment.confidence,
            }
            for moment in grounded_moments.values()
        ]
        concept_payload = self._complete(
            "clip_concepts",
            timeline,
            brief,
            {
                "campaign": self._campaign(brief),
                "editorial_profile": asdict(profile),
                "instruction": (
                    "Construct all independently understandable clip concepts from the "
                    "grounded moments. "
                    "Do not limit discovery to the campaign clip_count."
                ),
                "moments": moment_payloads,
            },
        )
        grounded_concepts: dict[str, GroundedClipConcept] = {}
        for proposal_index, raw in enumerate(self._array(concept_payload, "concepts")):
            try:
                concept = GroundedClipConcept.from_payload(raw, timeline)
                resolved_moment_ids: list[str] = []
                unknown: set[str] = set()
                for moment_id in concept.story_moment_ids:
                    resolved = self._resolve_story_moment_id(moment_id, concept, grounded_moments)
                    if resolved is None:
                        unknown.add(moment_id)
                    elif resolved not in resolved_moment_ids:
                        resolved_moment_ids.append(resolved)
                if unknown:
                    raise EditorialGroundingError(
                        f"concept {concept.concept_id} references unknown StoryMoments "
                        f"or ambiguous aliases: {sorted(unknown)}"
                    )
                if tuple(resolved_moment_ids) != concept.story_moment_ids:
                    concept = replace(concept, story_moment_ids=tuple(resolved_moment_ids))
                linked_moments = [grounded_moments[item] for item in resolved_moment_ids]
                required_prior = concept.required_prior_context or "; ".join(
                    dict.fromkeys(
                        item.required_prior_context
                        for item in linked_moments
                        if item.required_prior_context
                    )
                )
                required_followup = concept.required_followup_context or "; ".join(
                    dict.fromkeys(
                        item.required_followup_context
                        for item in linked_moments
                        if item.required_followup_context
                    )
                )
                if (
                    required_prior != concept.required_prior_context
                    or required_followup != concept.required_followup_context
                ):
                    concept = replace(
                        concept,
                        required_prior_context=required_prior,
                        required_followup_context=required_followup,
                    )
            except (ValueError, EditorialGroundingError) as exc:
                rejections.append(
                    {
                        "stage": "concept_grounding",
                        "proposal_index": proposal_index,
                        "concept_id": str(raw.get("concept_id") or ""),
                        "decision": "REJECT",
                        "reasons": ["invalid_grounded_clip_concept"],
                        "error": str(exc),
                    }
                )
                continue
            existing = grounded_concepts.get(concept.concept_id)
            if existing is None or concept.confidence > existing.confidence:
                grounded_concepts[concept.concept_id] = concept
        if not grounded_concepts:
            return self._empty_video_analysis(
                brief,
                timeline,
                visual_timeline,
                profile,
                grounded_moments,
                rejections,
            )

        representatives, clusters, duplicate_rejections = self._semantic_dedupe(
            brief, timeline, list(grounded_concepts.values())
        )
        rejections.extend(duplicate_rejections)
        grounded_concepts = {concept.concept_id: concept for concept in representatives}
        moments = [self._compat_moment(timeline, item) for item in grounded_moments.values()]
        concepts = [
            self._compat_concept(timeline, item, clusters[item.concept_id])
            for item in representatives
        ]
        source_hazards, hazard_rejections = self._classify_source_hazards(
            brief, timeline, visual_timeline
        )
        rejections.extend(hazard_rejections)
        return OpenVideoAnalysis(
            profile=profile,
            moments=moments,
            concepts=concepts,
            grounded_moments=grounded_moments,
            grounded_concepts=grounded_concepts,
            rejections=rejections,
            source_hazards=source_hazards,
        )

    def _semantic_dedupe(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        concepts: list[GroundedClipConcept],
    ) -> tuple[list[GroundedClipConcept], dict[str, str], list[dict[str, object]]]:
        ordered = sorted(concepts, key=lambda item: (-item.confidence, item.concept_id))
        vectors = self._embed(
            "concept_embeddings", timeline, brief, [item.semantic_summary for item in ordered]
        )
        if len(vectors) != len(ordered):
            raise ValueError("embedding provider returned the wrong number of vectors")
        kept: list[GroundedClipConcept] = []
        kept_vectors: list[list[float]] = []
        clusters: dict[str, str] = {}
        rejections: list[dict[str, object]] = []
        for concept, vector in zip(ordered, vectors, strict=True):
            duplicate_index: int | None = None
            duplicate_similarity = 0.0
            for index, other_vector in enumerate(kept_vectors):
                similarity = _cosine(vector, other_vector)
                if similarity >= self.semantic_duplicate_threshold:
                    duplicate_index = index
                    duplicate_similarity = similarity
                    break
            if duplicate_index is None:
                cluster = f"sem-{stable_hash({'concept': concept.concept_id})[:12]}"
                kept.append(concept)
                kept_vectors.append(vector)
                clusters[concept.concept_id] = cluster
                continue
            representative = kept[duplicate_index]
            clusters[concept.concept_id] = clusters[representative.concept_id]
            rejections.append(
                {
                    "concept_id": concept.concept_id,
                    "stage": "semantic_dedup",
                    "decision": "REJECT",
                    "reasons": ["learned_embedding_duplicate"],
                    "duplicate_of": representative.concept_id,
                    "similarity": round(duplicate_similarity, 6),
                }
            )
        return kept, clusters, rejections

    def _empty_batch(
        self,
        discovered_moments: list[StoryMoment],
        discovered_concepts: list[ClipConcept],
        selected: list[ClipConcept],
        variants: list[HookVariant],
        rejections: list[dict[str, object]],
        boundary_audits: list[dict[str, object]],
        campaign_policy_audits: list[dict[str, object]],
        source_hazard_evidence: list[dict[str, object]],
    ) -> OpenEditorialBatch:
        return OpenEditorialBatch(
            discovered_moments=discovered_moments,
            discovered_concepts=discovered_concepts,
            selected_concepts=selected,
            variants=variants,
            plans=[],
            rejections=rejections,
            model_invocations=list(self.invocations),
            boundary_audits=boundary_audits,
            campaign_policy_audits=campaign_policy_audits,
            source_hazards=source_hazard_evidence,
        )

    def plan_batch(
        self,
        brief: CampaignBrief,
        timelines: dict[str, CanonicalTimeline],
        analyses: list[OpenVideoAnalysis],
    ) -> OpenEditorialBatch:
        discovered_moments = [moment for analysis in analyses for moment in analysis.moments]
        discovered_concepts = [concept for analysis in analyses for concept in analysis.concepts]
        grounded: dict[str, GroundedClipConcept] = {}
        hazards_by_video: dict[str, list[SourceHazardSegment]] = {}
        source_hazard_evidence: list[dict[str, object]] = []
        rejections: list[dict[str, object]] = []
        for analysis in analyses:
            grounded.update(analysis.grounded_concepts)
            if analysis.source_hazards:
                hazard_video_ids = {concept.video_id for concept in analysis.concepts}
                if len(hazard_video_ids) == 1:
                    hazard_video_id = next(iter(hazard_video_ids))
                    hazards_by_video.setdefault(hazard_video_id, []).extend(analysis.source_hazards)
                    source_hazard_evidence.extend(
                        {**hazard.to_dict(), "video_id": hazard_video_id}
                        for hazard in analysis.source_hazards
                    )
        if not discovered_concepts:
            return self._empty_batch(
                discovered_moments,
                discovered_concepts,
                [],
                [],
                rejections,
                [],
                [],
                source_hazard_evidence,
            )

        selection_timeline = timelines[discovered_concepts[0].video_id]
        selection = self._complete(
            "global_concept_comparison",
            selection_timeline,
            brief,
            {
                "campaign": self._campaign(brief),
                "instruction": (
                    "Rank materially distinct concepts for planning. Consider the whole "
                    "episode/corpus; "
                    "do not use fixed domain categories. Return concept_ids in best-first order."
                ),
                "concepts": [
                    {
                        "concept_id": item.concept_id,
                        "video_id": item.video_id,
                        "semantic_summary": item.text,
                        "confidence": item.score / 10,
                    }
                    for item in discovered_concepts
                ],
                # Frozen compatibility hint retained in the cached request. It no longer limits
                # production yield; every remaining grounded concept is evaluated below.
                "planning_budget": brief.production.concept_count,
            },
        )
        raw_ids = selection.get("concept_ids")
        if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
            raise EditorialGroundingError("global comparison must return concept_ids")
        concept_index = {item.concept_id: item for item in discovered_concepts}
        selected_ids: list[str] = []
        for concept_id in raw_ids:
            if concept_id not in concept_index:
                raise EditorialGroundingError(
                    f"global comparison referenced unknown concept {concept_id}"
                )
            if concept_id not in selected_ids:
                selected_ids.append(concept_id)

        # Global comparison is a ranking aid, not an output quota. Append every remaining
        # grounded distinct concept so quality is decided by eligibility rather than count.
        for concept in sorted(
            discovered_concepts,
            key=lambda item: (-item.score, item.video_id, item.source_start, item.concept_id),
        ):
            if concept.concept_id not in selected_ids:
                selected_ids.append(concept.concept_id)
        selected = [concept_index[concept_id] for concept_id in selected_ids]

        variants: list[HookVariant] = []
        plans: list[EditPlan] = []
        boundary_audits: list[dict[str, object]] = []
        campaign_policy_audits: list[dict[str, object]] = []
        planning_model_successes = 0
        planning_model_failures = 0
        for concept in selected:
            timeline = timelines[concept.video_id]
            grounded_concept = grounded[concept.concept_id]
            try:
                hook_payload = self._complete(
                    f"hook_variants:{concept.concept_id}",
                    timeline,
                    brief,
                    {
                        "campaign": self._campaign(brief),
                        "instruction": (
                            "Return only materially different truthful hook constructions. "
                            "Return as many as are legitimate, not a fixed count. Spoken hooks "
                            "must reference source word IDs; "
                            "generated text may only appear in overlay_text."
                        ),
                        "concept": {
                            **asdict(grounded_concept),
                            "start_word_id": timeline.word_ref(
                                grounded_concept.supporting_word_ids[0]
                            ),
                            "end_word_id": timeline.word_ref(
                                grounded_concept.supporting_word_ids[-1]
                            ),
                            "supporting_word_ids": None,
                        },
                    },
                )
                planning_model_successes += 1
            except Exception as exc:
                planning_model_failures += 1
                rejections.append(
                    {
                        "concept_id": concept.concept_id,
                        "stage": "hook_generation",
                        "decision": "REJECT",
                        "reasons": ["model_completion_failed"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            grounded_hooks: list[GroundedHookVariant] = []
            for proposal_index, raw in enumerate(self._array(hook_payload, "variants")):
                try:
                    grounded_hooks.append(GroundedHookVariant.from_payload(raw, timeline))
                except ValueError as exc:
                    rejections.append(
                        {
                            "concept_id": concept.concept_id,
                            "stage": "hook_generation",
                            "proposal_index": proposal_index,
                            "variant_id": str(raw.get("variant_id") or ""),
                            "decision": "REJECT",
                            "reasons": ["invalid_grounded_hook_variant"],
                            "error": str(exc),
                        }
                    )
            grounded_hooks = self._dedupe_hooks(
                brief, timeline, grounded_hooks, concept.concept_id, rejections
            )
            if not grounded_hooks:
                rejections.append(
                    {
                        "concept_id": concept.concept_id,
                        "stage": "hook_generation",
                        "decision": "REJECT",
                        "reasons": ["no_grounded_hook_variants"],
                    }
                )
                continue
            variants.extend(
                self._compat_hook(timeline, concept.concept_id, hook) for hook in grounded_hooks
            )
            try:
                plan_payload = self._complete(
                    f"edit_plans:{concept.concept_id}",
                    timeline,
                    brief,
                    {
                        "campaign": self._campaign(brief),
                        "instruction": (
                            "Construct truthful source-grounded EditPlans. Preserve chronology. "
                            "Use only supplied canonical word IDs for spoken material. "
                            f"Every source range must be between {brief.min_clip_seconds:g} and "
                            f"{brief.max_clip_seconds:g} seconds and must contain its spoken hook. "
                            "Choose source_start_word_id/source_end_word_id from "
                            "source_context_words; do not return only the hook unless it already "
                            "satisfies duration bounds."
                        ),
                        "concept": {
                            **asdict(grounded_concept),
                            "start_word_id": timeline.word_ref(
                                grounded_concept.supporting_word_ids[0]
                            ),
                            "end_word_id": timeline.word_ref(
                                grounded_concept.supporting_word_ids[-1]
                            ),
                            "supporting_word_ids": None,
                        },
                        "hooks": [
                            {
                                **asdict(hook),
                                "source_start_word_id": timeline.word_ref(hook.source_word_ids[0]),
                                "source_end_word_id": timeline.word_ref(hook.source_word_ids[-1]),
                                "source_word_ids": None,
                            }
                            for hook in grounded_hooks
                        ],
                        "source_context_words": self._plan_context_words(
                            timeline, grounded_concept, brief
                        ),
                    },
                )
                planning_model_successes += 1
            except Exception as exc:
                planning_model_failures += 1
                rejections.append(
                    {
                        "concept_id": concept.concept_id,
                        "stage": "edit_plan",
                        "decision": "REJECT",
                        "reasons": ["model_completion_failed"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            known_hook_ids = {hook.variant_id for hook in grounded_hooks}
            for proposal_index, raw in enumerate(self._array(plan_payload, "plans")):
                try:
                    grounded_plan = GroundedEditPlan.from_payload(raw, timeline)
                    if grounded_plan.concept_id != concept.concept_id:
                        raise EditorialGroundingError("EditPlan references the wrong concept")
                    if grounded_plan.variant_id not in known_hook_ids:
                        raise EditorialGroundingError("EditPlan references an unknown hook variant")
                    repaired_plan, repair_evidence = self._repair_grounded_plan_duration(
                        brief, timeline, grounded_concept, grounded_plan
                    )
                    if repaired_plan is None:
                        rejections.append(repair_evidence)
                        continue
                    boundary = self._boundary_audit(
                        brief,
                        timeline,
                        grounded_concept,
                        repaired_plan,
                    )
                    policy = evaluate_campaign_policy(
                        brief,
                        boundary.source_start,
                        boundary.source_end,
                        tuple(hazards_by_video.get(concept.video_id, [])),
                        (),
                    )
                    eligibility = evaluate_pre_render_eligibility(
                        brief,
                        boundary,
                        policy,
                        repaired=bool(repair_evidence),
                    )
                    boundary_payload = boundary.to_dict(brief.acceptance_policy)
                    boundary_payload.update(
                        {"plan_id": repaired_plan.plan_id, "attempt": "initial"}
                    )
                    boundary_audits.append(boundary_payload)
                    policy_payload = policy.to_dict()
                    policy_payload.update({"plan_id": repaired_plan.plan_id, "attempt": "initial"})
                    campaign_policy_audits.append(policy_payload)
                    if eligibility.decision == GateDecision.REPAIR:
                        repaired_plan = self._apply_boundary_repair(
                            timeline, repaired_plan, boundary
                        )
                        boundary = self._boundary_audit(
                            brief,
                            timeline,
                            grounded_concept,
                            repaired_plan,
                            stage_suffix=":repair-1",
                        )
                        policy = evaluate_campaign_policy(
                            brief,
                            boundary.source_start,
                            boundary.source_end,
                            tuple(hazards_by_video.get(concept.video_id, [])),
                            (),
                        )
                        eligibility = evaluate_pre_render_eligibility(
                            brief, boundary, policy, repaired=True
                        )
                        boundary_payload = boundary.to_dict(brief.acceptance_policy)
                        boundary_payload.update(
                            {"plan_id": repaired_plan.plan_id, "attempt": "repair-1"}
                        )
                        boundary_audits.append(boundary_payload)
                        policy_payload = policy.to_dict()
                        policy_payload.update(
                            {"plan_id": repaired_plan.plan_id, "attempt": "repair-1"}
                        )
                        campaign_policy_audits.append(policy_payload)
                    if eligibility.decision != GateDecision.PASS:
                        rejections.append(
                            {
                                "concept_id": concept.concept_id,
                                "stage": "pre_render_editorial_integrity",
                                "proposal_index": proposal_index,
                                "plan_id": repaired_plan.plan_id,
                                "decision": eligibility.decision.value,
                                "reasons": list(eligibility.reasons)
                                or ["editorial_integrity_failed"],
                                "boundary_audit": boundary.to_dict(brief.acceptance_policy),
                                "campaign_policy_audit": policy.to_dict(),
                            }
                        )
                        continue
                    plan = repaired_plan.compile(timeline, stable_hash(timeline.to_dict()))
                    plan = replace(
                        plan,
                        boundary_audit=boundary.to_dict(brief.acceptance_policy),
                        campaign_policy_audit=policy.to_dict(),
                        pre_render_eligibility=eligibility.to_dict(),
                    )
                except ValueError as exc:
                    rejections.append(
                        {
                            "concept_id": concept.concept_id,
                            "stage": "edit_plan",
                            "proposal_index": proposal_index,
                            "plan_id": str(raw.get("plan_id") or ""),
                            "decision": "REJECT",
                            "reasons": ["invalid_grounded_edit_plan"],
                            "error": str(exc),
                        }
                    )
                    continue
                if repair_evidence:
                    rejections.append(repair_evidence)
                plans.append(plan)
        if not plans and planning_model_successes == 0 and planning_model_failures > 0:
            failures = [
                item
                for item in rejections
                if item.get("stage") in {"hook_generation", "edit_plan"}
                and "model_completion_failed" in (item.get("reasons") or [])
            ]
            raise EditorialGroundingError(
                "open editorial planner could not evaluate quality because every planning "
                f"model call failed; failures={failures[-8:]}"
            )
        return OpenEditorialBatch(
            discovered_moments=discovered_moments,
            discovered_concepts=discovered_concepts,
            selected_concepts=selected,
            variants=variants,
            plans=plans,
            rejections=rejections,
            model_invocations=list(self.invocations),
            boundary_audits=boundary_audits,
            campaign_policy_audits=campaign_policy_audits,
            source_hazards=source_hazard_evidence,
        )

    def _dedupe_hooks(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        hooks: list[GroundedHookVariant],
        concept_id: str,
        rejections: list[dict[str, object]],
    ) -> list[GroundedHookVariant]:
        if len(hooks) <= 1:
            return hooks
        texts = [
            " | ".join(
                [
                    hook.strategy_label,
                    hook.overlay_text or "",
                    _source_text(timeline, hook.source_word_ids),
                ]
            )
            for hook in hooks
        ]
        vectors = self._embed(f"hook_embeddings:{concept_id}", timeline, brief, texts)
        if len(vectors) != len(hooks):
            raise ValueError("embedding provider returned the wrong number of hook vectors")
        kept: list[GroundedHookVariant] = []
        kept_vectors: list[list[float]] = []
        for hook, vector in zip(hooks, vectors, strict=True):
            similarity = max((_cosine(vector, other) for other in kept_vectors), default=0.0)
            if similarity >= self.hook_duplicate_threshold:
                rejections.append(
                    {
                        "concept_id": concept_id,
                        "stage": "hook_generation",
                        "decision": "REJECT",
                        "reasons": ["learned_embedding_hook_duplicate"],
                        "variant_id": hook.variant_id,
                        "similarity": round(similarity, 6),
                    }
                )
                continue
            kept.append(hook)
            kept_vectors.append(vector)
        return kept

    @staticmethod
    def _compat_moment(timeline: CanonicalTimeline, item: GroundedStoryMoment) -> StoryMoment:
        spans = source_spans_from_word_ids(timeline, item.supporting_word_ids)
        start = min(span.start for span in spans)
        end = max(span.end for span in spans)
        return StoryMoment(
            moment_id=item.moment_id,
            video_id=timeline.video_id,
            start=start,
            end=end,
            text=_source_text(timeline, item.supporting_word_ids),
            moment_type=item.narrative_structure,
            topic=item.semantic_summary,
            setup=item.required_prior_context,
            payoff=item.editorial_reason,
            scores=_compat_scores(item.confidence),
            score=item.confidence * 10,
            transcript_fingerprint=stable_hash(item.supporting_word_ids),
        )

    @staticmethod
    def _compat_concept(
        timeline: CanonicalTimeline, item: GroundedClipConcept, semantic_cluster: str
    ) -> ClipConcept:
        spans = source_spans_from_word_ids(timeline, item.supporting_word_ids)
        start = min(span.start for span in spans)
        end = max(span.end for span in spans)
        return ClipConcept(
            concept_id=item.concept_id,
            video_id=timeline.video_id,
            source_start=start,
            source_end=end,
            text=_source_text(timeline, item.supporting_word_ids),
            topic=item.semantic_summary,
            setup=item.standalone_context,
            payoff=item.narrative_structure,
            moment_type=item.narrative_structure,
            recommended_duration=item.recommended_duration,
            scores=_compat_scores(item.confidence),
            score=item.confidence * 10,
            semantic_cluster=semantic_cluster,
            transcript_fingerprint=stable_hash(item.supporting_word_ids),
        )

    @staticmethod
    def _compat_hook(
        timeline: CanonicalTimeline, concept_id: str, item: GroundedHookVariant
    ) -> HookVariant:
        spans = source_spans_from_word_ids(timeline, item.source_word_ids)
        first = timeline.require_word_ids(item.source_word_ids)[0]
        return HookVariant(
            variant_id=item.variant_id,
            concept_id=concept_id,
            mode="curiosity_text" if item.overlay_text else "direct",
            source_spans=spans,
            overlay_text=item.overlay_text,
            score=item.confidence * 10,
            rationale=item.rationale,
            fingerprint=stable_hash(
                {
                    "concept_id": concept_id,
                    "source_word_ids": item.source_word_ids,
                    "overlay_text": item.overlay_text,
                    "strategy": item.strategy_label,
                }
            ),
            caption_start_source_time=first.source_start,
            caption_start_word=first.text,
        )

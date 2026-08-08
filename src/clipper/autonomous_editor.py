from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .ai_editorial import (
    EditorialGroundingError,
    EpisodeEditorialProfile,
    GroundedClipConcept,
    GroundedEditPlan,
    GroundedHookVariant,
    GroundedStoryMoment,
    source_spans_from_word_ids,
)
from .cache import FileCache, model_stage_cache_key, stable_hash
from .canonical import CanonicalTimeline
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


@dataclass(slots=True)
class OpenEditorialBatch:
    discovered_moments: list[StoryMoment]
    discovered_concepts: list[ClipConcept]
    selected_concepts: list[ClipConcept]
    variants: list[HookVariant]
    plans: list[EditPlan]
    rejections: list[dict[str, object]]
    model_invocations: list[dict[str, object]]


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

    def _profile_evidence(self, timeline: CanonicalTimeline) -> list[dict[str, object]]:
        if len(timeline.words) <= 1800:
            return self._word_payload(timeline, 0, len(timeline.words))
        # A global episode profile needs representative coverage, not a second full transcript.
        # Keep the sample stratified across the entire episode while bounding attention memory.
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
        for chunk_index, words in enumerate(self._chunks(timeline)):
            first_start = words[0].get("source_start") if words else None
            last_end = words[-1].get("source_end") if words else None
            chunk_start = float(first_start) if isinstance(first_start, int | float | str) else 0.0
            chunk_end = float(last_end) if isinstance(last_end, int | float | str) else 0.0
            payload = self._complete(
                f"story_moments:{chunk_index}",
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
            for raw in self._array(payload, "moments"):
                moment = GroundedStoryMoment.from_payload(raw, timeline)
                existing = grounded_moments.get(moment.moment_id)
                if existing is None or moment.confidence > existing.confidence:
                    grounded_moments[moment.moment_id] = moment
        if not grounded_moments:
            raise EditorialGroundingError("open editorial model returned no grounded StoryMoments")

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
        grounded_concepts = {
            concept.concept_id: concept
            for concept in (
                GroundedClipConcept.from_payload(raw, timeline)
                for raw in self._array(concept_payload, "concepts")
            )
        }
        if not grounded_concepts:
            raise EditorialGroundingError("open editorial model returned no grounded ClipConcepts")
        known_moments = set(grounded_moments)
        for concept in grounded_concepts.values():
            unknown = set(concept.story_moment_ids) - known_moments
            if unknown:
                raise EditorialGroundingError(
                    f"concept {concept.concept_id} references unknown StoryMoments: "
                    f"{sorted(unknown)}"
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
        return OpenVideoAnalysis(
            profile=profile,
            moments=moments,
            concepts=concepts,
            grounded_moments=grounded_moments,
            grounded_concepts=grounded_concepts,
            rejections=rejections,
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

    def plan_batch(
        self,
        brief: CampaignBrief,
        timelines: dict[str, CanonicalTimeline],
        analyses: list[OpenVideoAnalysis],
    ) -> OpenEditorialBatch:
        discovered_moments = [moment for analysis in analyses for moment in analysis.moments]
        discovered_concepts = [concept for analysis in analyses for concept in analysis.concepts]
        grounded: dict[str, GroundedClipConcept] = {}
        rejections: list[dict[str, object]] = []
        for analysis in analyses:
            grounded.update(analysis.grounded_concepts)
        if not discovered_concepts:
            raise EditorialGroundingError("open editorial analysis produced no concepts")

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
            if len(selected_ids) >= brief.production.concept_count:
                break
        if not selected_ids:
            raise EditorialGroundingError("global comparison selected no concepts")
        selected = [concept_index[concept_id] for concept_id in selected_ids]

        variants: list[HookVariant] = []
        plans: list[EditPlan] = []
        for concept in selected:
            timeline = timelines[concept.video_id]
            grounded_concept = grounded[concept.concept_id]
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
                        "start_word_id": timeline.word_ref(grounded_concept.supporting_word_ids[0]),
                        "end_word_id": timeline.word_ref(grounded_concept.supporting_word_ids[-1]),
                        "supporting_word_ids": None,
                    },
                },
            )
            grounded_hooks = [
                GroundedHookVariant.from_payload(raw, timeline)
                for raw in self._array(hook_payload, "variants")
            ]
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
            plan_payload = self._complete(
                f"edit_plans:{concept.concept_id}",
                timeline,
                brief,
                {
                    "campaign": self._campaign(brief),
                    "instruction": (
                        "Construct truthful source-grounded EditPlans. Preserve chronology. "
                        "Use only supplied "
                        "canonical word IDs for spoken material."
                    ),
                    "concept": {
                        **asdict(grounded_concept),
                        "start_word_id": timeline.word_ref(grounded_concept.supporting_word_ids[0]),
                        "end_word_id": timeline.word_ref(grounded_concept.supporting_word_ids[-1]),
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
                },
            )
            for raw in self._array(plan_payload, "plans"):
                grounded_plan = GroundedEditPlan.from_payload(raw, timeline)
                if grounded_plan.concept_id != concept.concept_id:
                    raise EditorialGroundingError("EditPlan references the wrong concept")
                if grounded_plan.variant_id not in {hook.variant_id for hook in grounded_hooks}:
                    raise EditorialGroundingError("EditPlan references an unknown hook variant")
                plan = grounded_plan.compile(timeline, stable_hash(timeline.to_dict()))
                if not brief.min_clip_seconds <= plan.duration <= brief.max_clip_seconds:
                    rejections.append(
                        {
                            "concept_id": concept.concept_id,
                            "stage": "edit_plan",
                            "decision": "REJECT",
                            "reasons": ["duration_outside_campaign_bounds"],
                            "plan_id": plan.plan_id,
                        }
                    )
                    continue
                plans.append(plan)
        if not plans:
            raise EditorialGroundingError("open editorial planner produced no valid EditPlans")
        return OpenEditorialBatch(
            discovered_moments=discovered_moments,
            discovered_concepts=discovered_concepts,
            selected_concepts=selected,
            variants=variants,
            plans=plans,
            rejections=rejections,
            model_invocations=list(self.invocations),
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

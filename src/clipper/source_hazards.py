from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import CanonicalTimeline
from .dag import DagStore, StageResult
from .editorial_capacity import stable_range_stage, token_aware_repartition
from .editorial_evidence import EditorialEvidenceProjection, project_multimodal_evidence
from .editorial_integrity import HazardClassification, SourceHazardSegment
from .models import CampaignBrief
from .multimodal_timeline import MultimodalTimeline
from .providers.base import EditorialCapacityError, EditorialProvider
from .providers.editorial_prompt import editorial_contract_fingerprint
from .stage_contracts import StageContract, StageIdentity, content_fingerprint, stage_identity

# Compatibility only: these values reconstruct stage identities written by the pre-capacity
# planner so an expensive successful run can resume without recomputing source hazards.
_LEGACY_CACHE_MAX_WORDS = 900
_LEGACY_CACHE_OVERLAP_WORDS = 160


@dataclass(frozen=True, slots=True)
class SourceHazardClassificationResult:
    hazards: tuple[SourceHazardSegment, ...]
    rejections: tuple[dict[str, object], ...]
    stage_cache_hits: int
    stage_executions: int

    def to_dict(self) -> dict[str, object]:
        return {
            "hazards": [item.to_dict() for item in self.hazards],
            "rejections": list(self.rejections),
            "stage_cache_hits": self.stage_cache_hits,
            "stage_executions": self.stage_executions,
        }


def campaign_context(brief: CampaignBrief) -> dict[str, object]:
    """Return campaign constraints that affect semantic/editorial decisions, never quotas."""

    return {
        "campaign_id": brief.campaign_id,
        "title": brief.title,
        "objective": brief.objective,
        "language": brief.language,
        "min_clip_seconds": brief.min_clip_seconds,
        "max_clip_seconds": brief.max_clip_seconds,
        "posting_requirements": list(brief.posting_requirements),
        "acceptance_policy": brief.acceptance_policy.to_dict(),
    }


class SourceHazardClassifier:
    """Classify source policy exceptions with grounded, fail-closed structured inference."""

    def __init__(
        self,
        editorial: EditorialProvider,
        dag: DagStore,
    ) -> None:
        self.editorial = editorial
        self.dag = dag
        self.cache_hits = 0
        self.executions = 0

    @staticmethod
    def _word_payload(
        timeline: CanonicalTimeline,
        start: int,
        end: int,
    ) -> list[dict[str, object]]:
        return [
            {
                "word_ref": timeline.word_ref(word.word_id),
                "text": word.text,
                "source_start": word.source_start,
                "source_end": word.source_end,
                "speaker_id": word.speaker_id,
                "confidence": word.confidence,
            }
            for word in timeline.words[start:end]
        ]

    @staticmethod
    def _legacy_multimodal_payload(
        multimodal: MultimodalTimeline | None,
        start: float,
        end: float,
    ) -> list[dict[str, object]]:
        if multimodal is None:
            return []
        return [item.to_dict() for item in multimodal.overlapping(start, end)]

    @staticmethod
    def _project_multimodal_payload(
        multimodal: MultimodalTimeline | None,
        start: float,
        end: float,
    ) -> EditorialEvidenceProjection:
        return project_multimodal_evidence(multimodal, start, end)

    def _identity(
        self,
        timeline: CanonicalTimeline,
        brief: CampaignBrief,
        stage: str,
        payload: dict[str, Any],
    ) -> StageIdentity:
        policy = brief.acceptance_policy.to_dict()
        contract = StageContract(
            name=stage,
            contract={
                "editorial_contract_fingerprint": editorial_contract_fingerprint(stage),
                "structured_output": True,
            },
            relevant_policy=policy,
        )
        return stage_identity(
            contract,
            source_hash=timeline.source_hash,
            dependency_output_hashes=(content_fingerprint(payload),),
            model_revision=self.editorial.identity.cache_fingerprint(),
            decoding_parameters={"do_sample": False},
        )

    def _complete(
        self,
        timeline: CanonicalTimeline,
        brief: CampaignBrief,
        stage: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        identity = self._identity(timeline, brief, stage, payload)

        def operation() -> StageResult:
            result = self.editorial.complete_json(task=stage, payload=payload)
            return StageResult(
                output=result.value,
                usage=asdict(result.usage),
                cost_usd=result.usage.estimated_cost_usd,
            )

        raw, cached = self.dag.execute(identity, operation)
        if cached:
            self.cache_hits += 1
        else:
            self.executions += 1
        if not isinstance(raw, dict):
            raise ValueError(f"{stage} returned a non-object payload")
        return {str(key): value for key, value in raw.items()}

    @staticmethod
    def _instruction() -> str:
        return (
            "Review the entire supplied source interval for policy exceptions. Fuse speech and "
            "multimodal evidence. Copy the first and last supplied word_ref into the coverage "
            "fields and set coverage_complete=true only after evaluating the whole interval. "
            "Return only exception spans; do not emit ordinary editorial_content. Use unknown "
            "for any uncertain subrange. An empty segments array is allowed only when complete "
            "coverage found no policy exceptions."
        )

    def _projected_payload_for_range(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        multimodal: MultimodalTimeline | None,
        start: int,
        end: int,
    ) -> tuple[dict[str, Any], EditorialEvidenceProjection]:
        words = timeline.words[start:end]
        chunk_start = words[0].source_start
        chunk_end = words[-1].source_end
        projection = self._project_multimodal_payload(multimodal, chunk_start, chunk_end)
        payload: dict[str, Any] = {
            "campaign": campaign_context(brief),
            "instruction": self._instruction(),
            "words": self._word_payload(timeline, start, end),
            "capacity_repartitionable": True,
            "multimodal_evidence": list(projection.events),
        }
        if projection.provenance:
            payload["multimodal_provenance"] = list(projection.provenance)
        return payload, projection

    def _payload_for_range(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        multimodal: MultimodalTimeline | None,
        start: int,
        end: int,
    ) -> dict[str, Any]:
        words = timeline.words[start:end]
        chunk_start = words[0].source_start
        chunk_end = words[-1].source_end
        return {
            "campaign": campaign_context(brief),
            "instruction": self._instruction(),
            "words": self._word_payload(timeline, start, end),
            "multimodal_evidence": self._legacy_multimodal_payload(
                multimodal,
                chunk_start,
                chunk_end,
            ),
        }

    @staticmethod
    def _legacy_ranges(timeline: CanonicalTimeline) -> tuple[tuple[int, int], ...]:
        if not timeline.words:
            return ()
        step = _LEGACY_CACHE_MAX_WORDS - _LEGACY_CACHE_OVERLAP_WORDS
        ranges: list[tuple[int, int]] = []
        for start in range(0, len(timeline.words), step):
            end = min(len(timeline.words), start + _LEGACY_CACHE_MAX_WORDS)
            ranges.append((start, end))
            if end == len(timeline.words):
                break
        return tuple(ranges)

    def _legacy_cached_work(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        multimodal: MultimodalTimeline | None,
    ) -> list[tuple[int, int, str | None]] | None:
        work: list[tuple[int, int, str | None]] = []
        for chunk_index, (start, end) in enumerate(self._legacy_ranges(timeline)):
            stage = f"source_hazards:{chunk_index}"
            payload = self._payload_for_range(brief, timeline, multimodal, start, end)
            identity = self._identity(timeline, brief, stage, payload)
            if self.dag.cached_output(identity) is None:
                return None
            work.append((start, end, stage))
        return work

    @staticmethod
    def _validated_segments(
        result: dict[str, Any],
        timeline: CanonicalTimeline,
        words: tuple[Any, ...],
        *,
        model_identity: dict[str, object],
    ) -> list[SourceHazardSegment]:
        raw_segments = result.get("segments")
        if not isinstance(raw_segments, list) or not all(
            isinstance(item, dict) for item in raw_segments
        ):
            raise ValueError("source hazard output must contain a segments object array")

        legal_word_ids = {word.word_id for word in words}
        parsed: list[SourceHazardSegment] = []
        for raw in raw_segments:
            hazard = SourceHazardSegment.from_payload(
                raw,
                timeline,
                model_identity=model_identity,
            )
            if not set(hazard.source_word_ids).issubset(legal_word_ids):
                raise ValueError("source hazard escaped the supplied chunk evidence")
            parsed.append(hazard)

        coverage_keys = {
            "coverage_start_word_id",
            "coverage_end_word_id",
            "coverage_complete",
        }
        coverage_present = any(key in result for key in coverage_keys)
        if coverage_present:
            expected_start = timeline.word_ref(words[0].word_id)
            expected_end = timeline.word_ref(words[-1].word_id)
            if result.get("coverage_complete") is not True:
                raise ValueError("source hazard coverage attestation is incomplete")
            if str(result.get("coverage_start_word_id") or "") != expected_start:
                raise ValueError("source hazard coverage start does not match supplied evidence")
            if str(result.get("coverage_end_word_id") or "") != expected_end:
                raise ValueError("source hazard coverage end does not match supplied evidence")
            if any(
                item.classification == HazardClassification.EDITORIAL_CONTENT for item in parsed
            ):
                raise ValueError("sparse source hazard output must omit ordinary editorial_content")
            return parsed

        # Backward-compatible acceptance for an old exhaustive result is safe only when its
        # returned segments explicitly cover every supplied canonical word. Contract fingerprints
        # prevent these legacy objects from being produced by the new production schema, but this
        # path preserves deterministic replay of already materialized exhaustive evidence.
        covered_word_ids = {word_id for item in parsed for word_id in item.source_word_ids}
        if not parsed or covered_word_ids != legal_word_ids:
            raise ValueError("source hazard output is missing complete coverage attestation")
        return parsed

    def classify(
        self,
        brief: CampaignBrief,
        timeline: CanonicalTimeline,
        *,
        multimodal: MultimodalTimeline | None,
    ) -> SourceHazardClassificationResult:
        if not brief.acceptance_policy.enabled:
            return SourceHazardClassificationResult((), (), 0, 0)
        if multimodal is not None and (
            multimodal.video_id != timeline.video_id
            or multimodal.source_hash != timeline.source_hash
        ):
            raise ValueError("multimodal and canonical timelines reference different sources")

        hazards: list[SourceHazardSegment] = []
        rejections: list[dict[str, object]] = []
        model_identity: dict[str, object] = dict(self.editorial.identity.to_dict())
        legacy = self._legacy_cached_work(brief, timeline, multimodal)
        work = legacy if legacy is not None else [(0, len(timeline.words), None)]
        while work:
            start, end, cached_stage = work.pop(0)
            words = timeline.words[start:end]
            chunk_start = words[0].source_start
            chunk_end = words[-1].source_end
            stage = cached_stage or stable_range_stage("source_hazards", timeline, start, end)
            if cached_stage is not None:
                payload = self._payload_for_range(brief, timeline, multimodal, start, end)
                projection = None
            else:
                payload, projection = self._projected_payload_for_range(
                    brief,
                    timeline,
                    multimodal,
                    start,
                    end,
                )
                print(
                    json.dumps(
                        projection.telemetry(
                            stage=stage,
                            start=chunk_start,
                            end=chunk_end,
                        ),
                        sort_keys=True,
                    )
                )
            try:
                result = self._complete(timeline, brief, stage, payload)
                hazards.extend(
                    self._validated_segments(
                        result,
                        timeline,
                        words,
                        model_identity=model_identity,
                    )
                )
            except EditorialCapacityError as exc:
                repartition = token_aware_repartition(timeline, start, end, exc.details)
                if repartition is not None:
                    print(json.dumps(repartition.telemetry(stage=stage), sort_keys=True))
                    work[0:0] = [
                        (child_start, child_end, None)
                        for child_start, child_end in repartition.ranges
                    ]
                    continue
                hazards.append(
                    SourceHazardSegment(
                        start=chunk_start,
                        end=chunk_end,
                        classification=HazardClassification.UNKNOWN,
                        confidence=0.0,
                        evidence=("source_hazard_capacity_exhausted",),
                        model_identity=model_identity,
                        source_word_ids=tuple(word.word_id for word in words),
                    )
                )
                rejections.append(
                    {
                        "video_id": timeline.video_id,
                        "stage": "source_hazard_classification",
                        "model_stage": stage,
                        "decision": "ESCALATE",
                        "reasons": ["policy_uncertain", "editorial_capacity_exhausted"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            except Exception as exc:
                hazards.append(
                    SourceHazardSegment(
                        start=chunk_start,
                        end=chunk_end,
                        classification=HazardClassification.UNKNOWN,
                        confidence=0.0,
                        evidence=("source_hazard_classification_failed",),
                        model_identity=model_identity,
                        source_word_ids=tuple(word.word_id for word in words),
                    )
                )
                rejections.append(
                    {
                        "video_id": timeline.video_id,
                        "stage": "source_hazard_classification",
                        "model_stage": stage,
                        "decision": "ESCALATE",
                        "reasons": ["policy_uncertain"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        hazards.sort(key=lambda item: (item.start, item.end, item.classification.value))
        return SourceHazardClassificationResult(
            hazards=tuple(hazards),
            rejections=tuple(rejections),
            stage_cache_hits=self.cache_hits,
            stage_executions=self.executions,
        )

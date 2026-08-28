from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import CanonicalTimeline
from .dag import DagStore, StageResult
from .editorial_capacity import (
    shrink_context_around_interval,
    stable_range_stage,
    token_aware_repartition,
)
from .editorial_evidence import EditorialEvidenceProjection, project_multimodal_evidence
from .modality_profile import SourceModalityProfile, assert_required_modalities_available
from .models import SourceSpan
from .multimodal_timeline import MultimodalTimeline
from .providers.base import EditorialCapacityError, EditorialProvider
from .providers.editorial_prompt import editorial_contract_fingerprint
from .quality_moments import QualityMoment, WindowQualityAssessment, choose_quality_moments
from .stage_contracts import StageContract, content_fingerprint, stage_identity
from .story_graph import NarrativeEnvelope, SemanticCore
from .window_solver import FeasibleDeliveryWindow, enumerate_feasible_windows


class AutonomousPlanningError(RuntimeError):
    """Raised when model output or evidence cannot satisfy the autonomous planning contract."""


@dataclass(frozen=True, slots=True)
class QualityPlanningResult:
    cores: tuple[SemanticCore, ...]
    envelopes: tuple[NarrativeEnvelope, ...]
    feasible_windows: tuple[FeasibleDeliveryWindow, ...]
    assessments: tuple[WindowQualityAssessment, ...]
    quality_moments: tuple[QualityMoment, ...]
    rejections: tuple[dict[str, object], ...]
    stage_cache_hits: int
    stage_executions: int

    def to_dict(self) -> dict[str, object]:
        return {
            "cores": [item.to_dict() for item in self.cores],
            "envelopes": [item.to_dict() for item in self.envelopes],
            "feasible_windows": [item.to_dict() for item in self.feasible_windows],
            "assessments": [item.to_dict() for item in self.assessments],
            "quality_moments": [item.to_dict() for item in self.quality_moments],
            "rejections": list(self.rejections),
            "stage_cache_hits": self.stage_cache_hits,
            "stage_executions": self.stage_executions,
        }


def _word_range(
    timeline: CanonicalTimeline,
    start_ref: object,
    end_ref: object,
) -> tuple[str, ...]:
    if not isinstance(start_ref, str) or not isinstance(end_ref, str):
        raise AutonomousPlanningError("model word boundaries must be string word references")
    try:
        start_id = timeline.resolve_word_ref(start_ref)
        end_id = timeline.resolve_word_ref(end_ref)
    except ValueError as exc:
        raise AutonomousPlanningError(str(exc)) from exc
    positions = {word.word_id: index for index, word in enumerate(timeline.words)}
    start_index = positions[start_id]
    end_index = positions[end_id]
    if end_index < start_index:
        raise AutonomousPlanningError("model word boundaries reverse source chronology")
    return tuple(word.word_id for word in timeline.words[start_index : end_index + 1])


def _stable_id(prefix: str, *parts: str) -> str:
    material = ":".join(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:16]}"


def semantic_cores_from_payload(
    timeline: CanonicalTimeline,
    payload: dict[str, Any],
) -> tuple[SemanticCore, ...]:
    raw_cores = payload.get("cores")
    if not isinstance(raw_cores, list):
        raise AutonomousPlanningError("semantic_cores output must contain a cores array")
    parsed: list[SemanticCore] = []
    for raw in raw_cores:
        if not isinstance(raw, dict):
            raise AutonomousPlanningError("semantic core entry must be an object")
        word_ids = _word_range(timeline, raw.get("start_word_id"), raw.get("end_word_id"))
        confidence = raw.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise AutonomousPlanningError("semantic core confidence must be numeric")
        core_id = _stable_id(
            "core",
            timeline.source_hash,
            word_ids[0],
            word_ids[-1],
        )
        parsed.append(
            SemanticCore.from_word_ids(
                timeline,
                core_id=core_id,
                source_word_ids=word_ids,
                semantic_summary=str(raw.get("semantic_summary") or "").strip(),
                editorial_reason=str(raw.get("editorial_reason") or "").strip(),
                confidence=float(confidence),
            )
        )
    return _dedupe_cores(tuple(parsed))


def _word_jaccard(left: SemanticCore, right: SemanticCore) -> float:
    left_words = set(left.source_word_ids)
    right_words = set(right.source_word_ids)
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 0.0


def _dedupe_cores(cores: tuple[SemanticCore, ...]) -> tuple[SemanticCore, ...]:
    selected: list[SemanticCore] = []
    for candidate in sorted(
        cores,
        key=lambda item: (-item.confidence, item.source_start, item.source_end, item.core_id),
    ):
        if any(_word_jaccard(candidate, existing) >= 0.9 for existing in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: (item.source_start, item.source_end, item.core_id))
    return tuple(selected)


def narrative_envelope_from_payload(
    timeline: CanonicalTimeline,
    core: SemanticCore,
    payload: dict[str, Any],
) -> NarrativeEnvelope:
    if str(payload.get("core_id") or "") != core.core_id:
        raise AutonomousPlanningError("narrative envelope references the wrong semantic core")
    word_ids = _word_range(timeline, payload.get("start_word_id"), payload.get("end_word_id"))
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise AutonomousPlanningError("narrative envelope confidence must be numeric")
    references = payload.get("reference_resolution")
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise AutonomousPlanningError("narrative reference resolution must be a string array")
    envelope_id = _stable_id(
        "envelope",
        timeline.source_hash,
        core.core_id,
        word_ids[0],
        word_ids[-1],
    )
    try:
        return NarrativeEnvelope.from_word_ids(
            timeline,
            core,
            envelope_id=envelope_id,
            source_word_ids=word_ids,
            required_prior_context=str(payload.get("required_prior_context") or "").strip(),
            required_followup_context=str(payload.get("required_followup_context") or "").strip(),
            setup_resolved=payload.get("setup_resolved") is True,
            payoff_resolved=payload.get("payoff_resolved") is True,
            reference_resolution=tuple(item.strip() for item in references if item.strip()),
            confidence=float(confidence),
        )
    except ValueError as exc:
        raise AutonomousPlanningError(str(exc)) from exc


def quality_assessment_from_payload(
    core: SemanticCore,
    windows: tuple[FeasibleDeliveryWindow, ...],
    payload: dict[str, Any],
) -> WindowQualityAssessment | None:
    if str(payload.get("core_id") or "") != core.core_id:
        raise AutonomousPlanningError("quality decision references the wrong semantic core")
    decision = str(payload.get("decision") or "")
    if decision not in {"PASS", "REJECT", "ESCALATE"}:
        raise AutonomousPlanningError("quality decision is unsupported")
    score = payload.get("quality_score")
    confidence = payload.get("confidence")
    opening_strategy = str(payload.get("opening_strategy") or "").strip()
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
    ):
        raise AutonomousPlanningError("quality score and confidence must be numeric")
    if not opening_strategy:
        raise AutonomousPlanningError("quality decision requires a source-derived opening strategy")
    selected = payload.get("selected_window_id")
    legal = {window.window_id: window for window in windows}
    if decision != "PASS":
        if selected is not None and selected not in legal:
            raise AutonomousPlanningError("non-PASS quality decision references an unknown window")
        return None
    if not isinstance(selected, str) or selected not in legal:
        raise AutonomousPlanningError(
            "PASS quality decision must select a supplied legal window ID"
        )
    return WindowQualityAssessment(
        core_id=core.core_id,
        window_id=selected,
        decision="PASS",
        quality_score=float(score),
        opening_strategy=opening_strategy,
        rationale=str(payload.get("rationale") or "").strip(),
        confidence=float(confidence),
    )


class AutonomousQualityPlanner:
    """Source-grounded planner where models judge semantics and deterministic code owns legality."""

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
    def _multimodal_projection(
        multimodal: MultimodalTimeline | None,
        start: float,
        end: float,
    ) -> EditorialEvidenceProjection:
        return project_multimodal_evidence(multimodal, start, end)

    @staticmethod
    def _attach_projection(
        payload: dict[str, Any],
        projection: EditorialEvidenceProjection,
    ) -> None:
        payload["multimodal_evidence"] = list(projection.events)
        if projection.provenance:
            payload["multimodal_provenance"] = list(projection.provenance)

    def _complete(
        self,
        timeline: CanonicalTimeline,
        stage: str,
        payload: dict[str, Any],
        *,
        relevant_policy: dict[str, object],
        dependency_hashes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        contract = StageContract(
            name=stage,
            contract={
                "editorial_contract_fingerprint": editorial_contract_fingerprint(stage),
                "structured_output": True,
            },
            relevant_policy=relevant_policy,
        )
        identity = stage_identity(
            contract,
            source_hash=timeline.source_hash,
            dependency_output_hashes=(content_fingerprint(payload), *dependency_hashes),
            model_revision=self.editorial.identity.revision,
            decoding_parameters={"do_sample": False},
        )

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
            raise AutonomousPlanningError(f"{stage} returned a non-object payload")
        return {str(key): value for key, value in raw.items()}

    def _semantic_payload(
        self,
        timeline: CanonicalTimeline,
        multimodal: MultimodalTimeline | None,
        campaign_context: dict[str, object],
        start: int,
        end: int,
    ) -> tuple[dict[str, Any], EditorialEvidenceProjection]:
        chunk_start = timeline.words[start].source_start
        chunk_end = timeline.words[end - 1].source_end
        projection = self._multimodal_projection(multimodal, chunk_start, chunk_end)
        payload: dict[str, Any] = {
            "campaign": campaign_context,
            "words": self._word_payload(timeline, start, end),
            "capacity_repartitionable": True,
            "instruction": (
                "Identify every independently worthwhile semantic nucleus in this source evidence. "
                "Do not pad to delivery duration and do not invent boundaries."
            ),
        }
        self._attach_projection(payload, projection)
        return payload, projection

    def _semantic_cores_adaptive(
        self,
        timeline: CanonicalTimeline,
        *,
        multimodal: MultimodalTimeline | None,
        campaign_context: dict[str, object],
        relevant_policy: dict[str, object],
    ) -> tuple[SemanticCore, ...]:
        if not timeline.words:
            return ()
        work: list[tuple[int, int]] = [(0, len(timeline.words))]
        raw_cores: list[SemanticCore] = []
        while work:
            start, end = work.pop(0)
            stage = stable_range_stage("semantic_cores", timeline, start, end)
            payload, projection = self._semantic_payload(
                timeline,
                multimodal,
                campaign_context,
                start,
                end,
            )
            print(
                json.dumps(
                    projection.telemetry(
                        stage=stage,
                        start=timeline.words[start].source_start,
                        end=timeline.words[end - 1].source_end,
                    ),
                    sort_keys=True,
                )
            )
            try:
                result = self._complete(
                    timeline,
                    stage,
                    payload,
                    relevant_policy=relevant_policy,
                )
            except EditorialCapacityError as exc:
                repartition = token_aware_repartition(timeline, start, end, exc.details)
                if repartition is None:
                    raise AutonomousPlanningError(
                        "smallest source-grounded semantic interval exceeds editorial capacity: "
                        f"{stage}: {exc}"
                    ) from exc
                print(json.dumps(repartition.telemetry(stage=stage), sort_keys=True))
                work[0:0] = list(repartition.ranges)
                continue
            parsed = semantic_cores_from_payload(timeline, result)
            legal = {word.word_id for word in timeline.words[start:end]}
            escaped = [
                core.core_id for core in parsed if not set(core.source_word_ids).issubset(legal)
            ]
            if escaped:
                raise AutonomousPlanningError(
                    f"{stage} returned semantic cores outside supplied evidence: {escaped[:3]}"
                )
            raw_cores.extend(parsed)
        return _dedupe_cores(tuple(raw_cores))

    def _core_positions(
        self,
        timeline: CanonicalTimeline,
        core: SemanticCore,
    ) -> tuple[int, int]:
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        return (
            positions[core.source_word_ids[0]],
            positions[core.source_word_ids[-1]] + 1,
        )

    def _narrative_envelope_adaptive(
        self,
        timeline: CanonicalTimeline,
        core: SemanticCore,
        *,
        multimodal: MultimodalTimeline | None,
        campaign_context: dict[str, object],
        relevant_policy: dict[str, object],
    ) -> NarrativeEnvelope:
        required_start, required_end = self._core_positions(timeline, core)
        context_start, context_end = 0, len(timeline.words)
        dependency = (content_fingerprint(core.to_dict()),)
        while True:
            source_start = timeline.words[context_start].source_start
            source_end = timeline.words[context_end - 1].source_end
            stage = f"narrative_envelope:{core.core_id}"
            projection = self._multimodal_projection(multimodal, source_start, source_end)
            payload = {
                "campaign": campaign_context,
                "core": core.to_dict(),
                "source_context_words": self._word_payload(
                    timeline,
                    context_start,
                    context_end,
                ),
                "capacity_repartitionable": True,
                "instruction": (
                    "Return the minimum complete narrative envelope containing this semantic core. "
                    "Resolve setup, references, causality and payoff without duration padding."
                ),
            }
            self._attach_projection(payload, projection)
            print(
                json.dumps(
                    projection.telemetry(stage=stage, start=source_start, end=source_end),
                    sort_keys=True,
                )
            )
            try:
                raw = self._complete(
                    timeline,
                    stage,
                    payload,
                    relevant_policy=relevant_policy,
                    dependency_hashes=dependency,
                )
                envelope = narrative_envelope_from_payload(timeline, core, raw)
                legal = {word.word_id for word in timeline.words[context_start:context_end]}
                if not set(envelope.source_word_ids).issubset(legal):
                    raise AutonomousPlanningError(
                        "narrative envelope escaped the supplied adaptive context"
                    )
                return envelope
            except EditorialCapacityError as exc:
                next_range = shrink_context_around_interval(
                    timeline,
                    context_start,
                    context_end,
                    required_start,
                    required_end,
                )
                if next_range is None or next_range == (context_start, context_end):
                    raise AutonomousPlanningError(
                        "minimum semantic-core context exceeds editorial capacity: "
                        f"{core.core_id}: {exc}"
                    ) from exc
                context_start, context_end = next_range

    def _quality_context_range(
        self,
        timeline: CanonicalTimeline,
        envelope: NarrativeEnvelope,
    ) -> tuple[int, int]:
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        return (
            positions[envelope.source_word_ids[0]],
            positions[envelope.source_word_ids[-1]] + 1,
        )

    def plan(
        self,
        timeline: CanonicalTimeline,
        *,
        multimodal: MultimodalTimeline | None,
        modality_profile: SourceModalityProfile | None,
        campaign_context: dict[str, object],
        relevant_policy: dict[str, object],
        min_seconds: float,
        max_seconds: float,
        forbidden_spans: tuple[SourceSpan, ...] = (),
    ) -> QualityPlanningResult:
        if min_seconds <= 0 or max_seconds < min_seconds:
            raise ValueError("campaign duration bounds are invalid")
        if multimodal is not None and (
            multimodal.video_id != timeline.video_id
            or multimodal.source_hash != timeline.source_hash
        ):
            raise AutonomousPlanningError(
                "multimodal and canonical timelines reference different sources"
            )
        if modality_profile is not None:
            assert_required_modalities_available(modality_profile)

        cores = self._semantic_cores_adaptive(
            timeline,
            multimodal=multimodal,
            campaign_context=campaign_context,
            relevant_policy=relevant_policy,
        )

        envelopes: list[NarrativeEnvelope] = []
        all_windows: list[FeasibleDeliveryWindow] = []
        assessments: list[WindowQualityAssessment] = []
        rejections: list[dict[str, object]] = []

        for core in cores:
            envelope = self._narrative_envelope_adaptive(
                timeline,
                core,
                multimodal=multimodal,
                campaign_context=campaign_context,
                relevant_policy=relevant_policy,
            )
            envelopes.append(envelope)
            if not envelope.complete:
                rejections.append(
                    {
                        "core_id": core.core_id,
                        "stage": "narrative_envelope",
                        "decision": "REJECT",
                        "reasons": ["incomplete_narrative_envelope"],
                    }
                )
                continue

            context_start, context_end = self._quality_context_range(timeline, envelope)
            context_words = self._word_payload(timeline, context_start, context_end)
            source_start = timeline.words[context_start].source_start
            source_end = timeline.words[context_end - 1].source_end

            windows = enumerate_feasible_windows(
                timeline,
                core,
                envelope,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                forbidden_spans=forbidden_spans,
            )
            all_windows.extend(windows)
            if not windows:
                rejections.append(
                    {
                        "core_id": core.core_id,
                        "stage": "feasible_windows",
                        "decision": "REJECT",
                        "reasons": ["no_campaign_legal_complete_window"],
                        "envelope_duration": envelope.duration,
                        "minimum_seconds": min_seconds,
                        "maximum_seconds": max_seconds,
                    }
                )
                continue

            quality_stage = f"quality_windows:{core.core_id}"
            quality_projection = self._multimodal_projection(
                multimodal,
                source_start,
                source_end,
            )
            quality_request: dict[str, Any] = {
                "campaign": campaign_context,
                "core": core.to_dict(),
                "envelope": envelope.to_dict(),
                "feasible_windows": [
                    {
                        "window_id": window.window_id,
                        "source_start": window.source_start,
                        "source_end": window.source_end,
                        "duration": window.duration,
                        "start_word_ref": timeline.word_ref(window.source_word_ids[0]),
                        "end_word_ref": timeline.word_ref(window.source_word_ids[-1]),
                    }
                    for window in windows
                ],
                "source_context_words": context_words,
                "instruction": (
                    "Judge whether this complete campaign-legal moment is genuinely worth "
                    "publishing. Select only a supplied feasible window ID; never invent "
                    "timestamps. Describe the selected opening from its actual source evidence "
                    "without assigning it to a predefined hook category."
                ),
            }
            self._attach_projection(quality_request, quality_projection)
            print(
                json.dumps(
                    quality_projection.telemetry(
                        stage=quality_stage,
                        start=source_start,
                        end=source_end,
                    ),
                    sort_keys=True,
                )
            )
            quality_payload = self._complete(
                timeline,
                quality_stage,
                quality_request,
                relevant_policy=relevant_policy,
                dependency_hashes=(
                    content_fingerprint(core.to_dict()),
                    content_fingerprint(envelope.to_dict()),
                    content_fingerprint([window.to_dict() for window in windows]),
                ),
            )
            assessment = quality_assessment_from_payload(core, windows, quality_payload)
            if assessment is None:
                rejections.append(
                    {
                        "core_id": core.core_id,
                        "stage": "quality_windows",
                        "decision": str(quality_payload.get("decision") or "ESCALATE"),
                        "reasons": [str(quality_payload.get("rationale") or "not_quality_worthy")],
                    }
                )
                continue
            assessments.append(assessment)

        moments = choose_quality_moments(
            cores,
            tuple(envelopes),
            tuple(all_windows),
            tuple(assessments),
        )
        return QualityPlanningResult(
            cores=cores,
            envelopes=tuple(envelopes),
            feasible_windows=tuple(all_windows),
            assessments=tuple(assessments),
            quality_moments=moments,
            rejections=tuple(rejections),
            stage_cache_hits=self.cache_hits,
            stage_executions=self.executions,
        )

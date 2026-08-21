from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .canonical import CanonicalTimeline
from .dag import DagStore, StageResult
from .modality_profile import SourceModalityProfile, assert_required_modalities_available
from .models import SourceSpan
from .multimodal_timeline import MultimodalTimeline
from .providers.base import EditorialProvider
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
        *,
        max_words_per_chunk: int = 900,
        chunk_overlap_words: int = 160,
        envelope_context_words: int = 700,
    ) -> None:
        if max_words_per_chunk < 200:
            raise ValueError("max_words_per_chunk must be at least 200")
        if not 0 <= chunk_overlap_words < max_words_per_chunk:
            raise ValueError("chunk overlap must be smaller than chunk size")
        if envelope_context_words < max_words_per_chunk // 2:
            raise ValueError("envelope context is too small for autonomous planning")
        self.editorial = editorial
        self.dag = dag
        self.max_words_per_chunk = max_words_per_chunk
        self.chunk_overlap_words = chunk_overlap_words
        self.envelope_context_words = envelope_context_words
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
    def _multimodal_payload(
        multimodal: MultimodalTimeline | None,
        start: float,
        end: float,
    ) -> list[dict[str, object]]:
        if multimodal is None:
            return []
        return [
            {
                "start": event.start,
                "end": event.end,
                "speaker_ids": list(event.speaker_ids),
                "scene_ids": list(event.scene_ids),
                "visible_people": list(event.visible_people),
                "actions": list(event.actions),
                "objects": list(event.objects),
                "ocr_text": list(event.ocr_text),
                "branding": list(event.branding),
                "hazards": list(event.hazards),
                "audio_events": list(event.audio_events),
                "visual_summaries": list(event.visual_summaries),
                "visual_salience": event.visual_salience,
                "motion_salience": event.motion_salience,
                "confidence": event.confidence,
            }
            for event in multimodal.overlapping(start, end)
        ]

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

    def _chunks(self, timeline: CanonicalTimeline) -> tuple[tuple[int, int], ...]:
        if not timeline.words:
            return ()
        step = self.max_words_per_chunk - self.chunk_overlap_words
        ranges: list[tuple[int, int]] = []
        for start in range(0, len(timeline.words), step):
            end = min(len(timeline.words), start + self.max_words_per_chunk)
            ranges.append((start, end))
            if end == len(timeline.words):
                break
        return tuple(ranges)

    def _context_range(
        self,
        timeline: CanonicalTimeline,
        core: SemanticCore,
    ) -> tuple[int, int]:
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        core_start = positions[core.source_word_ids[0]]
        core_end = positions[core.source_word_ids[-1]]
        core_width = core_end - core_start + 1
        if core_width >= self.envelope_context_words:
            return core_start, core_end + 1
        remaining = self.envelope_context_words - core_width
        before = remaining // 2
        start = max(0, core_start - before)
        end = min(len(timeline.words), start + self.envelope_context_words)
        if end <= core_end:
            end = core_end + 1
            start = max(0, end - self.envelope_context_words)
        return start, end

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

        raw_cores: list[SemanticCore] = []
        for chunk_index, (start, end) in enumerate(self._chunks(timeline)):
            words = self._word_payload(timeline, start, end)
            chunk_start = timeline.words[start].source_start
            chunk_end = timeline.words[end - 1].source_end
            stage = f"semantic_cores:{chunk_index}"
            payload = self._complete(
                timeline,
                stage,
                {
                    "campaign": campaign_context,
                    "words": words,
                    "multimodal_evidence": self._multimodal_payload(
                        multimodal,
                        chunk_start,
                        chunk_end,
                    ),
                    "instruction": (
                        "Identify every independently worthwhile semantic nucleus in this source "
                        "evidence. Do not pad to delivery duration and do not invent boundaries."
                    ),
                },
                relevant_policy=relevant_policy,
            )
            raw_cores.extend(semantic_cores_from_payload(timeline, payload))
        cores = _dedupe_cores(tuple(raw_cores))

        envelopes: list[NarrativeEnvelope] = []
        all_windows: list[FeasibleDeliveryWindow] = []
        assessments: list[WindowQualityAssessment] = []
        rejections: list[dict[str, object]] = []

        for core in cores:
            context_start, context_end = self._context_range(timeline, core)
            context_words = self._word_payload(timeline, context_start, context_end)
            source_start = timeline.words[context_start].source_start
            source_end = timeline.words[context_end - 1].source_end
            envelope_stage = f"narrative_envelope:{core.core_id}"
            envelope_payload = self._complete(
                timeline,
                envelope_stage,
                {
                    "campaign": campaign_context,
                    "core": core.to_dict(),
                    "source_context_words": context_words,
                    "multimodal_evidence": self._multimodal_payload(
                        multimodal,
                        source_start,
                        source_end,
                    ),
                    "instruction": (
                        "Return the minimum complete narrative envelope containing this semantic "
                        "core. Resolve setup, references, causality and payoff without duration "
                        "padding."
                    ),
                },
                relevant_policy=relevant_policy,
                dependency_hashes=(content_fingerprint(core.to_dict()),),
            )
            envelope = narrative_envelope_from_payload(timeline, core, envelope_payload)
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
            quality_payload = self._complete(
                timeline,
                quality_stage,
                {
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
                    "multimodal_evidence": self._multimodal_payload(
                        multimodal,
                        source_start,
                        source_end,
                    ),
                    "instruction": (
                        "Judge whether this complete campaign-legal moment is genuinely worth "
                        "publishing. Select only a supplied feasible window ID; never invent "
                        "timestamps. Describe the selected opening from its actual source evidence "
                        "without assigning it to a predefined hook category."
                    ),
                },
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

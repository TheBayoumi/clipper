from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .story_graph import NarrativeEnvelope, SemanticCore
from .window_solver import FeasibleDeliveryWindow

QualityDecision = Literal["PASS", "REJECT", "ESCALATE"]


@dataclass(frozen=True, slots=True)
class WindowQualityAssessment:
    core_id: str
    window_id: str
    decision: QualityDecision
    quality_score: float
    opening_strategy: str
    rationale: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.core_id.strip() or not self.window_id.strip():
            raise ValueError("quality assessment requires core_id and window_id")
        if self.decision not in {"PASS", "REJECT", "ESCALATE"}:
            raise ValueError("unsupported quality assessment decision")
        if not 0 <= self.quality_score <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("quality score and confidence must be between 0 and 1")
        if not self.opening_strategy.strip():
            raise ValueError("quality assessment requires a source-derived opening strategy")
        if not self.rationale.strip():
            raise ValueError("quality assessment requires rationale")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualityMoment:
    quality_moment_id: str
    core: SemanticCore
    envelope: NarrativeEnvelope
    delivery_window: FeasibleDeliveryWindow
    assessment: WindowQualityAssessment

    def __post_init__(self) -> None:
        if not self.quality_moment_id.strip():
            raise ValueError("quality moment requires a stable identifier")
        self.envelope.require_contains(self.core)
        if self.delivery_window.core_id != self.core.core_id:
            raise ValueError("quality moment delivery window references the wrong semantic core")
        if self.delivery_window.envelope_id != self.envelope.envelope_id:
            raise ValueError(
                "quality moment delivery window references the wrong narrative envelope"
            )
        if (
            self.delivery_window.video_id != self.envelope.video_id
            or self.delivery_window.source_hash != self.envelope.source_hash
        ):
            raise ValueError("quality moment delivery window references the wrong source")
        if self.delivery_window.source_start > self.envelope.source_start + 1e-6:
            raise ValueError("quality moment delivery window amputates narrative setup")
        if self.delivery_window.source_end < self.envelope.source_end - 1e-6:
            raise ValueError("quality moment delivery window amputates narrative payoff")
        if not set(self.envelope.source_word_ids).issubset(self.delivery_window.source_word_ids):
            raise ValueError("quality moment delivery window omits narrative-envelope evidence")
        if self.assessment.core_id != self.core.core_id:
            raise ValueError("quality assessment references the wrong semantic core")
        if self.assessment.window_id != self.delivery_window.window_id:
            raise ValueError("quality assessment references the wrong delivery window")
        if self.assessment.decision != "PASS":
            raise ValueError("only PASS assessments can become quality moments")

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_moment_id": self.quality_moment_id,
            "core": self.core.to_dict(),
            "envelope": self.envelope.to_dict(),
            "delivery_window": self.delivery_window.to_dict(),
            "assessment": self.assessment.to_dict(),
        }


def choose_quality_moments(
    cores: tuple[SemanticCore, ...],
    envelopes: tuple[NarrativeEnvelope, ...],
    windows: tuple[FeasibleDeliveryWindow, ...],
    assessments: tuple[WindowQualityAssessment, ...],
) -> tuple[QualityMoment, ...]:
    """Choose the best passing legal window for every quality-worthy semantic core."""

    core_index = {item.core_id: item for item in cores}
    envelope_index = {item.envelope_id: item for item in envelopes}
    window_index = {item.window_id: item for item in windows}
    if len(core_index) != len(cores):
        raise ValueError("duplicate semantic core IDs")
    if len(envelope_index) != len(envelopes):
        raise ValueError("duplicate narrative envelope IDs")
    if len(window_index) != len(windows):
        raise ValueError("duplicate feasible window IDs")

    passing: dict[str, list[WindowQualityAssessment]] = {}
    for assessment in assessments:
        window = window_index.get(assessment.window_id)
        if window is None:
            raise ValueError(f"quality assessment references unknown window {assessment.window_id}")
        if assessment.core_id != window.core_id:
            raise ValueError("quality assessment core/window identity mismatch")
        if assessment.decision == "PASS":
            passing.setdefault(assessment.core_id, []).append(assessment)

    moments: list[QualityMoment] = []
    for core in cores:
        candidates = passing.get(core.core_id, [])
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda item: (item.quality_score, item.confidence, item.window_id),
        )
        window = window_index[best.window_id]
        envelope = envelope_index.get(window.envelope_id)
        if envelope is None:
            raise ValueError("feasible window references unknown narrative envelope")
        moments.append(
            QualityMoment(
                quality_moment_id=f"quality:{core.core_id}",
                core=core,
                envelope=envelope,
                delivery_window=window,
                assessment=best,
            )
        )
    return tuple(moments)

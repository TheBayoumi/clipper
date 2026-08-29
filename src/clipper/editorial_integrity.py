from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from .canonical import CanonicalTimeline
from .models import AcceptancePolicy, CampaignBrief
from .stage_contracts import structural_contract_fingerprint


class BoundaryStatus(StrEnum):
    COMPLETE = "COMPLETE"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    INCOMPLETE = "INCOMPLETE"
    UNCERTAIN = "UNCERTAIN"


class GateDecision(StrEnum):
    PASS = "PASS"  # noqa: S105 - release verdict, not a credential
    REPAIR = "REPAIR"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


class BoundaryFailureReason(StrEnum):
    START_REQUIRES_PRIOR_CONTEXT = "start_requires_prior_context"
    START_FRAGMENT = "start_fragment"
    UNRESOLVED_REFERENCE = "unresolved_reference"
    END_INCOMPLETE = "end_incomplete"
    OPEN_QUESTION = "open_question"
    UNRESOLVED_SETUP = "unresolved_setup"
    UNRESOLVED_PAYOFF = "unresolved_payoff"
    PARTIAL_NUMBER_OR_UNIT = "partial_number_or_unit"
    FOLLOWUP_CONTEXT_REQUIRED = "followup_context_required"
    BOUNDARY_UNCERTAIN = "boundary_uncertain"
    DURATION_BELOW_MINIMUM = "duration_below_campaign_minimum"
    DURATION_REQUIRES_AMPUTATION = "duration_requires_amputation"


class HazardClassification(StrEnum):
    EDITORIAL_CONTENT = "editorial_content"
    ADVERTISEMENT = "advertisement"
    SPONSOR_READ = "sponsor_read"
    PROMO = "promo"
    INTRO = "intro"
    OUTRO = "outro"
    HOUSEKEEPING = "housekeeping"
    GRAPHIC_HEAVY = "graphic_heavy"
    UNKNOWN = "unknown"


class EvidenceOrigin(StrEnum):
    SOURCE = "source"
    CAMPAIGN_OVERLAY = "campaign_overlay"


def _status(value: object, field_name: str) -> BoundaryStatus:
    try:
        return BoundaryStatus(str(value).strip().upper())
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid boundary status") from exc


def _strings(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def _set_contract(instance: object, name: str, *types: type[Any]) -> None:
    object.__setattr__(
        instance,
        "contract_fingerprint",
        structural_contract_fingerprint(
            name,
            *types,
            exclude_fields=("contract_fingerprint",),
        ),
    )


@dataclass(frozen=True, slots=True)
class BoundaryAudit:
    source_start: float
    source_end: float
    first_source_word: str
    last_source_word: str
    pre_start_context: str
    post_end_context: str
    start_status: BoundaryStatus
    end_status: BoundaryStatus
    standalone_status: BoundaryStatus
    required_prior_context: str
    required_followup_context: str
    prior_context_included: bool
    followup_context_included: bool
    setup_resolved: bool
    payoff_resolved: bool
    open_questions: tuple[str, ...]
    open_references: tuple[str, ...]
    narrative_structure: str
    boundary_confidence: float
    failure_reasons: tuple[BoundaryFailureReason, ...]
    source_word_evidence: tuple[str, ...]
    repair_start_word_id: str | None = None
    repair_end_word_id: str | None = None
    model_identity: dict[str, object] | None = None
    contract_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _set_contract(self, "boundary-audit", BoundaryAudit)
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("boundary audit source timestamps are invalid")
        if not self.first_source_word.strip() or not self.last_source_word.strip():
            raise ValueError("boundary audit requires first and last source words")
        if not self.narrative_structure.strip():
            raise ValueError("boundary audit requires a narrative structure")
        if not 0.0 <= self.boundary_confidence <= 1.0:
            raise ValueError("boundary confidence must be between 0 and 1")
        if not self.source_word_evidence:
            raise ValueError("boundary audit requires canonical source-word evidence")

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        source_start: float,
        source_end: float,
        first_source_word: str,
        last_source_word: str,
        pre_start_context: str,
        post_end_context: str,
        source_word_evidence: tuple[str, ...],
        model_identity: dict[str, object],
        required_prior_context: str = "",
        required_followup_context: str = "",
    ) -> BoundaryAudit:
        raw_reasons = _strings(payload.get("failure_reasons", []), "failure_reasons")
        try:
            reasons = tuple(BoundaryFailureReason(item) for item in raw_reasons)
        except ValueError as exc:
            raise ValueError("boundary audit contains an unsupported failure reason") from exc
        repair_start = str(payload.get("repair_start_word_id") or "").strip() or None
        repair_end = str(payload.get("repair_end_word_id") or "").strip() or None
        return cls(
            source_start=source_start,
            source_end=source_end,
            first_source_word=first_source_word,
            last_source_word=last_source_word,
            pre_start_context=pre_start_context,
            post_end_context=post_end_context,
            start_status=_status(payload.get("start_status"), "start_status"),
            end_status=_status(payload.get("end_status"), "end_status"),
            standalone_status=_status(payload.get("standalone_status"), "standalone_status"),
            required_prior_context=(
                required_prior_context or str(payload.get("required_prior_context") or "").strip()
            ),
            required_followup_context=(
                required_followup_context
                or str(payload.get("required_followup_context") or "").strip()
            ),
            prior_context_included=bool(payload.get("prior_context_included", False)),
            followup_context_included=bool(payload.get("followup_context_included", False)),
            setup_resolved=bool(payload.get("setup_resolved", False)),
            payoff_resolved=bool(payload.get("payoff_resolved", False)),
            open_questions=_strings(payload.get("open_questions", []), "open_questions"),
            open_references=_strings(payload.get("open_references", []), "open_references"),
            narrative_structure=str(payload.get("narrative_structure") or "").strip(),
            boundary_confidence=float(payload.get("boundary_confidence", 0.0)),
            failure_reasons=reasons,
            source_word_evidence=source_word_evidence,
            repair_start_word_id=repair_start,
            repair_end_word_id=repair_end,
            model_identity=dict(model_identity),
        )

    def effective_reasons(self) -> tuple[str, ...]:
        reasons = [reason.value for reason in self.failure_reasons]

        def add(reason: BoundaryFailureReason) -> None:
            if reason.value not in reasons:
                reasons.append(reason.value)

        if self.required_prior_context and not self.prior_context_included:
            add(BoundaryFailureReason.START_REQUIRES_PRIOR_CONTEXT)
        if self.required_followup_context and not self.followup_context_included:
            add(BoundaryFailureReason.FOLLOWUP_CONTEXT_REQUIRED)
        if self.open_questions:
            add(BoundaryFailureReason.OPEN_QUESTION)
        if self.open_references:
            add(BoundaryFailureReason.UNRESOLVED_REFERENCE)
        if not self.setup_resolved:
            add(BoundaryFailureReason.UNRESOLVED_SETUP)
        if not self.payoff_resolved:
            add(BoundaryFailureReason.UNRESOLVED_PAYOFF)
        if self.start_status == BoundaryStatus.INCOMPLETE:
            add(BoundaryFailureReason.START_FRAGMENT)
        if self.end_status == BoundaryStatus.INCOMPLETE:
            add(BoundaryFailureReason.END_INCOMPLETE)
        if BoundaryStatus.UNCERTAIN in {
            self.start_status,
            self.end_status,
            self.standalone_status,
        }:
            add(BoundaryFailureReason.BOUNDARY_UNCERTAIN)
        return tuple(reasons)

    def decision(self, policy: AcceptancePolicy) -> GateDecision:
        if (
            self.boundary_confidence < policy.editorial.minimum_boundary_confidence
            or BoundaryStatus.UNCERTAIN
            in {self.start_status, self.end_status, self.standalone_status}
        ):
            return GateDecision.ESCALATE
        invalid = bool(self.effective_reasons())
        invalid = invalid or self.start_status != BoundaryStatus.COMPLETE
        invalid = invalid or self.end_status != BoundaryStatus.COMPLETE
        if policy.editorial.require_standalone_context:
            invalid = invalid or self.standalone_status != BoundaryStatus.COMPLETE
        if policy.editorial.require_resolved_ending:
            invalid = invalid or not self.payoff_resolved
        if not invalid:
            return GateDecision.PASS
        if self.repair_start_word_id or self.repair_end_word_id:
            return GateDecision.REPAIR
        return GateDecision.REJECT

    def to_dict(self, policy: AcceptancePolicy | None = None) -> dict[str, object]:
        data = asdict(self)
        data["start_status"] = self.start_status.value
        data["end_status"] = self.end_status.value
        data["standalone_status"] = self.standalone_status.value
        data["failure_reasons"] = list(self.effective_reasons())
        if policy is not None:
            data["decision"] = self.decision(policy).value
        return data


@dataclass(frozen=True, slots=True)
class SourceHazardSegment:
    start: float
    end: float
    classification: HazardClassification
    confidence: float
    evidence: tuple[str, ...]
    model_identity: dict[str, object]
    source_word_ids: tuple[str, ...] = ()
    contract_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _set_contract(self, "source-hazard", SourceHazardSegment)
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source hazard timestamps are invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("source hazard confidence must be between 0 and 1")
        if not self.evidence:
            raise ValueError("source hazard requires evidence")

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        timeline: CanonicalTimeline,
        *,
        model_identity: dict[str, object],
    ) -> SourceHazardSegment:
        start_ref = str(payload.get("start_word_id") or "").strip()
        end_ref = str(payload.get("end_word_id") or "").strip()
        if not start_ref or not end_ref:
            raise ValueError("source hazard requires start and end word references")
        start_id = timeline.resolve_word_ref(start_ref)
        end_id = timeline.resolve_word_ref(end_ref)
        positions = {word.word_id: index for index, word in enumerate(timeline.words)}
        start_index = positions[start_id]
        end_index = positions[end_id]
        if end_index < start_index:
            raise ValueError("source hazard word references must preserve chronology")
        words = timeline.words[start_index : end_index + 1]
        try:
            classification = HazardClassification(
                str(payload.get("classification") or "").strip().lower()
            )
        except ValueError as exc:
            raise ValueError("source hazard classification is invalid") from exc
        evidence = _strings(payload.get("evidence", []), "source hazard evidence")
        return cls(
            start=words[0].source_start,
            end=words[-1].source_end,
            classification=classification,
            confidence=float(payload.get("confidence", 0.0)),
            evidence=evidence,
            model_identity=dict(model_identity),
            source_word_ids=tuple(word.word_id for word in words),
        )

    def overlap_seconds(self, start: float, end: float, *, buffer: float = 0.0) -> float:
        buffered_start = max(0.0, self.start - buffer)
        buffered_end = self.end + buffer
        return max(0.0, min(end, buffered_end) - max(start, buffered_start))

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["classification"] = self.classification.value
        return data


@dataclass(frozen=True, slots=True)
class BrandingEvidence:
    start: float
    end: float
    origin: EvidenceOrigin
    description: str
    confidence: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("branding evidence timestamps are invalid")
        if not self.description.strip():
            raise ValueError("branding evidence description cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("branding evidence confidence must be between 0 and 1")

    def overlaps(self, start: float, end: float) -> bool:
        return min(end, self.end) > max(start, self.start)


@dataclass(frozen=True, slots=True)
class PolicyAudit:
    source_start: float
    source_end: float
    decision: GateDecision
    reasons: tuple[str, ...]
    hazard_intersections: tuple[dict[str, object], ...]
    branding_evidence: tuple[dict[str, object], ...]
    campaign_policy_checks: dict[str, object]
    contract_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _set_contract(self, "campaign-policy-audit", PolicyAudit)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


@dataclass(frozen=True, slots=True)
class PreRenderEligibility:
    decision: GateDecision
    reasons: tuple[str, ...]
    duration_seconds: float
    boundary_decision: GateDecision
    policy_decision: GateDecision
    revalidated_after_repair: bool
    boundary_audit: dict[str, object]
    policy_audit: dict[str, object]
    contract_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _set_contract(self, "pre-render-eligibility", PreRenderEligibility)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["decision"] = self.decision.value
        data["boundary_decision"] = self.boundary_decision.value
        data["policy_decision"] = self.policy_decision.value
        return data


def _decision_for_action(action: str) -> GateDecision:
    if action == "forbid":
        return GateDecision.REJECT
    if action == "escalate":
        return GateDecision.ESCALATE
    return GateDecision.PASS


def evaluate_campaign_policy(
    brief: CampaignBrief,
    source_start: float,
    source_end: float,
    hazards: tuple[SourceHazardSegment, ...],
    branding: tuple[BrandingEvidence, ...],
) -> PolicyAudit:
    policy = brief.acceptance_policy
    if source_start < 0 or source_end <= source_start:
        raise ValueError("campaign policy audit source timestamps are invalid")
    if not policy.enabled:
        return PolicyAudit(
            source_start,
            source_end,
            GateDecision.PASS,
            (),
            (),
            (),
            {"structured_policy_enabled": False, "backward_compatible": True},
        )

    reasons: list[str] = []
    decisions: list[GateDecision] = []
    intersections: list[dict[str, object]] = []
    covered_intervals: list[tuple[float, float]] = []
    segment_policy = policy.source_segments
    for hazard in hazards:
        raw_overlap_start = max(source_start, hazard.start)
        raw_overlap_end = min(source_end, hazard.end)
        if raw_overlap_end > raw_overlap_start:
            covered_intervals.append((raw_overlap_start, raw_overlap_end))
        overlap = hazard.overlap_seconds(
            source_start,
            source_end,
            buffer=(
                segment_policy.safety_buffer_seconds
                if hazard.classification.value in segment_policy.forbid
                else 0.0
            ),
        )
        if overlap <= 0:
            continue
        classification = hazard.classification.value
        intersection_payload = hazard.to_dict()
        intersection_payload["overlap_seconds"] = round(overlap, 6)
        intersections.append(intersection_payload)
        if hazard.confidence < policy.editorial.minimum_boundary_confidence:
            decisions.append(GateDecision.ESCALATE)
            reasons.append("policy_uncertain")
        elif classification in segment_policy.forbid:
            decisions.append(GateDecision.REJECT)
            reasons.append("forbidden_source_segment")
        elif classification not in segment_policy.allow:
            decision = _decision_for_action(segment_policy.unknown)
            decisions.append(decision)
            if decision == GateDecision.REJECT:
                reasons.append("forbidden_source_segment")
            elif decision == GateDecision.ESCALATE:
                reasons.append("policy_uncertain")

    covered_intervals.sort()
    merged: list[tuple[float, float]] = []
    for interval_start, interval_end in covered_intervals:
        if merged and interval_start <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))
        else:
            merged.append((interval_start, interval_end))
    covered_seconds = sum(interval_end - interval_start for interval_start, interval_end in merged)
    if covered_seconds + 0.05 < source_end - source_start:
        decisions.append(GateDecision.ESCALATE)
        reasons.append("policy_uncertain")

    branding_items: list[dict[str, object]] = []
    for brand in branding:
        if not brand.overlaps(source_start, source_end):
            continue
        branding_items.append({**asdict(brand), "origin": brand.origin.value})
        if brand.origin == EvidenceOrigin.CAMPAIGN_OVERLAY:
            if not policy.branding.supplied_campaign_assets_allowed:
                decisions.append(GateDecision.REJECT)
                reasons.append("campaign_branding_not_allowed")
            continue
        if brand.confidence < policy.branding.minimum_confidence:
            decisions.append(GateDecision.ESCALATE)
            reasons.append("policy_uncertain")
            continue
        decision = _decision_for_action(policy.branding.foreign_logos)
        decisions.append(decision)
        if decision == GateDecision.REJECT:
            reasons.append("foreign_branding")
        elif decision == GateDecision.ESCALATE:
            reasons.append("policy_uncertain")

    if GateDecision.REJECT in decisions:
        final = GateDecision.REJECT
    elif GateDecision.ESCALATE in decisions:
        final = GateDecision.ESCALATE
    else:
        final = GateDecision.PASS
    return PolicyAudit(
        source_start,
        source_end,
        final,
        tuple(dict.fromkeys(reasons)),
        tuple(intersections),
        tuple(branding_items),
        {
            "structured_policy_enabled": True,
            "source_segment_policy": asdict(segment_policy),
            "branding_policy": asdict(policy.branding),
            "generated_media_policy": policy.ai_generated_source_video,
            "portrayal_policy": policy.negative_creator_portrayal,
            "on_screen_text_language": policy.on_screen_text_language,
            "deferred_multimodal_checks": [
                "source_visible_foreign_branding",
                "ai_generated_source_video",
                "negative_creator_portrayal",
                "on_screen_text_language",
            ],
            "source_hazard_coverage_seconds": round(covered_seconds, 6),
            "source_hazard_coverage_complete": covered_seconds + 0.05 >= source_end - source_start,
        },
    )


def evaluate_pre_render_eligibility(
    brief: CampaignBrief,
    boundary: BoundaryAudit,
    policy: PolicyAudit,
    *,
    repaired: bool,
) -> PreRenderEligibility:
    duration = boundary.source_end - boundary.source_start
    reasons = list(boundary.effective_reasons())
    boundary_decision = boundary.decision(brief.acceptance_policy)
    if duration < brief.min_clip_seconds:
        reasons.append(BoundaryFailureReason.DURATION_BELOW_MINIMUM.value)
    if duration > brief.max_clip_seconds:
        reasons.append(BoundaryFailureReason.DURATION_REQUIRES_AMPUTATION.value)
    reasons.extend(policy.reasons)
    reasons = list(dict.fromkeys(reasons))

    if (
        duration < brief.min_clip_seconds
        or duration > brief.max_clip_seconds
        or GateDecision.REJECT in {boundary_decision, policy.decision}
    ):
        decision = GateDecision.REJECT
    elif GateDecision.ESCALATE in {boundary_decision, policy.decision}:
        decision = GateDecision.ESCALATE
    elif GateDecision.REPAIR in {boundary_decision, policy.decision}:
        decision = GateDecision.REPAIR
    else:
        decision = GateDecision.PASS
    return PreRenderEligibility(
        decision=decision,
        reasons=tuple(reasons),
        duration_seconds=round(duration, 6),
        boundary_decision=boundary_decision,
        policy_decision=policy.decision,
        revalidated_after_repair=repaired,
        boundary_audit=boundary.to_dict(brief.acceptance_policy),
        policy_audit=policy.to_dict(),
    )

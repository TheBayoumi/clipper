from __future__ import annotations

from dataclasses import dataclass

from .canonical import CanonicalTimeline
from .editorial_integrity import (
    BoundaryAudit,
    BoundaryStatus,
    BrandingEvidence,
    EvidenceOrigin,
    GateDecision,
    PolicyAudit,
    PreRenderEligibility,
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
    SourceSpan,
)
from .multimodal_timeline import MultimodalTimeline
from .quality_moments import QualityMoment
from .stage_contracts import content_fingerprint


@dataclass(frozen=True, slots=True)
class AdaptedQualityMoment:
    quality_moment: QualityMoment
    concept: ClipConcept
    variant: HookVariant
    plan: EditPlan
    boundary_audit: BoundaryAudit
    policy_audit: PolicyAudit
    eligibility: PreRenderEligibility

    def to_dict(self) -> dict[str, object]:
        return {
            "quality_moment": self.quality_moment.to_dict(),
            "concept": self.concept.to_dict(),
            "variant": self.variant.to_dict(),
            "plan": self.plan.to_dict(),
            "boundary_audit": self.boundary_audit.to_dict(),
            "policy_audit": self.policy_audit.to_dict(),
            "eligibility": self.eligibility.to_dict(),
        }


def source_branding_evidence(multimodal: MultimodalTimeline | None) -> tuple[BrandingEvidence, ...]:
    if multimodal is None:
        return ()
    evidence: list[BrandingEvidence] = []
    for event in multimodal.events:
        for description in event.branding:
            evidence.append(
                BrandingEvidence(
                    start=event.start,
                    end=event.end,
                    origin=EvidenceOrigin.SOURCE,
                    description=description,
                    confidence=event.confidence,
                )
            )
    return tuple(evidence)


def forbidden_spans_for_campaign(
    brief: CampaignBrief,
    hazards: tuple[SourceHazardSegment, ...],
    branding: tuple[BrandingEvidence, ...],
) -> tuple[SourceSpan, ...]:
    """Return intervals that cannot produce an automatic PASS for this campaign."""

    policy = brief.acceptance_policy
    if not policy.enabled:
        return ()
    spans: list[SourceSpan] = []
    segment_policy = policy.source_segments
    for hazard in hazards:
        classification = hazard.classification.value
        allowed = classification in segment_policy.allow
        if allowed:
            continue
        buffer = (
            segment_policy.safety_buffer_seconds if classification in segment_policy.forbid else 0.0
        )
        spans.append(SourceSpan(max(0.0, hazard.start - buffer), hazard.end + buffer))

    if policy.branding.foreign_logos != "allow":
        spans.extend(
            SourceSpan(item.start, item.end)
            for item in branding
            if item.origin == EvidenceOrigin.SOURCE
        )
    return _merge_spans(tuple(spans))


def _merge_spans(spans: tuple[SourceSpan, ...]) -> tuple[SourceSpan, ...]:
    if not spans:
        return ()
    ordered = sorted(spans, key=lambda item: (item.start, item.end))
    merged: list[SourceSpan] = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if item.start <= previous.end + 1e-6:
            merged[-1] = SourceSpan(previous.start, max(previous.end, item.end))
        else:
            merged.append(item)
    return tuple(merged)


def _quality_scores(score: float) -> EditorialScores:
    value = round(max(0.0, min(1.0, score)) * 10.0, 4)
    return EditorialScores(*(value for _ in range(12)))


def _source_text(timeline: CanonicalTimeline, word_ids: tuple[str, ...]) -> str:
    return " ".join(word.text for word in timeline.require_word_ids(word_ids)).strip()


def _boundary_audit(
    timeline: CanonicalTimeline,
    moment: QualityMoment,
) -> BoundaryAudit:
    window = moment.delivery_window
    words = timeline.require_word_ids(window.source_word_ids)
    positions = {word.word_id: index for index, word in enumerate(timeline.words)}
    first_index = positions[words[0].word_id]
    last_index = positions[words[-1].word_id]
    before = timeline.words[max(0, first_index - 24) : first_index]
    after = timeline.words[last_index + 1 : min(len(timeline.words), last_index + 25)]
    confidence = min(
        moment.core.confidence,
        moment.envelope.confidence,
        moment.assessment.confidence,
    )
    return BoundaryAudit(
        source_start=window.source_start,
        source_end=window.source_end,
        first_source_word=words[0].text,
        last_source_word=words[-1].text,
        pre_start_context=" ".join(word.text for word in before).strip(),
        post_end_context=" ".join(word.text for word in after).strip(),
        start_status=BoundaryStatus.COMPLETE,
        end_status=BoundaryStatus.COMPLETE,
        standalone_status=BoundaryStatus.COMPLETE,
        required_prior_context=moment.envelope.required_prior_context,
        required_followup_context=moment.envelope.required_followup_context,
        prior_context_included=True,
        followup_context_included=True,
        setup_resolved=moment.envelope.setup_resolved,
        payoff_resolved=moment.envelope.payoff_resolved,
        open_questions=(),
        open_references=(),
        narrative_structure=moment.core.semantic_summary,
        boundary_confidence=confidence,
        failure_reasons=(),
        source_word_evidence=window.source_word_ids,
        model_identity={"origin": "quality-moment-graph"},
        prompt_version="quality-moment-graph-v1",
        schema_version="boundary-audit-v1",
    )


def adapt_quality_moment(
    brief: CampaignBrief,
    timeline: CanonicalTimeline,
    moment: QualityMoment,
    *,
    hazards: tuple[SourceHazardSegment, ...],
    branding: tuple[BrandingEvidence, ...],
) -> AdaptedQualityMoment | None:
    """Compile one accepted QualityMoment into the existing renderer contract."""

    if moment.core.video_id != timeline.video_id or moment.core.source_hash != timeline.source_hash:
        raise ValueError("quality moment and canonical timeline reference different sources")
    window = moment.delivery_window
    boundary = _boundary_audit(timeline, moment)
    policy = evaluate_campaign_policy(
        brief,
        window.source_start,
        window.source_end,
        hazards,
        branding,
    )
    eligibility = evaluate_pre_render_eligibility(brief, boundary, policy, repaired=False)
    if eligibility.decision != GateDecision.PASS:
        return None

    score = max(0.0, min(1.0, moment.assessment.quality_score))
    score10 = round(score * 10.0, 4)
    transcript_fingerprint = content_fingerprint(
        {
            "source_hash": timeline.source_hash,
            "word_ids": list(window.source_word_ids),
        }
    )
    concept_id = moment.quality_moment_id
    source_text = _source_text(timeline, window.source_word_ids)
    concept = ClipConcept(
        concept_id=concept_id,
        video_id=timeline.video_id,
        source_start=window.source_start,
        source_end=window.source_end,
        text=source_text,
        topic=moment.core.semantic_summary,
        setup=moment.envelope.required_prior_context,
        payoff=moment.envelope.required_followup_context,
        moment_type="quality_moment",
        recommended_duration=window.duration,
        scores=_quality_scores(score),
        score=score10,
        semantic_cluster=moment.core.core_id,
        transcript_fingerprint=transcript_fingerprint,
    )
    span = SourceSpan(window.source_start, window.source_end)
    first_word = timeline.require_word_ids((window.source_word_ids[0],))[0]
    variant_id = f"{concept_id}:direct"
    variant = HookVariant(
        variant_id=variant_id,
        concept_id=concept_id,
        mode="direct",
        source_spans=(span,),
        overlay_text=None,
        score=score10,
        rationale=moment.assessment.rationale,
        fingerprint=content_fingerprint(
            {"quality_moment_id": concept_id, "window_id": window.window_id, "mode": "direct"}
        ),
        caption_start_source_time=window.source_start,
        caption_start_word=first_word.text,
    )
    plan = EditPlan(
        plan_id=f"{concept_id}:plan",
        video_id=timeline.video_id,
        concept_id=concept_id,
        variant_id=variant_id,
        hook_mode="direct",
        source_spans=(span,),
        hook_text=None,
        beats=(),
        caption_platform="tiktok",
        score=score10,
        transcript_fingerprint=transcript_fingerprint,
        caption_start_source_time=window.source_start,
        caption_start_word=first_word.text,
        boundary_audit=boundary.to_dict(brief.acceptance_policy),
        campaign_policy_audit=policy.to_dict(),
        pre_render_eligibility=eligibility.to_dict(),
    )
    return AdaptedQualityMoment(moment, concept, variant, plan, boundary, policy, eligibility)

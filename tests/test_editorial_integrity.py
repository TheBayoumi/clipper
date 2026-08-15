from __future__ import annotations

from dataclasses import replace

import pytest

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.editorial_integrity import (
    BoundaryAudit,
    BoundaryFailureReason,
    BoundaryStatus,
    BrandingEvidence,
    EvidenceOrigin,
    GateDecision,
    HazardClassification,
    PolicyAudit,
    SourceHazardSegment,
    evaluate_campaign_policy,
    evaluate_pre_render_eligibility,
)
from clipper.models import AcceptancePolicy, CampaignBrief, ProductionConfig


def _brief() -> CampaignBrief:
    return CampaignBrief(
        campaign_id="generic-campaign",
        title="Authorized interview clips",
        objective="Select complete, campaign-safe stories.",
        keywords=["interview"],
        allowed_video_ids=["video"],
        rights_confirmed=True,
        min_clip_seconds=8,
        max_clip_seconds=45,
        production=ProductionConfig(final_render_budget=6, minimum_distinct_finalist_concepts=3),
        acceptance_policy=AcceptancePolicy.from_dict(
            {
                "source_segments": {
                    "allow": ["editorial_content"],
                    "forbid": [
                        "advertisement",
                        "sponsor_read",
                        "promo",
                        "intro",
                        "outro",
                        "housekeeping",
                    ],
                    "unknown": "escalate",
                },
                "branding": {
                    "supplied_campaign_assets_allowed": True,
                    "foreign_logos": "forbid",
                },
                "generated_media": {"ai_generated_source_video": "forbid"},
                "portrayal": {"negative_creator_portrayal": "forbid"},
                "language": {"on_screen_text": "en"},
                "editorial": {
                    "require_standalone_context": True,
                    "require_resolved_ending": True,
                    "minimum_boundary_confidence": 0.75,
                },
            }
        ),
    )


def _boundary(
    *,
    start: BoundaryStatus = BoundaryStatus.COMPLETE,
    end: BoundaryStatus = BoundaryStatus.COMPLETE,
    standalone: BoundaryStatus = BoundaryStatus.COMPLETE,
    setup_resolved: bool = True,
    payoff_resolved: bool = True,
    required_prior_context: str = "",
    required_followup_context: str = "",
    prior_context_included: bool = True,
    followup_context_included: bool = True,
    open_questions: tuple[str, ...] = (),
    open_references: tuple[str, ...] = (),
    reasons: tuple[BoundaryFailureReason, ...] = (),
    confidence: float = 0.95,
    repair_start_word_id: str | None = None,
    repair_end_word_id: str | None = None,
) -> BoundaryAudit:
    return BoundaryAudit(
        source_start=10.0,
        source_end=37.0,
        first_source_word="The",
        last_source_word="finished.",
        pre_start_context="Earlier source context.",
        post_end_context="Later source context.",
        start_status=start,
        end_status=end,
        standalone_status=standalone,
        required_prior_context=required_prior_context,
        required_followup_context=required_followup_context,
        prior_context_included=prior_context_included,
        followup_context_included=followup_context_included,
        setup_resolved=setup_resolved,
        payoff_resolved=payoff_resolved,
        open_questions=open_questions,
        open_references=open_references,
        narrative_structure="setup-surprise-reaction",
        boundary_confidence=confidence,
        failure_reasons=reasons,
        source_word_evidence=("w10", "w37"),
        repair_start_word_id=repair_start_word_id,
        repair_end_word_id=repair_end_word_id,
        model_identity={"model_id": "semantic-boundary-test", "revision": "fixed"},
        prompt_version="editor-v2",
        schema_version="boundary-audit-v1",
    )


def _editorial_hazard(start: float = 0.0, end: float = 100.0) -> SourceHazardSegment:
    return SourceHazardSegment(
        start,
        end,
        HazardClassification.EDITORIAL_CONTENT,
        0.99,
        ("ordinary editorial conversation",),
        {"model_id": "hazard-test", "revision": "fixed"},
    )


def _policy_pass() -> PolicyAudit:
    return evaluate_campaign_policy(_brief(), 10.0, 37.0, (_editorial_hazard(),), ())


@pytest.mark.parametrize(
    ("case", "audit", "expected_reason"),
    [
        (
            "01-half-sentence",
            _boundary(
                start=BoundaryStatus.INCOMPLETE,
                reasons=(BoundaryFailureReason.START_FRAGMENT,),
            ),
            BoundaryFailureReason.START_FRAGMENT,
        ),
        (
            "02-answer-without-question",
            _boundary(
                start=BoundaryStatus.NEEDS_CONTEXT,
                required_prior_context="the question being answered",
                prior_context_included=False,
                reasons=(BoundaryFailureReason.START_REQUIRES_PRIOR_CONTEXT,),
            ),
            BoundaryFailureReason.START_REQUIRES_PRIOR_CONTEXT,
        ),
        (
            "03-unresolved-reference",
            _boundary(
                standalone=BoundaryStatus.NEEDS_CONTEXT,
                open_references=("it",),
                reasons=(BoundaryFailureReason.UNRESOLVED_REFERENCE,),
            ),
            BoundaryFailureReason.UNRESOLVED_REFERENCE,
        ),
        (
            "04-filler-removal-cannot-drop-setup",
            _boundary(
                start=BoundaryStatus.NEEDS_CONTEXT,
                required_prior_context="why the decision was necessary",
                prior_context_included=False,
                reasons=(BoundaryFailureReason.START_REQUIRES_PRIOR_CONTEXT,),
            ),
            BoundaryFailureReason.START_REQUIRES_PRIOR_CONTEXT,
        ),
    ],
)
def test_start_boundary_failure_classes_are_non_compensable(
    case: str, audit: BoundaryAudit, expected_reason: BoundaryFailureReason
) -> None:
    result = evaluate_pre_render_eligibility(_brief(), audit, _policy_pass(), repaired=False)
    assert case
    assert result.decision in {GateDecision.REPAIR, GateDecision.REJECT}
    assert expected_reason.value in result.reasons


def test_05_later_strong_hook_passes_only_when_resulting_clip_is_standalone() -> None:
    passed = evaluate_pre_render_eligibility(_brief(), _boundary(), _policy_pass(), repaired=False)
    failed = evaluate_pre_render_eligibility(
        _brief(),
        _boundary(
            standalone=BoundaryStatus.NEEDS_CONTEXT,
            open_references=("that",),
            reasons=(BoundaryFailureReason.UNRESOLVED_REFERENCE,),
        ),
        _policy_pass(),
        repaired=False,
    )
    assert passed.decision == GateDecision.PASS
    assert failed.decision != GateDecision.PASS


@pytest.mark.parametrize(
    ("case", "audit", "expected_reason"),
    [
        (
            "06-number-without-unit",
            _boundary(
                end=BoundaryStatus.INCOMPLETE,
                reasons=(BoundaryFailureReason.PARTIAL_NUMBER_OR_UNIT,),
            ),
            BoundaryFailureReason.PARTIAL_NUMBER_OR_UNIT,
        ),
        (
            "07-dangling-continuation",
            _boundary(
                end=BoundaryStatus.INCOMPLETE,
                reasons=(BoundaryFailureReason.END_INCOMPLETE,),
            ),
            BoundaryFailureReason.END_INCOMPLETE,
        ),
        (
            "08-question-without-answer",
            _boundary(
                end=BoundaryStatus.NEEDS_CONTEXT,
                open_questions=("What happened next?",),
                reasons=(BoundaryFailureReason.OPEN_QUESTION,),
            ),
            BoundaryFailureReason.OPEN_QUESTION,
        ),
        (
            "09-known-payoff-omitted",
            _boundary(
                end=BoundaryStatus.NEEDS_CONTEXT,
                payoff_resolved=False,
                required_followup_context="the explicitly promised result",
                followup_context_included=False,
                reasons=(BoundaryFailureReason.UNRESOLVED_PAYOFF,),
            ),
            BoundaryFailureReason.UNRESOLVED_PAYOFF,
        ),
        (
            "10-material-reaction-omitted",
            _boundary(
                payoff_resolved=False,
                reasons=(BoundaryFailureReason.UNRESOLVED_PAYOFF,),
            ),
            BoundaryFailureReason.UNRESOLVED_PAYOFF,
        ),
        (
            "11-punctuation-is-not-proof",
            replace(
                _boundary(
                    end=BoundaryStatus.INCOMPLETE,
                    reasons=(BoundaryFailureReason.END_INCOMPLETE,),
                ),
                last_source_word="Just.",
            ),
            BoundaryFailureReason.END_INCOMPLETE,
        ),
        (
            "12-pause-is-not-proof",
            _boundary(
                end=BoundaryStatus.INCOMPLETE,
                reasons=(BoundaryFailureReason.END_INCOMPLETE,),
            ),
            BoundaryFailureReason.END_INCOMPLETE,
        ),
    ],
)
def test_end_boundary_failure_classes_are_non_compensable(
    case: str, audit: BoundaryAudit, expected_reason: BoundaryFailureReason
) -> None:
    result = evaluate_pre_render_eligibility(_brief(), audit, _policy_pass(), repaired=False)
    assert case
    assert result.decision in {GateDecision.REPAIR, GateDecision.REJECT}
    assert expected_reason.value in result.reasons


def test_13_max_duration_never_forces_semantic_truncation() -> None:
    audit = replace(
        _boundary(
            end=BoundaryStatus.INCOMPLETE,
            reasons=(BoundaryFailureReason.END_INCOMPLETE,),
        ),
        source_end=55.0,
    )
    result = evaluate_pre_render_eligibility(_brief(), audit, _policy_pass(), repaired=True)
    assert result.decision == GateDecision.REJECT
    assert BoundaryFailureReason.END_INCOMPLETE.value in result.reasons


def test_14_complete_story_over_max_without_coherent_construction_is_rejected() -> None:
    audit = replace(_boundary(), source_end=60.0)
    result = evaluate_pre_render_eligibility(_brief(), audit, _policy_pass(), repaired=False)
    assert result.decision == GateDecision.REJECT
    assert BoundaryFailureReason.DURATION_REQUIRES_AMPUTATION.value in result.reasons


def test_15_repaired_candidate_re_runs_narrative_validation() -> None:
    repaired_audit = _boundary(
        end=BoundaryStatus.INCOMPLETE,
        reasons=(BoundaryFailureReason.END_INCOMPLETE,),
    )
    result = evaluate_pre_render_eligibility(
        _brief(), repaired_audit, _policy_pass(), repaired=True
    )
    assert result.revalidated_after_repair is True
    assert result.decision == GateDecision.REJECT


def test_16_short_complete_story_is_eligible_while_truncated_long_version_is_not() -> None:
    short_complete = evaluate_pre_render_eligibility(
        _brief(), _boundary(), _policy_pass(), repaired=False
    )
    truncated = evaluate_pre_render_eligibility(
        _brief(),
        replace(
            _boundary(
                end=BoundaryStatus.INCOMPLETE,
                reasons=(BoundaryFailureReason.END_INCOMPLETE,),
            ),
            source_end=55.0,
        ),
        _policy_pass(),
        repaired=False,
    )
    assert short_complete.decision == GateDecision.PASS
    assert truncated.decision == GateDecision.REJECT


def test_17_candidate_fully_inside_forbidden_source_hazard_fails() -> None:
    hazards = (
        SourceHazardSegment(
            0.0,
            60.0,
            HazardClassification.SPONSOR_READ,
            0.98,
            ("semantic sponsor read", "inserted product graphics"),
            {"model_id": "hazard-test", "revision": "fixed"},
        ),
    )
    result = evaluate_campaign_policy(_brief(), 10.0, 37.0, hazards, ())
    assert result.decision == GateDecision.REJECT
    assert "forbidden_source_segment" in result.reasons


def test_18_candidate_crossing_into_forbidden_source_hazard_fails() -> None:
    hazards = (
        SourceHazardSegment(
            35.0,
            60.0,
            HazardClassification.ADVERTISEMENT,
            0.99,
            ("ad transition",),
            {"model_id": "hazard-test", "revision": "fixed"},
        ),
    )
    result = evaluate_campaign_policy(_brief(), 10.0, 37.0, hazards, ())
    assert result.decision == GateDecision.REJECT
    assert result.hazard_intersections[0]["overlap_seconds"] == pytest.approx(2.0)


def test_19_clean_editorial_span_before_ad_remains_eligible() -> None:
    hazards = (
        _editorial_hazard(0.0, 38.0),
        SourceHazardSegment(
            38.0,
            60.0,
            HazardClassification.ADVERTISEMENT,
            0.99,
            ("ad transition",),
            {"model_id": "hazard-test", "revision": "fixed"},
        ),
    )
    result = evaluate_campaign_policy(_brief(), 10.0, 37.0, hazards, ())
    assert result.decision == GateDecision.PASS


def test_20_low_confidence_unknown_hazard_escalates() -> None:
    hazards = (
        SourceHazardSegment(
            15.0,
            25.0,
            HazardClassification.UNKNOWN,
            0.52,
            ("modalities disagree",),
            {"model_id": "hazard-test", "revision": "fixed"},
        ),
    )
    result = evaluate_campaign_policy(_brief(), 10.0, 37.0, hazards, ())
    assert result.decision == GateDecision.ESCALATE
    assert "policy_uncertain" in result.reasons


def test_21_approved_campaign_watermark_passes() -> None:
    branding = (
        BrandingEvidence(
            0.0,
            27.0,
            EvidenceOrigin.CAMPAIGN_OVERLAY,
            "approved campaign watermark",
            1.0,
        ),
    )
    result = evaluate_campaign_policy(_brief(), 10.0, 37.0, (_editorial_hazard(),), branding)
    assert result.decision == GateDecision.PASS


def test_22_source_visible_foreign_logo_fails_when_forbidden() -> None:
    branding = (BrandingEvidence(12.0, 20.0, EvidenceOrigin.SOURCE, "foreign logo", 0.96),)
    result = evaluate_campaign_policy(_brief(), 10.0, 37.0, (_editorial_hazard(),), branding)
    assert result.decision == GateDecision.REJECT
    assert "foreign_branding" in result.reasons


def test_23_campaign_overlay_and_source_branding_are_distinct_origins() -> None:
    campaign = BrandingEvidence(
        0.0, 27.0, EvidenceOrigin.CAMPAIGN_OVERLAY, "approved watermark", 1.0
    )
    source = BrandingEvidence(12.0, 20.0, EvidenceOrigin.SOURCE, "foreign logo", 0.96)
    assert campaign.origin != source.origin
    assert (
        evaluate_campaign_policy(_brief(), 10.0, 37.0, (_editorial_hazard(),), (campaign,)).decision
        == GateDecision.PASS
    )
    assert (
        evaluate_campaign_policy(_brief(), 10.0, 37.0, (_editorial_hazard(),), (source,)).decision
        == GateDecision.REJECT
    )


def test_boundary_uncertainty_never_silently_passes() -> None:
    result = evaluate_pre_render_eligibility(
        _brief(),
        _boundary(start=BoundaryStatus.UNCERTAIN, confidence=0.51),
        _policy_pass(),
        repaired=False,
    )
    assert result.decision == GateDecision.ESCALATE


def test_localized_boundary_repair_is_machine_distinct_from_rejection() -> None:
    repairable = _boundary(
        start=BoundaryStatus.NEEDS_CONTEXT,
        reasons=(BoundaryFailureReason.START_REQUIRES_PRIOR_CONTEXT,),
        repair_start_word_id="w08",
    )
    irreparable = replace(repairable, repair_start_word_id=None)
    assert repairable.decision(_brief().acceptance_policy) == GateDecision.REPAIR
    assert irreparable.decision(_brief().acceptance_policy) == GateDecision.REJECT


def test_source_hazard_payload_is_grounded_to_canonical_word_evidence() -> None:
    timeline = CanonicalTimeline(
        "video",
        "hash",
        tuple(
            CanonicalWord(
                f"w{index}",
                text,
                float(index),
                float(index) + 0.8,
                "A",
                0.99,
                "aligned",
                "test",
            )
            for index, text in enumerate(("ordinary", "conversation", "sponsor", "message"))
        ),
    )
    segment = SourceHazardSegment.from_payload(
        {
            "start_word_id": "w2",
            "end_word_id": "w3",
            "classification": "sponsor_read",
            "confidence": 0.97,
            "evidence": ["semantic sponsor transition", "promotional graphic"],
        },
        timeline,
        model_identity={"model_id": "hazard-test", "revision": "fixed"},
    )
    assert segment.start == pytest.approx(2.0)
    assert segment.end == pytest.approx(3.8)
    assert segment.source_word_ids == ("w2", "w3")


def test_missing_source_hazard_coverage_escalates_instead_of_passing() -> None:
    partial = (_editorial_hazard(10.0, 20.0),)
    result = evaluate_campaign_policy(_brief(), 10.0, 37.0, partial, ())
    assert result.decision == GateDecision.ESCALATE
    assert result.campaign_policy_checks["source_hazard_coverage_complete"] is False


def test_existing_required_context_cannot_be_cleared_by_boundary_model_payload() -> None:
    audit = BoundaryAudit.from_payload(
        {
            "start_status": "COMPLETE",
            "end_status": "COMPLETE",
            "standalone_status": "COMPLETE",
            "required_prior_context": "",
            "required_followup_context": "",
            "prior_context_included": False,
            "followup_context_included": False,
            "setup_resolved": True,
            "payoff_resolved": True,
            "open_questions": [],
            "open_references": [],
            "narrative_structure": "question-answer-consequence",
            "boundary_confidence": 0.95,
            "failure_reasons": [],
            "repair_start_word_id": None,
            "repair_end_word_id": None,
        },
        source_start=10.0,
        source_end=37.0,
        first_source_word="Yes",
        last_source_word="finished.",
        pre_start_context="Question before clip",
        post_end_context="",
        source_word_evidence=("w10", "w37"),
        model_identity={"model_id": "boundary-test", "revision": "fixed"},
        prompt_version="editor-v2",
        schema_version="boundary-audit-v1",
        required_prior_context="the question being answered",
    )
    assert audit.required_prior_context == "the question being answered"
    assert audit.decision(_brief().acceptance_policy) == GateDecision.REJECT


@pytest.mark.parametrize(
    "changes",
    [
        {"source_start": -1.0},
        {"first_source_word": ""},
        {"narrative_structure": ""},
        {"boundary_confidence": 2.0},
        {"source_word_evidence": ()},
    ],
)
def test_boundary_audit_rejects_incomplete_machine_evidence(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_boundary(), **changes)


def test_boundary_and_hazard_payload_validation_is_fail_closed() -> None:
    payload: dict[str, object] = {
        "start_status": "INVALID",
        "end_status": "COMPLETE",
        "standalone_status": "COMPLETE",
        "setup_resolved": True,
        "payoff_resolved": True,
        "open_questions": None,
        "open_references": [],
        "narrative_structure": "statement",
        "boundary_confidence": 0.9,
        "failure_reasons": [],
    }
    kwargs = {
        "source_start": 0.0,
        "source_end": 10.0,
        "first_source_word": "A",
        "last_source_word": "B",
        "pre_start_context": "",
        "post_end_context": "",
        "source_word_evidence": ("w0", "w1"),
        "model_identity": {"model_id": "m"},
        "prompt_version": "editor-v2",
        "schema_version": "boundary-audit-v1",
    }
    with pytest.raises(ValueError, match="boundary status"):
        BoundaryAudit.from_payload(payload, **kwargs)  # type: ignore[arg-type]
    payload["start_status"] = "COMPLETE"
    payload["failure_reasons"] = ["invented_reason"]
    with pytest.raises(ValueError, match="unsupported failure reason"):
        BoundaryAudit.from_payload(payload, **kwargs)  # type: ignore[arg-type]

    timeline = CanonicalTimeline(
        "video",
        "hash",
        (
            CanonicalWord("w0", "one", 0.0, 0.5, None, 1.0, "aligned", "test"),
            CanonicalWord("w1", "two", 1.0, 1.5, None, 1.0, "aligned", "test"),
        ),
    )
    with pytest.raises(ValueError, match="start and end"):
        SourceHazardSegment.from_payload({}, timeline, model_identity={})
    with pytest.raises(ValueError, match="chronology"):
        SourceHazardSegment.from_payload(
            {
                "start_word_id": "w1",
                "end_word_id": "w0",
                "classification": "promo",
                "confidence": 0.9,
                "evidence": ["promo"],
            },
            timeline,
            model_identity={},
        )
    with pytest.raises(ValueError, match="classification"):
        SourceHazardSegment.from_payload(
            {
                "start_word_id": "w0",
                "end_word_id": "w1",
                "classification": "invented",
                "confidence": 0.9,
                "evidence": ["unknown"],
            },
            timeline,
            model_identity={},
        )


def test_integrity_evidence_value_objects_reject_invalid_fields() -> None:
    with pytest.raises(ValueError, match="timestamps"):
        SourceHazardSegment(2.0, 1.0, HazardClassification.PROMO, 0.9, ("x",), {})
    with pytest.raises(ValueError, match="confidence"):
        SourceHazardSegment(1.0, 2.0, HazardClassification.PROMO, 2.0, ("x",), {})
    with pytest.raises(ValueError, match="evidence"):
        SourceHazardSegment(1.0, 2.0, HazardClassification.PROMO, 0.9, (), {})
    with pytest.raises(ValueError, match="timestamps"):
        BrandingEvidence(2.0, 1.0, EvidenceOrigin.SOURCE, "logo", 0.9)
    with pytest.raises(ValueError, match="description"):
        BrandingEvidence(1.0, 2.0, EvidenceOrigin.SOURCE, "", 0.9)
    with pytest.raises(ValueError, match="confidence"):
        BrandingEvidence(1.0, 2.0, EvidenceOrigin.SOURCE, "logo", 2.0)


def test_campaign_policy_conservative_edge_branches() -> None:
    with pytest.raises(ValueError, match="timestamps"):
        evaluate_campaign_policy(_brief(), 2.0, 1.0, (), ())
    disabled = replace(_brief(), acceptance_policy=AcceptancePolicy())
    assert evaluate_campaign_policy(disabled, 1.0, 2.0, (), ()).decision == GateDecision.PASS

    low_confidence_ad = SourceHazardSegment(
        0.0,
        50.0,
        HazardClassification.ADVERTISEMENT,
        0.4,
        ("weak ad evidence",),
        {},
    )
    assert (
        evaluate_campaign_policy(_brief(), 10.0, 37.0, (low_confidence_ad,), ()).decision
        == GateDecision.ESCALATE
    )

    forbid_unknown = replace(
        _brief(),
        acceptance_policy=replace(
            _brief().acceptance_policy,
            source_segments=replace(_brief().acceptance_policy.source_segments, unknown="forbid"),
        ),
    )
    graphic = SourceHazardSegment(
        0.0,
        50.0,
        HazardClassification.GRAPHIC_HEAVY,
        0.9,
        ("large source graphic",),
        {},
    )
    assert (
        evaluate_campaign_policy(forbid_unknown, 10.0, 37.0, (graphic,), ()).decision
        == GateDecision.REJECT
    )

    overlay_forbidden = replace(
        _brief(),
        acceptance_policy=replace(
            _brief().acceptance_policy,
            branding=replace(
                _brief().acceptance_policy.branding,
                supplied_campaign_assets_allowed=False,
            ),
        ),
    )
    overlay = BrandingEvidence(10.0, 37.0, EvidenceOrigin.CAMPAIGN_OVERLAY, "watermark", 1.0)
    assert (
        evaluate_campaign_policy(
            overlay_forbidden, 10.0, 37.0, (_editorial_hazard(),), (overlay,)
        ).decision
        == GateDecision.REJECT
    )

    uncertain_logo = BrandingEvidence(10.0, 20.0, EvidenceOrigin.SOURCE, "possible logo", 0.2)
    assert (
        evaluate_campaign_policy(
            _brief(), 10.0, 37.0, (_editorial_hazard(),), (uncertain_logo,)
        ).decision
        == GateDecision.ESCALATE
    )

    review_logo_brief = replace(
        _brief(),
        acceptance_policy=replace(
            _brief().acceptance_policy,
            branding=replace(_brief().acceptance_policy.branding, foreign_logos="escalate"),
        ),
    )
    certain_logo = replace(uncertain_logo, confidence=0.99)
    assert (
        evaluate_campaign_policy(
            review_logo_brief,
            10.0,
            37.0,
            (_editorial_hazard(),),
            (certain_logo,),
        ).decision
        == GateDecision.ESCALATE
    )


def test_pre_render_eligibility_covers_minimum_and_repair_outcomes() -> None:
    too_short = replace(_boundary(), source_end=15.0)
    result = evaluate_pre_render_eligibility(_brief(), too_short, _policy_pass(), repaired=False)
    assert result.decision == GateDecision.REJECT
    assert BoundaryFailureReason.DURATION_BELOW_MINIMUM.value in result.reasons

    repair_policy = replace(_policy_pass(), decision=GateDecision.REPAIR)
    repair = evaluate_pre_render_eligibility(_brief(), _boundary(), repair_policy, repaired=False)
    assert repair.decision == GateDecision.REPAIR

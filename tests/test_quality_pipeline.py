from __future__ import annotations

from dataclasses import replace

import pytest

from clipper.brief import load_brief
from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.editorial_integrity import (
    BrandingEvidence,
    EvidenceOrigin,
    HazardClassification,
    SourceHazardSegment,
)
from clipper.multimodal_timeline import MultimodalEvent, MultimodalTimeline
from clipper.quality_moments import QualityMoment, WindowQualityAssessment
from clipper.quality_pipeline import (
    adapt_quality_moment,
    forbidden_spans_for_campaign,
    source_branding_evidence,
)
from clipper.story_graph import NarrativeEnvelope, SemanticCore
from clipper.window_solver import enumerate_feasible_windows


def _timeline() -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:x",
                f"word-{index}",
                float(index),
                float(index + 1),
                "speaker",
                0.99,
                "word_exact",
                "test",
            )
            for index in range(30)
        ),
    )


def _moment(timeline: CanonicalTimeline, *, confidence: float = 0.95) -> QualityMoment:
    core = SemanticCore.from_word_ids(
        timeline,
        core_id="core",
        source_word_ids=tuple(word.word_id for word in timeline.words[10:13]),
        semantic_summary="complete worthwhile story",
        editorial_reason="strong independent moment",
        confidence=confidence,
    )
    envelope = NarrativeEnvelope.from_word_ids(
        timeline,
        core,
        envelope_id="envelope",
        source_word_ids=tuple(word.word_id for word in timeline.words[5:25]),
        required_prior_context="setup included",
        required_followup_context="payoff included",
        setup_resolved=True,
        payoff_resolved=True,
        confidence=confidence,
    )
    window = enumerate_feasible_windows(
        timeline,
        core,
        envelope,
        min_seconds=20,
        max_seconds=60,
    )[0]
    assessment = WindowQualityAssessment(
        core.core_id,
        window.window_id,
        "PASS",
        0.93,
        "complete and worth publishing",
        confidence,
    )
    return QualityMoment("quality:core", core, envelope, window, assessment)


def _hazard(
    classification: HazardClassification = HazardClassification.EDITORIAL_CONTENT,
    *,
    start: float = 0.0,
    end: float = 30.0,
    confidence: float = 0.99,
) -> SourceHazardSegment:
    return SourceHazardSegment(
        start,
        end,
        classification,
        confidence,
        ("source classification evidence",),
        {"model": "test"},
    )


def test_clean_quality_moment_compiles_to_one_renderer_compatible_plan() -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    moment = _moment(timeline)
    adapted = adapt_quality_moment(
        brief,
        timeline,
        moment,
        hazards=(_hazard(),),
        branding=(),
    )
    assert adapted is not None
    assert adapted.plan.concept_id == moment.quality_moment_id
    assert adapted.plan.source_spans[0].start == moment.delivery_window.source_start
    assert adapted.plan.source_spans[0].end == moment.delivery_window.source_end
    assert adapted.plan.pre_render_eligibility["decision"] == "PASS"
    assert adapted.plan.boundary_audit["decision"] == "PASS"
    assert adapted.plan.campaign_policy_audit["decision"] == "PASS"
    assert adapted.variant.mode == "direct"
    assert adapted.variant.caption_start_source_time == moment.delivery_window.source_start
    assert adapted.to_dict()["quality_moment"]["quality_moment_id"] == "quality:core"


def test_forbidden_and_uncertain_policy_evidence_cannot_become_automatic_plan() -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    moment = _moment(timeline)
    sponsor = _hazard(HazardClassification.SPONSOR_READ, start=8.0, end=12.0)
    spans = forbidden_spans_for_campaign(brief, (sponsor,), ())
    assert spans
    assert spans[0].start <= sponsor.start
    assert spans[0].end >= sponsor.end
    assert (
        adapt_quality_moment(
            brief,
            timeline,
            moment,
            hazards=(_hazard(), sponsor),
            branding=(),
        )
        is None
    )

    low_confidence = _moment(timeline, confidence=0.1)
    assert (
        adapt_quality_moment(
            brief,
            timeline,
            low_confidence,
            hazards=(_hazard(),),
            branding=(),
        )
        is None
    )


def test_source_branding_is_solver_forbidden_and_policy_rejected() -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    moment = _moment(timeline)
    multimodal = MultimodalTimeline(
        "video",
        "source",
        30.0,
        (
            MultimodalEvent(
                9.0,
                11.0,
                branding=("foreign source logo",),
                confidence=0.99,
            ),
        ),
    )
    branding = source_branding_evidence(multimodal)
    assert len(branding) == 1
    assert branding[0].origin == EvidenceOrigin.SOURCE
    assert forbidden_spans_for_campaign(brief, (_hazard(),), branding)
    assert (
        adapt_quality_moment(
            brief,
            timeline,
            moment,
            hazards=(_hazard(),),
            branding=branding,
        )
        is None
    )


def test_adapter_rejects_cross_source_quality_graph() -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    timeline = _timeline()
    moment = _moment(timeline)
    with pytest.raises(ValueError, match="different sources"):
        adapt_quality_moment(
            brief,
            replace(timeline, source_hash="other"),
            moment,
            hazards=(_hazard(),),
            branding=(),
        )


def test_disabled_structured_policy_produces_no_solver_exclusions() -> None:
    brief = load_brief("campaigns/reach-double-coverage-dedicated.yaml")
    disabled = replace(brief, acceptance_policy=replace(brief.acceptance_policy, enabled=False))
    brand = BrandingEvidence(1.0, 2.0, EvidenceOrigin.SOURCE, "logo", 1.0)
    assert forbidden_spans_for_campaign(disabled, (_hazard(HazardClassification.SPONSOR_READ),), (brand,)) == ()
    assert source_branding_evidence(None) == ()

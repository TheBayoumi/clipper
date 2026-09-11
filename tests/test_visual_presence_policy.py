from __future__ import annotations

from clipper.editorial_integrity import (
    GateDecision,
    HazardClassification,
    SourceHazardSegment,
    evaluate_campaign_policy,
)
from clipper.models import AcceptancePolicy, CampaignBrief
from clipper.multimodal_timeline import MultimodalEvent, MultimodalTimeline
from clipper.visual import VisualEvidenceSpan


def _brief(*required: str, minimum_confidence: float = 0.75) -> CampaignBrief:
    policy = AcceptancePolicy.from_dict(
        {
            "enabled": True,
            "source_segments": {
                "allow": ["editorial_content"],
                "forbid": [],
                "unknown": "escalate",
            },
            "branding": {"foreign_logos": "allow"},
            "visual_presence": {
                "require_any": list(required),
                "minimum_confidence": minimum_confidence,
            },
        }
    )
    return CampaignBrief(
        campaign_id="lovable-test",
        title="Lovable test",
        objective="Test required visual presence",
        allowed_video_ids=["video"],
        acceptance_policy=policy,
    )


def _hazards() -> tuple[SourceHazardSegment, ...]:
    return (
        SourceHazardSegment(
            start=0.0,
            end=10.0,
            classification=HazardClassification.EDITORIAL_CONTENT,
            confidence=0.99,
            evidence=("editorial",),
            model_identity={"model": "test"},
        ),
    )


def _timeline(
    *,
    visible_people: tuple[str, ...] = (),
    branding: tuple[str, ...] = (),
    ocr_text: tuple[str, ...] = (),
    confidence: float = 0.95,
    covered: bool = True,
) -> MultimodalTimeline:
    spans = (
        (VisualEvidenceSpan(0.0, 10.0, 5.0, "source_policy"),)
        if covered
        else (VisualEvidenceSpan(0.0, 4.0, 2.0, "source_policy"),)
    )
    return MultimodalTimeline(
        video_id="video",
        source_hash="a" * 64,
        duration=10.0,
        events=(
            MultimodalEvent(
                start=0.0,
                end=10.0,
                visible_people=visible_people,
                branding=branding,
                ocr_text=ocr_text,
                visual_summaries=("inspected source frame",),
                visual_salience=confidence,
                confidence=confidence,
            ),
        ),
        visual_evidence_spans=spans,
    )


def test_visual_presence_policy_parses_and_serializes() -> None:
    policy = AcceptancePolicy.from_dict(
        {"visual_presence": {"require_any": ["Anton", "Lovable"], "minimum_confidence": 0.8}}
    )
    assert policy.visual_presence.require_any == ("Anton", "Lovable")
    assert policy.to_dict()["visual_presence"] == {
        "require_any": ["Anton", "Lovable"],
        "minimum_confidence": 0.8,
    }


def test_required_visual_presence_passes_on_visible_person() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(visible_people=("Anton Osika",)),
    )
    assert audit.decision == GateDecision.PASS
    checks = audit.campaign_policy_checks["visual_presence_policy"]
    assert checks["matched_terms"] == ["Anton"]


def test_required_visual_presence_passes_on_branding() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(branding=("Lovable",)),
    )
    assert audit.decision == GateDecision.PASS


def test_required_visual_presence_rejects_when_absent_with_full_coverage() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(visible_people=("Ben",)),
    )
    assert audit.decision == GateDecision.REJECT
    assert "required_visual_presence_missing" in audit.reasons


def test_required_visual_presence_escalates_low_confidence_match() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable", minimum_confidence=0.8),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(branding=("Lovable",), confidence=0.6),
    )
    assert audit.decision == GateDecision.ESCALATE
    assert "required_visual_presence_uncertain" in audit.reasons


def test_required_visual_presence_escalates_incomplete_visual_coverage() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(visible_people=("Ben",), covered=False),
    )
    assert audit.decision == GateDecision.ESCALATE
    assert "required_visual_presence_uncertain" in audit.reasons


def test_required_visual_presence_rejects_substring_false_positives() -> None:
    ai_audit = evaluate_campaign_policy(
        _brief("AI"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(branding=("chair",)),
    )
    lovable_audit = evaluate_campaign_policy(
        _brief("Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(branding=("unlovable",)),
    )
    assert ai_audit.decision == GateDecision.REJECT
    assert lovable_audit.decision == GateDecision.REJECT


def test_required_visual_presence_matches_normalized_multiword_entity() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton Osika"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(visible_people=("Anton-Osika, founder",)),
    )
    assert audit.decision == GateDecision.PASS
    checks = audit.campaign_policy_checks["visual_presence_policy"]
    assert checks["matched_terms"] == ["Anton Osika"]

import pytest

from clipper.models import EditPlan, SourceSpan
from clipper.yield_policy import (
    QualityPlanGroup,
    accepted_quality_plans,
    group_quality_plans,
    quality_render_queue,
)


def _plan(concept: str, plan_id: str, score: float) -> EditPlan:
    return EditPlan(
        plan_id=plan_id,
        video_id="v",
        concept_id=concept,
        variant_id=f"variant-{plan_id}",
        hook_mode="direct",
        source_spans=(SourceSpan(0.0, 20.0),),
        hook_text=None,
        beats=(),
        caption_platform="tiktok",
        score=score,
        transcript_fingerprint="fingerprint",
    )


def test_quality_groups_are_derived_from_unique_concepts_not_a_budget() -> None:
    plans = [
        _plan("c1", "c1-low", 7.0),
        _plan("c1", "c1-high", 9.0),
        _plan("c2", "c2", 8.0),
        _plan("c3", "c3", 6.0),
    ]

    groups = group_quality_plans(plans)

    assert [group.concept_id for group in groups] == ["c1", "c2", "c3"]
    assert groups[0].primary.plan_id == "c1-high"
    assert [plan.plan_id for plan in groups[0].reserves] == ["c1-low"]


def test_quality_render_queue_keeps_reserves_adjacent_to_their_moment() -> None:
    groups = group_quality_plans(
        [
            _plan("c1", "c1-primary", 9.0),
            _plan("c1", "c1-reserve", 8.0),
            _plan("c2", "c2-primary", 7.0),
        ]
    )

    queue = quality_render_queue(groups)

    assert [(kind, plan.plan_id) for kind, plan in queue] == [
        ("primary", "c1-primary"),
        ("reserve", "c1-reserve"),
        ("primary", "c2-primary"),
    ]


def test_zero_quality_moments_is_a_valid_empty_yield() -> None:
    assert group_quality_plans([]) == ()
    assert quality_render_queue(()) == ()
    assert accepted_quality_plans([]) == ()


def test_accepted_quality_plans_never_count_two_variants_as_two_moments() -> None:
    accepted = accepted_quality_plans(
        [
            _plan("c1", "c1-a", 7.0),
            _plan("c1", "c1-b", 9.0),
            _plan("c2", "c2", 8.0),
        ]
    )

    assert [plan.plan_id for plan in accepted] == ["c1-b", "c2"]


def test_quality_plan_group_rejects_invalid_identity_and_cross_concept_plans() -> None:
    with pytest.raises(ValueError, match="requires a concept_id"):
        QualityPlanGroup("", (_plan("c1", "p1", 1.0),))
    with pytest.raises(ValueError, match="at least one plan"):
        QualityPlanGroup("c1", ())
    with pytest.raises(ValueError, match="another concept"):
        QualityPlanGroup("c1", (_plan("c2", "p2", 1.0),))

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .models import EditPlan


@dataclass(frozen=True, slots=True)
class QualityPlanGroup:
    """All viable edit alternatives for one independently worthwhile concept.

    A concept represents one quality moment. Multiple plans for that concept are recovery or
    presentation alternatives; they must never inflate the production yield.
    """

    concept_id: str
    plans: tuple[EditPlan, ...]

    def __post_init__(self) -> None:
        if not self.concept_id:
            raise ValueError("quality plan group requires a concept_id")
        if not self.plans:
            raise ValueError("quality plan group requires at least one plan")
        if any(plan.concept_id != self.concept_id for plan in self.plans):
            raise ValueError("quality plan group contains a plan from another concept")

    @property
    def primary(self) -> EditPlan:
        return self.plans[0]

    @property
    def reserves(self) -> tuple[EditPlan, ...]:
        return self.plans[1:]


def group_quality_plans(plans: Sequence[EditPlan]) -> tuple[QualityPlanGroup, ...]:
    """Return every unique quality moment with best-first recovery alternatives.

    No configured count is accepted. The number of groups is therefore the quality-derived
    production yield at the pre-render stage.
    """

    grouped: dict[str, list[EditPlan]] = defaultdict(list)
    for plan in plans:
        grouped[plan.concept_id].append(plan)

    result: list[QualityPlanGroup] = []
    for concept_id, alternatives in grouped.items():
        ranked = tuple(
            sorted(
                alternatives,
                key=lambda item: (-item.score, item.plan_id, item.variant_id),
            )
        )
        result.append(QualityPlanGroup(concept_id=concept_id, plans=ranked))

    result.sort(
        key=lambda group: (
            -group.primary.score,
            group.concept_id,
            group.primary.plan_id,
        )
    )
    return tuple(result)


def quality_render_queue(
    groups: Sequence[QualityPlanGroup],
) -> tuple[tuple[str, EditPlan], ...]:
    """Build a concept-local recovery queue without quota-filling promotion.

    Each concept's primary is immediately followed by only its own reserve variants. Runtime
    code skips remaining reserves as soon as that concept has one accepted render.
    """

    queue: list[tuple[str, EditPlan]] = []
    for group in groups:
        queue.append(("primary", group.primary))
        queue.extend(("reserve", plan) for plan in group.reserves)
    return tuple(queue)


def accepted_quality_plans(plans: Sequence[EditPlan]) -> tuple[EditPlan, ...]:
    """Normalize accepted plans to at most one result per quality moment."""

    by_concept: dict[str, EditPlan] = {}
    for plan in plans:
        current = by_concept.get(plan.concept_id)
        if current is None or plan.score > current.score:
            by_concept[plan.concept_id] = plan
    return tuple(
        sorted(
            by_concept.values(),
            key=lambda item: (-item.score, item.concept_id, item.plan_id),
        )
    )

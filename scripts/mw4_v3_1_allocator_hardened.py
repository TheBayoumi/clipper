from __future__ import annotations

from typing import Any


def _forbidden_candidate(plan: dict[str, Any]) -> bool:
    if str(plan.get("story_type", "")) == "semantic_montage":
        return True
    if str(plan.get("effect_profile", "")) == "semantic_montage":
        return True
    return any(
        str(segment.get("reason", "")) == "semantic_montage_moment"
        for segment in (plan.get("segments") or [])
    )


def install(legacy: Any) -> None:
    def select(source: str, candidates: list[dict[str, Any]], verified_count: int, config: dict[str, Any]) -> list[dict[str, Any]]:
        forbidden = [plan for plan in candidates if _forbidden_candidate(plan)]
        if forbidden:
            raise AssertionError(f"{source}: forbidden fallback candidate reached allocator")

        batch = config.get("batch_selection", {})
        maximum = int(batch.get("maximum_per_source", config.get("count_per_source_max", 8)))
        diversity = float(config.get("semantic_editor", {}).get("selection", {}).get("story_diversity_bonus", 0.06))
        selected: list[dict[str, Any]] = []
        if verified_count > 0 and bool(batch.get("require_verified_finishing_move_when_available", True)):
            finishers = [plan for plan in candidates if plan.get("finishing_move") is not None]
            if not finishers:
                raise AssertionError(f"{source}: verified Finishing Move has no valid chronological plan")
            selected.append(max(finishers, key=lambda plan: legacy._importance_score(plan, config)))
        remaining = [plan for plan in candidates if plan not in selected]
        while len(selected) < maximum:
            compatible = [plan for plan in remaining if not any(legacy._plans_conflict(plan, old, config) for old in selected)]
            if not compatible:
                break
            stories = {str(plan.get("story_type", "")) for plan in selected}
            best = max(
                compatible,
                key=lambda plan: (
                    legacy._importance_score(plan, config)
                    + (diversity if str(plan.get("story_type", "")) not in stories else 0.0),
                    str(plan.get("plan_key", "")),
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return sorted(
            selected,
            key=lambda plan: (
                plan.get("finishing_move") is None,
                -legacy._importance_score(plan, config),
                str(plan.get("plan_key", "")),
            ),
        )
    legacy._adaptive_source_selection = select

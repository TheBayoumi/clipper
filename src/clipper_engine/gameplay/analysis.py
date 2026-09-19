from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np

from . import combat, interaction, player_state, quality, refinement
from . import semantics as core

ShotSpan = core.ShotSpan
ConsolidatedEvent = core.ConsolidatedEvent
Engagement = core.Engagement
FinishingMoveSpan = core.FinishingMoveSpan
EditSegment = core.EditSegment
SemanticPlan = core.SemanticPlan
SemanticTimeline = core.SemanticTimeline
PAYOFF_KINDS = core.PAYOFF_KINDS
plan_integrity_violations = quality.plan_integrity_violations
_EPS = 1e-3


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_key(source: Path) -> str:
    stem = source.stem.lower()
    return next((key for key in ("batch2", "week2", "r1") if key in stem), stem)


def validate_configuration(config: dict[str, Any]) -> None:
    """Validate the single canonical semantic route and fail closed."""
    combat.validate_configuration(config)
    errors: list[str] = []

    if "fallback_policy" in config:
        errors.append("obsolete fallback_policy key must be removed")
    if "semantic_montage" in config.get("semantic_editor", {}):
        errors.append("obsolete semantic_montage configuration must be removed")

    for key in ("count_per_source_max", "minimum_count_per_source"):
        if key in config:
            errors.append(f"static clip-count configuration must be removed: {key}")
    batch = config.get("batch_selection", {})
    for key in (
        "candidate_pool_per_source",
        "maximum_per_source",
        "minimum_total_clips",
        "maximum_total_clips",
    ):
        if key in batch:
            errors.append(f"static batch clip-count configuration must be removed: {key}")

    semantic_editor = config.get("semantic_editor", {})
    minimum_output = _f(semantic_editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum_output = _f(semantic_editor.get("maximum_output_seconds", 12.0), 12.0)
    preferred_output = _f(semantic_editor.get("preferred_output_seconds", 11.0), 11.0)
    if abs(minimum_output - 10.0) > _EPS or abs(maximum_output - 12.0) > _EPS:
        errors.append("MW4 canonical clip duration contract must be exactly 10-12 seconds")
    if not minimum_output - _EPS <= preferred_output <= maximum_output + _EPS:
        errors.append("preferred_output_seconds must remain inside the 10-12 second contract")

    editorial = config.get("editorial", {})
    for key in (
        "finishing_move_allow_semantic_montage_continuation",
        "finishing_move_allow_verified_combat_island_continuation",
        "finishing_move_montage_max_output_seconds",
    ):
        if key in editorial:
            errors.append(f"obsolete editorial key must be removed: {key}")

    if "allow_planned_transition_for_semantic_montage" in config.get("source_integrity", {}):
        errors.append("obsolete semantic-montage source-transition key must be removed")
    if "allow_unverified_automatic" in config.get("finishing_move_detector", {}):
        errors.append(
            "automatic Finishing Move acceptance switch must be removed; "
            "discovery is diagnostics-only"
        )

    continuation = config.get("combat_state_verifier", {}).get("finishing_continuation", {})
    if continuation.get("require_verified_payoff") is not True:
        errors.append("Finishing Move continuation must require verified payoff")
    for key in (
        "minimum_verified_hostile_anchors_without_payoff",
        "minimum_retention_without_payoff",
    ):
        if key in continuation:
            errors.append(f"obsolete no-payoff continuation key must be removed: {key}")

    first_gap = _f(editorial.get("finishing_move_max_continuation_gap_seconds", 0.0))
    body_gap = _f(editorial.get("finishing_move_body_hard_cut_max_source_gap_seconds", 0.0))
    if first_gap <= 0.0:
        errors.append("finishing_move_max_continuation_gap_seconds must be positive")
    if body_gap <= 0.0 or body_gap > first_gap + _EPS:
        errors.append(
            "Finishing Move body hard-cut gap must be positive and no larger than "
            "first-continuation reach"
        )

    if errors:
        raise RuntimeError("gameplay profile configuration violation: " + "; ".join(errors))


def _verified_cut_windows(
    config: dict[str, Any],
    source_key: str,
) -> tuple[tuple[float, float], ...]:
    return quality._verified_cut_windows(config, source_key)


def _build_hardened_shots(
    source: Path,
    duration: float,
    config: dict[str, Any],
) -> tuple[ShotSpan, ...]:
    automatic = list(core.scene_cut_times(source, config))
    verified = [
        (left + right) / 2.0 for left, right in _verified_cut_windows(config, _source_key(source))
    ]
    cuts: list[float] = []
    for value in sorted(automatic + verified):
        if 0.35 < value < duration - 0.35 and all(abs(value - old) >= 0.18 for old in cuts):
            cuts.append(value)
    guard = float(config.get("semantic_analysis", {}).get("scene_change_guard_seconds", 0.12))
    boundaries = [0.0, *cuts, duration]
    shots: list[ShotSpan] = []
    for index, (left, right) in enumerate(itertools.pairwise(boundaries)):
        start = left if index == 0 else min(right, left + guard)
        end = right if index == len(boundaries) - 2 else max(start, right - guard)
        if end - start >= 1.0:
            shots.append(ShotSpan(round(start, 3), round(end, 3)))
    return tuple(shots or [ShotSpan(0.0, duration)])


def _verified_finishing_moves(
    shots: tuple[ShotSpan, ...],
    source: Path,
    config: dict[str, Any],
) -> tuple[FinishingMoveSpan, ...]:
    """Only manually verified campaign spans can enter production planning."""
    verified: list[FinishingMoveSpan] = []
    items = (
        config.get("finishing_move_detector", {})
        .get("verified_spans", {})
        .get(_source_key(source), [])
    )
    for item in items:
        start = float(item["start"])
        payoff = float(item["payoff"])
        end = float(item["end"])
        midpoint = (start + end) / 2.0
        shot_index = core._shot_index(shots, midpoint)
        shot = shots[shot_index]
        if shot.start <= midpoint <= shot.end:
            verified.append(
                FinishingMoveSpan(
                    round(max(float(shot.start), start), 3),
                    round(payoff, 3),
                    round(min(float(shot.end), end), 3),
                    shot_index,
                    0.99,
                    {
                        "visual_verification": 1.0,
                        "third_person_execution_regime": 1.0,
                    },
                )
            )
    return tuple(sorted(verified, key=lambda item: item.start))


def _verified_hostile_signal(timeline: SemanticTimeline) -> np.ndarray:
    signal = np.zeros(len(timeline.times), dtype=np.float32)
    for engagement in timeline.engagements:
        i0 = max(0, int(np.floor(float(engagement.start) * timeline.fps)))
        i1 = min(
            len(signal),
            int(np.ceil(float(engagement.end) * timeline.fps)),
        )
        signal[i0:i1] = 1.0
    return signal


def analyze_source(
    source: Path,
    config: dict[str, Any],
) -> SemanticTimeline:
    """Canonical evidence -> verification -> combat-island analysis route."""
    validate_configuration(config)
    base = core.base.analyze_source(source, config)
    shots = _build_hardened_shots(source, float(base.duration), config)
    consolidated = core.consolidate_events(base, shots)
    coarse = core.cluster_engagements(base, shots, consolidated, config)
    timeline = SemanticTimeline(
        base,
        shots,
        consolidated,
        coarse,
        _verified_finishing_moves(shots, source, config),
    )
    timeline._source_key = _source_key(source)

    refinement.annotate_timeline(source, timeline, config)
    player_state.annotate_timeline(timeline, config)
    timeline.engagements = combat.build(
        timeline,
        coarse,
        config,
        core,
        interaction.hostile_decision,
        Engagement,
    )
    timeline.signals["combat_island_gap"] = combat.gap_signal(timeline, config)
    timeline.signals["verified_hostile"] = _verified_hostile_signal(timeline)
    timeline._semantic_diagnostics = {
        "coarse_engagement_count": len(coarse),
        "verified_combat_island_count": len(timeline.engagements),
        "contact_only_can_anchor_hostile": False,
        "unknown_actor_defaults_to_hostile": False,
        "combat_islands_are_exact_gap_split_bounds": True,
        "body_segments_preserve_exact_combat_island_bounds": True,
        "alternate_planner_available": False,
        "player_state_detector": dict(getattr(timeline, "_player_state_diagnostics", {}) or {}),
    }
    return timeline


def _protected_finishing_overlap(
    timeline: SemanticTimeline,
    start: float,
    end: float,
    config: dict[str, Any],
) -> bool:
    if end <= start + _EPS:
        return False
    editorial = config["editorial"]
    lead = _f(
        editorial.get("finishing_move_opening_lead_seconds", 0.28),
        0.28,
    )
    hold = _f(
        editorial.get("finishing_move_payoff_hold_seconds", 0.45),
        0.45,
    )
    for span in timeline.finishing_moves:
        shot = timeline.shots[span.shot_index]
        protected_start = max(float(shot.start), float(span.start) - lead)
        protected_end = min(float(shot.end), float(span.end) + hold)
        if max(start, protected_start) < min(end, protected_end) - _EPS:
            return True
    return False


def _component_metric_failures(
    metrics: dict[str, float],
    config: dict[str, Any],
    *,
    require_ending: bool,
) -> list[str]:
    cfg = config["combat_state_verifier"]["finishing_continuation"]
    min_retention = max(
        _f(
            config.get("performance_targets", {}).get("retention_quality_min", 0.36),
            0.36,
        ),
        _f(cfg.get("minimum_retention_quality", 0.40), 0.40),
    )
    min_payoff = max(
        _f(
            config.get("performance_targets", {}).get("payoff_quality_min", 0.34),
            0.34,
        ),
        _f(cfg.get("minimum_payoff_quality", 0.40), 0.40),
    )
    checks = [
        (
            metrics["opening"]
            >= _f(
                cfg.get("minimum_verified_island_opening_quality", 0.42),
                0.42,
            ),
            "verified combat island opening below floor",
        ),
        (
            metrics["retention"] >= min_retention,
            "verified combat island retention below floor",
        ),
        (
            metrics["payoff"] >= min_payoff,
            "verified combat island payoff below floor",
        ),
        (
            metrics["weakest"] >= _f(cfg.get("minimum_weakest_quarter_interest", 0.30), 0.30),
            "verified combat island weakest quarter below floor",
        ),
        (
            metrics["residual"]
            <= _f(
                cfg.get(
                    "maximum_unexplained_low_interest_run_seconds",
                    0.75,
                ),
                0.75,
            ),
            "verified combat island unexplained inactivity exceeds floor",
        ),
        (
            metrics["low_fraction"]
            <= _f(
                config["semantic_editor"]["dull"].get("maximum_low_interest_fraction", 0.38),
                0.38,
            ),
            "verified combat island low-interest fraction exceeds floor",
        ),
    ]
    if require_ending:
        checks.append(
            (
                metrics["ending"] >= _f(cfg.get("minimum_ending_quality", 0.45), 0.45),
                "verified combat island ending below floor",
            )
        )
    return [message for passed, message in checks if not passed]


def _required_body_duration(
    hero: EditSegment,
    final_minimum: float,
) -> float:
    return max(0.0, final_minimum - quality._segment_duration(hero))


def _group_structure_failures(
    engagements: tuple[Engagement, ...],
    hero_end: float,
    config: dict[str, Any],
) -> list[str]:
    if not engagements:
        return ["empty Finishing Move body group"]
    max_first_gap = _f(
        config["editorial"].get("finishing_move_max_continuation_gap_seconds", 18.0),
        18.0,
    )
    hard_cut_gap = quality.finishing_body_hard_cut_gap(config)
    failures: list[str] = []
    if float(engagements[0].start) - hero_end > max_first_gap + _EPS:
        failures.append(
            "first verified body island starts beyond Finishing Move continuation reach"
        )
    for left, right in itertools.pairwise(engagements):
        gap = float(right.start) - float(left.end)
        if gap < -_EPS:
            failures.append("verified body islands overlap or reverse")
        elif gap > hard_cut_gap + _EPS:
            failures.append("verified body hard-cut source gap exceeds contract")
    return list(dict.fromkeys(failures))


def build_plans_for_source(
    timeline: SemanticTimeline,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlan]:
    from . import planning

    validate_configuration(config)
    return planning.build_plans_for_source(timeline, config, excluded, source_key)


def build_plans(
    timeline: SemanticTimeline,
    config: dict[str, Any],
    excluded: list[list[float]],
) -> list[SemanticPlan]:
    return build_plans_for_source(
        timeline,
        config,
        excluded,
        getattr(timeline, "_source_key", ""),
    )


def diagnose_source(
    timeline: SemanticTimeline,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> dict[str, Any]:
    plans = build_plans_for_source(timeline, config, excluded, source_key)
    result = dict(getattr(timeline, "_semantic_diagnostics", {}) or {})
    result.update(
        {
            "shot_count": len(timeline.shots),
            "engagement_count": len(timeline.engagements),
            "verified_finishing_move_count": len(timeline.finishing_moves),
            "candidate_count_after_semantic_gates": len(plans),
            "local_interaction_verifier": dict(
                getattr(timeline, "_local_interaction_diagnostics", {}) or {}
            ),
            "proposal_diagnostics": dict(getattr(timeline, "_proposal_diagnostics", {}) or {}),
            "finishing_move_continuation_diagnostics": list(
                getattr(timeline, "_finishing_continuation_diagnostics", []) or []
            ),
        }
    )
    return result


def self_test() -> None:
    core.architecture_self_test()
    interaction.self_test()
    combat.self_test()
    refinement.self_test()

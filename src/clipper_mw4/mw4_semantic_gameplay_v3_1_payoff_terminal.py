from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from . import mw4_semantic_gameplay_v3_1_payoff_complete as canonical

SemanticPlanV31 = canonical.SemanticPlanV31
_EPS = 1e-3

plan_integrity_violations = canonical.plan_integrity_violations
finishing_open_plans = canonical.finishing_open_plans


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _verified_payoff_times_in_region(
    timeline: Any,
    region_start: float,
    region_end: float,
    config: dict[str, Any],
) -> tuple[float, ...]:
    return tuple(
        round(float(event.time), 3)
        for event in canonical._verified_payoff_anchors(timeline, config)
        if region_start - _EPS <= float(event.time) <= region_end + _EPS
    )


def _terminal_aware_variants(
    region_start: float,
    region_end: float,
    anchor_time: float,
    payoff_times: tuple[float, ...],
    config: dict[str, Any],
) -> tuple[tuple[float, float], ...]:
    """Search source-native stories where a later verified payoff may close the clip.

    The opening anchor must remain inside every proposal. No source time is skipped,
    no duration is padded, and all normal story-quality/integrity gates run later.
    This only expands boundary search to a missing canonical case: verified kill A
    opens the action and verified kill B supplies the terminal payoff.
    """
    editor = config["semantic_editor"]
    minimum = _f(editor.get("minimum_output_seconds", 10.0), 10.0)
    maximum = _f(editor.get("maximum_output_seconds", 20.0), 20.0)
    preferred = min(
        maximum,
        _f(editor.get("preferred_output_seconds", 12.5), 12.5),
    )
    preferred_tail = _f(
        editor["ending"].get("preferred_payoff_tail_seconds", 0.45),
        0.45,
    )
    maximum_tail = _f(
        editor["ending"].get("maximum_payoff_tail_seconds", 0.75),
        0.75,
    )

    candidates = set(
        canonical._anchor_variants(
            region_start,
            region_end,
            anchor_time,
            config,
        )
    )
    action_start = max(region_start, anchor_time - 0.30)
    usable_terminals: set[float] = {anchor_time}
    for payoff_time in payoff_times:
        if payoff_time < anchor_time - _EPS:
            continue
        if payoff_time - action_start > maximum + _EPS:
            continue
        usable_terminals.add(float(payoff_time))

    preferred_endings: set[float] = set()
    for terminal_time in sorted(usable_terminals):
        for tail in (preferred_tail, maximum_tail):
            terminal_end = min(region_end, terminal_time + tail)
            preferred_endings.add(terminal_end)

            # Exact action-to-later-payoff span is the missing case in the old
            # 10/12.5/20 target-only search. It is admitted only when naturally
            # inside the unchanged campaign duration range.
            duration = terminal_end - action_start
            if minimum - _EPS <= duration <= maximum + _EPS:
                candidates.add((round(action_start, 3), round(terminal_end, 3)))

            # Also test standard campaign lengths ending cleanly on the later
            # verified payoff. The opening kill must remain inside the span.
            for target in (minimum, preferred, maximum):
                start = max(region_start, terminal_end - target)
                end = terminal_end
                if end - start < minimum - _EPS:
                    continue
                if not (start - _EPS <= anchor_time <= end + _EPS):
                    continue
                candidates.add((round(start, 3), round(end, 3)))

    endings = tuple(preferred_endings) or (anchor_time + preferred_tail,)
    ordered = sorted(
        candidates,
        key=lambda item: (
            min(abs(item[1] - end) for end in endings),
            abs((item[1] - item[0]) - preferred),
            item[0],
            item[1],
        ),
    )
    return tuple(ordered[:64])


def _anchor_plans(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    anchors = canonical._verified_payoff_anchors(timeline, config)
    qualified: dict[float, list[SemanticPlanV31]] = defaultdict(list)
    attempted = 0
    anchor_diagnostics: list[dict[str, Any]] = []

    for event in anchors:
        anchor_time = round(float(event.time), 3)
        region = canonical._verified_source_region(
            timeline,
            source_key,
            anchor_time,
            config,
        )
        if region is None:
            anchor_diagnostics.append(
                {
                    "anchor_time": anchor_time,
                    "source_region": None,
                    "terminal_payoff_candidates": [],
                    "attempted_variant_count": 0,
                    "qualified_variant_count": 0,
                    "rejections": {"inside_verified_cut_or_no_region": 1},
                }
            )
            continue

        payoff_times = _verified_payoff_times_in_region(
            timeline,
            region[0],
            region[1],
            config,
        )
        terminal_candidates = tuple(
            value
            for value in payoff_times
            if value >= anchor_time - _EPS
            and value - max(region[0], anchor_time - 0.30)
            <= _f(
                config["semantic_editor"].get("maximum_output_seconds", 20.0),
                20.0,
            )
            + _EPS
        )
        variants = _terminal_aware_variants(
            region[0],
            region[1],
            anchor_time,
            payoff_times,
            config,
        )
        rejections: Counter[str] = Counter()
        for start, end in variants:
            attempted += 1
            plan, reason = canonical._plan_for_span(
                base,
                timeline,
                config,
                excluded,
                source_key,
                start,
                end,
                anchor_time,
            )
            if plan is None:
                rejections[reason] += 1
            else:
                qualified[anchor_time].append(plan)

        anchor_diagnostics.append(
            {
                "anchor_time": anchor_time,
                "source_region": [region[0], region[1]],
                "terminal_payoff_candidates": list(terminal_candidates),
                "attempted_variant_count": len(variants),
                "qualified_variant_count": len(qualified[anchor_time]),
                "rejections": dict(sorted(rejections.items())),
            }
        )

    selected: list[SemanticPlanV31] = []
    qualified_anchor_count = 0
    for anchor_time in sorted(qualified):
        unique: list[SemanticPlanV31] = []
        seen: set[tuple[float, float]] = set()
        for plan in sorted(
            qualified[anchor_time],
            key=lambda item: item.score,
            reverse=True,
        ):
            key = (
                round(float(plan.segments[0].start), 3),
                round(float(plan.segments[0].end), 3),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(plan)
        if unique:
            qualified_anchor_count += 1
            selected.extend(unique[:4])

    existing = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    existing.update(
        {
            "verified_payoff_anchor_count": len(anchors),
            "payoff_anchor_times": [round(float(item.time), 3) for item in anchors],
            "payoff_anchor_span_variant_count": attempted,
            "payoff_anchor_qualified_anchor_count": qualified_anchor_count,
            "payoff_anchor_unrepresented_count": len(anchors) - qualified_anchor_count,
            "payoff_anchor_independent_search": True,
            "later_verified_payoff_may_close_story": True,
            "automatic_scene_cuts_are_soft_for_normal_payoff_proposals": True,
            "verified_stringout_cuts_remain_hard": True,
            "payoff_anchor_diagnostics": anchor_diagnostics,
        }
    )
    timeline._proposal_diagnostics = existing
    return selected


def normal_plans(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    anchored = _anchor_plans(
        base,
        timeline,
        config,
        excluded,
        source_key,
    )
    legacy = canonical.canonical.normal_plans(
        base,
        timeline,
        config,
        excluded,
        source_key,
    )
    merged: list[SemanticPlanV31] = []
    seen: set[tuple[float, float]] = set()
    for plan in sorted(
        anchored + legacy,
        key=lambda item: item.score,
        reverse=True,
    ):
        key = (
            round(float(plan.segments[0].start), 3),
            round(float(plan.segments[-1].end), 3),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(plan)
    return merged


def build_plans_for_source(
    base: Any,
    timeline: Any,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> list[SemanticPlanV31]:
    base.validate_configuration(config)
    normal = normal_plans(
        base,
        timeline,
        config,
        excluded,
        source_key,
    )
    finishing = canonical._one_per_finishing_move(
        canonical.canonical.finishing_open_plans(
            base,
            timeline,
            config,
            excluded,
            source_key,
        )
    )
    plans = sorted(
        finishing + normal,
        key=lambda item: item.score,
        reverse=True,
    )
    diagnostics = dict(getattr(timeline, "_proposal_diagnostics", {}) or {})
    diagnostics.update(
        {
            "proposal_architecture": "every verified payoff anchor to verified source region with later verified terminal payoff search",
            "finishing_move_variants_collapsed_per_verified_move": True,
            "automatic_scene_cuts_are_soft_for_normal_payoff_proposals": True,
            "verified_stringout_cuts_remain_hard": True,
            "semantic_montage_enabled": False,
            "fallback_planner_enabled": False,
            "threshold_reduction_used": False,
            "qualified_plan_count": len(plans),
        }
    )
    timeline._proposal_diagnostics = diagnostics
    return plans


def self_test(base: Any) -> None:
    canonical.self_test(base)
    config = {
        "semantic_editor": {
            "minimum_output_seconds": 10.0,
            "maximum_output_seconds": 20.0,
            "preferred_output_seconds": 12.5,
            "ending": {
                "preferred_payoff_tail_seconds": 0.45,
                "maximum_payoff_tail_seconds": 0.75,
            },
        }
    }
    variants = _terminal_aware_variants(
        0.0,
        30.0,
        10.0,
        (10.0, 27.0),
        config,
    )
    expected = (9.7, 27.45)
    if expected not in variants:
        raise AssertionError(
            "later verified terminal payoff did not produce natural kill-to-kill story bounds"
        )
    if any(end - start > 20.0 + _EPS for start, end in variants):
        raise AssertionError("terminal-aware proposal search exceeded unchanged campaign maximum")
    print("MW4 terminal-aware verified-payoff proposal self-test: PASS")

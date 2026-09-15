from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_ENGINE = "deterministic-gameplay-v3.1-final"
EXPECTED_EDITOR = "semantic-editor-v3.1-final"
EXPECTED_MODE = "engagement_driven_hardened_source_integrity"
EXPECTED_SOURCES = ("r1", "batch2", "week2")
_EPS = 1e-3


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_configuration(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if "fallback_policy" in config:
        errors.append("obsolete fallback_policy key must be removed")
    if "semantic_montage" in config.get("semantic_editor", {}):
        errors.append("obsolete semantic_montage configuration must be removed")

    editorial = config.get("editorial", {})
    for key in (
        "finishing_move_allow_semantic_montage_continuation",
        "finishing_move_allow_verified_combat_island_continuation",
        "finishing_move_montage_max_output_seconds",
    ):
        if key in editorial:
            errors.append(f"obsolete editorial key must be removed: {key}")
    first_gap = float(editorial.get("finishing_move_max_continuation_gap_seconds", 0.0))
    body_gap = float(editorial.get("finishing_move_body_hard_cut_max_source_gap_seconds", 0.0))
    if first_gap <= 0.0:
        errors.append("finishing_move_max_continuation_gap_seconds must be positive")
    if body_gap <= 0.0 or body_gap > first_gap + _EPS:
        errors.append("finishing_move_body_hard_cut_max_source_gap_seconds must be positive and <= first continuation reach")

    if "allow_planned_transition_for_semantic_montage" in config.get("source_integrity", {}):
        errors.append("obsolete semantic-montage source transition key must be removed")
    if "allow_unverified_automatic" in config.get("finishing_move_detector", {}):
        errors.append("automatic Finishing Move acceptance switch must be removed")

    verifier = config.get("combat_state_verifier", {})
    local = verifier.get("local_interaction_verifier", {})
    if verifier.get("enabled") is not True:
        errors.append("combat_state_verifier.enabled must be true")
    if local.get("enabled") is not True:
        errors.append("local_interaction_verifier.enabled must be true")
    continuation = verifier.get("finishing_continuation", {})
    if continuation.get("require_verified_payoff") is not True:
        errors.append("Finishing Move continuation must require verified payoff")
    for key in ("minimum_verified_hostile_anchors_without_payoff", "minimum_retention_without_payoff"):
        if key in continuation:
            errors.append(f"obsolete no-payoff continuation key must be removed: {key}")

    if errors:
        raise AssertionError("MW4 V3.1 canonical contract configuration violation: " + "; ".join(errors))


def _gate(config: dict[str, Any], story: str, metric: str, default: float) -> float:
    table = config.get("story_quality_gates", {}).get(metric, {})
    return float(table.get(story, table.get("default", default)))


def plan_key(plan: dict[str, Any]) -> str:
    finishing = plan.get("finishing_move")
    payload = {
        "story_type": str(plan.get("story_type", "")),
        "effect_profile": str(plan.get("effect_profile", "")),
        "segments": [
            [
                round(float(segment["start"]), 3),
                round(float(segment["end"]), 3),
                round(float(segment.get("speed", 1.0)), 3),
                str(segment.get("reason", "")),
            ]
            for segment in (plan.get("segments") or [])
        ],
        "finishing_move": None
        if finishing is None
        else [
            round(float(finishing["start"]), 3),
            round(float(finishing["payoff"]), 3),
            round(float(finishing["end"]), 3),
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _source_intervals(plan: dict[str, Any]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for segment in plan.get("segments") or []:
        start = float(segment["start"])
        end = float(segment["end"])
        if end > start:
            intervals.append((start, end))
    return intervals


def _merged_length(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _shared_source_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    intersections: list[tuple[float, float]] = []
    for a0, a1 in _source_intervals(left):
        for b0, b1 in _source_intervals(right):
            start, end = max(a0, b0), min(a1, b1)
            if end > start:
                intersections.append((start, end))
    return _merged_length(intersections)


def _semantic_anchor_times(plan: dict[str, Any]) -> list[float]:
    anchors: set[float] = set()
    finishing = plan.get("finishing_move")
    if finishing is not None:
        anchors.add(round(float(finishing.get("payoff", finishing["start"])), 3))
    for event in plan.get("effect_events") or []:
        if str(event.get("kind", "")) in {"outcome_like", "impact"} and "time" in event:
            anchors.add(round(float(event["time"]), 3))
    for engagement in plan.get("engagements") or []:
        for event in engagement.get("events") or []:
            kinds = {str(item) for item in (event.get("kinds") or [])}
            if kinds.intersection({"outcome_like", "impact"}) and "time" in event:
                anchors.add(round(float(event["time"]), 3))
    return sorted(anchors)


def _same_finishing_move(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    a, b = left.get("finishing_move"), right.get("finishing_move")
    if a is None or b is None:
        return False
    if abs(float(a.get("payoff", a["start"])) - float(b.get("payoff", b["start"]))) <= tolerance:
        return True
    return max(float(a["start"]), float(b["start"])) < min(float(a["end"]), float(b["end"]))


def _plans_conflict(left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]) -> bool:
    policy = config.get("duplicate_policy", {})
    anchor_tolerance = float(policy.get("semantic_anchor_tolerance_seconds", 0.35))
    max_context = float(policy.get("max_shared_context_seconds", 2.5))
    substantial_seconds = float(policy.get("substantial_overlap_seconds", 6.0))
    substantial_fraction = float(policy.get("substantial_overlap_fraction_shorter", 0.60))

    if bool(policy.get("finishing_move_exclusive", True)) and _same_finishing_move(left, right, anchor_tolerance):
        return True
    if any(
        abs(a - b) <= anchor_tolerance
        for a in _semantic_anchor_times(left)
        for b in _semantic_anchor_times(right)
    ):
        return True

    shared = _shared_source_seconds(left, right)
    if shared <= max_context + 1e-9:
        return False
    shorter = min(_merged_length(_source_intervals(left)), _merged_length(_source_intervals(right)))
    fraction = shared / shorter if shorter > 0 else 0.0
    return shared >= substantial_seconds or fraction >= substantial_fraction


def _ordering_failures(source: str, index: int, plan: dict[str, Any]) -> list[str]:
    segments = list(plan.get("segments") or [])
    if not segments:
        return [f"{source} clip {index}: plan has no source segments"]
    failures: list[str] = []
    start = float(plan.get("start", segments[0]["start"]))
    end = float(plan.get("end", segments[-1]["end"]))
    if abs(start - float(segments[0]["start"])) > _EPS:
        failures.append(f"{source} clip {index}: plan start does not match first segment")
    if abs(end - float(segments[-1]["end"])) > _EPS or end <= start:
        failures.append(f"{source} clip {index}: plan bounds are non-monotonic")
    for pos, segment in enumerate(segments):
        s0 = float(segment["start"])
        s1 = float(segment["end"])
        if s1 <= s0:
            failures.append(f"{source} clip {index}: segment {pos + 1} has invalid bounds")
        if abs(float(segment.get("speed", 1.0)) - 1.0) > 1e-6:
            failures.append(f"{source} clip {index}: segment {pos + 1} is not source-native 1.0x")
        if pos and s0 < float(segments[pos - 1]["end"]) - _EPS:
            failures.append(f"{source} clip {index}: source chronology reverses/overlaps at segment {pos + 1}")
    return failures


def _event_has_payoff(event: dict[str, Any]) -> bool:
    return bool({str(item) for item in (event.get("kinds") or [])}.intersection({"outcome_like", "impact"}))


def _canonical_terminal_end(engagement: dict[str, Any], config: dict[str, Any]) -> float | None:
    payoffs = [event for event in (engagement.get("events") or []) if _event_has_payoff(event)]
    if not payoffs:
        return None
    ending = config["semantic_editor"]["ending"]
    preferred = float(ending.get("preferred_payoff_tail_seconds", 0.45))
    maximum = float(ending.get("maximum_payoff_tail_seconds", 0.75))
    tail = min(max(0.0, preferred), max(0.0, maximum))
    return round(min(float(engagement["end"]), float(payoffs[-1]["time"]) + tail), 3)


def _finishing_plan_failures(source: str, index: int, plan: dict[str, Any], config: dict[str, Any]) -> list[str]:
    finishing = plan.get("finishing_move")
    if finishing is None:
        return []
    failures: list[str] = []
    segments = list(plan.get("segments") or [])
    engagements = list(plan.get("engagements") or [])
    editorial = config["editorial"]

    if str(plan.get("story_type", "")) != "finishing_move_open" or str(plan.get("effect_profile", "")) != "finishing_move_hero":
        failures.append(f"{source} clip {index}: Finishing Move routing is not opening hero")
    if not segments:
        return failures + [f"{source} clip {index}: Finishing Move has no segments"]

    hero = segments[0]
    if str(hero.get("reason", "")) != "finishing_move_open_hero":
        failures.append(f"{source} clip {index}: first segment is not protected Finishing Move hero")
    if not (float(hero["start"]) <= float(finishing["start"]) <= float(hero["end"])):
        failures.append(f"{source} clip {index}: verified Finishing Move is not contained in hero segment")
    delay = float(finishing["start"]) - float(hero["start"])
    if delay > 0.48 + _EPS:
        failures.append(f"{source} clip {index}: Finishing Move opens too late ({delay:.3f}s)")

    body = segments[1:]
    if not body:
        failures.append(f"{source} clip {index}: Finishing Move has no verified combat-island body")
        return failures
    if len(body) != len(engagements):
        failures.append(f"{source} clip {index}: Finishing Move body/engagement cardinality mismatch")
        return failures

    later_floor = max(float(hero["end"]), float(finishing["end"]) + 0.25)
    max_first_gap = float(editorial["finishing_move_max_continuation_gap_seconds"])
    hard_cut_gap = float(editorial["finishing_move_body_hard_cut_max_source_gap_seconds"])
    if float(body[0]["start"]) < later_floor - _EPS:
        failures.append(f"{source} clip {index}: Finishing Move continuation starts before later-content floor")
    if float(body[0]["start"]) - float(hero["end"]) > max_first_gap + _EPS:
        failures.append(f"{source} clip {index}: Finishing Move first continuation exceeds reach")

    for pos, (segment, engagement) in enumerate(zip(body, engagements)):
        if str(segment.get("reason", "")) != "verified_combat_island_body":
            failures.append(f"{source} clip {index}: body segment {pos + 1} is not a verified combat island")
        if abs(float(segment["start"]) - float(engagement["start"])) > _EPS:
            failures.append(f"{source} clip {index}: body segment {pos + 1} does not start at canonical island boundary")
        expected_end = (
            _canonical_terminal_end(engagement, config)
            if pos == len(body) - 1
            else round(float(engagement["end"]), 3)
        )
        if expected_end is None or abs(float(segment["end"]) - float(expected_end)) > _EPS:
            failures.append(f"{source} clip {index}: body segment {pos + 1} violates canonical island boundary")
        if not any(_event_has_payoff(event) for event in (engagement.get("events") or [])):
            failures.append(f"{source} clip {index}: body island {pos + 1} lacks verified payoff anchor")
        if pos:
            gap = float(segment["start"]) - float(body[pos - 1]["end"])
            if gap < -_EPS:
                failures.append(f"{source} clip {index}: Finishing Move body segments overlap/reverse")
            elif gap > hard_cut_gap + _EPS:
                failures.append(f"{source} clip {index}: Finishing Move body hard-cut source gap exceeds contract")
    return failures


def validate_plan(source: str, index: int, plan: dict[str, Any], config: dict[str, Any]) -> list[str]:
    validate_configuration(config)
    failures = _ordering_failures(source, index, plan)
    editor = config["semantic_editor"]
    story = str(plan.get("story_type", ""))
    duration = float(plan.get("output_duration", 0.0))
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_seconds", 20.0))
    if not (minimum <= duration <= maximum):
        failures.append(f"{source} clip {index}: duration out of campaign range")
    checks = (
        ("opening_quality", "opening_min", float(editor["opening"].get("minimum_quality", 0.42)), "opening gate failed"),
        ("ending_quality", "ending_min", float(editor["ending"].get("minimum_quality", 0.40)), "ending gate failed"),
        ("retention_quality", "retention_min", float(config["performance_targets"].get("retention_quality_min", 0.36)), "retention-quality gate failed"),
        ("payoff_quality", "payoff_min", float(config["performance_targets"].get("payoff_quality_min", 0.34)), "payoff-quality gate failed"),
    )
    for field, metric, default, message in checks:
        if float(plan.get(field, 0.0)) < _gate(config, story, metric, default):
            failures.append(f"{source} clip {index}: {message}")
    if float(plan.get("max_unexplained_low_interest_run_seconds", 99.0)) > float(editor["dull"].get("maximum_unexplained_low_interest_run_seconds", 0.90)):
        failures.append(f"{source} clip {index}: unexplained dull run too long")
    if float(plan.get("weakest_quarter_interest", 0.0)) < float(editor["dull"].get("minimum_weak_quarter_interest", 0.27)):
        failures.append(f"{source} clip {index}: weakest quarter gate failed")
    if float(plan.get("low_interest_fraction", 1.0)) > float(editor["dull"].get("maximum_low_interest_fraction", 0.38)):
        failures.append(f"{source} clip {index}: unexplained low-interest fraction too high")
    failures.extend(_finishing_plan_failures(source, index, plan, config))
    return list(dict.fromkeys(failures))


def _importance_score(plan: dict[str, Any], config: dict[str, Any]) -> float:
    score = float(plan.get("score", 0.0))
    if plan.get("finishing_move") is not None:
        score += float(config.get("semantic_editor", {}).get("selection", {}).get("finishing_move_bonus", 0.14))
    return score


def _adaptive_source_selection(
    source: str,
    candidates: list[dict[str, Any]],
    verified_count: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    batch = config.get("batch_selection", {})
    maximum = int(batch.get("maximum_per_source", config.get("count_per_source_max", 8)))
    diversity = float(config.get("semantic_editor", {}).get("selection", {}).get("story_diversity_bonus", 0.06))
    selected: list[dict[str, Any]] = []

    if verified_count > 0 and bool(batch.get("require_verified_finishing_move_when_available", True)):
        finishers = [plan for plan in candidates if plan.get("finishing_move") is not None]
        if not finishers:
            raise AssertionError(f"{source}: verified Finishing Move has no valid canonical plan")
        selected.append(max(finishers, key=lambda plan: (_importance_score(plan, config), str(plan.get("plan_key", "")))))

    remaining = [plan for plan in candidates if plan not in selected]
    while len(selected) < maximum:
        compatible = [plan for plan in remaining if not any(_plans_conflict(plan, old, config) for old in selected)]
        if not compatible:
            break
        stories = {str(plan.get("story_type", "")) for plan in selected}
        best = max(
            compatible,
            key=lambda plan: (
                _importance_score(plan, config)
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
            -_importance_score(plan, config),
            str(plan.get("plan_key", "")),
        ),
    )


def _apply_global_ceiling(
    selections: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    maximum_total = int(config.get("batch_selection", {}).get("maximum_total_clips", 20))
    flattened = [(source, plan) for source in EXPECTED_SOURCES for plan in selections[source]]
    if len(flattened) <= maximum_total:
        return selections
    mandatory = [item for item in flattened if item[1].get("finishing_move") is not None]
    optional = [item for item in flattened if item[1].get("finishing_move") is None]
    optional.sort(key=lambda item: (-_importance_score(item[1], config), item[0], str(item[1].get("plan_key", ""))))
    keep = mandatory + optional[: max(0, maximum_total - len(mandatory))]
    keep_keys = {(source, str(plan["plan_key"])) for source, plan in keep}
    return {
        source: [plan for plan in selections[source] if (source, str(plan["plan_key"])) in keep_keys]
        for source in EXPECTED_SOURCES
    }


def allocate_batch(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    validate_configuration(config)
    manifests = [_load(path) for path in sorted(root.rglob("*_analysis_v3_1.json"))]
    by_source = {str(item.get("source_key", "")): item for item in manifests}
    failures: list[str] = []
    if set(by_source) != set(EXPECTED_SOURCES):
        failures.append(f"analysis manifests cover {sorted(by_source)}, expected {sorted(EXPECTED_SOURCES)}")
    if failures:
        raise AssertionError("\n".join(failures))

    batch = config.get("batch_selection", {})
    pool_limit = int(batch.get("candidate_pool_per_source", 40))
    minimum_total = int(batch.get("minimum_total_clips", 1))
    maximum_total = int(batch.get("maximum_total_clips", 20))
    candidate_pools: dict[str, list[dict[str, Any]]] = {}
    verified_counts: dict[str, int] = {}
    automatic_candidate_counts: dict[str, int] = {}
    rejected_by_contract: dict[str, int] = {}

    for source in EXPECTED_SOURCES:
        manifest = by_source[source]
        if manifest.get("semantic_engine") != EXPECTED_ENGINE:
            failures.append(f"{source}: wrong semantic engine in analysis")
        if manifest.get("editorial_planner") != EXPECTED_EDITOR:
            failures.append(f"{source}: wrong editorial planner in analysis")
        if manifest.get("candidate_mode") != EXPECTED_MODE:
            failures.append(f"{source}: wrong candidate mode in analysis")
        if manifest.get("failure"):
            failures.append(f"{source}: analysis failure: {manifest['failure']}")

        valid: list[dict[str, Any]] = []
        rejected = 0
        for index, raw_plan in enumerate(list(manifest.get("candidate_pool") or []), 1):
            plan = dict(raw_plan)
            plan["plan_key"] = str(plan.get("plan_key") or plan_key(plan))
            plan_failures = validate_plan(source, index, plan, config)
            if plan_failures:
                rejected += 1
                continue
            valid.append(plan)
        valid.sort(key=lambda item: (-_importance_score(item, config), str(item.get("plan_key", ""))))
        candidate_pools[source] = valid[:pool_limit]
        rejected_by_contract[source] = rejected
        verified_counts[source] = int(manifest.get("verified_finishing_move_count", 0))
        automatic_candidate_counts[source] = len(manifest.get("automatic_finishing_move_candidates") or [])

    if failures:
        raise AssertionError("\n".join(failures))

    selections = {
        source: _adaptive_source_selection(source, candidate_pools[source], verified_counts[source], config)
        for source in EXPECTED_SOURCES
    }
    selections = _apply_global_ceiling(selections, config)
    total_selected = sum(len(items) for items in selections.values())
    if not (minimum_total <= total_selected <= maximum_total):
        raise AssertionError(
            f"adaptive allocator selected {total_selected}; allowed total is {minimum_total}..{maximum_total}"
        )

    allocations: dict[str, Any] = {}
    selected_finishers = 0
    for source in EXPECTED_SOURCES:
        selected = selections[source]
        selected_finishers += sum(1 for plan in selected if plan.get("finishing_move") is not None)
        allocations[source] = {
            "count": len(selected),
            "plan_keys": [str(plan["plan_key"]) for plan in selected],
            "plans": selected,
            "qualified_candidate_count": len(candidate_pools[source]),
            "rejected_by_contract": rejected_by_contract[source],
        }

    total_verified = sum(verified_counts.values())
    if total_verified > 0 and selected_finishers < 1:
        raise AssertionError("verified Finishing Move exists but allocator selected none")

    return {
        "version": "3.1",
        "semantic_engine": EXPECTED_ENGINE,
        "editorial_planner": EXPECTED_EDITOR,
        "candidate_mode": EXPECTED_MODE,
        "allocation_mode": "adaptive_important_scenes_before_render",
        "selection_basis": "canonical quality-qualified semantically distinct important scenes",
        "target_count": None,
        "selected_count": total_selected,
        "minimum_total_clips": minimum_total,
        "maximum_total_clips": maximum_total,
        "distribution": {source: len(selections[source]) for source in EXPECTED_SOURCES},
        "verified_finishing_move_count": total_verified,
        "selected_finishing_move_count": selected_finishers,
        "automatic_finishing_move_candidate_counts": automatic_candidate_counts,
        "source_allocations": allocations,
        "status": "PASS",
    }


def validate_source_manifest(
    manifest: dict[str, Any],
    config: dict[str, Any],
    allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_configuration(config)
    failures: list[str] = []
    source = str(manifest.get("source_key", ""))
    selected = list(manifest.get("selected") or [])
    outputs = list(manifest.get("outputs") or [])
    mode = str(manifest.get("mode", ""))

    if manifest.get("semantic_engine") != EXPECTED_ENGINE:
        failures.append(f"{source}: wrong semantic engine")
    if manifest.get("editorial_planner") != EXPECTED_EDITOR:
        failures.append(f"{source}: wrong editorial planner")
    if manifest.get("candidate_mode") != EXPECTED_MODE:
        failures.append(f"{source}: wrong candidate mode")
    if mode not in {"shadow", "production"}:
        failures.append(f"{source}: invalid mode {mode!r}")
    if manifest.get("failure"):
        failures.append(f"{source}: pipeline reported failure: {manifest['failure']}")

    maximum = int(config.get("batch_selection", {}).get("maximum_per_source", config.get("count_per_source_max", 8)))
    if not (0 <= len(selected) <= maximum):
        failures.append(f"{source}: selected {len(selected)}, allowed range is 0..{maximum}")
    if len(outputs) != len(selected):
        failures.append(f"{source}: rendered {len(outputs)} != selected {len(selected)}")

    selected_keys = [str(plan.get("plan_key") or plan_key(plan)) for plan in selected]
    if allocation is not None:
        expected = allocation.get("source_allocations", {}).get(source, {}).get("plan_keys", [])
        if selected_keys != list(expected):
            failures.append(f"{source}: rendered selection does not match pre-render allocation")

    for index, plan in enumerate(selected, 1):
        failures.extend(validate_plan(source, index, plan, config))
    if int(manifest.get("unplanned_source_cut_count", -1)) != 0:
        failures.append(f"{source}: unplanned source-cut violations present")
    verified = int(manifest.get("verified_finishing_move_count", 0))
    selected_finishers = int(manifest.get("selected_finishing_move_count", 0))
    if verified > 0 and selected_finishers < 1:
        failures.append(f"{source}: verified Finishing Move exists but none was selected")

    for index, output in enumerate(outputs, 1):
        checks = (output.get("qa") or {}).get("checks") or {}
        if not checks or not all(bool(value) for value in checks.values()):
            failures.append(f"{source} output {index}: technical QA failed")
        if mode == "production" and not checks.get("high_bitrate_near_250mbps", False):
            failures.append(f"{source} output {index}: production bitrate contract failed")
    if failures:
        raise AssertionError("\n".join(failures))
    return {
        "source_key": source,
        "mode": mode,
        "selected_count": len(selected),
        "rendered_count": len(outputs),
        "verified_finishing_move_count": verified,
        "selected_finishing_move_count": selected_finishers,
        "status": "PASS",
    }


def validate_batch(
    root: Path,
    config: dict[str, Any],
    allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_configuration(config)
    summaries = [_load(path) for path in sorted(root.rglob("*_pipeline_summary.json"))]
    failures: list[str] = []
    sources = {str(item.get("source_key", "")) for item in summaries}
    if sources != set(EXPECTED_SOURCES):
        failures.append(f"batch summaries cover {sorted(sources)}, expected {sorted(EXPECTED_SOURCES)}")
    if len(summaries) != len(EXPECTED_SOURCES):
        failures.append(f"expected exactly 3 source summaries, found {len(summaries)}")
    modes = {str(item.get("mode", "")) for item in summaries}
    if len(modes) != 1 or not modes.issubset({"shadow", "production"}):
        failures.append(f"batch has inconsistent/invalid modes: {sorted(modes)}")

    batch = config.get("batch_selection", {})
    maximum_per_source = int(batch.get("maximum_per_source", config.get("count_per_source_max", 8)))
    minimum_total = int(batch.get("minimum_total_clips", 1))
    maximum_total = int(batch.get("maximum_total_clips", 20))
    total_selected = total_rendered = total_verified = total_selected_finishers = 0

    for item in summaries:
        source = str(item.get("source_key", ""))
        if item.get("failure"):
            failures.append(f"{source}: {item['failure']}")
        selected = int(item.get("selected_count", 0))
        rendered = int(item.get("rendered_count", 0))
        if not (0 <= selected <= maximum_per_source):
            failures.append(f"{source}: selected {selected}, allowed range is 0..{maximum_per_source}")
        if rendered != selected:
            failures.append(f"{source}: rendered {rendered} != selected {selected}")
        if not bool(item.get("technical_qa_passed", False)):
            failures.append(f"{source}: technical QA did not pass")
        if int(item.get("unplanned_source_cut_count", -1)) != 0:
            failures.append(f"{source}: unplanned source cuts present")
        if allocation is not None:
            expected_count = int(allocation.get("source_allocations", {}).get(source, {}).get("count", -1))
            if selected != expected_count:
                failures.append(f"{source}: rendered count {selected} != allocation {expected_count}")
        total_selected += selected
        total_rendered += rendered
        total_verified += int(item.get("verified_finishing_move_count", 0))
        total_selected_finishers += int(item.get("selected_finishing_move_count", 0))

    if not (minimum_total <= total_selected <= maximum_total):
        failures.append(f"batch selected {total_selected}; allowed total is {minimum_total}..{maximum_total}")
    if total_rendered != total_selected:
        failures.append("batch rendered count does not equal selected count")
    if total_verified > 0 and total_selected_finishers < 1:
        failures.append("verified Finishing Move exists in batch but none was rendered")
    if allocation is not None and int(allocation.get("selected_count", -1)) != total_selected:
        failures.append("rendered batch does not match pre-render allocation total")
    if failures:
        raise AssertionError("\n".join(failures))
    return {
        "mode": next(iter(modes)),
        "sources": sorted(sources),
        "selected_count": total_selected,
        "rendered_count": total_rendered,
        "selection_mode": "adaptive_important_scenes",
        "minimum_total_clips": minimum_total,
        "maximum_total_clips": maximum_total,
        "verified_finishing_move_count": total_verified,
        "selected_finishing_move_count": total_selected_finishers,
        "allocation_enforced": allocation is not None,
        "status": "PASS",
    }


def _self_test() -> None:
    config = {
        "batch_selection": {
            "maximum_per_source": 8,
            "minimum_total_clips": 1,
            "maximum_total_clips": 20,
            "require_verified_finishing_move_when_available": True,
        },
        "duplicate_policy": {
            "semantic_anchor_tolerance_seconds": 0.35,
            "max_shared_context_seconds": 2.5,
            "substantial_overlap_seconds": 6.0,
            "substantial_overlap_fraction_shorter": 0.60,
            "finishing_move_exclusive": True,
        },
        "editorial": {
            "finishing_move_max_continuation_gap_seconds": 18.0,
            "finishing_move_body_hard_cut_max_source_gap_seconds": 18.0,
        },
        "source_integrity": {},
        "finishing_move_detector": {},
        "combat_state_verifier": {
            "enabled": True,
            "local_interaction_verifier": {"enabled": True},
            "finishing_continuation": {"require_verified_payoff": True},
        },
        "semantic_editor": {
            "minimum_output_seconds": 10.0,
            "maximum_output_seconds": 20.0,
            "opening": {"minimum_quality": 0.42},
            "ending": {"minimum_quality": 0.40, "preferred_payoff_tail_seconds": 0.45, "maximum_payoff_tail_seconds": 0.75},
            "dull": {
                "maximum_unexplained_low_interest_run_seconds": 0.90,
                "minimum_weak_quarter_interest": 0.27,
                "maximum_low_interest_fraction": 0.38,
            },
            "selection": {"story_diversity_bonus": 0.06, "finishing_move_bonus": 0.14},
        },
        "performance_targets": {"retention_quality_min": 0.36, "payoff_quality_min": 0.34},
        "story_quality_gates": {},
    }
    validate_configuration(config)

    hidden_payoff = {
        "effect_events": [{"kind": "combat_burst", "time": 4.0}],
        "engagements": [{"events": [{"kinds": ["outcome_like"], "time": 5.0}]}],
    }
    if _semantic_anchor_times(hidden_payoff) != [5.0]:
        raise AssertionError("payoff anchor hidden behind non-payoff effect event")
    if _semantic_anchor_times({"effect_events": [{"kind": "combat_burst", "time": 4.0}]}) != []:
        raise AssertionError("non-payoff event became semantic duplicate anchor")

    bad_order = {
        "start": 10.0,
        "end": 5.0,
        "segments": [{"start": 10.0, "end": 12.0}, {"start": 1.0, "end": 5.0}],
    }
    if not _ordering_failures("test", 1, bad_order):
        raise AssertionError("backward source chronology was accepted")

    high = {"plan_key": "high", "score": 1.0, "story_type": "impact_payoff", "segments": [{"start": 0.0, "end": 10.0}]}
    low_a = {"plan_key": "low-a", "score": 0.6, "story_type": "impact_payoff", "segments": [{"start": 0.0, "end": 4.0}]}
    low_b = {"plan_key": "low-b", "score": 0.6, "story_type": "impact_payoff", "segments": [{"start": 6.0, "end": 10.0}]}
    selected = _adaptive_source_selection("test", [low_a, low_b, high], 0, config)
    if [plan["plan_key"] for plan in selected] != ["high"]:
        raise AssertionError("allocator is cardinality-first instead of quality-first")

    payoff = {"time": 175.917, "kinds": ["outcome_like"]}
    finisher = {
        "story_type": "finishing_move_open",
        "effect_profile": "finishing_move_hero",
        "start": 148.07,
        "end": 176.367,
        "output_duration": 10.164,
        "opening_quality": 0.9,
        "ending_quality": 0.9,
        "retention_quality": 0.9,
        "payoff_quality": 0.9,
        "weakest_quarter_interest": 0.9,
        "low_interest_fraction": 0.0,
        "max_unexplained_low_interest_run_seconds": 0.0,
        "segments": [
            {"start": 148.07, "end": 150.5, "speed": 1.0, "reason": "finishing_move_open_hero"},
            {"start": 153.533, "end": 157.8, "speed": 1.0, "reason": "verified_combat_island_body"},
            {"start": 172.9, "end": 176.367, "speed": 1.0, "reason": "verified_combat_island_body"},
        ],
        "engagements": [
            {"start": 153.533, "end": 157.8, "events": [{"time": 156.0, "kinds": ["impact"]}]},
            {"start": 172.9, "end": 181.0, "events": [payoff]},
        ],
        "finishing_move": {"start": 148.35, "payoff": 149.45, "end": 150.05},
    }
    failures = _finishing_plan_failures("batch2", 1, finisher, config)
    if failures:
        raise AssertionError(f"canonical multi-island Finishing Move body was rejected: {failures}")

    print(json.dumps({
        "self_test": "PASS",
        "contract": "canonical-v3.1-adaptive-important-scenes-contract",
        "selection_mode": "quality-first-adaptive",
        "alternate_allocator_available": False,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--allocate-from", type=Path)
    parser.add_argument("--allocation-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.config is None:
        parser.error("--config is required")
    config = _load(args.config)
    validate_configuration(config)

    if args.allocate_from is not None:
        if args.allocation_out is None:
            parser.error("--allocation-out is required with --allocate-from")
        result = allocate_batch(args.allocate_from, config)
        _write(args.allocation_out, result)
        print(json.dumps(result, indent=2))
        return

    if (args.source_manifest is None) == (args.batch_root is None):
        parser.error("provide exactly one of --source-manifest or --batch-root")
    allocation = _load(args.allocation) if args.allocation is not None else None
    result = (
        validate_source_manifest(_load(args.source_manifest), config, allocation)
        if args.source_manifest is not None
        else validate_batch(args.batch_root, config, allocation)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

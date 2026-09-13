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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def _source_intervals(plan: dict[str, Any]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for segment in plan.get("segments") or []:
        start = float(segment["start"])
        end = float(segment["end"])
        if end > start:
            intervals.append((start, end))
    return intervals


def _merged_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
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
            start = max(a0, b0)
            end = min(a1, b1)
            if end > start:
                intersections.append((start, end))
    return _merged_length(intersections)


def _semantic_anchor_times(plan: dict[str, Any]) -> list[float]:
    finishing = plan.get("finishing_move")
    if finishing is not None:
        return [float(finishing.get("payoff", finishing.get("start", 0.0)))]

    anchors: list[float] = []
    for event in plan.get("effect_events") or []:
        kind = str(event.get("kind", ""))
        if kind in {"outcome_like", "impact"} and "time" in event:
            anchors.append(float(event["time"]))

    if not anchors:
        for engagement in plan.get("engagements") or []:
            for event in engagement.get("events") or []:
                kinds = {str(item) for item in (event.get("kinds") or [])}
                if kinds.intersection({"outcome_like", "impact"}) and "time" in event:
                    anchors.append(float(event["time"]))

    if not anchors:
        for event in plan.get("effect_events") or []:
            if "time" in event:
                anchors.append(float(event["time"]))

    return sorted(set(round(value, 3) for value in anchors))


def _same_finishing_move(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    a = left.get("finishing_move")
    b = right.get("finishing_move")
    if a is None or b is None:
        return False
    if abs(float(a.get("payoff", a["start"])) - float(b.get("payoff", b["start"]))) <= tolerance:
        return True
    return max(float(a["start"]), float(b["start"])) < min(float(a["end"]), float(b["end"]))


def _plans_conflict(left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]) -> bool:
    policy = config.get("duplicate_policy", {})
    anchor_tolerance = float(policy.get("semantic_anchor_tolerance_seconds", 0.70))
    max_context = float(policy.get("max_shared_context_seconds", 0.90))
    substantial_seconds = float(policy.get("substantial_overlap_seconds", 1.75))
    substantial_fraction = float(policy.get("substantial_overlap_fraction_shorter", 0.22))

    if bool(policy.get("finishing_move_exclusive", True)) and _same_finishing_move(
        left, right, anchor_tolerance
    ):
        return True

    left_anchors = _semantic_anchor_times(left)
    right_anchors = _semantic_anchor_times(right)
    if any(abs(a - b) <= anchor_tolerance for a in left_anchors for b in right_anchors):
        return True

    shared = _shared_source_seconds(left, right)
    if shared <= max_context + 1e-9:
        return False

    left_used = _merged_length(_source_intervals(left))
    right_used = _merged_length(_source_intervals(right))
    shorter = min(left_used, right_used)
    fraction = shared / shorter if shorter > 0 else 0.0
    return shared >= substantial_seconds or fraction >= substantial_fraction


def _finishing_plan_failures(
    source: str,
    index: int,
    plan: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    finishing = plan.get("finishing_move")
    if finishing is None:
        return []

    failures: list[str] = []
    editorial = config.get("editorial", {})
    story = str(plan.get("story_type", ""))
    profile = str(plan.get("effect_profile", ""))
    segments = list(plan.get("segments") or [])

    if story != "finishing_move_open" or profile != "finishing_move_hero":
        failures.append(f"{source} clip {index}: Finishing Move routing is not opening hero")
    if not segments:
        failures.append(f"{source} clip {index}: Finishing Move has no segments")
        return failures

    first = segments[0]
    if str(first.get("reason", "")) != "finishing_move_open_hero":
        failures.append(f"{source} clip {index}: first segment is not protected Finishing Move hero")
    if abs(float(first.get("speed", 1.0)) - 1.0) > 1e-6:
        failures.append(f"{source} clip {index}: Finishing Move choreography is not 1.0x")
    delay = (float(finishing["start"]) - float(first["start"])) / float(first.get("speed", 1.0))
    if delay > 0.48:
        failures.append(f"{source} clip {index}: Finishing Move opens too late ({delay:.3f}s)")

    if not bool(editorial.get("finishing_move_allow_semantic_montage_continuation", False)):
        if any(str(segment.get("reason", "")) == "semantic_montage_moment" for segment in segments[1:]):
            failures.append(
                f"{source} clip {index}: Finishing Move continuation uses semantic montage moments"
            )

    if len(segments) > 1:
        gap = max(0.0, float(segments[1]["start"]) - float(first["end"]))
        max_gap = float(editorial.get("finishing_move_max_continuation_gap_seconds", 18.0))
        if gap > max_gap:
            failures.append(
                f"{source} clip {index}: Finishing Move continuation gap {gap:.3f}s exceeds {max_gap:.3f}s"
            )
    return failures


def validate_plan(
    source: str,
    index: int,
    plan: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    editor = config["semantic_editor"]
    story = str(plan.get("story_type", ""))
    default_open = float(editor["opening"].get("minimum_quality", 0.42))
    default_end = float(editor["ending"].get("minimum_quality", 0.40))
    default_ret = float(config["performance_targets"].get("retention_quality_min", 0.36))
    default_pay = float(config["performance_targets"].get("payoff_quality_min", 0.34))
    max_dull = float(editor["dull"].get("maximum_unexplained_low_interest_run_seconds", 0.90))
    min_weak = float(editor["dull"].get("minimum_weak_quarter_interest", 0.27))
    max_low_fraction = float(editor["dull"].get("maximum_low_interest_fraction", 0.38))

    if not (10.0 <= float(plan.get("output_duration", 0.0)) <= 20.0):
        failures.append(f"{source} clip {index}: duration out of campaign range")
    if float(plan.get("opening_quality", 0.0)) < _gate(config, story, "opening_min", default_open):
        failures.append(f"{source} clip {index}: opening gate failed")
    if float(plan.get("ending_quality", 0.0)) < _gate(config, story, "ending_min", default_end):
        failures.append(f"{source} clip {index}: ending gate failed")
    if float(plan.get("retention_quality", 0.0)) < _gate(config, story, "retention_min", default_ret):
        failures.append(f"{source} clip {index}: retention-quality gate failed")
    if float(plan.get("payoff_quality", 0.0)) < _gate(config, story, "payoff_min", default_pay):
        failures.append(f"{source} clip {index}: payoff-quality gate failed")
    if float(plan.get("max_unexplained_low_interest_run_seconds", 99.0)) > max_dull:
        failures.append(f"{source} clip {index}: unexplained dull run too long")
    if float(plan.get("weakest_quarter_interest", 0.0)) < min_weak:
        failures.append(f"{source} clip {index}: weakest quarter gate failed")
    if float(plan.get("low_interest_fraction", 1.0)) > max_low_fraction:
        failures.append(f"{source} clip {index}: unexplained low-interest fraction too high")

    failures.extend(_finishing_plan_failures(source, index, plan, config))
    return failures


def _importance_score(plan: dict[str, Any], config: dict[str, Any]) -> float:
    selection = config.get("semantic_editor", {}).get("selection", {})
    score = float(plan.get("score", 0.0))
    if plan.get("finishing_move") is not None:
        score += float(selection.get("finishing_move_bonus", 0.14))
    if plan.get("story_type") == "semantic_montage" and bool(
        config.get("batch_selection", {}).get("prefer_normal_over_semantic_montage", True)
    ):
        score -= float(config.get("semantic_editor", {}).get("semantic_montage", {}).get("selection_penalty", 0.015))
    return score


def _adaptive_source_selection(
    source: str,
    candidates: list[dict[str, Any]],
    verified_finishing_move_count: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    batch = config.get("batch_selection", {})
    maximum = int(batch.get("maximum_per_source", config.get("count_per_source_max", 8)))
    beam_width = int(batch.get("beam_width", 2048))
    diversity_bonus = float(
        config.get("semantic_editor", {}).get("selection", {}).get("story_diversity_bonus", 0.06)
    )
    require_finisher = (
        verified_finishing_move_count > 0
        and bool(batch.get("require_verified_finishing_move_when_available", True))
    )

    ordered = sorted(
        candidates,
        key=lambda item: (
            item.get("finishing_move") is None,
            -_importance_score(item, config),
            str(item.get("plan_key", "")),
        ),
    )

    mandatory: list[int] = []
    if require_finisher:
        finishing_indices = [
            index for index, plan in enumerate(ordered)
            if plan.get("finishing_move") is not None
        ]
        if not finishing_indices:
            raise AssertionError(f"{source}: verified Finishing Move exists but no valid finishing plan survived gates")
        best_finisher = max(finishing_indices, key=lambda index: _importance_score(ordered[index], config))
        mandatory = [best_finisher]

    initial = tuple(mandatory)
    remaining = [index for index in range(len(ordered)) if index not in mandatory]
    states: list[tuple[int, ...]] = [initial]

    def state_value(state: tuple[int, ...]) -> tuple[int, float]:
        plans = [ordered[index] for index in state]
        stories = {str(plan.get("story_type", "")) for plan in plans}
        score = sum(_importance_score(plan, config) for plan in plans)
        score += diversity_bonus * len(stories)
        return len(state), score

    for candidate_index in remaining:
        candidate = ordered[candidate_index]
        expanded = list(states)
        for state in states:
            if len(state) >= maximum:
                continue
            selected = [ordered[index] for index in state]
            if any(_plans_conflict(candidate, existing, config) for existing in selected):
                continue
            expanded.append(tuple(sorted((*state, candidate_index))))

        deduped = list(dict.fromkeys(expanded))
        deduped.sort(key=state_value, reverse=True)
        states = deduped[:beam_width]

    best = max(states, key=state_value)
    selected = [ordered[index] for index in best]
    selected.sort(
        key=lambda item: (
            item.get("finishing_move") is None,
            -_importance_score(item, config),
            str(item.get("plan_key", "")),
        )
    )
    return selected


def _apply_global_ceiling(
    selections: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    batch = config.get("batch_selection", {})
    maximum_total = int(batch.get("maximum_total_clips", 20))
    flattened: list[tuple[str, dict[str, Any]]] = [
        (source, plan)
        for source in EXPECTED_SOURCES
        for plan in selections[source]
    ]
    if len(flattened) <= maximum_total:
        return selections

    mandatory = [item for item in flattened if item[1].get("finishing_move") is not None]
    optional = [item for item in flattened if item[1].get("finishing_move") is None]
    optional.sort(
        key=lambda item: (
            -_importance_score(item[1], config),
            item[0],
            str(item[1].get("plan_key", "")),
        )
    )
    keep = mandatory + optional[: max(0, maximum_total - len(mandatory))]
    keep_keys = {(source, str(plan["plan_key"])) for source, plan in keep}
    return {
        source: [
            plan for plan in selections[source]
            if (source, str(plan["plan_key"])) in keep_keys
        ]
        for source in EXPECTED_SOURCES
    }


def allocate_batch(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifests = [_load(path) for path in sorted(root.rglob("*_analysis_v3_1.json"))]
    by_source = {str(item.get("source_key", "")): item for item in manifests}
    failures: list[str] = []

    if set(by_source) != set(EXPECTED_SOURCES):
        failures.append(
            f"analysis manifests cover {sorted(by_source)}, expected {sorted(EXPECTED_SOURCES)}"
        )
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

        raw = list(manifest.get("candidate_pool") or [])
        valid: list[dict[str, Any]] = []
        rejected = 0
        for index, raw_plan in enumerate(raw, 1):
            plan = dict(raw_plan)
            plan["plan_key"] = str(plan.get("plan_key") or plan_key(plan))
            plan_failures = validate_plan(source, index, plan, config)
            if plan_failures:
                rejected += 1
                continue
            valid.append(plan)
        valid.sort(key=lambda item: -_importance_score(item, config))
        candidate_pools[source] = valid[:pool_limit]
        rejected_by_contract[source] = rejected
        verified_counts[source] = int(manifest.get("verified_finishing_move_count", 0))
        automatic_candidate_counts[source] = len(manifest.get("automatic_finishing_move_candidates") or [])

    if failures:
        raise AssertionError("\n".join(failures))

    selections = {
        source: _adaptive_source_selection(
            source,
            candidate_pools[source],
            verified_counts[source],
            config,
        )
        for source in EXPECTED_SOURCES
    }
    selections = _apply_global_ceiling(selections, config)

    total_selected = sum(len(items) for items in selections.values())
    if total_selected < minimum_total:
        raise AssertionError(
            f"adaptive allocator found only {total_selected} important unique clips; minimum safety floor is {minimum_total}"
        )
    if total_selected > maximum_total:
        raise AssertionError(
            f"adaptive allocator selected {total_selected}, above maximum_total_clips={maximum_total}"
        )

    allocations: dict[str, Any] = {}
    selected_verified_finishers = 0
    for source in EXPECTED_SOURCES:
        selected = selections[source]
        keys = [str(plan["plan_key"]) for plan in selected]
        selected_verified_finishers += sum(1 for plan in selected if plan.get("finishing_move") is not None)
        allocations[source] = {
            "count": len(selected),
            "plan_keys": keys,
            "plans": selected,
            "qualified_candidate_count": len(candidate_pools[source]),
            "rejected_by_contract": rejected_by_contract[source],
        }

    total_verified = sum(verified_counts.values())
    if total_verified > 0 and selected_verified_finishers < 1:
        raise AssertionError("verified Finishing Move exists but adaptive allocator selected none")

    return {
        "version": "3.1",
        "semantic_engine": EXPECTED_ENGINE,
        "editorial_planner": EXPECTED_EDITOR,
        "candidate_mode": EXPECTED_MODE,
        "allocation_mode": "adaptive_important_scenes_before_render",
        "selection_basis": "all quality-qualified semantically distinct important scenes, capped only by editorial safety ceilings",
        "target_count": None,
        "selected_count": total_selected,
        "minimum_total_clips": minimum_total,
        "maximum_total_clips": maximum_total,
        "distribution": {source: len(selections[source]) for source in EXPECTED_SOURCES},
        "verified_finishing_move_count": total_verified,
        "selected_finishing_move_count": selected_verified_finishers,
        "automatic_finishing_move_candidate_counts": automatic_candidate_counts,
        "source_allocations": allocations,
        "status": "PASS",
    }


def validate_source_manifest(
    manifest: dict[str, Any],
    config: dict[str, Any],
    allocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        failures.append(f"{source}: source contract requires shadow or production mode, got {mode!r}")
    if manifest.get("failure"):
        failures.append(f"{source}: pipeline reported failure: {manifest['failure']}")

    maximum = int(config.get("batch_selection", {}).get("maximum_per_source", config.get("count_per_source_max", 8)))
    if not (0 <= len(selected) <= maximum):
        failures.append(f"{source}: selected {len(selected)} clips, allowed adaptive range is 0..{maximum}")
    if len(outputs) != len(selected):
        failures.append(f"{source}: rendered {len(outputs)} != selected {len(selected)}")

    selected_keys = [str(plan.get("plan_key") or plan_key(plan)) for plan in selected]
    if allocation is not None:
        expected = allocation.get("source_allocations", {}).get(source, {}).get("plan_keys", [])
        if selected_keys != list(expected):
            failures.append(f"{source}: rendered selection does not match adaptive pre-render allocation")

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

    maximum_per_source = int(config.get("batch_selection", {}).get("maximum_per_source", config.get("count_per_source_max", 8)))
    minimum_total = int(config.get("batch_selection", {}).get("minimum_total_clips", 1))
    maximum_total = int(config.get("batch_selection", {}).get("maximum_total_clips", 20))
    total_selected = 0
    total_rendered = 0
    total_verified_finishers = 0
    total_selected_finishers = 0

    for item in summaries:
        source = str(item.get("source_key", ""))
        if item.get("failure"):
            failures.append(f"{source}: {item['failure']}")
        selected = int(item.get("selected_count", 0))
        rendered = int(item.get("rendered_count", 0))
        if not (0 <= selected <= maximum_per_source):
            failures.append(f"{source}: selected {selected}, allowed adaptive range is 0..{maximum_per_source}")
        if rendered != selected:
            failures.append(f"{source}: rendered {rendered} != selected {selected}")
        if not bool(item.get("technical_qa_passed", False)):
            failures.append(f"{source}: technical QA did not pass")
        if int(item.get("unplanned_source_cut_count", -1)) != 0:
            failures.append(f"{source}: unplanned source cuts present")
        if allocation is not None:
            expected_count = int(allocation.get("source_allocations", {}).get(source, {}).get("count", -1))
            if selected != expected_count:
                failures.append(f"{source}: rendered count {selected} does not match adaptive allocation {expected_count}")
        total_selected += selected
        total_rendered += rendered
        total_verified_finishers += int(item.get("verified_finishing_move_count", 0))
        total_selected_finishers += int(item.get("selected_finishing_move_count", 0))

    if not (minimum_total <= total_selected <= maximum_total):
        failures.append(
            f"batch selected {total_selected} clips; adaptive allowed total is {minimum_total}..{maximum_total}"
        )
    if total_rendered != total_selected:
        failures.append("batch rendered count does not equal selected count")
    if total_verified_finishers > 0 and total_selected_finishers < 1:
        failures.append("verified Finishing Move exists in batch but no finishing_move_open clip was rendered")
    if allocation is not None and int(allocation.get("selected_count", -1)) != total_selected:
        failures.append("rendered batch does not match adaptive pre-render allocation total")

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
        "verified_finishing_move_count": total_verified_finishers,
        "selected_finishing_move_count": total_selected_finishers,
        "allocation_enforced": allocation is not None,
        "status": "PASS",
    }


def _self_test() -> None:
    cfg = {
        "story_quality_gates": {
            "opening_min": {"default": 0.42, "finishing_move_open": 0.50}
        },
        "editorial": {
            "finishing_move_allow_semantic_montage_continuation": False,
            "finishing_move_max_continuation_gap_seconds": 18.0,
        },
        "duplicate_policy": {
            "semantic_anchor_tolerance_seconds": 0.70,
            "max_shared_context_seconds": 0.90,
            "substantial_overlap_seconds": 1.75,
            "substantial_overlap_fraction_shorter": 0.22,
            "finishing_move_exclusive": True,
        },
        "batch_selection": {
            "maximum_per_source": 8,
            "maximum_total_clips": 20,
            "minimum_total_clips": 1,
            "beam_width": 128,
        },
        "semantic_editor": {
            "selection": {"story_diversity_bonus": 0.06, "finishing_move_bonus": 0.14},
            "semantic_montage": {"selection_penalty": 0.015},
        },
    }

    assert _gate(cfg, "finishing_move_open", "opening_min", 0.42) == 0.50
    assert _gate(cfg, "precision_outcome", "opening_min", 0.42) == 0.42

    def sample(start: float, end: float, anchor: float, score: float = 0.8) -> dict[str, Any]:
        return {
            "story_type": "engagement_chain",
            "effect_profile": "chain_escalation",
            "segments": [{"start": start, "end": end, "speed": 1.0, "reason": "keep"}],
            "effect_events": [{"time": anchor, "kind": "outcome_like"}],
            "score": score,
        }

    assert plan_key(sample(1.0, 11.5, 6.0)) == plan_key(sample(1.0, 11.5, 6.0))
    assert not _plans_conflict(sample(0.0, 10.0, 4.0), sample(9.3, 19.3, 14.0), cfg)
    assert _plans_conflict(sample(0.0, 10.0, 4.0), sample(8.0, 18.0, 14.0), cfg)
    assert _plans_conflict(sample(0.0, 10.0, 4.0), sample(3.5, 13.5, 4.4), cfg)

    candidates = [
        sample(0.0, 10.0, 4.0, 0.9),
        sample(10.0, 20.0, 14.0, 0.85),
        sample(20.0, 30.0, 24.0, 0.8),
    ]
    for plan in candidates:
        plan["plan_key"] = plan_key(plan)
    selected = _adaptive_source_selection("r1", candidates, 0, cfg)
    assert len(selected) == 3

    print(json.dumps({
        "self_test": "PASS",
        "contract": "single-v3.1-adaptive-important-scenes-contract",
        "selection_mode": "adaptive-not-fixed-count",
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

    if args.source_manifest is not None:
        result = validate_source_manifest(_load(args.source_manifest), config, allocation)
    else:
        result = validate_batch(args.batch_root, config, allocation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

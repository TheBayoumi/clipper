from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mw4_v3_1_contract as base

KNOWN_SOURCE_ORDER = ("r1", "batch2", "week2")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_order(keys: Any) -> tuple[str, ...]:
    values = {str(item) for item in keys if str(item)}
    ordered = [key for key in KNOWN_SOURCE_ORDER if key in values]
    ordered.extend(sorted(values - set(ordered)))
    return tuple(ordered)


def _primary_scene_key(plan: dict[str, Any]) -> tuple[str, float | str]:
    finishing = plan.get("finishing_move")
    if finishing is not None:
        return ("finishing_move", round(float(finishing.get("payoff", finishing["start"])), 3))
    anchors = base._semantic_anchor_times(plan)
    if anchors:
        return ("terminal_payoff", round(float(anchors[-1]), 3))
    return ("plan", str(plan.get("plan_key") or base.plan_key(plan)))


def _diversified_candidate_pool(
    valid: list[dict[str, Any]],
    limit: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered = sorted(
        valid,
        key=lambda item: (-base._importance_score(item, config), str(item.get("plan_key", ""))),
    )
    best_per_scene: dict[tuple[str, float | str], dict[str, Any]] = {}
    for plan in ordered:
        best_per_scene.setdefault(_primary_scene_key(plan), plan)
    first_pass = sorted(
        best_per_scene.values(),
        key=lambda item: (-base._importance_score(item, config), str(item.get("plan_key", ""))),
    )
    selected = first_pass[:limit]
    selected_ids = {str(item.get("plan_key", "")) for item in selected}
    if len(selected) < limit:
        for plan in ordered:
            key = str(plan.get("plan_key", ""))
            if key in selected_ids:
                continue
            selected.append(plan)
            selected_ids.add(key)
            if len(selected) >= limit:
                break
    return selected


def _apply_global_ceiling(
    selections: dict[str, list[dict[str, Any]]],
    source_order: tuple[str, ...],
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    maximum_total = int(config.get("batch_selection", {}).get("maximum_total_clips", 20))
    flattened = [(source, plan) for source in source_order for plan in selections[source]]
    if len(flattened) <= maximum_total:
        return selections
    mandatory = [item for item in flattened if item[1].get("finishing_move") is not None]
    optional = [item for item in flattened if item[1].get("finishing_move") is None]
    optional.sort(
        key=lambda item: (
            -base._importance_score(item[1], config),
            item[0],
            str(item[1].get("plan_key", "")),
        )
    )
    keep = mandatory + optional[: max(0, maximum_total - len(mandatory))]
    keep_keys = {(source, str(plan["plan_key"])) for source, plan in keep}
    return {
        source: [
            plan for plan in selections[source] if (source, str(plan["plan_key"])) in keep_keys
        ]
        for source in source_order
    }


def allocation_rejection_diagnostics(root: Path) -> dict[str, Any]:
    manifests = [_load(path) for path in sorted(root.rglob("*_analysis_v3_1.json"))]
    by_source = {
        str(item.get("source_key", "")): item for item in manifests if item.get("source_key")
    }
    sources: dict[str, Any] = {}
    for source in _source_order(by_source):
        manifest = by_source[source]
        semantic = dict(manifest.get("diagnostics") or manifest.get("semantic_diagnostics") or {})
        sources[source] = {
            "candidate_count_after_semantic_gates": int(
                semantic.get(
                    "candidate_count_after_semantic_gates",
                    len(manifest.get("candidate_pool") or []),
                )
            ),
            "verified_finishing_move_count": int(manifest.get("verified_finishing_move_count", 0)),
            "automatic_finishing_move_candidate_count": len(
                manifest.get("automatic_finishing_move_candidates") or []
            ),
            "verified_combat_island_count": int(semantic.get("verified_combat_island_count", 0)),
            "local_interaction_verifier": dict(semantic.get("local_interaction_verifier") or {}),
            "proposal_diagnostics": dict(semantic.get("proposal_diagnostics") or {}),
            "finishing_move_continuation_diagnostics": list(
                semantic.get("finishing_move_continuation_diagnostics") or []
            ),
        }
    return {
        "version": "3.1",
        "semantic_engine": base.EXPECTED_ENGINE,
        "editorial_planner": base.EXPECTED_EDITOR,
        "candidate_mode": base.EXPECTED_MODE,
        "status": "REJECTED",
        "source_order": list(_source_order(by_source)),
        "sources": sources,
    }


def allocate_batch(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    base.validate_configuration(config)
    manifests = [_load(path) for path in sorted(root.rglob("*_analysis_v3_1.json"))]
    by_source = {
        str(item.get("source_key", "")): item for item in manifests if item.get("source_key")
    }
    if not by_source:
        raise AssertionError("no analysis manifests found")
    if len(by_source) != len(manifests):
        raise AssertionError("duplicate or missing source_key in analysis manifests")
    source_order = _source_order(by_source)

    batch = config.get("batch_selection", {})
    pool_limit = int(batch.get("candidate_pool_per_source", 40))
    minimum_total = int(batch.get("minimum_total_clips", 1))
    maximum_total = int(batch.get("maximum_total_clips", 20))
    candidate_pools: dict[str, list[dict[str, Any]]] = {}
    verified_counts: dict[str, int] = {}
    automatic_candidate_counts: dict[str, int] = {}
    rejected_by_contract: dict[str, int] = {}
    discovered_scene_counts: dict[str, int] = {}
    failures: list[str] = []

    for source in source_order:
        manifest = by_source[source]
        if manifest.get("semantic_engine") != base.EXPECTED_ENGINE:
            failures.append(f"{source}: wrong semantic engine in analysis")
        if manifest.get("editorial_planner") != base.EXPECTED_EDITOR:
            failures.append(f"{source}: wrong editorial planner in analysis")
        if manifest.get("candidate_mode") != base.EXPECTED_MODE:
            failures.append(f"{source}: wrong candidate mode in analysis")
        if manifest.get("failure"):
            failures.append(f"{source}: analysis failure: {manifest['failure']}")

        valid: list[dict[str, Any]] = []
        rejected = 0
        for index, raw_plan in enumerate(list(manifest.get("candidate_pool") or []), 1):
            plan = dict(raw_plan)
            plan["plan_key"] = str(plan.get("plan_key") or base.plan_key(plan))
            if base.validate_plan(source, index, plan, config):
                rejected += 1
                continue
            valid.append(plan)
        candidate_pools[source] = _diversified_candidate_pool(valid, pool_limit, config)
        discovered_scene_counts[source] = len({_primary_scene_key(plan) for plan in valid})
        rejected_by_contract[source] = rejected
        verified_counts[source] = int(manifest.get("verified_finishing_move_count", 0))
        automatic_candidate_counts[source] = len(
            manifest.get("automatic_finishing_move_candidates") or []
        )

    if failures:
        raise AssertionError("\n".join(failures))

    selections = {
        source: base._adaptive_source_selection(
            source, candidate_pools[source], verified_counts[source], config
        )
        for source in source_order
    }
    selections = _apply_global_ceiling(selections, source_order, config)
    total_selected = sum(len(items) for items in selections.values())
    if not (minimum_total <= total_selected <= maximum_total):
        raise AssertionError(
            f"adaptive allocator selected {total_selected}; allowed total is {minimum_total}..{maximum_total}"
        )

    allocations: dict[str, Any] = {}
    selected_finishers = 0
    for source in source_order:
        selected = selections[source]
        selected_finishers += sum(1 for plan in selected if plan.get("finishing_move") is not None)
        allocations[source] = {
            "count": len(selected),
            "plan_keys": [str(plan["plan_key"]) for plan in selected],
            "plans": selected,
            "qualified_candidate_count": len(candidate_pools[source]),
            "distinct_scene_count_before_pool_limit": discovered_scene_counts[source],
            "rejected_by_contract": rejected_by_contract[source],
        }

    total_verified = sum(verified_counts.values())
    if total_verified > 0 and selected_finishers < 1:
        raise AssertionError("verified Finishing Move exists but allocator selected none")

    return {
        "version": "3.1",
        "semantic_engine": base.EXPECTED_ENGINE,
        "editorial_planner": base.EXPECTED_EDITOR,
        "candidate_mode": base.EXPECTED_MODE,
        "allocation_mode": "adaptive_important_scenes_before_render",
        "selection_basis": "canonical quality-qualified semantically distinct important scenes with distinct-payoff pool preservation",
        "target_count": None,
        "source_order": list(source_order),
        "selected_count": total_selected,
        "minimum_total_clips": minimum_total,
        "maximum_total_clips": maximum_total,
        "distribution": {source: len(selections[source]) for source in source_order},
        "verified_finishing_move_count": total_verified,
        "selected_finishing_move_count": selected_finishers,
        "automatic_finishing_move_candidate_counts": automatic_candidate_counts,
        "source_allocations": allocations,
        "status": "PASS",
    }


def validate_batch(
    root: Path, config: dict[str, Any], allocation: dict[str, Any] | None = None
) -> dict[str, Any]:
    base.validate_configuration(config)
    summaries = [_load(path) for path in sorted(root.rglob("*_pipeline_summary.json"))]
    source_order = (
        tuple(allocation.get("source_order") or [])
        if allocation
        else _source_order(str(item.get("source_key", "")) for item in summaries)
    )
    expected = set(source_order)
    actual = {str(item.get("source_key", "")) for item in summaries}
    failures: list[str] = []
    if actual != expected:
        failures.append(f"batch summaries cover {sorted(actual)}, expected {sorted(expected)}")
    if len(summaries) != len(expected):
        failures.append(
            f"expected exactly {len(expected)} source summaries, found {len(summaries)}"
        )
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
            failures.append(
                f"{source}: selected {selected}, allowed range is 0..{maximum_per_source}"
            )
        if rendered != selected:
            failures.append(f"{source}: rendered {rendered} != selected {selected}")
        if not bool(item.get("technical_qa_passed", False)):
            failures.append(f"{source}: technical QA did not pass")
        if int(item.get("unplanned_source_cut_count", -1)) != 0:
            failures.append(f"{source}: unplanned source cuts present")
        if allocation is not None:
            expected_count = int(
                allocation.get("source_allocations", {}).get(source, {}).get("count", -1)
            )
            if selected != expected_count:
                failures.append(
                    f"{source}: rendered count {selected} != allocation {expected_count}"
                )
        total_selected += selected
        total_rendered += rendered
        total_verified += int(item.get("verified_finishing_move_count", 0))
        total_selected_finishers += int(item.get("selected_finishing_move_count", 0))

    if not (minimum_total <= total_selected <= maximum_total):
        failures.append(
            f"batch selected {total_selected}; allowed total is {minimum_total}..{maximum_total}"
        )
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
        "sources": list(source_order),
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
    base._self_test()
    sample = [
        {"plan_key": "a", "score": 0.9, "effect_events": [{"kind": "outcome_like", "time": 10.0}]},
        {"plan_key": "b", "score": 0.8, "effect_events": [{"kind": "outcome_like", "time": 30.0}]},
        {"plan_key": "c", "score": 0.95, "effect_events": [{"kind": "outcome_like", "time": 10.0}]},
    ]
    config = {"semantic_editor": {"selection": {"finishing_move_bonus": 0.14}}}
    pool = _diversified_candidate_pool(sample, 2, config)
    if {_primary_scene_key(item) for item in pool} != {
        ("terminal_payoff", 10.0),
        ("terminal_payoff", 30.0),
    }:
        raise AssertionError(
            "candidate pool did not preserve distinct payoff scenes before score-fill"
        )
    print(
        json.dumps(
            {
                "self_test": "PASS",
                "dynamic_source_set": True,
                "distinct_payoff_pool_preservation": True,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--allocate-from", type=Path)
    parser.add_argument("--allocation-out", type=Path)
    parser.add_argument("--rejection-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.config is None:
        parser.error("--config is required")
    config = _load(args.config)
    base.validate_configuration(config)
    if args.allocate_from is not None:
        if args.allocation_out is None:
            parser.error("--allocation-out is required with --allocate-from")
        try:
            result = allocate_batch(args.allocate_from, config)
        except AssertionError as exc:
            rejection = allocation_rejection_diagnostics(args.allocate_from)
            rejection["reason"] = str(exc)
            if args.rejection_out is not None:
                _write(args.rejection_out, rejection)
            print(json.dumps(rejection, indent=2))
            raise
        _write(args.allocation_out, result)
        print(json.dumps(result, indent=2))
        return
    if (args.source_manifest is None) == (args.batch_root is None):
        parser.error("provide exactly one of --source-manifest or --batch-root")
    allocation = _load(args.allocation) if args.allocation is not None else None
    result = (
        base.validate_source_manifest(_load(args.source_manifest), config, allocation)
        if args.source_manifest is not None
        else validate_batch(args.batch_root, config, allocation)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

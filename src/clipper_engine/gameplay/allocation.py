from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import contract


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_order(keys: Any) -> tuple[str, ...]:
    preferred = ("r1", "batch2", "week2")
    values = {str(item) for item in keys if str(item)}
    ordered = [key for key in preferred if key in values]
    ordered.extend(sorted(values - set(ordered)))
    return tuple(ordered)


def _required_outcome_anchors(manifest: dict[str, Any]) -> tuple[float, ...]:
    semantic = dict(manifest.get("diagnostics") or manifest.get("semantic_diagnostics") or {})
    local = dict(semantic.get("local_interaction_verifier") or {})
    anchors = {
        round(float(item["time"]), 3)
        for item in (local.get("events") or [])
        if item.get("confirmed") and "outcome_like" in (item.get("kinds") or [])
    }
    return tuple(sorted(anchors))


def _candidate_coverage(plan: dict[str, Any]) -> frozenset[float]:
    explicit = plan.get("covered_outcome_anchor_times") or []
    if explicit:
        return frozenset(round(float(item), 3) for item in explicit)

    anchors: set[float] = set()
    for event in plan.get("effect_events") or []:
        if event.get("kind") == "outcome_like" and "time" in event:
            anchors.add(round(float(event["time"]), 3))
    for engagement in plan.get("engagements") or []:
        for event in engagement.get("events") or []:
            if "outcome_like" in (event.get("kinds") or []) and "time" in event:
                anchors.add(round(float(event["time"]), 3))
    return frozenset(anchors)


def _anchor_diagnostics(manifest: dict[str, Any]) -> dict[float, dict[str, Any]]:
    semantic = dict(manifest.get("diagnostics") or manifest.get("semantic_diagnostics") or {})
    proposal = dict(semantic.get("proposal_diagnostics") or {})
    return {
        round(float(item["anchor_time"]), 3): dict(item)
        for item in (proposal.get("payoff_anchor_diagnostics") or [])
        if "anchor_time" in item
    }


def _selection_objective(
    selected: tuple[int, ...],
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[int, float, float, tuple[str, ...]]:
    plans = [candidates[index] for index in selected]
    shared = 0.0
    for left_index, left in enumerate(plans):
        for right in plans[left_index + 1 :]:
            shared += contract._shared_source_seconds(left, right)
    quality = sum(contract._importance_score(plan, config) for plan in plans)
    keys = tuple(sorted(str(plan.get("plan_key", "")) for plan in plans))
    return (len(plans), round(shared, 6), -round(quality, 8), keys)


def _solve_source(
    source: str,
    candidates: list[dict[str, Any]],
    required_outcomes: set[float],
    require_finishing_move: bool,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    coverage = [_candidate_coverage(plan) for plan in candidates]
    finishing = [plan.get("finishing_move") is not None for plan in candidates]

    outcome_union: set[float] = set()
    for item in coverage:
        outcome_union.update(item)
    missing = sorted(required_outcomes - outcome_union)
    if missing:
        raise AssertionError(
            f"{source}: qualified allocation input cannot cover outcome anchors {missing}"
        )
    if require_finishing_move and not any(finishing):
        raise AssertionError(f"{source}: verified Finishing Move has no valid candidate")

    conflicts = [
        [
            left != right and contract.plans_conflict(candidates[left], candidates[right], config)
            for right in range(len(candidates))
        ]
        for left in range(len(candidates))
    ]

    token_options: dict[tuple[str, float | str], tuple[int, ...]] = {}
    for anchor in sorted(required_outcomes):
        token_options[("outcome", anchor)] = tuple(
            index for index, item in enumerate(coverage) if anchor in item
        )
    if require_finishing_move:
        token_options[("finishing", source)] = tuple(
            index for index, is_finishing in enumerate(finishing) if is_finishing
        )

    required_tokens = frozenset(token_options)
    candidate_tokens: list[frozenset[tuple[str, float | str]]] = []
    for index in range(len(candidates)):
        tokens: set[tuple[str, float | str]] = {
            ("outcome", anchor) for anchor in coverage[index] if anchor in required_outcomes
        }
        if require_finishing_move and finishing[index]:
            tokens.add(("finishing", source))
        candidate_tokens.append(frozenset(tokens))

    best: tuple[int, ...] | None = None
    best_objective: tuple[int, float, float, tuple[str, ...]] | None = None
    visited: set[frozenset[int]] = set()

    def search(selected: frozenset[int], covered: frozenset[tuple[str, float | str]]) -> None:
        nonlocal best, best_objective
        if selected in visited:
            return
        visited.add(selected)

        if required_tokens.issubset(covered):
            ordered = tuple(sorted(selected))
            objective = _selection_objective(ordered, candidates, config)
            if best_objective is None or objective < best_objective:
                best = ordered
                best_objective = objective
            return

        if best is not None and len(selected) >= len(best):
            return

        uncovered = required_tokens - covered
        viable: list[tuple[int, tuple[str, float | str], tuple[int, ...]]] = []
        for token in uncovered:
            options = tuple(
                index
                for index in token_options[token]
                if index not in selected
                and not any(conflicts[index][chosen] for chosen in selected)
            )
            if not options:
                return
            viable.append((len(options), token, options))
        _, _, options = min(viable, key=lambda item: (item[0], str(item[1])))

        for index in sorted(
            options,
            key=lambda item: (
                -len(candidate_tokens[item] & uncovered),
                -contract._importance_score(candidates[item], config),
                str(candidates[item].get("plan_key", "")),
            ),
        ):
            search(
                selected | {index},
                covered | candidate_tokens[index],
            )

    search(frozenset(), frozenset())
    if best is None:
        conflict_summary = {
            str(candidates[index].get("plan_key", "")): [
                str(candidates[other].get("plan_key", ""))
                for other in range(len(candidates))
                if conflicts[index][other]
            ]
            for index in range(len(candidates))
        }
        raise AssertionError(
            f"{source}: no conflict-free allocation covers all qualified outcomes; "
            f"required={sorted(required_outcomes)} conflicts={conflict_summary}"
        )

    return [candidates[index] for index in best]


def allocation_rejection_diagnostics(root: Path) -> dict[str, Any]:
    manifests = [_load(path) for path in sorted(root.rglob("*_analysis.json"))]
    by_source = {
        str(item.get("source_key", "")): item for item in manifests if item.get("source_key")
    }
    return {
        "schema_version": 1,
        "status": "REJECTED",
        "sources": {
            source: {
                "required_outcome_anchors": list(_required_outcome_anchors(manifest)),
                "proposal_diagnostics": dict(
                    (manifest.get("diagnostics") or {}).get("proposal_diagnostics") or {}
                ),
            }
            for source, manifest in by_source.items()
        },
    }


def allocate(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    contract.validate_configuration(config)
    manifests = [_load(path) for path in sorted(root.rglob("*_analysis.json"))]
    by_source = {
        str(item.get("source_key", "")): item for item in manifests if item.get("source_key")
    }
    if not by_source:
        raise AssertionError("no analysis manifests found")
    if len(by_source) != len(manifests):
        raise AssertionError("duplicate or missing source_key in analysis manifests")

    source_order = _source_order(by_source)
    allocations: dict[str, Any] = {}
    total_selected = 0
    total_verified_finishers = 0
    total_selected_finishers = 0

    for source in source_order:
        manifest = by_source[source]
        if manifest.get("semantic_engine") != contract.EXPECTED_ENGINE:
            raise AssertionError(f"{source}: wrong semantic engine in analysis")
        if manifest.get("editorial_planner") != contract.EXPECTED_EDITOR:
            raise AssertionError(f"{source}: wrong editorial planner in analysis")
        if manifest.get("candidate_mode") != contract.EXPECTED_MODE:
            raise AssertionError(f"{source}: wrong candidate mode in analysis")
        if manifest.get("failure"):
            raise AssertionError(f"{source}: analysis failure: {manifest['failure']}")

        valid: list[dict[str, Any]] = []
        rejected_by_contract = 0
        for index, raw_plan in enumerate(manifest.get("candidate_pool") or [], 1):
            plan = dict(raw_plan)
            plan["plan_key"] = str(plan.get("plan_key") or contract.plan_key(plan))
            failures = contract.validate_plan(source, index, plan, config)
            if failures:
                rejected_by_contract += 1
                continue
            valid.append(plan)

        required = set(_required_outcome_anchors(manifest))
        union = set().union(*(_candidate_coverage(plan) for plan in valid)) if valid else set()
        qualified = required & union
        diagnostics = _anchor_diagnostics(manifest)
        dispositions: list[dict[str, Any]] = []
        unresolved: list[float] = []
        for anchor in sorted(required):
            if anchor in qualified:
                dispositions.append(
                    {
                        "anchor_time": anchor,
                        "disposition": "qualified_for_allocation",
                    }
                )
                continue
            detail = diagnostics.get(anchor)
            if detail is None:
                unresolved.append(anchor)
                continue
            dispositions.append(
                {
                    "anchor_time": anchor,
                    "disposition": "no_admissible_candidate",
                    "attempted_variant_count": int(detail.get("attempted_variant_count", 0)),
                    "rejections": dict(detail.get("rejections") or {}),
                }
            )
        if unresolved:
            raise AssertionError(
                f"{source}: verified outcomes lack proposal disposition: {unresolved}"
            )

        verified_finishers = int(manifest.get("verified_finishing_move_count", 0))
        require_finisher = bool(
            verified_finishers
            and config.get("batch_selection", {}).get(
                "require_verified_finishing_move_when_available", True
            )
        )
        selected = _solve_source(source, valid, qualified, require_finisher, config)
        selected_coverage = set().union(
            *(_candidate_coverage(plan) for plan in selected)
        ) if selected else set()
        uncovered_qualified = sorted(qualified - selected_coverage)
        if uncovered_qualified:
            raise AssertionError(
                f"{source}: allocation left qualified outcomes uncovered: {uncovered_qualified}"
            )

        for item in dispositions:
            if item["disposition"] == "qualified_for_allocation":
                item["disposition"] = "covered_by_selected_clip"
                item["selected_plan_keys"] = [
                    str(plan["plan_key"])
                    for plan in selected
                    if float(item["anchor_time"]) in _candidate_coverage(plan)
                ]

        pair_conflicts = [
            (str(left["plan_key"]), str(right["plan_key"]))
            for index, left in enumerate(selected)
            for right in selected[index + 1 :]
            if contract.plans_conflict(left, right, config)
        ]
        if pair_conflicts:
            raise AssertionError(f"{source}: selected plans conflict: {pair_conflicts}")

        selected_finishers = sum(1 for plan in selected if plan.get("finishing_move") is not None)
        allocations[source] = {
            "count": len(selected),
            "plan_keys": [str(plan["plan_key"]) for plan in selected],
            "plans": selected,
            "qualified_candidate_count": len(valid),
            "rejected_by_contract": rejected_by_contract,
            "required_outcome_anchor_count": len(required),
            "qualified_outcome_anchor_count": len(qualified),
            "covered_qualified_outcome_anchor_count": len(qualified),
            "outcome_dispositions": dispositions,
            "selected_pair_conflict_count": 0,
        }
        total_selected += len(selected)
        total_verified_finishers += verified_finishers
        total_selected_finishers += selected_finishers

    if total_verified_finishers > 0 and total_selected_finishers < 1:
        raise AssertionError("verified Finishing Move exists but allocation selected none")

    return {
        "schema_version": 1,
        "semantic_engine": contract.EXPECTED_ENGINE,
        "editorial_planner": contract.EXPECTED_EDITOR,
        "candidate_mode": contract.EXPECTED_MODE,
        "allocation_mode": "coverage_conflict_before_render",
        "selection_basis": (
            "cover every qualified verified outcome, minimize clip count and source overlap, "
            "then maximize editorial quality deterministically"
        ),
        "source_order": list(source_order),
        "target_count": total_selected,
        "selected_count": total_selected,
        "duration_contract": {"minimum_seconds": 10.0, "maximum_seconds": 12.0},
        "distribution": {
            source: int(allocations[source]["count"]) for source in source_order
        },
        "verified_finishing_move_count": total_verified_finishers,
        "selected_finishing_move_count": total_selected_finishers,
        "source_allocations": allocations,
        "status": "PASS",
    }

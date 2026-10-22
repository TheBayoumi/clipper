from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .. import build_identity
from . import contract

Anchor = float | str


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
    if "verified_hostile_event_assignment" not in local:
        raise AssertionError("analysis manifest missing canonical verified hostile assignment")
    anchors = {
        round(float(item["time"]), 3)
        for item in (local.get("verified_hostile_event_assignment") or [])
        if item.get("hostile_verified") and "outcome_like" in (item.get("kinds") or [])
    }
    return tuple(sorted(anchors))


def _required_anchors(manifest: dict[str, Any]) -> tuple[Anchor, ...]:
    semantic = list(manifest.get("required_semantic_anchors") or [])
    ids = [str(item.get("id") or "") for item in semantic]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise AssertionError("required_semantic_anchors require unique non-empty ids")
    diagnostics = dict(manifest.get("diagnostics") or {})
    local = dict(diagnostics.get("local_interaction_verifier") or {})
    outcomes = (
        _required_outcome_anchors(manifest) if "verified_hostile_event_assignment" in local else ()
    )
    return (*sorted(ids), *outcomes)


def _candidate_coverage(plan: dict[str, Any]) -> frozenset[Anchor]:
    """Use only explicitly declared coverage produced by canonical source verification.

    Reconstructing outcome anchors from engagement or effect-event text can turn
    unverified model claims into allocation requirements or false source coverage.
    An explicit empty list means no directly verified outcome in this framing.
    """
    raw_outcomes = plan.get("covered_outcome_anchor_times")
    raw_semantics = plan.get("covered_semantic_anchor_ids")
    if not isinstance(raw_outcomes, list):
        raise AssertionError("candidate missing explicit covered_outcome_anchor_times")
    if not isinstance(raw_semantics, list):
        raise AssertionError("candidate missing explicit covered_semantic_anchor_ids")
    if any(not isinstance(anchor, str) or not anchor for anchor in raw_semantics):
        raise AssertionError("candidate semantic anchor IDs must be nonempty strings")
    if any(
        isinstance(anchor, bool)
        or not isinstance(anchor, int | float)
        or not math.isfinite(anchor)
        or anchor < 0
        for anchor in raw_outcomes
    ):
        raise AssertionError("candidate outcome anchors must be finite positive source times")
    return frozenset(
        [str(item) for item in plan["covered_semantic_anchor_ids"]]
        + [round(float(item), 3) for item in plan["covered_outcome_anchor_times"]]
    )


def _validate_declared_coverage(plan: dict[str, Any], required: set[Anchor]) -> list[str]:
    """Validate source-local, explicitly verified anchor coverage before allocation."""
    problems: list[str] = []
    try:
        covered = _candidate_coverage(plan)
    except (AssertionError, ValueError, TypeError, OverflowError) as exc:
        return [f"explicit_anchor_coverage_invalid:{exc}"]
    if not covered.issubset(required):
        problems.append("unverified_or_foreign_anchor_coverage")
    segments = plan.get("segments") or []
    if not isinstance(segments, list):
        return ["invalid_source_segments"]
    try:
        intervals = [
            (float(item["start"]), float(item["end"]))
            for item in segments
            if isinstance(item, dict) and "start" in item and "end" in item
        ]
    except (TypeError, ValueError, OverflowError):
        return ["invalid_source_segments"]
    if not intervals or any(
        not math.isfinite(left) or not math.isfinite(right) or right <= left
        for left, right in intervals
    ):
        return ["invalid_source_segments"]
    for anchor in covered:
        if isinstance(anchor, float) and (
            not math.isfinite(anchor)
            or not any(left - 0.001 <= anchor <= right + 0.001 for left, right in intervals)
        ):
            problems.append("outcome_anchor_outside_actual_source_segments")
            break
    if len(plan["covered_outcome_anchor_times"]) != len(
        set(map(str, plan["covered_outcome_anchor_times"]))
    ) or len(plan["covered_semantic_anchor_ids"]) != len(
        set(map(str, plan["covered_semantic_anchor_ids"]))
    ):
        problems.append("duplicate_declared_anchor_coverage")
    return sorted(set(problems))


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
    required_outcomes: set[Anchor],
    require_finishing_move: bool,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    coverage = [_candidate_coverage(plan) for plan in candidates]
    finishing = [plan.get("finishing_move") is not None for plan in candidates]

    coverage_union: set[Anchor] = set()
    for item in coverage:
        coverage_union.update(item)
    missing = sorted(required_outcomes - coverage_union, key=str)
    if missing:
        raise AssertionError(
            f"{source}: qualified allocation input cannot cover semantic anchors {missing}"
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

    token_options: dict[tuple[str, Anchor], tuple[int, ...]] = {}
    for anchor in sorted(required_outcomes, key=str):
        token_options[("anchor", anchor)] = tuple(
            index for index, item in enumerate(coverage) if anchor in item
        )
    if require_finishing_move:
        token_options[("finishing", source)] = tuple(
            index for index, is_finishing in enumerate(finishing) if is_finishing
        )

    required_tokens = frozenset(token_options)
    candidate_tokens: list[frozenset[tuple[str, Anchor]]] = []
    for index in range(len(candidates)):
        tokens: set[tuple[str, Anchor]] = {
            ("anchor", anchor) for anchor in coverage[index] if anchor in required_outcomes
        }
        if require_finishing_move and finishing[index]:
            tokens.add(("finishing", source))
        candidate_tokens.append(frozenset(tokens))

    best: tuple[int, ...] | None = None
    best_objective: tuple[int, float, float, tuple[str, ...]] | None = None
    visited: set[frozenset[int]] = set()

    def search(selected: frozenset[int], covered: frozenset[tuple[str, Anchor]]) -> None:
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
        viable: list[tuple[int, tuple[str, Anchor], tuple[int, ...]]] = []
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
            search(selected | {index}, covered | candidate_tokens[index])

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
            f"{source}: no conflict-free allocation covers all qualified semantic anchors; "
            f"required={sorted(required_outcomes, key=str)} conflicts={conflict_summary}"
        )

    return [candidates[index] for index in best]


def allocation_rejection_diagnostics(root: Path) -> dict[str, Any]:
    manifests = [_load(path) for path in sorted(root.rglob("*_analysis.json"))]
    by_source = {
        str(item.get("source_key", "")): item for item in manifests if item.get("source_key")
    }
    sources: dict[str, Any] = {}
    for source, manifest in by_source.items():
        if manifest.get("required_semantic_anchors"):
            sources[source] = {
                "required_semantic_anchors": list(manifest.get("required_semantic_anchors") or []),
                "spoken_content_diagnostics": dict(
                    (manifest.get("diagnostics") or {}).get("spoken_content") or {}
                ),
            }
        else:
            sources[source] = {
                "required_outcome_anchors": list(_required_outcome_anchors(manifest)),
                "proposal_diagnostics": dict(
                    (manifest.get("diagnostics") or {}).get("proposal_diagnostics") or {}
                ),
            }
    return {"schema_version": 2, "status": "REJECTED", "sources": sources}


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
    profile_sources = [
        str(item.get("source_key") or "")
        for item in ((config.get("source_profile") or {}).get("assets") or [])
    ]
    if profile_sources:
        if len(set(profile_sources)) != len(profile_sources) or not all(profile_sources):
            raise AssertionError("source profile declares duplicate or empty asset keys")
        if set(profile_sources) != set(by_source):
            raise AssertionError(
                f"incomplete configured-source qualification: "
                f"missing={sorted(set(profile_sources) - set(by_source))}, "
                f"unexpected={sorted(set(by_source) - set(profile_sources))}"
            )

    source_order = _source_order(by_source)
    allocations: dict[str, Any] = {}
    total_selected = 0
    total_verified_finishers = 0
    total_selected_finishers = 0
    expected_engine = contract.expected_semantic_engine(config)
    expected_editor = contract.expected_editorial_planner(config)
    expected_mode = contract.expected_candidate_mode(config)

    for source in source_order:
        manifest = by_source[source]
        if manifest.get("semantic_engine") != expected_engine:
            raise AssertionError(f"{source}: wrong semantic engine in analysis")
        if manifest.get("editorial_planner") != expected_editor:
            raise AssertionError(f"{source}: wrong editorial planner in analysis")
        if manifest.get("candidate_mode") != expected_mode:
            raise AssertionError(f"{source}: wrong candidate mode in analysis")
        if manifest.get("failure"):
            raise AssertionError(f"{source}: analysis failure: {manifest['failure']}")
        build_identity.validate_artifact_sha(
            manifest.get("execution_sha"), stage=f"{source} analysis"
        )

        required = set(_required_anchors(manifest))
        valid: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for index, raw_plan in enumerate(manifest.get("candidate_pool") or [], 1):
            plan = dict(raw_plan)
            expected_key = contract.plan_key(plan)
            reasons: list[str] = []
            if plan.get("plan_key") != expected_key:
                reasons.append("candidate_plan_key_identity_mismatch")
            reasons.extend(_validate_declared_coverage(plan, required))
            if not reasons:
                reasons.extend(contract.validate_plan(source, index, plan, config))
            if reasons:
                rejected.append(
                    {
                        "candidate_index": index,
                        "plan_key": expected_key,
                        "reasons": sorted(set(reasons)),
                    }
                )
                continue
            valid.append(plan)
        rejected_by_contract = len(rejected)

        union = set().union(*(_candidate_coverage(plan) for plan in valid)) if valid else set()
        has_semantic_anchors = bool(manifest.get("required_semantic_anchors"))
        semantic_details = {
            str(item.get("id")): dict(item)
            for item in (manifest.get("required_semantic_anchors") or [])
            if item.get("id")
        }
        qualified: set[Anchor] = {
            anchor for anchor in required if isinstance(anchor, float) and anchor in union
        }
        if has_semantic_anchors:
            unresolved_semantic: list[Anchor] = []
            for anchor in sorted(semantic_details, key=str):
                detail = semantic_details.get(str(anchor))
                if detail is None:
                    unresolved_semantic.append(anchor)
                    continue
                disposition = str(detail.get("disposition") or "")
                if disposition == "search_required":
                    if anchor not in union:
                        raise AssertionError(
                            f"{source}: qualified semantic anchor has no legal candidate: {anchor}"
                        )
                    qualified.add(anchor)
                elif disposition == "no_admissible_candidate":
                    if int(detail.get("legal_candidate_count", 0)) > 0:
                        raise AssertionError(
                            f"{source}: semantic anchor {anchor} had legal proposals "
                            "but was incorrectly marked no_admissible_candidate"
                        )
                else:
                    unresolved_semantic.append(anchor)
            if unresolved_semantic:
                raise AssertionError(
                    f"{source}: semantic anchors lack explicit discovery disposition: "
                    f"{unresolved_semantic}"
                )

        dispositions: list[dict[str, Any]] = []
        unresolved: list[Anchor] = []
        diagnostics = _anchor_diagnostics(manifest)
        for anchor in sorted(required, key=str):
            if anchor in qualified:
                dispositions.append({"anchor": anchor, "disposition": "qualified_for_allocation"})
                continue
            if isinstance(anchor, str) and anchor in semantic_details:
                detail = semantic_details.get(str(anchor))
                if detail is None:
                    unresolved.append(anchor)
                    continue
                dispositions.append(
                    {
                        "anchor": anchor,
                        "disposition": "no_admissible_candidate",
                        "detail": detail,
                    }
                )
                continue
            detail = diagnostics.get(float(anchor))
            if detail is None:
                unresolved.append(anchor)
                continue
            if int(detail.get("qualified_variant_count", 0)) > 0:
                raise AssertionError(
                    f"{source}: verified outcome {anchor} had legal independent "
                    "proposals but none survived source/contract allocation"
                )
            if int(detail.get("attempted_variant_count", 0)) == 0 and not (
                detail.get("rejections") or {}
            ):
                raise AssertionError(
                    f"{source}: verified outcome {anchor} has no structured "
                    "independent search rejection"
                )
            dispositions.append(
                {
                    "anchor": anchor,
                    "anchor_time": anchor,
                    "disposition": "no_admissible_candidate",
                    "attempted_variant_count": int(detail.get("attempted_variant_count", 0)),
                    "rejections": dict(detail.get("rejections") or {}),
                }
            )
        if unresolved:
            raise AssertionError(
                f"{source}: semantic anchors lack proposal disposition: {unresolved}"
            )

        verified_finishers = int(manifest.get("verified_finishing_move_count", 0))
        require_finisher = bool(
            str(config.get("content_strategy", {}).get("mode") or "") != "spoken_content"
            and verified_finishers
            and config.get("batch_selection", {}).get(
                "require_verified_finishing_move_when_available", True
            )
        )
        selected = _solve_source(source, valid, qualified, require_finisher, config)
        selected_coverage = (
            set().union(*(_candidate_coverage(plan) for plan in selected)) if selected else set()
        )
        uncovered_qualified = sorted(qualified - selected_coverage, key=str)
        if uncovered_qualified:
            raise AssertionError(
                f"{source}: allocation left qualified semantic anchors uncovered: "
                f"{uncovered_qualified}"
            )

        for item in dispositions:
            if item["disposition"] == "qualified_for_allocation":
                item["disposition"] = "covered_by_selected_clip"
                item["selected_plan_keys"] = [
                    str(plan["plan_key"])
                    for plan in selected
                    if item["anchor"] in _candidate_coverage(plan)
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
            "content_type": manifest.get("content_type"),
            "plan_keys": [str(plan["plan_key"]) for plan in selected],
            "plans": selected,
            "qualified_candidate_count": len(valid),
            "rejected_by_contract": rejected_by_contract,
            "rejected_candidates": rejected,
            "required_anchor_count": len(required),
            "qualified_anchor_count": len(qualified),
            "covered_qualified_anchor_count": len(qualified),
            "anchor_dispositions": dispositions,
            "outcome_dispositions": [
                item for item in dispositions if isinstance(item["anchor"], float)
            ],
            "semantic_anchor_dispositions": [
                item for item in dispositions if isinstance(item["anchor"], str)
            ],
            "selected_pair_conflict_count": 0,
        }
        total_selected += len(selected)
        total_verified_finishers += verified_finishers
        total_selected_finishers += selected_finishers

    if total_verified_finishers > 0 and total_selected_finishers < 1:
        raise AssertionError("verified Finishing Move exists but allocation selected none")

    return {
        "schema_version": 2,
        "execution_sha": build_identity.current_execution_sha(),
        "semantic_engine": expected_engine,
        "editorial_planner": expected_editor,
        "candidate_mode": expected_mode,
        "allocation_mode": "coverage_conflict_before_render",
        "selection_basis": (
            "cover every qualified required semantic anchor, avoid source conflicts, "
            "minimize redundant clips, then maximize editorial quality deterministically"
        ),
        "source_order": list(source_order),
        "target_count": total_selected,
        "selected_count": total_selected,
        "duration_contract": {"minimum_seconds": 10.0, "maximum_seconds": 12.0},
        "distribution": {source: int(allocations[source]["count"]) for source in source_order},
        "verified_finishing_move_count": total_verified_finishers,
        "selected_finishing_move_count": total_selected_finishers,
        "source_allocations": allocations,
        "status": "PASS",
    }


def self_test() -> None:
    manifest = {
        "diagnostics": {
            "local_interaction_verifier": {
                "events": [{"time": 1.0, "kinds": ["outcome_like"], "confirmed": True}],
                "verified_hostile_event_assignment": [
                    {
                        "time": 2.0,
                        "kinds": ["outcome_like"],
                        "hostile_verified": True,
                        "disposition": "non_islandable_hard_gap_event",
                    },
                    {
                        "time": 3.0,
                        "kinds": ["outcome_like"],
                        "hostile_verified": False,
                    },
                ],
            }
        }
    }
    if _required_outcome_anchors(manifest) != (2.0,):
        raise AssertionError(
            "allocation required outcomes are not sourced from canonical hostile verification"
        )
    spoken_manifest = {
        "required_semantic_anchors": [
            {"id": "developer:freedom"},
            {"id": "developer:persistence"},
        ]
    }
    if _required_anchors(spoken_manifest) != (
        "developer:freedom",
        "developer:persistence",
    ):
        raise AssertionError("allocation did not preserve declarative semantic anchors")
    print("gameplay required-anchor authority self-test: PASS")

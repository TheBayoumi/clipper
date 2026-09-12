from __future__ import annotations

import argparse
import hashlib
import itertools
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


def _plans_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for a in left.get("segments") or []:
        for b in right.get("segments") or []:
            if max(float(a["start"]), float(b["start"])) < min(
                float(a["end"]), float(b["end"])
            ):
                return True
    return False


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


def _subset_score(plans: tuple[dict[str, Any], ...], config: dict[str, Any]) -> float:
    diversity_bonus = float(
        config.get("semantic_editor", {}).get("selection", {}).get("story_diversity_bonus", 0.06)
    )
    score = sum(float(plan.get("score", 0.0)) for plan in plans)
    score += diversity_bonus * len({str(plan.get("story_type", "")) for plan in plans})
    if bool(config.get("batch_selection", {}).get("prefer_normal_over_semantic_montage", True)):
        score -= 0.015 * sum(1 for plan in plans if plan.get("story_type") == "semantic_montage")
    return score


def _best_source_subset(
    source: str,
    candidates: list[dict[str, Any]],
    count: int,
    verified_finishing_move_count: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    require_finisher = (
        verified_finishing_move_count > 0
        and bool(config.get("batch_selection", {}).get("require_verified_finishing_move_when_available", True))
    )
    best: tuple[dict[str, Any], ...] | None = None
    best_score = float("-inf")

    for subset in itertools.combinations(candidates, count):
        if require_finisher and not any(plan.get("finishing_move") is not None for plan in subset):
            continue
        if any(_plans_overlap(a, b) for a, b in itertools.combinations(subset, 2)):
            continue
        score = _subset_score(subset, config)
        if score > best_score:
            best = subset
            best_score = score
    return best


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

    minimum = int(config.get("minimum_count_per_source", 2))
    maximum = int(config.get("count_per_source_max", 4))
    target = int(config.get("batch_selection", {}).get("target_count_exact", config.get("target_count", 10)))
    pool_limit = int(config.get("batch_selection", {}).get("candidate_pool_per_source", 18))

    candidate_pools: dict[str, list[dict[str, Any]]] = {}
    verified_counts: dict[str, int] = {}
    automatic_candidate_counts: dict[str, int] = {}

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
        for index, raw_plan in enumerate(raw, 1):
            plan = dict(raw_plan)
            plan["plan_key"] = str(plan.get("plan_key") or plan_key(plan))
            plan_failures = validate_plan(source, index, plan, config)
            if not plan_failures:
                valid.append(plan)
        valid.sort(
            key=lambda item: (
                item.get("finishing_move") is None,
                -float(item.get("score", 0.0)),
            )
        )
        candidate_pools[source] = valid[:pool_limit]
        verified_counts[source] = int(manifest.get("verified_finishing_move_count", 0))
        automatic_candidate_counts[source] = len(manifest.get("automatic_finishing_move_candidates") or [])
        if len(candidate_pools[source]) < minimum:
            failures.append(
                f"{source}: only {len(candidate_pools[source])} valid candidates available; minimum is {minimum}"
            )

    if failures:
        raise AssertionError("\n".join(failures))

    best_by_source_count: dict[str, dict[int, tuple[dict[str, Any], ...]]] = {
        source: {} for source in EXPECTED_SOURCES
    }
    for source in EXPECTED_SOURCES:
        for count in range(minimum, maximum + 1):
            subset = _best_source_subset(
                source,
                candidate_pools[source],
                count,
                verified_counts[source],
                config,
            )
            if subset is not None:
                best_by_source_count[source][count] = subset

    best_distribution: tuple[int, int, int] | None = None
    best_total_score = float("-inf")
    best_selection: dict[str, tuple[dict[str, Any], ...]] | None = None

    for distribution in itertools.product(range(minimum, maximum + 1), repeat=len(EXPECTED_SOURCES)):
        if sum(distribution) != target:
            continue
        selected: dict[str, tuple[dict[str, Any], ...]] = {}
        possible = True
        total_score = 0.0
        for source, count in zip(EXPECTED_SOURCES, distribution):
            subset = best_by_source_count[source].get(count)
            if subset is None:
                possible = False
                break
            selected[source] = subset
            total_score += _subset_score(subset, config)
        if possible and total_score > best_total_score:
            best_distribution = distribution
            best_total_score = total_score
            best_selection = selected

    if best_selection is None or best_distribution is None:
        availability = {
            source: sorted(best_by_source_count[source]) for source in EXPECTED_SOURCES
        }
        raise AssertionError(
            f"global allocator cannot build exactly {target} clips without overlap/quality violations; "
            f"feasible per-source counts={availability}"
        )

    allocations: dict[str, Any] = {}
    selected_verified_finishers = 0
    for source in EXPECTED_SOURCES:
        subset = best_selection[source]
        keys = [str(plan["plan_key"]) for plan in subset]
        selected_verified_finishers += sum(1 for plan in subset if plan.get("finishing_move") is not None)
        allocations[source] = {
            "count": len(subset),
            "plan_keys": keys,
            "plans": subset,
        }

    total_verified = sum(verified_counts.values())
    if total_verified > 0 and selected_verified_finishers < 1:
        raise AssertionError("verified Finishing Move exists but global allocator selected none")

    return {
        "version": "3.1",
        "semantic_engine": EXPECTED_ENGINE,
        "editorial_planner": EXPECTED_EDITOR,
        "candidate_mode": EXPECTED_MODE,
        "allocation_mode": "global_before_render",
        "target_count": target,
        "selected_count": sum(best_distribution),
        "distribution": {
            source: count for source, count in zip(EXPECTED_SOURCES, best_distribution)
        },
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

    minimum = int(config.get("minimum_count_per_source", 2))
    maximum = int(config.get("count_per_source_max", 4))
    if not (minimum <= len(selected) <= maximum):
        failures.append(f"{source}: selected {len(selected)} clips, required {minimum}..{maximum}")
    if len(outputs) != len(selected):
        failures.append(f"{source}: rendered {len(outputs)} != selected {len(selected)}")

    selected_keys = [str(plan.get("plan_key") or plan_key(plan)) for plan in selected]
    if allocation is not None:
        expected = allocation.get("source_allocations", {}).get(source, {}).get("plan_keys", [])
        if selected_keys != list(expected):
            failures.append(f"{source}: rendered selection does not match global pre-render allocation")

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

    minimum = int(config.get("minimum_count_per_source", 2))
    maximum = int(config.get("count_per_source_max", 4))
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
        if not (minimum <= selected <= maximum):
            failures.append(f"{source}: selected {selected}, required {minimum}..{maximum}")
        if rendered != selected:
            failures.append(f"{source}: rendered {rendered} != selected {selected}")
        if not bool(item.get("technical_qa_passed", False)):
            failures.append(f"{source}: technical QA did not pass")
        if int(item.get("unplanned_source_cut_count", -1)) != 0:
            failures.append(f"{source}: unplanned source cuts present")
        if allocation is not None:
            expected_count = int(allocation.get("source_allocations", {}).get(source, {}).get("count", -1))
            if selected != expected_count:
                failures.append(f"{source}: rendered count {selected} does not match allocation {expected_count}")
        total_selected += selected
        total_rendered += rendered
        total_verified_finishers += int(item.get("verified_finishing_move_count", 0))
        total_selected_finishers += int(item.get("selected_finishing_move_count", 0))

    target = int(config.get("batch_selection", {}).get("target_count_exact", config.get("target_count", 10)))
    if total_selected != target:
        failures.append(f"batch selected {total_selected} clips; exact target is {target}")
    if total_rendered != total_selected:
        failures.append("batch rendered count does not equal selected count")
    if total_verified_finishers > 0 and total_selected_finishers < 1:
        failures.append("verified Finishing Move exists in batch but no finishing_move_open clip was rendered")
    if allocation is not None and int(allocation.get("selected_count", -1)) != total_selected:
        failures.append("rendered batch does not match pre-render allocation total")

    if failures:
        raise AssertionError("\n".join(failures))

    return {
        "mode": next(iter(modes)),
        "sources": sorted(sources),
        "selected_count": total_selected,
        "rendered_count": total_rendered,
        "target_count": target,
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
    }
    assert _gate(cfg, "finishing_move_open", "opening_min", 0.42) == 0.50
    assert _gate(cfg, "precision_outcome", "opening_min", 0.42) == 0.42
    sample = {
        "story_type": "engagement_chain",
        "effect_profile": "chain_escalation",
        "segments": [{"start": 1.0, "end": 11.5, "speed": 1.0, "reason": "keep"}],
    }
    assert plan_key(sample) == plan_key(sample)
    print(json.dumps({"self_test": "PASS", "contract": "single-v3.1-global-pre-render-contract"}))


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

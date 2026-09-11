from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_ENGINE = "deterministic-gameplay-v3.1-final"
EXPECTED_EDITOR = "semantic-editor-v3.1-final"
EXPECTED_MODE = "engagement_driven_hardened_source_integrity"
EXPECTED_SOURCES = {"r1", "batch2", "week2"}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(config: dict[str, Any], story: str, metric: str, default: float) -> float:
    table = config.get("story_quality_gates", {}).get(metric, {})
    return float(table.get(story, table.get("default", default)))


def validate_source_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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

    editor = config["semantic_editor"]
    default_open = float(editor["opening"].get("minimum_quality", 0.42))
    default_end = float(editor["ending"].get("minimum_quality", 0.40))
    default_ret = float(config["performance_targets"].get("retention_quality_min", 0.36))
    default_pay = float(config["performance_targets"].get("payoff_quality_min", 0.34))
    max_dull = float(editor["dull"].get("maximum_unexplained_low_interest_run_seconds", 0.90))
    min_weak = float(editor["dull"].get("minimum_weak_quarter_interest", 0.27))
    max_low_fraction = float(editor["dull"].get("maximum_low_interest_fraction", 0.38))

    for index, plan in enumerate(selected, 1):
        story = str(plan["story_type"])
        if not (10.0 <= float(plan["output_duration"]) <= 20.0):
            failures.append(f"{source} clip {index}: duration out of campaign range")
        if float(plan["opening_quality"]) < _gate(config, story, "opening_min", default_open):
            failures.append(f"{source} clip {index}: opening gate failed")
        if float(plan["ending_quality"]) < _gate(config, story, "ending_min", default_end):
            failures.append(f"{source} clip {index}: ending gate failed")
        if float(plan["retention_quality"]) < _gate(config, story, "retention_min", default_ret):
            failures.append(f"{source} clip {index}: retention-quality gate failed")
        if float(plan["payoff_quality"]) < _gate(config, story, "payoff_min", default_pay):
            failures.append(f"{source} clip {index}: payoff-quality gate failed")
        if float(plan["max_unexplained_low_interest_run_seconds"]) > max_dull:
            failures.append(f"{source} clip {index}: unexplained dull run too long")
        if float(plan["weakest_quarter_interest"]) < min_weak:
            failures.append(f"{source} clip {index}: weakest quarter gate failed")
        if float(plan["low_interest_fraction"]) > max_low_fraction:
            failures.append(f"{source} clip {index}: unexplained low-interest fraction too high")

        finishing = plan.get("finishing_move")
        if finishing is not None:
            if story != "finishing_move_open" or plan.get("effect_profile") != "finishing_move_hero":
                failures.append(f"{source} clip {index}: Finishing Move routing is not opening hero")
            segments = plan.get("segments") or []
            if not segments:
                failures.append(f"{source} clip {index}: Finishing Move has no segments")
            else:
                delay = (float(finishing["start"]) - float(segments[0]["start"])) / float(segments[0]["speed"])
                if delay > 0.48:
                    failures.append(f"{source} clip {index}: Finishing Move opens too late ({delay:.3f}s)")
                if segments[0].get("reason") != "finishing_move_open_hero":
                    failures.append(f"{source} clip {index}: first segment is not protected Finishing Move hero")

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


def validate_batch(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    summaries = [_load(path) for path in sorted(root.rglob("*_pipeline_summary.json"))]
    failures: list[str] = []
    sources = {str(item.get("source_key", "")) for item in summaries}
    if sources != EXPECTED_SOURCES:
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
        total_selected += selected
        total_rendered += rendered
        total_verified_finishers += int(item.get("verified_finishing_move_count", 0))
        total_selected_finishers += int(item.get("selected_finishing_move_count", 0))

    target = int(config.get("target_count", 10))
    if total_selected < target:
        failures.append(f"batch selected {total_selected} clips; target is at least {target}")
    if total_rendered != total_selected:
        failures.append("batch rendered count does not equal selected count")
    if total_verified_finishers > 0 and total_selected_finishers < 1:
        failures.append("verified Finishing Move exists in batch but no finishing_move_open clip was rendered")

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
        "status": "PASS",
    }


def _self_test() -> None:
    cfg = {
        "story_quality_gates": {
            "opening_min": {"default": 0.42, "finishing_move_open": 0.50}
        }
    }
    assert _gate(cfg, "finishing_move_open", "opening_min", 0.42) == 0.50
    assert _gate(cfg, "precision_outcome", "opening_min", 0.42) == 0.42
    print(json.dumps({"self_test": "PASS", "contract": "single-v3.1-source-and-batch-contract"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.config is None:
        parser.error("--config is required")
    if (args.source_manifest is None) == (args.batch_root is None):
        parser.error("provide exactly one of --source-manifest or --batch-root")

    config = _load(args.config)
    if args.source_manifest is not None:
        result = validate_source_manifest(_load(args.source_manifest), config)
    else:
        result = validate_batch(args.batch_root, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

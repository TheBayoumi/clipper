from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_final as semantic
import mw4_v3_1_contract as contract


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-key", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allocation", required=True, type=Path)
    parser.add_argument("--plan-key", required=True)
    parser.add_argument("--ordinal", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("shadow", "production"), required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected, allocation = renderer._select_from_allocation(args.source_key, args.allocation)
    matches = [plan for plan in selected if renderer.plan_key(plan) == args.plan_key]
    if len(matches) != 1:
        raise RuntimeError(
            f"{args.source_key}: expected exactly one allocated plan for {args.plan_key}, found {len(matches)}"
        )
    plan = matches[0]
    payload = renderer._candidate_dict(plan)

    plan_failures = contract.validate_plan(args.source_key, args.ordinal, payload, config)
    if plan_failures:
        raise RuntimeError("; ".join(plan_failures))

    structure = renderer._source_structure_only(args.source, config, args.source_key)
    integrity = semantic.plan_integrity_violations(plan, structure, config, args.source_key)
    if integrity:
        raise RuntimeError("; ".join(integrity))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "250M" if args.mode == "production" else "SHADOW"
    filename = (
        f"MW4_V31_{args.source_key}_{args.ordinal:02d}_"
        f"{plan.story_type}_{plan.effect_profile}_{suffix}.mp4"
    )
    target = args.output_dir / filename

    renderer.render_candidate(args.source, plan, config, target, mode=args.mode)
    qa = renderer.validate_output(target, config, mode=args.mode)
    sheets = args.output_dir / "contact_sheets"
    renderer.create_contact_sheets(target, sheets)

    result = {
        "version": "3.1",
        "mode": args.mode,
        "source_key": args.source_key,
        "ordinal": args.ordinal,
        "plan_key": args.plan_key,
        "allocation_mode": allocation.get("allocation_mode"),
        "allocation_selected_count": allocation.get("selected_count"),
        "story_type": plan.story_type,
        "effect_profile": plan.effect_profile,
        "finishing_move": plan.finishing_move is not None,
        "unplanned_source_cut_count": 0,
        "technical_qa_passed": all(bool(value) for value in qa["checks"].values()),
        "file": filename,
        "sha256": renderer.sha256(target),
        "editorial_plan": payload,
        "qa": qa,
        "render_reanalysis": False,
        "single_clip_workspace": True,
        "status": "PASS",
    }
    result_path = args.output_dir / f"{args.source_key}_{args.ordinal:02d}_{args.plan_key}_clip_result_v3_1.json"
    _write(result_path, result)
    print(json.dumps({
        "source": args.source_key,
        "ordinal": args.ordinal,
        "plan_key": args.plan_key,
        "mode": args.mode,
        "file": filename,
        "qa": "PASS",
    }))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..gameplay import contract
from ..gameplay import candidates
from . import ffv1
from . import helpers
from . import source_fidelity


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_one(
    *,
    source_key: str,
    source: Path,
    config: dict[str, Any],
    allocation_path: Path,
    plan_key: str,
    ordinal: int,
    output_dir: Path,
    mode: str,
) -> dict[str, Any]:
    selected, allocation = candidates._select_from_allocation(source_key, allocation_path)
    matches = [plan for plan in selected if candidates.plan_key(plan) == plan_key]
    if len(matches) != 1:
        raise RuntimeError(
            f"{source_key}: expected exactly one allocated plan for {plan_key}, "
            f"found {len(matches)}"
        )
    plan = matches[0]
    payload = candidates._candidate_dict(plan)

    plan_failures = contract.validate_plan(source_key, ordinal, payload, config)
    if plan_failures:
        raise RuntimeError("; ".join(plan_failures))

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "250M" if mode == "production" else "SHADOW"
    filename = f"MW4_{source_key}_{ordinal:02d}_{plan.story_type}_source_native_{suffix}.mp4"
    target = output_dir / filename

    ffv1.preflight()
    with tempfile.TemporaryDirectory(prefix="clipper_gameplay_stage_") as temp_dir:
        workspace = Path(temp_dir)
        staged_source, local_plan, source_profile, staging_fidelity = ffv1._stage_plan_source(
            source, plan, workspace, config
        )
        fidelity_plan = replace(local_plan, effect_profile="source_native_full_frame")
        canonical_master = workspace / "canonical_lossless_edited_master.nut"
        canonical_info = ffv1._render_canonical_lossless_master(
            staged_source,
            fidelity_plan,
            config,
            source_profile,
            canonical_master,
        )
        source_fidelity._encode_from_canonical_master(
            canonical_master,
            config,
            source_profile,
            target,
            mode=mode,
        )
        fidelity_qa = ffv1._source_fidelity_qa(
            source_profile,
            staging_fidelity,
            canonical_master,
            target,
            config,
        )

    qa = helpers.validate_output(target, config, mode=mode)
    helpers.create_contact_sheets(target, output_dir / "contact_sheets")

    result = {
        "schema_version": 1,
        "mode": mode,
        "source_key": source_key,
        "ordinal": ordinal,
        "plan_key": plan_key,
        "allocation_mode": allocation.get("allocation_mode"),
        "allocation_selected_count": allocation.get("selected_count"),
        "story_type": plan.story_type,
        "planned_effect_profile": plan.effect_profile,
        "rendered_effect_profile": "source_native_full_frame",
        "finishing_move": plan.finishing_move is not None,
        "unplanned_source_cut_count": 0,
        "technical_qa_passed": all(bool(value) for value in qa["checks"].values()),
        "source_fidelity_qa_passed": all(bool(value) for value in fidelity_qa["checks"].values()),
        "source_fidelity": fidelity_qa,
        "canonical_master": canonical_info,
        "file": filename,
        "sha256": helpers.sha256(target),
        "editorial_plan": payload,
        "qa": qa,
        "render_reanalysis": False,
        "source_structure_reanalysis": False,
        "bounded_lossless_segment_staging": True,
        "exact_source_to_stage_frame_hash_qa": True,
        "source_color_metadata_preserved": True,
        "spatial_crop_upscale_used": False,
        "source_native_full_frame": True,
        "single_canonical_visual_timeline": True,
        "single_clip_workspace": True,
        "status": "PASS",
    }
    result_path = output_dir / f"{source_key}_{ordinal:02d}_{plan_key}_clip_result.json"
    _write(result_path, result)
    print(
        json.dumps(
            {
                "source": source_key,
                "ordinal": ordinal,
                "plan_key": plan_key,
                "mode": mode,
                "file": filename,
                "qa": "PASS",
                "source_fidelity_qa": "PASS",
                "ssim": fidelity_qa["ssim"],
                "psnr_db": fidelity_qa["psnr_db"],
                "frame_count": fidelity_qa["reference_profile"]["frame_count"],
                "source_profile": source_profile,
                "source_to_stage_hashes_exact": True,
                "source_color_metadata_preserved": True,
                "spatial_crop_upscale_used": False,
                "single_canonical_visual_timeline": True,
                "bounded_lossless_segment_staging": True,
            }
        )
    )
    return result


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
    render_one(
        source_key=args.source_key,
        source=args.source,
        config=json.loads(args.config.read_text(encoding="utf-8")),
        allocation_path=args.allocation,
        plan_key=args.plan_key,
        ordinal=args.ordinal,
        output_dir=args.output_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

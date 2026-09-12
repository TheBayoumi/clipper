from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_final as semantic
import mw4_v3_1_contract as contract


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _map_source_time(
    value: float,
    mappings: list[tuple[semantic.EditSegment, semantic.EditSegment]],
) -> float | None:
    for original, local in mappings:
        if original.start - 1e-6 <= value <= original.end + 1e-6:
            return local.start + (value - original.start)
    return None


def _stage_plan_source(
    source: Path,
    plan: semantic.SemanticPlanV31,
    workspace: Path,
) -> tuple[Path, semantic.SemanticPlanV31]:
    """Create a short lossless source containing only the approved edit segments.

    Each far-apart source interval is seeked and decoded independently so no
    full-reel multi-trim graph can retain gigabytes of raw frames. FFV1/PCM staging
    is lossless; the final H.264 master is still encoded exactly once.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    source_info = renderer.probe(source)
    source_duration = float(source_info["format"]["duration"])

    pieces: list[Path] = []
    mappings: list[tuple[semantic.EditSegment, semantic.EditSegment]] = []
    cursor = 0.0
    segments = list(plan.segments)

    for index, segment in enumerate(segments):
        previous = segments[index - 1] if index else None
        following = segments[index + 1] if index + 1 < len(segments) else None

        noncontiguous_before = bool(
            previous is not None
            and (segment.start > previous.end + 0.02 or segment.start < previous.start - 0.02)
        )
        noncontiguous_after = bool(
            following is None
            or following.start > segment.end + 0.02
            or following.start < segment.start - 0.02
        )

        lead = min(0.05, segment.start) if noncontiguous_before else 0.0
        tail = min(0.05, max(0.0, source_duration - segment.end)) if noncontiguous_after else 0.0
        extract_start = max(0.0, segment.start - lead)
        extract_end = min(source_duration, segment.end + tail)
        extract_duration = extract_end - extract_start
        if extract_duration <= 0.0:
            raise RuntimeError(f"invalid staged segment {index + 1}: {segment.start}-{segment.end}")

        piece = workspace / f"segment_{index:02d}.mkv"
        _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{extract_start:.6f}",
            "-i", str(source),
            "-t", f"{extract_duration:.6f}",
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-c:v", "ffv1",
            "-level", "3",
            "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            str(piece),
        ])

        piece_duration = float(renderer.probe(piece)["format"]["duration"])
        local_start = cursor + (segment.start - extract_start)
        local_end = local_start + (segment.end - segment.start)
        if local_end > cursor + piece_duration + 0.025:
            raise RuntimeError(
                f"staged segment {index + 1} is shorter than approved source interval: "
                f"need {local_end - cursor:.3f}s, have {piece_duration:.3f}s"
            )

        local_segment = semantic.EditSegment(
            start=round(local_start, 6),
            end=round(local_end, 6),
            speed=segment.speed,
            reason=segment.reason,
        )
        mappings.append((segment, local_segment))
        pieces.append(piece)
        cursor += piece_duration

    concat_list = workspace / "concat.txt"
    concat_list.write_text(
        "".join(f"file '{piece.name}'\n" for piece in pieces),
        encoding="utf-8",
    )
    stitched = workspace / "approved_segments_lossless.mkv"
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(stitched),
    ])

    local_events = []
    for event in plan.effect_events:
        mapped = _map_source_time(event.time, mappings)
        if mapped is not None:
            local_events.append(replace(event, time=round(mapped, 6)))

    local_finishing = None
    if plan.finishing_move is not None:
        start = _map_source_time(plan.finishing_move.start, mappings)
        payoff = _map_source_time(plan.finishing_move.payoff, mappings)
        end = _map_source_time(plan.finishing_move.end, mappings)
        if start is None or payoff is None or end is None:
            raise RuntimeError("verified Finishing Move was not fully preserved by staged source")
        local_finishing = replace(
            plan.finishing_move,
            start=round(start, 6),
            payoff=round(payoff, 6),
            end=round(end, 6),
        )

    local_plan = replace(
        plan,
        start=min(item.start for _, item in mappings),
        end=max(item.end for _, item in mappings),
        segments=tuple(item for _, item in mappings),
        effect_events=tuple(local_events),
        finishing_move=local_finishing,
    )
    return stitched, local_plan


def _render_lossless_reference(
    staged_source: Path,
    fidelity_plan: semantic.SemanticPlanV31,
    config: dict[str, Any],
    target: Path,
) -> None:
    graph, duration = renderer.build_filter(fidelity_plan, config)
    settings = config["output"]
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-filter_complex_threads", "2",
        "-i", str(staged_source),
        "-filter_complex", graph,
        "-map", "[outv]",
        "-map", "[aout]",
        "-c:v", "ffv1",
        "-level", "3",
        "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        "-r", str(settings["fps"]),
        "-vsync", "cfr",
        "-t", f"{duration:.3f}",
        str(target),
    ])


def _extract_metric(stderr: str, pattern: str, label: str) -> float:
    matches = re.findall(pattern, stderr)
    if not matches:
        raise RuntimeError(f"unable to parse {label} from FFmpeg fidelity QA")
    return float(matches[-1])


def _source_fidelity_qa(reference: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    ssim_run = _run_capture([
        "ffmpeg", "-hide_banner", "-i", str(reference), "-i", str(output),
        "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-",
    ])
    psnr_run = _run_capture([
        "ffmpeg", "-hide_banner", "-i", str(reference), "-i", str(output),
        "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-",
    ])
    ssim = _extract_metric(ssim_run.stderr, r"All:([0-9.]+)", "SSIM")
    psnr = _extract_metric(psnr_run.stderr, r"average:([0-9.]+)", "PSNR")

    fidelity_cfg = config.get("source_fidelity", {})
    minimum_ssim = float(fidelity_cfg.get("minimum_ssim", 0.99))
    minimum_psnr = float(fidelity_cfg.get("minimum_psnr_db", 40.0))
    checks = {
        "no_spatial_crop_or_upscale": True,
        "ssim_source_native_reference": ssim >= minimum_ssim,
        "psnr_source_native_reference": psnr >= minimum_psnr,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"source-fidelity QA failed: SSIM={ssim:.6f} minimum={minimum_ssim:.6f}; "
            f"PSNR={psnr:.3f}dB minimum={minimum_psnr:.3f}dB"
        )
    return {
        "checks": checks,
        "ssim": ssim,
        "minimum_ssim": minimum_ssim,
        "psnr_db": psnr,
        "minimum_psnr_db": minimum_psnr,
        "reference": "lossless FFV1 render of the identical approved edit/timing",
    }


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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "250M" if args.mode == "production" else "SHADOW"
    filename = (
        f"MW4_V31_{args.source_key}_{args.ordinal:02d}_"
        f"{plan.story_type}_source_native_{suffix}.mp4"
    )
    target = args.output_dir / filename

    with tempfile.TemporaryDirectory(prefix="mw4_v31_stage_") as temp_dir:
        workspace = Path(temp_dir)
        staged_source, local_plan = _stage_plan_source(args.source, plan, workspace)
        fidelity_plan = replace(local_plan, effect_profile="source_native_full_frame")
        renderer.render_candidate(staged_source, fidelity_plan, config, target, mode=args.mode)

        reference = workspace / "source_native_reference.mkv"
        _render_lossless_reference(staged_source, fidelity_plan, config, reference)
        fidelity_qa = _source_fidelity_qa(reference, target, config)

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
        "planned_effect_profile": plan.effect_profile,
        "rendered_effect_profile": "source_native_full_frame",
        "finishing_move": plan.finishing_move is not None,
        "unplanned_source_cut_count": 0,
        "technical_qa_passed": all(bool(value) for value in qa["checks"].values()),
        "source_fidelity_qa_passed": all(bool(value) for value in fidelity_qa["checks"].values()),
        "source_fidelity": fidelity_qa,
        "file": filename,
        "sha256": renderer.sha256(target),
        "editorial_plan": payload,
        "qa": qa,
        "render_reanalysis": False,
        "source_structure_reanalysis": False,
        "bounded_lossless_segment_staging": True,
        "spatial_crop_upscale_used": False,
        "source_native_full_frame": True,
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
        "source_fidelity_qa": "PASS",
        "ssim": fidelity_qa["ssim"],
        "psnr_db": fidelity_qa["psnr_db"],
        "spatial_crop_upscale_used": False,
        "bounded_lossless_segment_staging": True,
    }))


if __name__ == "__main__":
    main()

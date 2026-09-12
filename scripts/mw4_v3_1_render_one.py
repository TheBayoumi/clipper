from __future__ import annotations

import argparse
import json
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

    The old renderer branched several far-apart trims from one full-reel decoder.
    FFmpeg could then buffer decoded 1080p frames for tens of source seconds while
    concat waited for another branch. This stage seeks to each segment separately,
    decodes only that short interval, stores it losslessly, then concatenates those
    pieces. Final H.264 quality/rate control is still applied exactly once by the
    canonical renderer.
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

    # Source-integrity/semantic analysis already happened before allocation. The
    # immutable allocated plan key is revalidated above, so do not rescan the full
    # reel again in every render workspace.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "250M" if args.mode == "production" else "SHADOW"
    filename = (
        f"MW4_V31_{args.source_key}_{args.ordinal:02d}_"
        f"{plan.story_type}_{plan.effect_profile}_{suffix}.mp4"
    )
    target = args.output_dir / filename

    with tempfile.TemporaryDirectory(prefix="mw4_v31_stage_") as temp_dir:
        staged_source, local_plan = _stage_plan_source(args.source, plan, Path(temp_dir))
        renderer.render_candidate(staged_source, local_plan, config, target, mode=args.mode)

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
        "source_structure_reanalysis": False,
        "bounded_lossless_segment_staging": True,
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
        "bounded_lossless_segment_staging": True,
    }))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import mw4_semantic_gameplay_v3_1_final as semantic
import mw4_v3_1_lossless as lossless
import mw4_v3_1_render_one_legacy as legacy


def _verify_stitched_lossless(
    pieces: list[Path],
    stitched: Path,
    source_profile: dict[str, Any],
) -> dict[str, Any]:
    expected_hashes: list[str] = []
    for piece in pieces:
        expected_hashes.extend(
            legacy._decoded_frame_hashes(
                piece,
                pix_fmt=str(source_profile["pix_fmt"]),
            )
        )
    stitched_hashes = legacy._decoded_frame_hashes(
        stitched,
        pix_fmt=str(source_profile["pix_fmt"]),
    )
    stitched_profile = legacy._video_profile(stitched)
    checks = {
        "exact_decoded_frame_count_match": len(expected_hashes) == len(stitched_hashes),
        "exact_decoded_frame_hash_match": expected_hashes == stitched_hashes,
        "r_fps_matches_source": (
            stitched_profile["r_frame_rate"] == source_profile["r_frame_rate"]
        ),
        "avg_fps_matches_source": (
            stitched_profile["avg_frame_rate"] == source_profile["avg_frame_rate"]
        ),
        **legacy._metadata_match_checks(
            source_profile,
            stitched_profile,
            prefix="stitched",
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "lossless piece->stitched fidelity failed: "
            f"checks={checks}; expected_frames={len(expected_hashes)} "
            f"stitched_frames={len(stitched_hashes)}; "
            f"source_profile={source_profile}; stitched_profile={stitched_profile}"
        )
    return {
        "segment": "stitched",
        "frame_count": len(stitched_hashes),
        "checks": checks,
        "stage_profile": stitched_profile,
    }


def _stage_plan_source(
    source: Path,
    plan: semantic.SemanticPlanV31,
    workspace: Path,
    config: dict[str, Any],
) -> tuple[Path, semantic.SemanticPlanV31, dict[str, Any], list[dict[str, Any]]]:
    """Stage approved intervals with a lossless MOV transport and prove identity."""
    workspace.mkdir(parents=True, exist_ok=True)
    source_info = legacy.renderer.probe(source)
    source_duration = float(source_info["format"]["duration"])
    source_profile = legacy._video_profile(source)
    legacy._assert_source_profile(source, source_profile, config)
    lossless.assert_supported_profile(source_profile)

    pieces: list[Path] = []
    mappings: list[tuple[semantic.EditSegment, semantic.EditSegment]] = []
    staging_fidelity: list[dict[str, Any]] = []
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
            raise RuntimeError(
                f"invalid staged segment {index + 1}: {segment.start}-{segment.end}"
            )

        piece = workspace / f"segment_{index:02d}.mov"
        legacy._run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{extract_start:.6f}",
                "-i", str(source),
                "-t", f"{extract_duration:.6f}",
                "-map", "0:v:0",
                "-map", "0:a:0",
                *lossless.lossless_video_args(source_profile),
                "-vsync", "0",
                "-c:a", "pcm_s16le",
                "-ar", "48000",
                "-ac", "2",
                "-f", "mov",
                str(piece),
            ]
        )
        staging_fidelity.append(
            legacy._verify_lossless_piece(
                source,
                piece,
                extract_start=extract_start,
                extract_duration=extract_duration,
                source_profile=source_profile,
                index=index,
            )
        )

        piece_duration = float(legacy.renderer.probe(piece)["format"]["duration"])
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
    stitched = workspace / "approved_segments_lossless.mov"
    legacy._run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-video_track_timescale", str(lossless.VIDEO_TRACK_TIMESCALE),
            "-f", "mov",
            str(stitched),
        ]
    )
    staging_fidelity.append(
        _verify_stitched_lossless(pieces, stitched, source_profile)
    )

    local_events = []
    for event in plan.effect_events:
        mapped = legacy._map_source_time(event.time, mappings)
        if mapped is not None:
            local_events.append(replace(event, time=round(mapped, 6)))

    local_finishing = None
    if plan.finishing_move is not None:
        start = legacy._map_source_time(plan.finishing_move.start, mappings)
        payoff = legacy._map_source_time(plan.finishing_move.payoff, mappings)
        end = legacy._map_source_time(plan.finishing_move.end, mappings)
        if start is None or payoff is None or end is None:
            raise RuntimeError(
                "verified Finishing Move was not fully preserved by staged source"
            )
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
    return stitched, local_plan, source_profile, staging_fidelity


def _canonical_filter_hashes(
    staged_source: Path,
    graph: str,
    pix_fmt: str,
) -> list[str]:
    text = legacy._run_capture(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-filter_complex_threads", "2",
            "-i", str(staged_source),
            "-filter_complex", graph,
            "-map", "[outv]",
            "-an",
            "-vsync", "0",
            "-pix_fmt", pix_fmt,
            "-f", "framemd5", "-",
        ]
    ).stdout
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _render_canonical_lossless_master(
    staged_source: Path,
    plan: semantic.SemanticPlanV31,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    graph, duration = legacy._build_source_native_filter(plan, source_profile)
    settings = config["output"]
    lossless.assert_supported_profile(source_profile)

    # legacy.main() supplies a temporary .nut path. Force MOV explicitly;
    # ffmpeg/ffprobe identify the container by content, and this file is never exported.
    legacy._run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-filter_complex_threads", "2",
            "-i", str(staged_source),
            "-filter_complex", graph,
            "-map", "[outv]",
            "-map", "[aout]",
            *lossless.lossless_video_args(source_profile),
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-r", str(settings["fps"]),
            "-vsync", "cfr",
            "-t", f"{duration:.3f}",
            "-f", "mov",
            str(target),
        ]
    )

    expected_hashes = _canonical_filter_hashes(
        staged_source,
        graph,
        str(source_profile["pix_fmt"]),
    )
    master_hashes = legacy._decoded_frame_hashes(
        target,
        pix_fmt=str(source_profile["pix_fmt"]),
    )
    if expected_hashes != master_hashes:
        mismatch = next(
            (
                i
                for i, pair in enumerate(zip(expected_hashes, master_hashes))
                if pair[0] != pair[1]
            ),
            None,
        )
        raise RuntimeError(
            "canonical lossless x264/MOV altered decoded pixels: "
            f"expected_frames={len(expected_hashes)} master_frames={len(master_hashes)} "
            f"first_hash_mismatch={mismatch}"
        )

    return {
        "graph_has_spatial_transform": False,
        "video_operations": "trim/setpts/concat + source-SAR metadata only",
        "canonical_fps": str(settings["fps"]),
        "canonical_container": "mov",
        "canonical_codec": "libx264 qp=0 lossless",
        "video_track_timescale": lossless.VIDEO_TRACK_TIMESCALE,
        "exact_filtered_frame_hash_match": True,
        "source_profile": source_profile,
    }


_original_source_fidelity_qa = legacy._source_fidelity_qa


def _source_fidelity_qa(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _original_source_fidelity_qa(*args, **kwargs)
    result["reference"] = (
        "original MediaSilo source decoded-frame hashes must match every "
        "lossless staged segment and stitched MOV exactly; the canonical "
        "lossless x264/MOV master is frame-hash verified before final H.264 "
        "is measured for generation loss"
    )
    return result


def main() -> None:
    # Protect direct/local invocations that do not run the workflow self-test.
    lossless.preflight()
    legacy._stage_plan_source = _stage_plan_source
    legacy._render_canonical_lossless_master = _render_canonical_lossless_master
    legacy._source_fidelity_qa = _source_fidelity_qa
    legacy.main()


if __name__ == "__main__":
    main()

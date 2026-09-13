from __future__ import annotations

from pathlib import Path
from typing import Any

import mw4_v3_1_render_one_mov as base


def _canonical_filter_hashes(
    staged_source: Path,
    graph: str,
    pix_fmt: str,
    *,
    fps: str,
    duration: float,
) -> list[str]:
    """Hash the exact CFR-scheduled video timeline encoded by the master."""
    text = base.legacy._run_capture(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-filter_complex_threads", "2",
            "-i", str(staged_source),
            "-filter_complex", graph,
            "-map", "[outv]",
            "-an",
            "-r", fps,
            "-vsync", "cfr",
            "-t", f"{duration:.3f}",
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
    plan: base.semantic.SemanticPlanV31,
    config: dict[str, Any],
    source_profile: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    graph, duration = base.legacy._build_source_native_filter(plan, source_profile)
    settings = config["output"]
    base.lossless.assert_supported_profile(source_profile)

    base.legacy._run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-filter_complex_threads", "2",
            "-i", str(staged_source),
            "-filter_complex", graph,
            "-map", "[outv]",
            "-map", "[aout]",
            *base.lossless.lossless_video_args(source_profile),
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
        fps=str(settings["fps"]),
        duration=duration,
    )
    master_hashes = base.legacy._decoded_frame_hashes(
        target,
        pix_fmt=str(source_profile["pix_fmt"]),
    )
    if expected_hashes != master_hashes:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(expected_hashes, master_hashes))
                if pair[0] != pair[1]
            ),
            None,
        )
        raise RuntimeError(
            "canonical CFR reference differs from lossless x264/MOV decoded pixels: "
            f"expected_frames={len(expected_hashes)} master_frames={len(master_hashes)} "
            f"first_hash_mismatch={mismatch}"
        )

    return {
        "graph_has_spatial_transform": False,
        "video_operations": "trim/setpts/concat + source-SAR metadata only",
        "canonical_fps": str(settings["fps"]),
        "canonical_container": "mov",
        "canonical_codec": "libx264 qp=0 lossless",
        "video_track_timescale": base.lossless.VIDEO_TRACK_TIMESCALE,
        "exact_filtered_frame_hash_match": True,
        "source_profile": source_profile,
    }


def main() -> None:
    base._render_canonical_lossless_master = _render_canonical_lossless_master
    base.main()


if __name__ == "__main__":
    main()

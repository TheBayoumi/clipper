from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_FPS = "60000/1001"
VIDEO_TRACK_TIMESCALE = 60000

QUALIFICATION_PROFILE: dict[str, str] = {
    "pix_fmt": "yuv420p",
    "color_range": "tv",
    "color_space": "bt709",
    "color_transfer": "bt709",
    "color_primaries": "bt709",
    "chroma_location": "left",
}

_PROFILE_OPTIONS = (
    ("color_range", "-color_range"),
    ("color_space", "-colorspace"),
    ("color_transfer", "-color_trc"),
    ("color_primaries", "-color_primaries"),
    ("chroma_location", "-chroma_sample_location"),
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _capture(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def assert_supported_profile(profile: dict[str, Any]) -> None:
    failures = [
        f"{field}={profile.get(field)!r} expected={expected!r}"
        for field, expected in QUALIFICATION_PROFILE.items()
        if str(profile.get(field) or "") != expected
    ]
    if failures:
        raise RuntimeError(
            "MW4 lossless MOV transport is qualified only for the current "
            "MediaSilo SDR profile; refusing an unqualified source: "
            + "; ".join(failures)
        )


def profile_output_args(profile: dict[str, Any]) -> list[str]:
    assert_supported_profile(profile)
    args = ["-pix_fmt", str(profile["pix_fmt"])]
    for field, option in _PROFILE_OPTIONS:
        args += [option, str(profile[field])]
    return args


def lossless_video_args(profile: dict[str, Any]) -> list[str]:
    """Lossless x264 + explicit VUI + exact MOV video timescale."""
    assert_supported_profile(profile)
    x264_vui = ":".join(
        (
            "fullrange=off",
            f"colorprim={profile['color_primaries']}",
            f"transfer={profile['color_transfer']}",
            f"colormatrix={profile['color_space']}",
            "chromaloc=0",
        )
    )
    return [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-qp", "0",
        *profile_output_args(profile),
        "-x264-params", x264_vui,
        "-video_track_timescale", str(VIDEO_TRACK_TIMESCALE),
        "-threads:v", "4",
    ]


def _probe(path: Path) -> dict[str, str]:
    entries = (
        "stream=pix_fmt,sample_aspect_ratio,display_aspect_ratio,color_range,color_space,"
        "color_transfer,color_primaries,chroma_location,r_frame_rate,avg_frame_rate,time_base"
    )
    payload = json.loads(
        _capture(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", entries,
                "-of", "json",
                str(path),
            ]
        )
    )
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"lossless preflight expected one video stream: {path}")
    stream = streams[0]
    return {key: str(value) for key, value in stream.items()}


def _frame_hashes(path: Path) -> list[str]:
    text = _capture(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(path),
            "-map", "0:v:0",
            "-an",
            "-vsync", "0",
            "-pix_fmt", QUALIFICATION_PROFILE["pix_fmt"],
            "-f", "framemd5", "-",
        ]
    )
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def preflight() -> dict[str, Any]:
    """Prove the installed FFmpeg/libx264 can satisfy the MW4 transport contract."""
    profile: dict[str, Any] = dict(QUALIFICATION_PROFILE)
    with tempfile.TemporaryDirectory(prefix="mw4_lossless_preflight_") as tmp:
        root = Path(tmp)
        source = root / "source.mov"
        staged = root / "staged.mov"

        # Match the official source's unspecified SAR while retaining the full
        # 60000/1001 SDR color/chroma contract.
        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi",
                "-i", "testsrc2=size=128x72:rate=60000/1001:duration=0.25,setsar=0/1",
                "-frames:v", "12",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "12",
                *profile_output_args(profile),
                "-x264-params",
                "fullrange=off:colorprim=bt709:transfer=bt709:"
                "colormatrix=bt709:chromaloc=0",
                "-video_track_timescale", str(VIDEO_TRACK_TIMESCALE),
                "-an",
                "-f", "mov",
                str(source),
            ]
        )

        _run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(source),
                "-map", "0:v:0",
                *lossless_video_args(profile),
                "-vsync", "0",
                "-an",
                "-f", "mov",
                str(staged),
            ]
        )

        source_profile = _probe(source)
        staged_profile = _probe(staged)
        source_hashes = _frame_hashes(source)
        staged_hashes = _frame_hashes(staged)

        checks: dict[str, bool] = {
            "source_has_frames": bool(source_hashes),
            "exact_decoded_frame_count_match": len(source_hashes) == len(staged_hashes),
            "exact_decoded_frame_hash_match": source_hashes == staged_hashes,
            "source_r_fps": source_profile.get("r_frame_rate") == EXPECTED_FPS,
            "source_avg_fps": source_profile.get("avg_frame_rate") == EXPECTED_FPS,
            "stage_r_fps": staged_profile.get("r_frame_rate") == EXPECTED_FPS,
            "stage_avg_fps": staged_profile.get("avg_frame_rate") == EXPECTED_FPS,
            "source_time_base": source_profile.get("time_base") == "1/60000",
            "stage_time_base": staged_profile.get("time_base") == "1/60000",
            "sample_aspect_ratio_match": (
                staged_profile.get("sample_aspect_ratio", "")
                == source_profile.get("sample_aspect_ratio", "")
            ),
            "display_aspect_ratio_match": (
                staged_profile.get("display_aspect_ratio", "")
                == source_profile.get("display_aspect_ratio", "")
            ),
        }
        for field, expected in QUALIFICATION_PROFILE.items():
            checks[f"source_{field}"] = source_profile.get(field) == expected
            checks[f"stage_{field}"] = staged_profile.get(field) == expected

        if not all(checks.values()):
            raise RuntimeError(
                "MW4 lossless transport preflight failed: "
                f"checks={checks}; source={source_profile}; stage={staged_profile}"
            )

        result = {
            "lossless_transport_preflight": "PASS",
            "checks": checks,
            "source_profile": source_profile,
            "stage_profile": staged_profile,
        }
        print(json.dumps(result))
        return result

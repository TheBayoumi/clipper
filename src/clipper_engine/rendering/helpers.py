from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)

def probe(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,bit_rate,sample_rate,channels:format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    return json.loads(result.stdout)

def _encode_args(config: dict[str, Any], mode: str) -> list[str]:
    settings = config["output"]
    if mode == "production":
        bitrate = int(settings["video_bitrate_kbps"])
        return [
            "-c:v",
            str(settings["codec"]),
            "-preset",
            str(settings["preset"]),
            "-profile:v",
            str(settings["profile"]),
            "-level:v",
            "5.2",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            f"{bitrate}k",
            "-minrate",
            f"{bitrate}k",
            "-maxrate",
            f"{bitrate}k",
            "-bufsize",
            f"{bitrate * 2}k",
            "-x264-params",
            "nal-hrd=cbr",
            "-c:a",
            str(settings["audio_codec"]),
            "-b:a",
            f"{int(settings['audio_bitrate_kbps'])}k",
            "-ar",
            str(settings["audio_sample_rate"]),
            "-ac",
            str(settings["audio_channels"]),
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "superfast",
        "-profile:v",
        "high",
        "-level:v",
        "5.2",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "8M",
        "-maxrate",
        "10M",
        "-bufsize",
        "20M",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
    ]

def validate_output(
    path: Path, config: dict[str, Any], *, mode: str = "production"
) -> dict[str, Any]:
    data = probe(path)
    video = next(item for item in data["streams"] if item["codec_type"] == "video")
    audio = next(item for item in data["streams"] if item["codec_type"] == "audio")
    fmt = data["format"]
    settings = config["output"]
    duration = float(fmt["duration"])
    bitrate = int(fmt.get("bit_rate") or video.get("bit_rate") or 0)
    checks = {
        "duration_10_to_12s": (
            float(config["semantic_editor"].get("minimum_output_seconds", 10.0))
            <= duration
            <= float(config["semantic_editor"].get("maximum_output_seconds", 12.0))
        ),
        "full_frame_1920x1080": (video["width"], video["height"]) == (1920, 1080),
        "codec_h264": video["codec_name"] == "h264",
        "profile_high": str(video.get("profile", "")).lower() == "high",
        "r_frame_rate": video["r_frame_rate"] == settings["fps"],
        "avg_frame_rate": video["avg_frame_rate"] == settings["fps"],
        "audio_48khz": audio["sample_rate"] == "48000",
        "audio_stereo": audio["channels"] == 2,
    }
    if mode == "production":
        expected = int(settings["video_bitrate_kbps"]) * 1000
        checks["high_bitrate_near_250mbps"] = bitrate >= int(expected * 0.94)
    run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"])
    if not all(checks.values()):
        raise RuntimeError(
            f"QA failed for {path.name}: {checks}; actual_r_frame_rate={video.get('r_frame_rate')} "
            f"actual_avg_frame_rate={video.get('avg_frame_rate')}"
        )
    return {"checks": checks, "probe": data}

def create_contact_sheets(video: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=4,scale=320:-1,tile=6x5:padding=2:margin=2",
            "-frames:v",
            "3",
            str(output_dir / f"{video.stem}_%02d.jpg"),
        ]
    )

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


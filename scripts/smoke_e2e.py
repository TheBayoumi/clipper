from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from clipper.models import CampaignBrief, VideoCandidate
from clipper.pipeline import PipelineSettings, run_pipeline

VIDEO_ID = "smoke-video"
CHANNEL_ID = "smoke-channel"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generate_source(path: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(path),
        ]
    )


def _write_vtt(path: Path) -> None:
    path.write_text(
        """WEBVTT

00:00:00.000 --> 00:00:04.000
Here is the problem with manual business work.

00:00:04.000 --> 00:00:08.000
AI automation can save time for every business.

00:00:08.000 --> 00:00:10.000
Never repeat the same task again.
""",
        encoding="utf-8",
    )


class SmokeSource:
    def __init__(self, source_path: Path, subtitle_path: Path) -> None:
        self.source_path = source_path
        self.subtitle_path = subtitle_path

    def discover(self, brief: CampaignBrief) -> list[VideoCandidate]:
        del brief
        return [
            VideoCandidate(
                video_id=VIDEO_ID,
                title="Deterministic clipping smoke source",
                channel_id=CHANNEL_ID,
                channel_title="Clipper CI",
                url="synthetic://smoke-video",
                duration_seconds=10.0,
            )
        ]

    def download_subtitles(
        self,
        video: VideoCandidate,
        work_dir: Path,
        language: str,
    ) -> Path:
        del video, language
        work_dir.mkdir(parents=True, exist_ok=True)
        destination = work_dir / "smoke.en.vtt"
        shutil.copyfile(self.subtitle_path, destination)
        return destination

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
        del video, work_dir
        return self.source_path


def _probe(path: Path) -> dict[str, Any]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ]
    )
    payload: object = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("ffprobe returned an invalid JSON payload")
    return payload


def _validate_probe(probe: dict[str, Any]) -> None:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError("ffprobe did not return stream metadata")
    videos = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audios = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    if len(videos) != 1 or len(audios) != 1:
        raise RuntimeError(
            f"expected one video and one audio stream, got {len(videos)} and {len(audios)}"
        )
    video = videos[0]
    if (video.get("width"), video.get("height")) != (1080, 1920):
        raise RuntimeError(
            f"unexpected output geometry: {video.get('width')}x{video.get('height')}"
        )
    if video.get("codec_name") != "h264" or audios[0].get("codec_name") != "aac":
        raise RuntimeError("unexpected output codecs")
    format_data = probe.get("format")
    if not isinstance(format_data, dict):
        raise RuntimeError("ffprobe did not return format metadata")
    duration = float(str(format_data.get("duration", "0")))
    if not 7.5 <= duration <= 8.5:
        raise RuntimeError(f"unexpected clip duration: {duration}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic deployed-image smoke test")
    parser.add_argument("--output-root", type=Path, default=Path("smoke-artifacts"))
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    input_dir = output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_path = input_dir / "source.mp4"
    subtitle_path = input_dir / "source.vtt"
    brief_path = input_dir / "brief.json"

    _generate_source(source_path)
    _write_vtt(subtitle_path)
    _write_json(
        brief_path,
        {
            "campaign_id": "deployment-smoke",
            "title": "AI automation for business",
            "objective": "Create a concise vertical clip about saving time",
            "keywords": ["automation", "business"],
            "required_phrases": ["save time"],
            "allowed_video_ids": [VIDEO_ID],
            "clip_count": 1,
            "min_clip_seconds": 8,
            "max_clip_seconds": 9,
            "source_limit": 1,
            "max_clips_per_source": 1,
            "rights_confirmed": True,
            "attribution_required": False,
        },
    )

    run_dir = run_pipeline(
        brief_path,
        settings=PipelineSettings(artifact_root=output_root / "pipeline"),
        source_client=SmokeSource(source_path, subtitle_path),
        render=True,
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise RuntimeError(f"pipeline reported errors: {manifest['errors']}")
    if len(manifest.get("planned_clips", [])) != 1:
        raise RuntimeError("pipeline did not plan exactly one clip")
    rendered = manifest.get("rendered_clips", [])
    if len(rendered) != 1:
        raise RuntimeError("pipeline did not render exactly one clip")

    rendered_path = Path(rendered[0]["output_path"])
    if not rendered_path.is_file() or rendered_path.stat().st_size < 100_000:
        raise RuntimeError("rendered MP4 is absent or unexpectedly small")
    caption_path = rendered_path.with_suffix(".ass")
    if not caption_path.is_file() or r"{\ko" not in caption_path.read_text(encoding="utf-8"):
        raise RuntimeError("word-reveal ASS captions were not generated")
    tracking_path = rendered_path.with_suffix(".tracking.json")
    if not tracking_path.is_file():
        raise RuntimeError("face-tracking evidence was not generated")
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    if float(tracking.get("zoom_factor", 0)) <= 1.0:
        raise RuntimeError("tracking plan did not apply a zoom")
    if tracking.get("framing_mode") != "portrait_smart_crop":
        raise RuntimeError("tracking plan did not use portrait smart crop")
    if tracking.get("background_fill") != "none":
        raise RuntimeError("portrait render unexpectedly uses background fill")
    crop_width = int(tracking.get("crop_width", 0))
    crop_height = int(tracking.get("crop_height", 0))
    if crop_width <= 0 or crop_height <= 0:
        raise RuntimeError("tracking plan did not record portrait crop dimensions")
    if abs(crop_width / crop_height - 9 / 16) > 0.01:
        raise RuntimeError("tracking crop is not approximately 9:16")

    _run(["ffmpeg", "-v", "error", "-i", str(rendered_path), "-f", "null", "-"])
    probe = _probe(rendered_path)
    _validate_probe(probe)

    clip_path = output_root / "smoke.mp4"
    preview_path = output_root / "preview.png"
    shutil.copyfile(rendered_path, clip_path)
    shutil.copyfile(manifest_path, output_root / "manifest.json")
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "1",
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            str(preview_path),
        ]
    )

    report = {
        "status": "passed",
        "image_ref": os.getenv("CLIPPER_IMAGE_REF", "local"),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "clip": str(clip_path),
        "clip_sha256": _sha256(clip_path),
        "clip_size_bytes": clip_path.stat().st_size,
        "preview": str(preview_path),
        "caption_path": str(caption_path),
        "tracking_path": str(tracking_path),
        "tracking": tracking,
        "probe": probe,
        "manifest": manifest,
    }
    _write_json(output_root / "smoke-report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

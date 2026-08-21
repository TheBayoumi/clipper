from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.models import VideoCandidate
from clipper.pipeline import PipelineSettings, run_pipeline
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult

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


def _usage() -> InferenceUsage:
    return InferenceUsage("container-smoke", "2026-08-22T00:00:00Z", 0.0)


class SmokeSource:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
        del video, work_dir
        return self.source_path


class SmokeTranscription:
    identity = ModelIdentity("smoke-asr", "v1", "none", "deterministic", "none", "canonical-v1")

    def transcribe(
        self, source: Path, *, video_id: str, source_hash: str
    ) -> ProviderResult[CanonicalTimeline]:
        if not source.is_file():
            raise FileNotFoundError(source)
        texts = (
            "This",
            "deterministic",
            "container",
            "smoke",
            "proves",
            "the",
            "current",
            "multimodal",
            "render",
            "path",
        )
        words = tuple(
            CanonicalWord(
                f"{video_id}:w{index:07d}",
                text,
                float(index),
                float(index) + 0.8,
                None,
                0.99,
                "word_exact",
                "container-smoke",
            )
            for index, text in enumerate(texts)
        )
        return ProviderResult(
            CanonicalTimeline(video_id, source_hash, words),
            self.identity,
            _usage(),
        )


class SmokeAlignment:
    identity = ModelIdentity("smoke-align", "v1", "none", "deterministic", "none", "canonical-v1")

    def align(self, source: Path, timeline: CanonicalTimeline) -> ProviderResult[CanonicalTimeline]:
        if not source.is_file():
            raise FileNotFoundError(source)
        words = tuple(
            CanonicalWord(
                word.word_id,
                word.text,
                word.source_start,
                word.source_end,
                word.speaker_id,
                word.confidence,
                "aligned",
                word.transcript_source,
            )
            for word in timeline.words
        )
        return ProviderResult(
            CanonicalTimeline(timeline.video_id, timeline.source_hash, words),
            self.identity,
            _usage(),
        )


class SmokeDiarization:
    identity = ModelIdentity("smoke-diar", "v1", "none", "deterministic", "none", "canonical-v1")

    def diarize(
        self, source: Path, timeline: CanonicalTimeline
    ) -> ProviderResult[CanonicalTimeline]:
        if not source.is_file():
            raise FileNotFoundError(source)
        words = tuple(
            CanonicalWord(
                word.word_id,
                word.text,
                word.source_start,
                word.source_end,
                "SPEAKER_00",
                word.confidence,
                word.timing_mode,
                word.transcript_source,
            )
            for word in timeline.words
        )
        return ProviderResult(
            CanonicalTimeline(timeline.video_id, timeline.source_hash, words),
            self.identity,
            _usage(),
        )


class SmokeEditorial:
    identity = ModelIdentity(
        "smoke-editor", "v1", "none", "deterministic", "editor", "structured-json"
    )

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        if task.startswith("source_hazards:"):
            words = payload["words"]
            if not isinstance(words, list) or not words:
                raise ValueError("smoke source-hazard payload has no words")
            refs = [item["word_ref"] for item in words if isinstance(item, dict)]
            value: dict[str, Any] = {
                "segments": [
                    {
                        "start_word_id": refs[0],
                        "end_word_id": refs[-1],
                        "classification": "editorial_content",
                        "confidence": 0.99,
                        "evidence": ["deterministic smoke fixture contains editorial content"],
                    }
                ]
            }
        elif task.startswith("semantic_cores:"):
            words = payload["words"]
            if not isinstance(words, list) or not words:
                raise ValueError("smoke semantic-core payload has no words")
            refs = [item["word_ref"] for item in words if isinstance(item, dict)]
            value = {
                "cores": [
                    {
                        "core_id": "smoke-core",
                        "start_word_id": refs[0],
                        "end_word_id": refs[-1],
                        "semantic_summary": "The current multimodal render path works end to end",
                        "editorial_reason": "The fixture is a complete deterministic statement",
                        "confidence": 0.99,
                    }
                ]
            }
        elif task.startswith("narrative_envelope:"):
            core = payload["core"]
            words = payload["source_context_words"]
            if not isinstance(core, dict) or not isinstance(words, list) or not words:
                raise ValueError("smoke narrative-envelope payload is invalid")
            refs = [item["word_ref"] for item in words if isinstance(item, dict)]
            value = {
                "envelope_id": "smoke-envelope",
                "core_id": core["core_id"],
                "start_word_id": refs[0],
                "end_word_id": refs[-1],
                "required_prior_context": "",
                "required_followup_context": "",
                "setup_resolved": True,
                "payoff_resolved": True,
                "reference_resolution": [],
                "confidence": 0.99,
            }
        elif task.startswith("quality_windows:"):
            core = payload["core"]
            windows = payload["feasible_windows"]
            if not isinstance(core, dict) or not isinstance(windows, list) or not windows:
                raise ValueError("smoke quality-window payload is invalid")
            window = windows[0]
            if not isinstance(window, dict):
                raise ValueError("smoke quality-window entry is invalid")
            value = {
                "core_id": core["core_id"],
                "selected_window_id": window["window_id"],
                "decision": "PASS",
                "quality_score": 0.99,
                "rationale": "Deterministic fixture has one complete feasible moment",
                "confidence": 0.99,
            }
        else:
            raise ValueError(f"unsupported smoke editorial task: {task}")
        return ProviderResult(value, self.identity, _usage())


class SmokeVision:
    identity = ModelIdentity(
        "smoke-vision", "v1", "none", "deterministic", "vision", "structured-json"
    )

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, Any]]:
        if not frames:
            raise ValueError("smoke visual inference requires extracted frames")
        if task == "visual_timeline_scout":
            value: dict[str, Any] = {
                "events": [
                    {
                        "start": 0.0,
                        "end": 9.8,
                        "scene_id": "smoke-scene",
                        "summary": "Synthetic test pattern with a stable frame",
                        "visible_speakers": ["SPEAKER_00"],
                        "event_labels": ["deterministic_smoke"],
                        "confidence": 0.99,
                    }
                ]
            }
        elif task in {"rendered_clip_review", "rendered_clip_review_escalation"}:
            value = {
                "decision": "PASS",
                "summary": "Deterministic rendered clip is visually coherent.",
                "overall_confidence": 0.99,
                "issues": [],
            }
        else:
            raise ValueError(f"unsupported smoke vision task: {task}")
        del context
        return ProviderResult(value, self.identity, _usage())


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


def _validate_probe(probe: dict[str, Any], *, expected_duration: float) -> None:
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
    if abs(duration - expected_duration) > 0.6:
        raise RuntimeError(
            f"unexpected clip duration: {duration}; expected approximately {expected_duration}"
        )


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
    brief_path = input_dir / "brief.json"

    _generate_source(source_path)
    _write_json(
        brief_path,
        {
            "campaign_id": "deployment-smoke",
            "title": "Deterministic multimodal deployment smoke",
            "objective": "Exercise the current quality-derived production graph and render path",
            "targets": {
                "mode": "explicit",
                "videos": [
                    {
                        "video_id": VIDEO_ID,
                        "url": "https://www.youtube.com/watch?v=smoke-video",
                        "channel_id": CHANNEL_ID,
                    }
                ],
            },
            "rights": {"confirmed": True, "authorized_channels": [CHANNEL_ID]},
            "content_constraints": {"min_clip_seconds": 8, "max_clip_seconds": 10},
            "attribution_required": False,
        },
    )

    run_dir = run_pipeline(
        brief_path,
        settings=PipelineSettings(
            artifact_root=output_root / "pipeline",
            cache_root=output_root / "cache",
            render_profile="smoke",
        ),
        source_client=SmokeSource(source_path),
        editorial_provider=SmokeEditorial(),
        visual_scout_provider=SmokeVision(),
        visual_review_provider=SmokeVision(),
        transcription_provider=SmokeTranscription(),
        alignment_provider=SmokeAlignment(),
        diarization_provider=SmokeDiarization(),
        render=True,
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise RuntimeError(f"pipeline reported errors: {manifest['errors']}")
    if manifest.get("targets", {}).get("eligible_quality_moments") != 1:
        raise RuntimeError("deterministic fixture did not produce its one eligible quality moment")
    rendered = manifest.get("rendered_clips", [])
    if len(rendered) != 1:
        raise RuntimeError("deterministic eligible quality moment was not rendered and accepted")

    rendered_path = Path(rendered[0]["output_path"])
    if not rendered_path.is_file() or rendered_path.stat().st_size < 100_000:
        raise RuntimeError("rendered MP4 is absent or unexpectedly small")
    caption_path = rendered_path.with_suffix(".ass")
    if not caption_path.is_file() or r"{\ko" not in caption_path.read_text(encoding="utf-8"):
        raise RuntimeError("word-reveal ASS captions were not generated")
    tracking_path = rendered_path.with_suffix(".tracking.json")
    if not tracking_path.is_file():
        raise RuntimeError("speaker-framing evidence was not generated")
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    if abs(float(tracking.get("zoom_factor", 0)) - 1.0) > 1e-6:
        raise RuntimeError("speaker framing plan discarded source pixels with a base digital zoom")
    image_quality = tracking.get("image_quality") or {}
    if image_quality.get("digital_zoom_used") is not False:
        raise RuntimeError("tracking evidence reports digital zoom")
    if tracking.get("framing_mode") not in {
        "speaker_locked_portrait",
        "stable_portrait_fallback",
    }:
        raise RuntimeError("render did not use speaker-locked portrait framing")
    if tracking.get("background_fill") != "none":
        raise RuntimeError("render reintroduced background filler")
    if tracking.get("speaker_focus") is not True:
        raise RuntimeError("speaker focus evidence was not enabled")
    anchors = tracking.get("anchors") or []
    if len(anchors) < 2:
        raise RuntimeError("speaker framing plan did not persist a stable crop trajectory")
    crop_width = int(tracking.get("crop_width", 0))
    crop_height = int(tracking.get("crop_height", 0))
    if crop_width <= 0 or crop_height <= 0:
        raise RuntimeError("tracking plan did not record portrait crop dimensions")
    if abs(crop_width / crop_height - 9 / 16) > 0.01:
        raise RuntimeError("tracking crop is not approximately 9:16")
    render_path = rendered_path.with_suffix(".render.json")
    if not render_path.is_file():
        raise RuntimeError("render profile evidence was not generated")
    render_evidence = json.loads(render_path.read_text(encoding="utf-8"))
    if render_evidence.get("profile") != "smoke":
        raise RuntimeError("container smoke did not use the smoke render profile")
    if render_evidence.get("resampling_stages") != 1:
        raise RuntimeError("container smoke introduced redundant image resampling")
    if render_evidence.get("post_upscale_punch_in") is not False:
        raise RuntimeError("container smoke reintroduced post-upscale punch-in")

    _run(["ffmpeg", "-v", "error", "-i", str(rendered_path), "-f", "null", "-"])
    probe = _probe(rendered_path)
    expected_duration = float(rendered[0]["end"]) - float(rendered[0]["start"])
    _validate_probe(probe, expected_duration=expected_duration)

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

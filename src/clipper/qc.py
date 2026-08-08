from __future__ import annotations

import json
import re
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

from .captions import platform_caption_layout

_LOUDNORM_JSON_RE = re.compile(r"\{\s*\"input_i\".*?\}", re.DOTALL)
_SILENCE_RE = re.compile(r"silence_duration:\s*([0-9.]+)")


def _run(command: list[str], *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _fps(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" not in value:
        return _float(value)
    numerator, denominator = value.split("/", 1)
    return _float(numerator) / max(_float(denominator, 1.0), 1e-9)


def _caption_margin(path: Path) -> int | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Style: Default,"):
            fields = line.split(",")
            if len(fields) >= 22:
                try:
                    return int(fields[-2])
                except ValueError:
                    return None
    return None


def _caption_timing_mode(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines()[:8]:
        if line.startswith("; TimingMode: "):
            return line.split(":", 1)[1].strip() or None
    return None


def _caption_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tracking_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "framing_mode": payload.get("framing_mode"),
        "background_fill": payload.get("background_fill"),
        "zoom_factor": payload.get("zoom_factor"),
        "source_width": payload.get("source_width"),
        "source_height": payload.get("source_height"),
        "crop_width": payload.get("crop_width"),
        "crop_height": payload.get("crop_height"),
        "speaker_tracks": payload.get("speaker_tracks"),
        "speaker_switches": payload.get("speaker_switches"),
        "reframe_events": payload.get("reframe_events"),
        "transitions": payload.get("transitions") or [],
        "source_cuts": payload.get("source_cuts") or [],
        "image_quality": payload.get("image_quality") or {},
    }


def _render_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _transition_issues(transitions: object) -> list[str]:
    if not isinstance(transitions, list):
        return ["tracking transition evidence is malformed"]
    issues: list[str] = []
    moving: list[dict[str, Any]] = []
    for raw in transitions:
        if not isinstance(raw, dict):
            issues.append("tracking transition evidence is malformed")
            continue
        mode = str(raw.get("mode") or "")
        reason = str(raw.get("reason") or "")
        start = _float(raw.get("start"))
        end = _float(raw.get("end"))
        normalized = _float(raw.get("normalized_distance"))
        target_visible_at = _float(raw.get("target_visible_at"), start)
        if mode == "hold":
            continue
        moving.append(raw)
        if reason == "source_cut" and mode != "hard_cut":
            issues.append("source camera cut is incorrectly rendered as sliding crop motion")
        if mode == "hard_cut" and end - start > 0.05:
            issues.append("hard crop cut has a non-zero transition duration")
        if mode == "eased_reframe":
            duration = end - start
            if duration <= 0:
                issues.append("eased crop reframe has invalid duration")
            elif normalized / duration > 1.0:
                issues.append("crop reframe velocity exceeds one crop width per second")
        if reason != "source_cut" and start + 0.05 < target_visible_at:
            issues.append("crop starts reframing before the target face is visible")
    for previous, current in pairwise(moving):
        gap = _float(current.get("start")) - _float(previous.get("end"))
        previous_dx = _float(previous.get("to_x")) - _float(previous.get("from_x"))
        current_dx = _float(current.get("to_x")) - _float(current.get("from_x"))
        if (
            0 <= gap < 1.5
            and previous_dx * current_dx < 0
            and _float(previous.get("normalized_distance")) > 0.15
            and _float(current.get("normalized_distance")) > 0.15
        ):
            issues.append("back-and-forth crop oscillation detected")
            break
    return issues


def run_technical_qc(
    video_path: str | Path,
    *,
    expected_duration: float,
    caption_path: str | Path,
    tracking_path: str | Path,
    caption_platform: str = "tiktok",
    watermark_required: bool = False,
    watermark_present: bool = False,
    expected_width: int = 1080,
    expected_height: int = 1920,
    expected_fps: float = 30.0,
    render_metadata_path: str | Path | None = None,
    caption_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    video = Path(video_path)
    caption = Path(caption_path)
    tracking = Path(tracking_path)
    render_metadata = (
        Path(render_metadata_path)
        if render_metadata_path is not None
        else video.with_suffix(".render.json")
    )
    caption_audit_file = (
        Path(caption_audit_path)
        if caption_audit_path is not None
        else video.with_suffix(".caption-audit.json")
    )
    issues: list[str] = []
    if not video.is_file() or video.stat().st_size == 0:
        return {"status": "FAIL", "issues": ["rendered video is missing or empty"]}
    if not shutil.which("ffprobe") or not shutil.which("ffmpeg"):
        return {"status": "FAIL", "issues": ["ffprobe/ffmpeg is unavailable for QC"]}
    probe = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_name,codec_type,width,height,r_frame_rate,duration,bit_rate:format=duration,size,bit_rate",
            "-of",
            "json",
            str(video),
        ]
    )
    if probe.returncode != 0:
        return {"status": "FAIL", "issues": ["ffprobe could not read rendered video"]}
    payload = json.loads(probe.stdout)
    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), None)
    duration = _float(payload.get("format", {}).get("duration"))
    fps = _fps(video_stream.get("r_frame_rate") if video_stream else None)
    if video_stream is None:
        issues.append("video stream is missing")
    else:
        if int(video_stream.get("width", 0)) != expected_width:
            issues.append("unexpected output width")
        if int(video_stream.get("height", 0)) != expected_height:
            issues.append("unexpected output height")
        if abs(fps - expected_fps) > 0.15:
            issues.append("unexpected output fps")
        if video_stream.get("codec_name") != "h264":
            issues.append("video codec is not h264")
    if audio_stream is None:
        issues.append("audio stream is missing")
    elif audio_stream.get("codec_name") != "aac":
        issues.append("audio codec is not aac")
    if abs(duration - expected_duration) > 0.8:
        issues.append("render duration differs materially from edit plan")
    decode = _run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"])
    decode_pass = decode.returncode == 0
    if not decode_pass:
        issues.append("full media decode failed")
    loudness: dict[str, float | None] = {
        "integrated_lufs": None,
        "true_peak_dbfs": None,
        "lra_lu": None,
    }
    if audio_stream is not None:
        loud = _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(video),
                "-vn",
                "-af",
                "loudnorm=I=-14:LRA=11:TP=-1.5:print_format=json",
                "-f",
                "null",
                "-",
            ]
        )
        match = _LOUDNORM_JSON_RE.search(loud.stderr)
        if match:
            data = json.loads(match.group(0))
            integrated = _float(data.get("input_i"), float("nan"))
            peak = _float(data.get("input_tp"), float("nan"))
            lra = _float(data.get("input_lra"), float("nan"))
            loudness = {"integrated_lufs": integrated, "true_peak_dbfs": peak, "lra_lu": lra}
            if not -18.0 <= integrated <= -10.0:
                issues.append("integrated loudness is outside the speech-safe target band")
            if peak > -0.5:
                issues.append("true peak is too high")
        else:
            issues.append("objective loudness analysis failed")
    silence_events: list[float] = []
    if audio_stream is not None:
        silence = _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(video),
                "-vn",
                "-af",
                "silencedetect=noise=-45dB:d=1.0",
                "-f",
                "null",
                "-",
            ]
        )
        silence_events = [_float(value) for value in _SILENCE_RE.findall(silence.stderr)]
        if any(value >= 2.0 for value in silence_events):
            issues.append("long audio silence detected")
    margin = _caption_margin(caption)
    timing_mode = _caption_timing_mode(caption)
    required_margin = platform_caption_layout(caption_platform).bottom_margin_px(expected_height)
    caption_safe = margin is not None and margin >= required_margin
    if not caption_safe:
        issues.append("caption bottom margin is outside configured platform safe region")
    caption_audit = _caption_audit(caption_audit_file)
    if not caption_audit:
        issues.append("caption audit is missing or malformed")
    else:
        if caption_audit.get("alignment") != "PASS":
            issues.append("first caption does not match the first audible words")
        timing_delta = _float(caption_audit.get("first_caption_timing_delta_seconds"), 999.0)
        if timing_delta > 0.08:
            issues.append("first caption timing is not aligned with the first audible word")
    tracking_info = _tracking_evidence(tracking)
    no_filler = tracking_info.get("background_fill") == "none"
    if not no_filler:
        issues.append("tracking evidence does not confirm no-filler portrait composition")
    crop_width = int(tracking_info.get("crop_width") or 0)
    crop_height = int(tracking_info.get("crop_height") or 0)
    valid_crop = (
        crop_width > 0 and crop_height > 0 and abs(crop_width / crop_height - 9 / 16) < 0.02
    )
    if not valid_crop:
        issues.append("tracking crop is not a valid portrait crop")
    transition_issues = _transition_issues(tracking_info.get("transitions", []))
    issues.extend(transition_issues)
    image_quality = tracking_info.get("image_quality")
    image_quality = image_quality if isinstance(image_quality, dict) else {}
    max_crop_height = int(image_quality.get("max_portrait_crop_height") or 0)
    if max_crop_height and crop_height < max_crop_height * 0.92:
        issues.append("crop resolution is materially below the maximum portrait crop")
    render_info = _render_evidence(render_metadata)
    if render_info:
        if int(render_info.get("resampling_stages") or 0) > 1:
            issues.append("render uses more than one image resampling stage")
        if bool(render_info.get("post_upscale_punch_in")):
            issues.append("render uses prohibited post-upscale digital punch-in")
        if bool(render_info.get("digital_zoom_used")):
            issues.append("render uses digital zoom that discards source pixels")
    if watermark_required and not watermark_present:
        issues.append("required campaign watermark was not supplied to renderer")
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "video": {
            "path": str(video),
            "size_bytes": video.stat().st_size,
            "duration_seconds": duration,
            "expected_duration_seconds": expected_duration,
            "width": video_stream.get("width") if video_stream else None,
            "height": video_stream.get("height") if video_stream else None,
            "fps": fps,
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "decode_pass": decode_pass,
            "bit_rate_bps": _float(
                (video_stream or {}).get("bit_rate") or payload.get("format", {}).get("bit_rate")
            ),
        },
        "audio": {**loudness, "silence_events_seconds": silence_events},
        "captions": {
            "platform": caption_platform,
            "bottom_margin_px": margin,
            "required_bottom_margin_px": required_margin,
            "safe_region_pass": caption_safe,
            "timing_mode": timing_mode,
            "word_exact": timing_mode == "word_exact",
            "audit_path": str(caption_audit_file),
            "first_audio_word": caption_audit.get("first_audio_word") if caption_audit else None,
            "first_audio_words": caption_audit.get("first_audio_words") if caption_audit else None,
            "first_audio_word_time": caption_audit.get("first_audio_word_time")
            if caption_audit
            else None,
            "first_caption_text": caption_audit.get("first_caption_text")
            if caption_audit
            else None,
            "first_caption_time": caption_audit.get("first_caption_time")
            if caption_audit
            else None,
            "first_caption_timing_delta_seconds": (
                caption_audit.get("first_caption_timing_delta_seconds") if caption_audit else None
            ),
            "alignment": caption_audit.get("alignment") if caption_audit else "FAIL",
            "partial_words_dropped": caption_audit.get("partial_words_dropped")
            if caption_audit
            else None,
            "hook_overlay_suppressed_duplicate": (
                caption_audit.get("hook_overlay_suppressed_duplicate") if caption_audit else None
            ),
        },
        "framing": {
            **tracking_info,
            "no_filler_pass": no_filler,
            "valid_crop_pass": valid_crop,
            "transition_qc_pass": not transition_issues,
        },
        "image_quality": {
            **image_quality,
            "resampling_stages": render_info.get("resampling_stages") if render_info else None,
            "digital_zoom_used": render_info.get("digital_zoom_used") if render_info else None,
            "render_profile": render_info.get("profile") if render_info else None,
            "encoder_preset": render_info.get("preset") if render_info else None,
            "crf": render_info.get("crf") if render_info else None,
            "output_bit_rate_bps": _float(
                (video_stream or {}).get("bit_rate") or payload.get("format", {}).get("bit_rate")
            ),
        },
        "watermark": {"required": watermark_required, "renderer_asset_present": watermark_present},
    }

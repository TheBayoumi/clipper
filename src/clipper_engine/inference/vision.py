"""Grounded source-wide visual evidence. Model claims never certify gameplay outcomes."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

from ..sources import qa as source_qa

# Pinned model and revision already qualified as a vision contract in Clipper's
# ContentRewards work. A worker must report this exact identity and deployment SHA.
VISION_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
VISION_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
VISION_PROMPT_VERSION = "grounded-source-visual-observations-2"
VISION_SCHEMA_VERSION = "clipper-visual-observations-1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def source_visual_prompt(timestamps: list[float], *, recovery_attempt: int = 0) -> str:
    """Required exact output structure; no guessed events for unsupplied frames."""
    if not timestamps or any(
        isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t) or t < 0
        for t in timestamps
    ):
        raise ValueError("vision prompt requires finite source-clock timestamps")
    if recovery_attempt not in {0, 1}:
        raise ValueError("vision schema recovery attempt is invalid")
    skeleton = {
        "observations": [
            {
                "timestamp": value,
                "summary": "directly visible facts or no visible action",
                "event_labels": [],
                "confidence": 0.0,
            }
            for value in timestamps
        ]
    }
    recovery = (
        " Your previous reply did not match this schema. A standalone event "
        "object, events array, or omitted timestamp is invalid; regenerate "
        "the entire observations array exactly as specified."
        if recovery_attempt
        else ""
    )
    return (
        "Watch the supplied source-indexed images in order. Return ONE JSON object "
        "and nothing else. The top-level key MUST be 'observations' (plural). "
        f"There MUST be exactly {len(timestamps)} observation objects, with "
        "timestamps exactly matching the provided values in the same order. "
        "Every observation MUST have numeric timestamp and confidence from 0 to 1, "
        "string summary, and event_labels as an array of short strings. "
        "Use an EMPTY event_labels array if no relevant action is visible. "
        "Describe only visible direct evidence of gameplay action, confrontation, "
        "possible outcome, traversal or reveal. Do NOT infer unseen kills, character "
        "identity, dialogue, motives or actions between sampled frames. "
        "Use precisely this JSON structure (replace summaries, labels and confidences "
        "with your observations, but preserve timestamps and all keys): "
        + json.dumps(skeleton, separators=(",", ":"))
        + recovery
    )


class VisionProvider(Protocol):
    def inspect(self, request: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class VisionSettings:
    enabled: bool = False
    app_name: str = "clipper-neural-qualification"
    sample_interval_seconds: float = 4.0
    frame_max_edge: int = 640
    batch_frames: int = 8
    local_refine_radius_seconds: float = 0.5
    local_refine_interval_seconds: float = 0.25
    max_refinement_anchors: int = 120
    max_sample_frames: int = 1024
    deadline_seconds: float = 180.0
    max_gpu_seconds: float = 900.0
    max_estimated_usd: float = 1.0
    deployed_sha: str = ""

    @classmethod
    def from_profile(cls, config: dict[str, Any]) -> VisionSettings:
        raw = (config.get("capabilities") or {}).get("neural_visual_events") or {}
        if not isinstance(raw, dict):
            raise ValueError("neural_visual_events must be an object")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("neural_visual_events.enabled must be boolean")
        if not enabled:
            return cls()
        allowed = {
            "enabled",
            "backend",
            "app_name",
            "deployed_sha",
            "model_id",
            "model_revision",
            "sample_interval_seconds",
            "frame_max_edge",
            "batch_frames",
            "local_refine_radius_seconds",
            "local_refine_interval_seconds",
            "max_refinement_anchors",
            "max_sample_frames",
            "deadline_seconds",
            "max_gpu_seconds",
            "max_estimated_usd",
            "notes",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unsupported neural visual capability fields: {sorted(unknown)}")
        if raw.get("backend") != "modal":
            raise ValueError("enabled neural visual events require Clipper's modal backend")
        if (
            raw.get("model_id", VISION_MODEL_ID) != VISION_MODEL_ID
            or raw.get("model_revision", VISION_REVISION) != VISION_REVISION
        ):
            raise ValueError("neural profile requires the pinned validated vision model")
        values: dict[str, Any] = {}
        for key in (
            "sample_interval_seconds",
            "frame_max_edge",
            "batch_frames",
            "local_refine_radius_seconds",
            "local_refine_interval_seconds",
            "max_refinement_anchors",
            "max_sample_frames",
            "deadline_seconds",
            "max_gpu_seconds",
            "max_estimated_usd",
        ):
            if key in raw:
                values[key] = raw[key]
        values["enabled"] = True
        values["app_name"] = str(raw.get("app_name") or "clipper-neural-qualification")
        requested_sha = str(raw.get("deployed_sha") or "").strip().lower()
        github_sha = os.getenv("GITHUB_SHA", "").strip().lower()
        acceptance_sha = os.getenv("CLIPPER_ACCEPTANCE_SHA", "").strip().lower()
        if github_sha and acceptance_sha and github_sha != acceptance_sha:
            raise ValueError("neural inference runner has conflicting source SHA identities")
        run_sha = acceptance_sha or github_sha
        if requested_sha == "current-run":
            requested_sha = run_sha
        if run_sha and requested_sha != run_sha:
            raise ValueError("neural inference deployment SHA must equal execution SHA")
        values["deployed_sha"] = requested_sha
        s = cls(**values)
        if not s.app_name.startswith("clipper-") or s.app_name in {
            "clipper-open-editor",
            "clipper-production-pipeline",
        }:
            raise ValueError(
                "neural inference may not target existing shared production Modal apps"
            )
        for name in (
            "frame_max_edge",
            "batch_frames",
            "max_refinement_anchors",
            "max_sample_frames",
        ):
            value = getattr(s, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for name in (
            "sample_interval_seconds",
            "local_refine_radius_seconds",
            "local_refine_interval_seconds",
            "deadline_seconds",
            "max_gpu_seconds",
            "max_estimated_usd",
        ):
            value = getattr(s, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a numeric limit")
        if not SHA_PATTERN.fullmatch(s.deployed_sha):
            raise ValueError("neural inference requires explicit deployed 40-character Git SHA")
        if (
            not 0.5 <= s.sample_interval_seconds <= 15
            or not 128 <= s.frame_max_edge <= 960
            or not 1 <= s.batch_frames <= 16
            or not 0.10 <= s.local_refine_radius_seconds <= 2
            or not 0.10 <= s.local_refine_interval_seconds <= 1
            or not 1 <= s.max_refinement_anchors <= 300
            or not 8 <= s.max_sample_frames <= 4096
            or not 10 <= s.deadline_seconds <= 900
            or not 1 <= s.max_gpu_seconds <= 21600
            or not 0 < s.max_estimated_usd <= 100
        ):
            raise ValueError("neural inference sampling or budget is outside bounded limits")
        for value in (
            s.sample_interval_seconds,
            s.local_refine_radius_seconds,
            s.local_refine_interval_seconds,
            s.deadline_seconds,
            s.max_gpu_seconds,
            s.max_estimated_usd,
        ):
            if not math.isfinite(value):
                raise ValueError("neural inference limits must be finite")
        return s


def _source_identity(source: Path, manifest_path: Path) -> dict[str, str]:
    manifest = source_qa.verify(source, manifest_path, require_original=False)
    fields = {
        "source_key": str(manifest.get("source_key") or ""),
        "provider_asset_id": str(manifest.get("provider_asset_id") or ""),
        "provider_review_url": str(manifest.get("provider_review_url") or ""),
        "analysis_sha256": str(manifest.get("sha256") or ""),
        "derivative_type": str(manifest.get("derivative_type") or ""),
    }
    if (
        not fields["source_key"]
        or not fields["provider_asset_id"]
        or urlparse(fields["provider_review_url"]).scheme != "https"
        or not urlparse(fields["provider_review_url"]).netloc
        or not SHA256_PATTERN.fullmatch(fields["analysis_sha256"])
        or fields["derivative_type"] not in {"proxy", "source"}
    ):
        raise ValueError("neural vision requires certified source provenance")
    return fields


def _sample_times(duration: float, interval: float, frame_rate: Fraction) -> tuple[float, ...]:
    """Sample at actual certified source-frame times, not rounded wall-clock guesses."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("certified source duration must be finite and positive")
    if not isinstance(frame_rate, Fraction) or frame_rate <= 0:
        raise ValueError("neural source timing requires certified positive rational frame rate")
    maximum_index = math.floor((duration - 1e-6) * float(frame_rate))
    if maximum_index < 0:
        raise ValueError("certified source cannot supply a complete video frame")
    indices = {
        min(maximum_index, round(Fraction(str(i * interval)) * frame_rate))
        for i in range(math.ceil(duration / interval))
    }
    return tuple(round(float(Fraction(index, 1) / frame_rate), 6) for index in sorted(indices))


def _refined_sample_times(
    coarse: tuple[float, ...],
    duration: float,
    rate: Fraction,
    timeline: Any,
    config: dict[str, Any],
    source_key: str,
    settings: VisionSettings,
) -> tuple[float, ...]:
    """Cost-bounded, source-grid adaptive observations around verified and uncertain cues.

    Every independently verified outcome and source-reviewed cinematic cue gets
    priority. Remaining slots are allocated to independent audiovisual change
    peaks across the *entire* reel, not just the earliest CPU events. Coarse
    coverage is never replaced; unverified GPU output never creates an anchor.
    """
    from ..gameplay import interaction

    protected: set[float] = set()
    optional: dict[float, float] = {}
    for event in getattr(timeline, "consolidated_events", ()):
        kinds = set(getattr(event, "kinds", ()) or ())
        if not kinds.intersection({"outcome_like", "impact", "combat_burst", "contact"}):
            continue
        t = float(event.time)
        if not math.isfinite(t) or not 0 <= t < duration:
            continue
        if "outcome_like" in kinds and interaction.hostile_decision(event, config).hostile:
            protected.add(t)
        else:
            optional[t] = max(optional.get(t, 0.0), float(getattr(event, "confidence", 0.5)))
    cinematic = (config.get("cinematic_moments") or {}).get("verified_anchors") or {}
    for moment in cinematic.get(source_key) or []:
        if moment.get("verification") and "time" in moment:
            t = float(moment["time"])
            if math.isfinite(t) and 0 <= t < duration:
                protected.add(t)
    if len(protected) > settings.max_refinement_anchors:
        raise RuntimeError("verified source anchors exceed neural local-refinement budget")

    # Uniform coverage detects the detector's blind spots; audiovisual peaks
    # nominate higher-resolution sampling even without a CPU-hostile event.
    # The source timeline's low-rate fps is its own certified feature clock.
    signals = getattr(timeline, "signals", {}) or {}
    feature_fps = float(getattr(timeline, "fps", 0) or 0)
    if feature_fps > 0 and isinstance(signals, dict):
        available = [
            np.asarray(signals[name], dtype=np.float32)
            for name in ("interest", "audio_transient", "center_change", "global_motion")
            if name in signals
        ]
        if available:
            shortest = min(len(array) for array in available)
            # 0.5-second cells avoid quantizing a fast event to a 15s grid.
            stride = max(1, round(feature_fps * 0.5))
            for start in range(0, shortest, stride):
                end = min(shortest, start + stride)
                values = [
                    float(np.max(np.nan_to_num(array[start:end], nan=0.0))) for array in available
                ]
                strength = max(values)
                if strength < 0.38:
                    continue
                t = min(duration - 1e-6, (start + (end - start) / 2) / feature_fps)
                optional[round(t, 4)] = max(optional.get(round(t, 4), 0), strength)

    max_frame = math.floor((duration - 1e-6) * float(rate))
    indices = {round(Fraction(str(t)) * rate) for t in coarse}
    if len(indices) > settings.max_sample_frames:
        raise RuntimeError("source-wide neural sampling exceeds configured frame budget")

    def neighborhood(stamp: float) -> set[int]:
        radius = settings.local_refine_radius_seconds
        step = settings.local_refine_interval_seconds
        count = math.ceil((2 * radius) / step)
        return {
            max(
                0,
                min(
                    max_frame,
                    round(
                        Fraction(str(max(0.0, min(duration - 1e-6, stamp - radius + k * step))))
                        * rate
                    ),
                ),
            )
            for k in range(count + 1)
        }

    # Required anchors must never be removed to accommodate arbitrary model
    # budget constraints: fail visibly if they cannot all fit.
    for stamp in sorted(protected):
        indices.update(neighborhood(stamp))
    if len(indices) > settings.max_sample_frames:
        raise RuntimeError("verified source anchors exceed neural frame budget")
    slots = settings.max_refinement_anchors - len(protected)
    selected = sorted(protected)
    # Rank by actionable audiovisual evidence; ties prefer earlier source time
    # deterministically. Enforce spatial diversity so one explosion does not
    # consume every refinement slot.
    ordered = sorted(optional.items(), key=lambda item: (-item[1], item[0]))
    for stamp, _ in ordered:
        if slots <= 0:
            break
        if any(abs(stamp - existing) < 1.5 for existing in selected):
            continue
        additions = neighborhood(stamp)
        if len(indices | additions) > settings.max_sample_frames:
            continue
        indices.update(additions)
        selected.append(stamp)
        slots -= 1
    return tuple(round(float(Fraction(index, 1) / rate), 6) for index in sorted(indices))


def _extract_samples(
    source: Path, times: tuple[float, ...], edge: int, directory: Path
) -> tuple[Path, ...]:
    """Bounded per-sample source seek retains explicit source-clock sampling points."""
    paths: list[Path] = []
    for index, timestamp in enumerate(times):
        path = directory / f"sample_{index:06d}.jpg"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(source),
            "-vf",
            f"scale=w='min({edge},iw)':h='min({edge},ih)':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not path.is_file() or not path.stat().st_size:
            raise RuntimeError(
                f"unable to decode source-grounded visual sample {timestamp:.3f}s: "
                f"{completed.stderr[-500:]}"
            )
        paths.append(path)
    return tuple(paths)


def _parse_batch(
    response: dict[str, Any],
    *,
    identity: dict[str, str],
    requested: tuple[float, ...],
    deployed_sha: str,
) -> tuple[list[dict[str, Any]], float, float]:
    if not isinstance(response, dict):
        raise ValueError("vision worker response must be an object")
    if response.get("deployed_git_sha") != deployed_sha:
        raise RuntimeError("vision worker deployed SHA does not match requested SHA")
    model = response.get("model")
    if not isinstance(model, dict) or any(
        model.get(k) != v
        for k, v in {
            "model_id": VISION_MODEL_ID,
            "revision": VISION_REVISION,
            "prompt_version": VISION_PROMPT_VERSION,
            "schema_version": VISION_SCHEMA_VERSION,
        }.items()
    ):
        raise RuntimeError("vision model/prompt/schema identity mismatch")
    if response.get("source_identity") != identity:
        raise RuntimeError("vision worker source identity mismatch")
    raw = response.get("observations")
    if not isinstance(raw, list) or len(raw) != len(requested):
        raise ValueError("vision worker observations do not cover every supplied source frame")
    parsed: list[dict[str, Any]] = []
    for expected, item in zip(requested, raw, strict=True):
        if not isinstance(item, dict) or item.get("timestamp") != expected:
            raise ValueError("vision observation has invalid, missing, or reordered timestamp")
        labels = item.get("event_labels")
        if not isinstance(labels, list) or not all(isinstance(v, str) for v in labels):
            raise ValueError("vision observations require string event_labels")
        confidence = item.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("vision observation confidence must be finite and between 0 and 1")
        summary = item.get("summary")
        if not isinstance(summary, str) or len(summary) > 2000:
            raise ValueError("vision observation summary invalid")
        parsed.append(
            {
                "timestamp": expected,
                "event_labels": labels,
                "summary": summary,
                "confidence": float(confidence),
                "verification": "PROPOSED",
                "review_required": bool(labels),
            }
        )
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("vision inference missing usage evidence")
    gpu_seconds = usage.get("gpu_seconds")
    cost = usage.get("estimated_cost_usd")
    if any(
        isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x) or x < 0
        for x in (gpu_seconds, cost)
    ) or not usage.get("gpu_type"):
        raise ValueError("vision inference missing measured GPU accounting")
    return parsed, float(gpu_seconds), float(cost)


def analyze_source(
    source: Path,
    source_qa_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    *,
    provider: VisionProvider | None = None,
    source_key: str | None = None,
    timeline: Any | None = None,
) -> dict[str, Any] | None:
    """Optional neural scouting; none of its claims are verified semantic anchors."""
    settings = VisionSettings.from_profile(config)
    if not settings.enabled:
        return None
    started_at = time.monotonic()
    identity = _source_identity(source, source_qa_path)
    if identity["source_key"] != (source_key or source.stem):
        raise RuntimeError("neural source key does not match certified source identity")
    manifest = json.loads(source_qa_path.read_text(encoding="utf-8"))
    timing = (manifest.get("media_contract") or {}).get("video", {}).get("timing") or {}
    try:
        source_rate = Fraction(str(timing.get("nominal_rate") or ""))
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError("certified source has no rational nominal frame rate") from exc
    times = _sample_times(
        float(manifest["duration_seconds"]), settings.sample_interval_seconds, source_rate
    )
    if timeline is not None:
        times = _refined_sample_times(
            times,
            float(manifest["duration_seconds"]),
            source_rate,
            timeline,
            config,
            identity["source_key"],
            settings,
        )
    if not times:
        raise RuntimeError("certified source has no neural visual samples")
    if len(times) > settings.max_sample_frames:
        raise RuntimeError("source-wide neural sampling exceeds configured frame budget")
    from .modal import ModalVisionProvider

    engine = provider if provider is not None else ModalVisionProvider(settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "neural_checkpoints"
    cache_dir.mkdir(exist_ok=True)
    observations: list[dict[str, Any]] = []
    total_gpu_seconds = 0.0
    total_cost = 0.0
    hits = 0
    model = {
        "model_id": VISION_MODEL_ID,
        "revision": VISION_REVISION,
        "prompt_version": VISION_PROMPT_VERSION,
        "schema_version": VISION_SCHEMA_VERSION,
    }
    with tempfile.TemporaryDirectory(prefix="clipper-neural-frames-") as temporary:
        directory = Path(temporary)
        for offset in range(0, len(times), settings.batch_frames):
            batch = times[offset : offset + settings.batch_frames]
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "source_identity": identity,
                        "times": batch,
                        "model": model,
                        "deployed_sha": settings.deployed_sha,
                        "edge": settings.frame_max_edge,
                        "interval": settings.sample_interval_seconds,
                        "local_refine_radius": settings.local_refine_radius_seconds,
                        "local_refine_step": settings.local_refine_interval_seconds,
                        "max_refinement_anchors": settings.max_refinement_anchors,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            checkpoint = cache_dir / f"{fingerprint}.json"
            if checkpoint.is_file():
                persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
                if persisted.get("fingerprint") != fingerprint:
                    raise RuntimeError("neural checkpoint fingerprint mismatch")
                parsed = persisted["observations"]
                if (
                    not isinstance(parsed, list)
                    or len(parsed) != len(batch)
                    or any(
                        not isinstance(o, dict) or o.get("timestamp") != t
                        for o, t in zip(parsed, batch, strict=True)
                    )
                ):
                    raise RuntimeError("neural checkpoint contains invalid source timestamps")
                expected_integrity = hashlib.sha256(
                    json.dumps(
                        {"fingerprint": fingerprint, "observations": parsed},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if persisted.get("observations_sha256") != expected_integrity:
                    raise RuntimeError("neural checkpoint integrity mismatch")
                hits += 1
                observations.extend(parsed)
                continue
            if (
                total_gpu_seconds >= settings.max_gpu_seconds
                or total_cost >= settings.max_estimated_usd
            ):
                raise RuntimeError("neural source-wide inference budget exhausted")
            paths = _extract_samples(source, batch, settings.frame_max_edge, directory)
            request = {
                "task": "source_gameplay_visual_scout",
                "source_identity": identity,
                "frame_timestamps": list(batch),
                "frames_base64": [base64.b64encode(p.read_bytes()).decode("ascii") for p in paths],
                "expected_git_sha": settings.deployed_sha,
                "model": model,
            }
            response = engine.inspect(request)
            parsed, gpu_seconds, cost = _parse_batch(
                response, identity=identity, requested=batch, deployed_sha=settings.deployed_sha
            )
            total_gpu_seconds += gpu_seconds
            total_cost += cost
            if (
                total_gpu_seconds > settings.max_gpu_seconds
                or total_cost > settings.max_estimated_usd
            ):
                raise RuntimeError("neural source-wide inference budget exceeded")
            checkpoint_content = {"fingerprint": fingerprint, "observations": parsed}
            checkpoint_content["observations_sha256"] = hashlib.sha256(
                json.dumps(checkpoint_content, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            checkpoint.write_text(json.dumps(checkpoint_content, sort_keys=True), encoding="utf-8")
            observations.extend(parsed)
            for path in paths:
                path.unlink(missing_ok=True)
    report = {
        "source_identity": identity,
        "model": model,
        "deployed_git_sha": settings.deployed_sha,
        "source_clock": {
            "nominal_rate": str(source_rate),
            "sample_frame_indices": [round(Fraction(str(t)) * source_rate) for t in times],
        },
        "status": "PROPOSALS_ONLY",
        "coverage": {
            "expected_samples": len(times),
            "observed_samples": len(observations),
            "checkpoint_hits": hits,
            "source_wide": True,
        },
        "usage": {
            "gpu_seconds": total_gpu_seconds,
            "estimated_cost_usd": total_cost,
            "source_analysis_wall_seconds": round(time.monotonic() - started_at, 3),
            "fresh_model_batches": math.ceil(len(times) / settings.batch_frames) - hits,
            "cache_reused_batches": hits,
        },
        "observations": observations,
    }
    path = output_dir / "neural_visual_evidence.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

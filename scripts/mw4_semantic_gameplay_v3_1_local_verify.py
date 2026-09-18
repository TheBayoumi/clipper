from __future__ import annotations

import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_OUTER_RADIUS_FRACTION = 0.105


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("combat_state_verifier", {}).get("local_interaction_verifier", {})


def _candidate(event: Any) -> bool:
    kinds = set(getattr(event, "kinds", ()) or ())
    return bool(kinds & {"combat_burst", "outcome_like", "impact"})


def _merge_windows(
    events: list[tuple[int, Any]], radius: float, duration: float
) -> list[tuple[float, float, list[tuple[int, Any]]]]:
    raw: list[tuple[float, float, int, Any]] = []
    for index, event in events:
        value = float(event.time)
        raw.append((max(0.0, value - radius), min(duration, value + radius), index, event))
    raw.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, list[tuple[int, Any]]]] = []
    for start, end, index, event in raw:
        if not merged or start > merged[-1][1] + 0.05:
            merged.append((start, end, [(index, event)]))
        else:
            old_start, old_end, members = merged[-1]
            merged[-1] = (old_start, max(old_end, end), members + [(index, event)])
    return merged


def _extract_frames(
    source: Path, start: float, end: float, fps: float, width: int, height: int
) -> np.ndarray:
    duration = max(0.0, end - start)
    if duration <= 0.0:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source),
            "-an",
            "-vf",
            f"fps={fps},scale={width}:{height}:flags=fast_bilinear,format=rgb24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frame_size = width * height * 3
    count = len(raw) // frame_size
    if count <= 0:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    return np.frombuffer(raw[: count * frame_size], dtype=np.uint8).reshape(count, height, width, 3)


def _masks(
    height: int,
    width: int,
    outer_radius_fraction: float = _DEFAULT_OUTER_RADIUS_FRACTION,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Build the centered X-shaped hitmarker support and its local control annulus.

    The previous 0.085 outer-radius fraction clipped real MW4 hitmarker arms at
    the 384x216 local-refine resolution. 0.105 covers the complete centered
    transient while remaining tightly local to the reticle; the control ring,
    temporal baseline, balanced-arm term and unchanged acceptance threshold
    continue to reject global flashes and off-center HUD activity.
    """
    if not 0.085 <= float(outer_radius_fraction) <= 0.14:
        raise ValueError("hitmarker_outer_radius_fraction must be within 0.085..0.14")
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    annulus = (radius >= max(3.0, min(width, height) * 0.020)) & (
        radius <= max(11.0, min(width, height) * float(outer_radius_fraction))
    )
    band = max(1.5, min(width, height) * 0.012)
    diagonal = annulus & (np.abs(np.abs(dx) - np.abs(dy)) <= band)
    control = annulus & ~diagonal
    quadrants = (
        diagonal & (dx >= 0) & (dy >= 0),
        diagonal & (dx >= 0) & (dy < 0),
        diagonal & (dx < 0) & (dy >= 0),
        diagonal & (dx < 0) & (dy < 0),
    )
    return diagonal, control, quadrants


def _appearance(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = rgb.astype(np.float32) / 255.0
    low = np.min(frame, axis=2)
    high = np.max(frame, axis=2)
    neutral = 1.0 - (high - low)
    white = np.clip((low - 0.55) / 0.45, 0.0, 1.0) * np.clip((neutral - 0.45) / 0.55, 0.0, 1.0)
    red = np.clip(
        (frame[..., 0] - np.maximum(frame[..., 1], frame[..., 2]) - 0.08) / 0.55, 0.0, 1.0
    )
    gray = np.mean(frame, axis=2, dtype=np.float32)
    return white, red, gray


def _frame_score(
    frame: np.ndarray,
    baseline: np.ndarray,
    diagonal: np.ndarray,
    control: np.ndarray,
    quadrants: tuple[np.ndarray, ...],
) -> dict[str, float]:
    white, red, gray = _appearance(frame)
    base_white, base_red, base_gray = _appearance(baseline)
    white_delta = np.maximum(white - base_white, 0.0)
    red_delta = np.maximum(red - base_red, 0.0)
    diff = np.abs(gray - base_gray)
    diagonal_white = max(
        0.0, float(np.mean(white_delta[diagonal])) - 0.95 * float(np.mean(white_delta[control]))
    )
    diagonal_red = max(
        0.0, float(np.mean(red_delta[diagonal])) - 0.95 * float(np.mean(red_delta[control]))
    )
    specificity = max(0.0, float(np.mean(diff[diagonal])) - 0.90 * float(np.mean(diff[control])))
    control_appearance = float(np.mean(white_delta[control] + 0.75 * red_delta[control]))
    arms: list[float] = []
    for arm in quadrants:
        if not np.any(arm):
            arms.append(0.0)
            continue
        arm_signal = float(np.mean(white_delta[arm] + 0.75 * red_delta[arm]))
        arms.append(max(0.0, arm_signal - 0.95 * control_appearance))
    arms.sort(reverse=True)
    balanced = float(arms[2]) if len(arms) >= 3 else 0.0
    score = float(
        np.clip(
            3.2 * diagonal_white + 2.6 * diagonal_red + 2.0 * specificity + 1.15 * balanced,
            0.0,
            1.0,
        )
    )
    return {
        "score": score,
        "white": diagonal_white,
        "red": diagonal_red,
        "specificity": specificity,
        "arms": balanced,
    }


def hitmarker_metrics(
    frames: np.ndarray,
    event_index: int,
    fps: float,
    *,
    search_before_seconds: float = 0.45,
    search_after_seconds: float = 0.38,
    baseline_far_seconds: float = 0.60,
    baseline_near_seconds: float = 0.16,
    outer_radius_fraction: float = _DEFAULT_OUTER_RADIUS_FRACTION,
) -> dict[str, float]:
    """Find a transient centered hitmarker near a consolidated semantic event."""
    empty = {
        "score": 0.0,
        "white": 0.0,
        "red": 0.0,
        "specificity": 0.0,
        "arms": 0.0,
        "offset_frames": 0.0,
    }
    if len(frames) < 4:
        return empty
    event_index = max(0, min(len(frames) - 1, int(event_index)))
    diagonal, control, quadrants = _masks(
        frames.shape[1],
        frames.shape[2],
        outer_radius_fraction,
    )
    if not np.any(diagonal) or not np.any(control):
        return empty
    before = max(1, int(math.ceil(search_before_seconds * fps)))
    after = max(1, int(math.ceil(search_after_seconds * fps)))
    left = max(0, event_index - before)
    right = min(len(frames) - 1, event_index + after)
    far = max(2, int(round(baseline_far_seconds * fps)))
    near = max(1, int(round(baseline_near_seconds * fps)))
    best = dict(empty)
    for candidate_index in range(left, right + 1):
        baseline_start = max(0, candidate_index - far)
        baseline_end = max(baseline_start, candidate_index - near)
        if baseline_end <= baseline_start:
            baseline_start = 0
            baseline_end = candidate_index
        if baseline_end <= baseline_start:
            continue
        baseline = np.median(frames[baseline_start:baseline_end].astype(np.float32), axis=0).astype(
            np.uint8
        )
        metrics = _frame_score(frames[candidate_index], baseline, diagonal, control, quadrants)
        if metrics["score"] > best["score"]:
            best = {**metrics, "offset_frames": float(candidate_index - event_index)}
    return best


def annotate_timeline(source: Path, timeline: Any, config: dict[str, Any]) -> Any:
    cfg = _cfg(config)
    if not bool(cfg.get("enabled", True)):
        raise RuntimeError(
            "local interaction verifier may not be disabled for MW4 V3.1 qualification"
        )
    events = list(timeline.consolidated_events)
    indexed = [(index, event) for index, event in enumerate(events) if _candidate(event)]
    if not indexed:
        timeline._local_interaction_diagnostics = {
            "candidate_event_count": 0,
            "verified_event_count": 0,
            "decode_window_count": 0,
        }
        return timeline
    semantic = config.get("semantic_analysis", {})
    fps = float(semantic.get("local_refine_fps", 12.0))
    width = int(semantic.get("local_refine_width", 384))
    height = int(semantic.get("local_refine_height", 216))
    radius = float(semantic.get("local_refine_radius_seconds", 0.8))
    minimum = float(cfg.get("minimum_hitmarker_score", 0.34))
    outer_radius_fraction = float(
        cfg.get("hitmarker_outer_radius_fraction", _DEFAULT_OUTER_RADIUS_FRACTION)
    )
    if not 0.085 <= outer_radius_fraction <= 0.14:
        raise RuntimeError(
            "local_interaction_verifier.hitmarker_outer_radius_fraction must be within 0.085..0.14"
        )
    search_before = float(cfg.get("event_search_before_seconds", 0.45))
    search_after = float(cfg.get("event_search_after_seconds", 0.38))
    required_radius = search_before + float(cfg.get("baseline_far_seconds", 0.60)) + 0.10
    radius = max(radius, required_radius)
    windows = _merge_windows(indexed, radius, float(timeline.duration))
    updated = list(events)
    diagnostics: list[dict[str, Any]] = []
    for start, end, members in windows:
        frames = _extract_frames(Path(source), start, end, fps, width, height)
        if len(frames) == 0:
            raise RuntimeError(
                f"local interaction verifier decoded zero frames for {start:.3f}-{end:.3f}s"
            )
        for index, event in members:
            event_index = int(round((float(event.time) - start) * fps))
            metrics = hitmarker_metrics(
                frames,
                event_index,
                fps,
                search_before_seconds=search_before,
                search_after_seconds=search_after,
                baseline_far_seconds=float(cfg.get("baseline_far_seconds", 0.60)),
                baseline_near_seconds=float(cfg.get("baseline_near_seconds", 0.16)),
                outer_radius_fraction=outer_radius_fraction,
            )
            evidence = dict(getattr(event, "evidence", {}) or {})
            evidence.update(
                {
                    "local_refine_attempted": 1.0,
                    "local_hitmarker_score": round(float(metrics["score"]), 4),
                    "local_hitmarker_white": round(float(metrics["white"]), 4),
                    "local_hitmarker_red": round(float(metrics["red"]), 4),
                    "local_hitmarker_specificity": round(float(metrics["specificity"]), 4),
                    "local_hitmarker_balanced_arms": round(float(metrics["arms"]), 4),
                    "local_hitmarker_offset_seconds": round(
                        float(metrics["offset_frames"]) / fps, 4
                    ),
                    "local_direct_interaction": 1.0 if float(metrics["score"]) >= minimum else 0.0,
                }
            )
            updated[index] = replace(event, evidence=evidence)
            diagnostics.append(
                {
                    "time": round(float(event.time), 3),
                    "kinds": list(getattr(event, "kinds", ()) or ()),
                    "score": round(float(metrics["score"]), 4),
                    "offset_seconds": round(float(metrics["offset_frames"]) / fps, 4),
                    "confirmed": float(metrics["score"]) >= minimum,
                }
            )
    timeline.consolidated_events = tuple(updated)
    timeline._local_interaction_diagnostics = {
        "candidate_event_count": len(indexed),
        "verified_event_count": sum(1 for item in diagnostics if item["confirmed"]),
        "decode_window_count": len(windows),
        "minimum_hitmarker_score": minimum,
        "hitmarker_outer_radius_fraction": round(outer_radius_fraction, 4),
        "event_search_before_seconds": search_before,
        "event_search_after_seconds": search_after,
        "policy": "candidate-centered temporal search for a spatially specific centered hitmarker using full MW4 marker-arm geometry; ambiguous motion, color and off-center HUD changes remain unknown",
        "events": diagnostics,
    }
    return timeline


def self_test() -> None:
    fps = 12.0
    height, width = 216, 384
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    hit = (radius >= 5) & (radius <= 17) & (np.abs(np.abs(dx) - np.abs(dy)) <= 2.2)
    frames = np.full((20, height, width, 3), 45, dtype=np.uint8)
    frames[10][hit] = 245
    positive = hitmarker_metrics(frames, 10, fps)
    if positive["score"] < 0.34:
        raise AssertionError(f"synthetic centered hitmarker was not confirmed: {positive}")

    wide_hit = (radius >= 18) & (radius <= 22) & (np.abs(np.abs(dx) - np.abs(dy)) <= 2.2)
    wide = np.full((20, height, width, 3), 45, dtype=np.uint8)
    wide[10][wide_hit] = 245
    wide_positive = hitmarker_metrics(wide, 10, fps)
    if wide_positive["score"] < 0.34:
        raise AssertionError(
            f"full-size centered MW4 hitmarker arms were clipped by verifier geometry: {wide_positive}"
        )

    shifted = np.full((22, height, width, 3), 45, dtype=np.uint8)
    shifted[8][hit] = 245
    shifted_positive = hitmarker_metrics(shifted, 12, fps)
    if shifted_positive["score"] < 0.34 or shifted_positive["offset_frames"] >= 0:
        raise AssertionError(
            f"early hitmarker near consolidated event was missed: {shifted_positive}"
        )
    flash = np.full((20, height, width, 3), 45, dtype=np.uint8)
    flash[10] = 180
    negative = hitmarker_metrics(flash, 10, fps)
    if negative["score"] >= 0.34:
        raise AssertionError(f"global flash was misclassified as direct interaction: {negative}")
    offcenter = np.full((20, height, width, 3), 45, dtype=np.uint8)
    offcenter[10, 35:55, 35:55] = 245
    negative2 = hitmarker_metrics(offcenter, 10, fps)
    if negative2["score"] >= 0.34:
        raise AssertionError(
            f"off-center HUD flash was misclassified as direct interaction: {negative2}"
        )
    static = np.full((20, height, width, 3), 45, dtype=np.uint8)
    static[:, hit] = 245
    negative3 = hitmarker_metrics(static, 10, fps)
    if negative3["score"] >= 0.34:
        raise AssertionError(
            f"static reticle geometry was misclassified as transient hitmarker: {negative3}"
        )
    print("MW4 local direct-interaction verifier self-test: PASS")

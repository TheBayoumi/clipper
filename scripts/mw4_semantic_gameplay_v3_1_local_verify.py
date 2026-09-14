from __future__ import annotations

import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("combat_state_verifier", {}).get("local_interaction_verifier", {})


def _candidate(event: Any) -> bool:
    kinds = set(getattr(event, "kinds", ()) or ())
    return bool(kinds & {"combat_burst", "outcome_like", "impact"})


def _merge_windows(events: list[tuple[int, Any]], radius: float, duration: float) -> list[tuple[float, float, list[tuple[int, Any]]]]:
    raw: list[tuple[float, float, int, Any]] = []
    for index, event in events:
        t = float(event.time)
        raw.append((max(0.0, t - radius), min(duration, t + radius), index, event))
    raw.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, list[tuple[int, Any]]]] = []
    for start, end, index, event in raw:
        if not merged or start > merged[-1][1] + 0.05:
            merged.append((start, end, [(index, event)]))
        else:
            old_start, old_end, members = merged[-1]
            merged[-1] = (old_start, max(old_end, end), members + [(index, event)])
    return merged


def _extract_frames(source: Path, start: float, end: float, fps: float, width: int, height: int) -> np.ndarray:
    duration = max(0.0, end - start)
    if duration <= 0.0:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", str(source),
        "-an", "-vf", f"fps={fps},scale={width}:{height}:flags=fast_bilinear,format=rgb24",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    frame_size = width * height * 3
    count = len(raw) // frame_size
    if count <= 0:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    return np.frombuffer(raw[:count * frame_size], dtype=np.uint8).reshape(count, height, width, 3)


def _masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    annulus = (radius >= max(3.0, min(width, height) * 0.020)) & (radius <= max(11.0, min(width, height) * 0.085))
    band = max(1.5, min(width, height) * 0.012)
    diag = annulus & (np.abs(np.abs(dx) - np.abs(dy)) <= band)
    control = annulus & ~diag
    quadrants = (
        diag & (dx >= 0) & (dy >= 0),
        diag & (dx >= 0) & (dy < 0),
        diag & (dx < 0) & (dy >= 0),
        diag & (dx < 0) & (dy < 0),
    )
    return diag, control, quadrants


def _appearance(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = rgb.astype(np.float32) / 255.0
    lo = np.min(f, axis=2)
    hi = np.max(f, axis=2)
    neutral = 1.0 - (hi - lo)
    white = np.clip((lo - 0.55) / 0.45, 0.0, 1.0) * np.clip((neutral - 0.45) / 0.55, 0.0, 1.0)
    red = np.clip((f[..., 0] - np.maximum(f[..., 1], f[..., 2]) - 0.08) / 0.55, 0.0, 1.0)
    gray = np.mean(f, axis=2, dtype=np.float32)
    return white, red, gray


def hitmarker_metrics(frames: np.ndarray, event_index: int, fps: float) -> dict[str, float]:
    if len(frames) < 4:
        return {"score": 0.0, "white": 0.0, "red": 0.0, "specificity": 0.0, "arms": 0.0}
    event_index = max(1, min(len(frames) - 1, int(event_index)))
    pre_far = max(0, event_index - max(2, int(round(0.55 * fps))))
    pre_near = max(pre_far + 1, event_index - max(1, int(round(0.16 * fps))))
    baseline_frames = frames[pre_far:pre_near]
    if len(baseline_frames) == 0:
        baseline_frames = frames[:event_index]
    if len(baseline_frames) == 0:
        return {"score": 0.0, "white": 0.0, "red": 0.0, "specificity": 0.0, "arms": 0.0}
    baseline = np.median(baseline_frames.astype(np.float32), axis=0).astype(np.uint8)
    base_white, base_red, base_gray = _appearance(baseline)
    diag, control, quadrants = _masks(frames.shape[1], frames.shape[2])
    if not np.any(diag) or not np.any(control):
        return {"score": 0.0, "white": 0.0, "red": 0.0, "specificity": 0.0, "arms": 0.0}

    left = max(0, event_index - 1)
    right = min(len(frames), event_index + max(2, int(round(0.32 * fps))) + 1)
    best = {"score": 0.0, "white": 0.0, "red": 0.0, "specificity": 0.0, "arms": 0.0}
    for frame in frames[left:right]:
        white, red, gray = _appearance(frame)
        white_delta = np.maximum(white - base_white, 0.0)
        red_delta = np.maximum(red - base_red, 0.0)
        diff = np.abs(gray - base_gray)
        diag_white = max(0.0, float(np.mean(white_delta[diag])) - 0.95 * float(np.mean(white_delta[control])))
        diag_red = max(0.0, float(np.mean(red_delta[diag])) - 0.95 * float(np.mean(red_delta[control])))
        specificity = max(0.0, float(np.mean(diff[diag])) - 0.90 * float(np.mean(diff[control])))
        control_appearance = float(np.mean(white_delta[control] + 0.75 * red_delta[control]))
        arm_values = []
        for arm in quadrants:
            if not np.any(arm):
                arm_values.append(0.0)
                continue
            arm_signal = float(np.mean(white_delta[arm] + 0.75 * red_delta[arm]))
            arm_values.append(max(0.0, arm_signal - 0.95 * control_appearance))
        arm_values.sort(reverse=True)
        balanced_arms = float(arm_values[2]) if len(arm_values) >= 3 else 0.0
        score = float(np.clip(
            3.2 * diag_white + 2.6 * diag_red + 2.0 * specificity + 1.15 * balanced_arms,
            0.0,
            1.0,
        ))
        if score > best["score"]:
            best = {
                "score": score,
                "white": diag_white,
                "red": diag_red,
                "specificity": specificity,
                "arms": balanced_arms,
            }
    return best


def annotate_timeline(source: Path, timeline: Any, config: dict[str, Any]) -> Any:
    cfg = _cfg(config)
    if not bool(cfg.get("enabled", True)):
        raise RuntimeError("local interaction verifier may not be disabled for MW4 V3.1 qualification")
    events = list(timeline.consolidated_events)
    indexed = [(index, event) for index, event in enumerate(events) if _candidate(event)]
    if not indexed:
        setattr(timeline, "_local_interaction_diagnostics", {
            "candidate_event_count": 0,
            "verified_event_count": 0,
            "decode_window_count": 0,
        })
        return timeline

    semantic = config.get("semantic_analysis", {})
    fps = float(semantic.get("local_refine_fps", 12.0))
    width = int(semantic.get("local_refine_width", 384))
    height = int(semantic.get("local_refine_height", 216))
    radius = float(semantic.get("local_refine_radius_seconds", 0.8))
    minimum = float(cfg.get("minimum_hitmarker_score", 0.34))
    windows = _merge_windows(indexed, radius, float(timeline.duration))
    updated = list(events)
    diagnostics: list[dict[str, Any]] = []

    for start, end, members in windows:
        frames = _extract_frames(Path(source), start, end, fps, width, height)
        if len(frames) == 0:
            raise RuntimeError(f"local interaction verifier decoded zero frames for {start:.3f}-{end:.3f}s")
        for index, event in members:
            event_index = int(round((float(event.time) - start) * fps))
            metrics = hitmarker_metrics(frames, event_index, fps)
            evidence = dict(getattr(event, "evidence", {}) or {})
            evidence.update({
                "local_refine_attempted": 1.0,
                "local_hitmarker_score": round(float(metrics["score"]), 4),
                "local_hitmarker_white": round(float(metrics["white"]), 4),
                "local_hitmarker_red": round(float(metrics["red"]), 4),
                "local_hitmarker_specificity": round(float(metrics["specificity"]), 4),
                "local_hitmarker_balanced_arms": round(float(metrics["arms"]), 4),
                "local_direct_interaction": 1.0 if float(metrics["score"]) >= minimum else 0.0,
            })
            updated[index] = replace(event, evidence=evidence)
            diagnostics.append({
                "time": round(float(event.time), 3),
                "kinds": list(getattr(event, "kinds", ()) or ()),
                "score": round(float(metrics["score"]), 4),
                "confirmed": float(metrics["score"]) >= minimum,
            })

    timeline.consolidated_events = tuple(updated)
    setattr(timeline, "_local_interaction_diagnostics", {
        "candidate_event_count": len(indexed),
        "verified_event_count": sum(1 for item in diagnostics if item["confirmed"]),
        "decode_window_count": len(windows),
        "minimum_hitmarker_score": minimum,
        "policy": "high-precision local center hitmarker geometry; ambiguous motion remains unknown and color is not used as actor identity",
        "events": diagnostics,
    })
    return timeline


def self_test() -> None:
    fps = 12.0
    height, width = 216, 384
    frames = np.full((14, height, width, 3), 45, dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    hit = (radius >= 5) & (radius <= 17) & (np.abs(np.abs(dx) - np.abs(dy)) <= 2.2)
    frames[7][hit] = 245
    positive = hitmarker_metrics(frames, 7, fps)
    if positive["score"] < 0.34:
        raise AssertionError(f"synthetic four-arm hitmarker was not confirmed: {positive}")

    flash = np.full((14, height, width, 3), 45, dtype=np.uint8)
    flash[7] = 180
    negative = hitmarker_metrics(flash, 7, fps)
    if negative["score"] >= 0.34:
        raise AssertionError(f"global flash was misclassified as hitmarker: {negative}")

    offcenter = np.full((14, height, width, 3), 45, dtype=np.uint8)
    offcenter[7, 35:55, 35:55] = 245
    negative2 = hitmarker_metrics(offcenter, 7, fps)
    if negative2["score"] >= 0.34:
        raise AssertionError(f"off-center HUD flash was misclassified as hitmarker: {negative2}")
    print("MW4 local interaction verifier self-test: PASS")

from __future__ import annotations

import importlib
from bisect import bisect_right
from dataclasses import dataclass, field
from itertools import pairwise
from math import hypot
from pathlib import Path
from statistics import median
from typing import Any

from .models import ClipCandidate, TranscriptSegment

cv2: Any = importlib.import_module("cv2")

DEFAULT_TARGET_ASPECT = 9 / 16
FACE_VERTICAL_POSITION = 0.38
FACE_ANALYSIS_WIDTH = 480
FACE_RECOVERY_ANALYSIS_WIDTH = 960


@dataclass(frozen=True, slots=True)
class FaceAnchor:
    time: float
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"time": self.time, "x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class CameraTransition:
    reason: str
    start: float
    end: float
    distance_px: float
    crop_width: int
    normalized_distance: float
    mode: str
    from_x: float
    from_y: float
    to_x: float
    to_y: float
    target_visible_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "start": self.start,
            "end": self.end,
            "distance_px": self.distance_px,
            "crop_width": self.crop_width,
            "normalized_distance": self.normalized_distance,
            "mode": self.mode,
            "from_x": self.from_x,
            "from_y": self.from_y,
            "to_x": self.to_x,
            "to_y": self.to_y,
            "target_visible_at": self.target_visible_at,
        }


@dataclass(frozen=True, slots=True)
class FaceObservation:
    track_id: int
    time: float
    x: float
    y: float
    width: float
    height: float
    mouth_motion: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def to_dict(self) -> dict[str, float | int]:
        return {
            "track_id": self.track_id,
            "time": self.time,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(slots=True)
class _TrackState:
    track_id: int
    observations: list[FaceObservation] = field(default_factory=list)
    last_box: tuple[float, float, float, float] | None = None
    last_mouth: Any | None = None


@dataclass(frozen=True, slots=True)
class _SpeakerWindow:
    start: float
    end: float
    speaker_id: str | None = None


@dataclass(frozen=True, slots=True)
class TrackingPlan:
    zoom_factor: float
    source_width: int
    source_height: int
    anchors: tuple[FaceAnchor, ...] = ()
    face_detected: bool = False
    crop_width: int = 0
    crop_height: int = 0
    target_aspect: float = DEFAULT_TARGET_ASPECT
    framing_mode: str = "speaker_locked_portrait"
    background_fill: str = "none"
    speaker_focus: bool = True
    speaker_tracks: int = 0
    speaker_switches: int = 0
    reframe_events: int = 0
    transitions: tuple[CameraTransition, ...] = ()
    source_cuts: tuple[float, ...] = ()
    selected_faces: tuple[FaceObservation, ...] = ()
    safe_fallback_shots: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, object]:
        max_crop_width, max_crop_height = portrait_crop_dimensions(
            self.source_width, self.source_height, target_aspect=self.target_aspect, zoom_factor=1.0
        )
        horizontal_scale = 1080 / self.crop_width if self.crop_width else 0.0
        vertical_scale = 1920 / self.crop_height if self.crop_height else 0.0
        composition = _composition_evidence(self)
        shot_coverage = _shot_coverage_evidence(self)
        return {
            "zoom_factor": self.zoom_factor,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "target_aspect": self.target_aspect,
            "framing_mode": self.framing_mode,
            "background_fill": self.background_fill,
            "speaker_focus": self.speaker_focus,
            "speaker_tracks": self.speaker_tracks,
            "speaker_switches": self.speaker_switches,
            "reframe_events": self.reframe_events,
            "face_detected": self.face_detected,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "transitions": [item.to_dict() for item in self.transitions],
            "source_cuts": list(self.source_cuts),
            "selected_face_samples": composition["samples"],
            "composition": composition["summary"],
            "shot_coverage": shot_coverage,
            "image_quality": {
                "source_width": self.source_width,
                "source_height": self.source_height,
                "crop_width": self.crop_width,
                "crop_height": self.crop_height,
                "max_portrait_crop_width": max_crop_width,
                "max_portrait_crop_height": max_crop_height,
                "horizontal_scale": horizontal_scale,
                "vertical_scale": vertical_scale,
                "effective_upscale_factor": max(horizontal_scale, vertical_scale),
                "digital_zoom_used": self.zoom_factor > 1.0001,
            },
        }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _scene_signature(frame: Any) -> Any:
    height, width = frame.shape[:2]
    thumb_width = 96
    thumb_height = max(1, round(height * thumb_width / max(width, 1)))
    thumbnail = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)


def _scene_change_score(previous: Any, current: Any) -> float:
    if previous is None or current is None or previous.shape != current.shape:
        return 0.0
    return float(cv2.mean(cv2.absdiff(previous, current))[0]) / 255.0


def _nearest_source_cut(
    source_cuts: tuple[float, ...], at: float, tolerance: float = 0.45
) -> float | None:
    if not source_cuts:
        return None
    candidate = min(source_cuts, key=lambda value: abs(value - at))
    return candidate if abs(candidate - at) <= tolerance else None


def _even_floor(value: float) -> int:
    rounded = max(2, int(value))
    return rounded if rounded % 2 == 0 else rounded - 1


def portrait_crop_dimensions(
    source_width: int,
    source_height: int,
    *,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    zoom_factor: float = 1.0,
) -> tuple[int, int]:
    """Return an even-sized source crop that fills the target portrait aspect ratio."""
    if source_width <= 0 or source_height <= 0:
        return 0, 0
    if target_aspect <= 0:
        raise ValueError("target_aspect must be positive")
    if zoom_factor < 1.0:
        raise ValueError("zoom_factor must be at least 1.0")

    source_aspect = source_width / source_height
    if source_aspect >= target_aspect:
        max_height = source_height / zoom_factor
        crop_width = _even_floor(max_height * target_aspect)
        crop_height = _even_floor(crop_width / target_aspect)
    else:
        max_width = source_width / zoom_factor
        crop_height = _even_floor(max_width / target_aspect)
        crop_width = _even_floor(crop_height * target_aspect)

    crop_width = min(crop_width, source_width - source_width % 2)
    crop_height = min(crop_height, source_height - source_height % 2)
    return max(2, crop_width), max(2, crop_height)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, width, height = box
    return x + width / 2, y + height / 2


def _box_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _face_box_plausible(
    box: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> bool:
    """Reject small edge graphics that Haar can mistake for faces."""
    x, y, width, height = box
    if frame_width <= 0 or frame_height <= 0:
        return False
    center_x = (x + width / 2) / frame_width
    center_y = (y + height / 2) / frame_height
    size_fraction = max(width / frame_width, height / frame_height)
    return size_fraction >= 0.15 and 0.05 <= center_x <= 0.95 and 0.12 <= center_y <= 0.72


def _dedupe_face_boxes(
    boxes: list[tuple[int, int, int, int]], *, iou_threshold: float = 0.35
) -> list[tuple[int, int, int, int]]:
    selected: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        candidate = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if any(
            _box_iou(
                candidate,
                (float(current[0]), float(current[1]), float(current[2]), float(current[3])),
            )
            >= iou_threshold
            for current in selected
        ):
            continue
        selected.append(box)
    return sorted(selected, key=lambda item: item[0])


def _detect_face_boxes(
    gray: Any, frontal_detector: Any, profile_detector: Any
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not frontal_detector.empty():
        detected = frontal_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
        )
        boxes.extend((int(x), int(y), int(width), int(height)) for x, y, width, height in detected)
    if not profile_detector.empty():
        detected = profile_detector.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24)
        )
        boxes.extend((int(x), int(y), int(width), int(height)) for x, y, width, height in detected)
        flipped = cv2.flip(gray, 1)
        mirrored = profile_detector.detectMultiScale(
            flipped, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24)
        )
        frame_width = gray.shape[1]
        boxes.extend(
            (frame_width - int(x) - int(width), int(y), int(width), int(height))
            for x, y, width, height in mirrored
        )
    plausible = [box for box in boxes if _face_box_plausible(box, gray.shape[1], gray.shape[0])]
    return _dedupe_face_boxes(plausible)


def _detect_face_boxes_with_recovery(
    frame: Any,
    frontal_detector: Any,
    profile_detector: Any,
) -> tuple[Any, float, list[tuple[int, int, int, int]]]:
    """Retry missed faces at higher resolution without taxing every sampled frame."""
    source_width = int(frame.shape[1])
    for analysis_width in (FACE_ANALYSIS_WIDTH, FACE_RECOVERY_ANALYSIS_WIDTH):
        scale = min(1.0, analysis_width / max(source_width, 1))
        analysis = (
            cv2.resize(frame, (round(source_width * scale), round(frame.shape[0] * scale)))
            if scale < 1.0
            else frame
        )
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        boxes = _detect_face_boxes(gray, frontal_detector, profile_detector)
        if boxes or scale >= 1.0:
            return gray, scale, boxes
    return gray, scale, boxes


def _match_track(
    box: tuple[float, float, float, float],
    tracks: list[_TrackState],
    used_track_ids: set[int],
    frame_width: int,
    frame_height: int,
) -> _TrackState | None:
    diagonal = max(1.0, (frame_width**2 + frame_height**2) ** 0.5)
    center = _box_center(box)
    best: _TrackState | None = None
    best_score = float("-inf")
    for track in tracks:
        if track.track_id in used_track_ids or track.last_box is None:
            continue
        previous_center = _box_center(track.last_box)
        distance = (
            (center[0] - previous_center[0]) ** 2 + (center[1] - previous_center[1]) ** 2
        ) ** 0.5
        iou = _box_iou(box, track.last_box)
        score = 1.8 * iou - distance / diagonal
        if score > best_score:
            best = track
            best_score = score
    return best if best is not None and best_score >= -0.16 else None


def _mouth_patch(gray: Any, box: tuple[int, int, int, int]) -> Any | None:
    x, y, width, height = box
    left = max(0, x + round(width * 0.18))
    right = min(gray.shape[1], x + round(width * 0.82))
    top = max(0, y + round(height * 0.55))
    bottom = min(gray.shape[0], y + round(height * 0.93))
    if right - left < 4 or bottom - top < 4:
        return None
    return cv2.resize(gray[top:bottom, left:right], (32, 16))


def _mouth_motion(previous: Any | None, current: Any | None) -> float:
    if previous is None or current is None:
        return 0.0
    difference = cv2.absdiff(previous, current)
    return float(cv2.mean(difference)[0] / 255.0)


def _segment_windows(
    clip: ClipCandidate,
    segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...],
    *,
    max_window_seconds: float = 0.8,
) -> tuple[_SpeakerWindow, ...]:
    if max_window_seconds <= 0:
        raise ValueError("max_window_seconds must be positive")
    windows: list[_SpeakerWindow] = []
    for segment in segments:
        start = max(clip.start, segment.start)
        end = min(clip.end, segment.end)
        if end - start < 0.08:
            continue
        cursor = start
        while cursor < end - 1e-9:
            window_end = min(end, cursor + max_window_seconds)
            if window_end - cursor >= 0.08:
                windows.append(
                    _SpeakerWindow(
                        cursor - clip.start,
                        window_end - clip.start,
                        segment.speaker_id,
                    )
                )
            cursor = window_end
    if not windows:
        return (_SpeakerWindow(0.0, clip.duration),)
    return tuple(sorted(windows, key=lambda item: item.start))


def _split_windows_at_source_cuts(
    windows: tuple[_SpeakerWindow, ...], source_cuts: tuple[float, ...]
) -> tuple[_SpeakerWindow, ...]:
    """Keep speaker decisions inside one source-camera shot."""
    if not source_cuts:
        return windows
    split: list[_SpeakerWindow] = []
    for window in windows:
        boundaries = [cut for cut in source_cuts if window.start + 0.08 <= cut <= window.end - 0.08]
        cursor = window.start
        for boundary in boundaries:
            split.append(_SpeakerWindow(cursor, boundary, window.speaker_id))
            cursor = boundary
        if window.end - cursor >= 0.08:
            split.append(_SpeakerWindow(cursor, window.end, window.speaker_id))
    return tuple(split)


def _shot_stable_targets(
    windows: tuple[_SpeakerWindow, ...],
    assignments: tuple[int | None, ...],
    targets: tuple[tuple[float, float] | None, ...],
    source_cuts: tuple[float, ...],
) -> tuple[tuple[float, float] | None, ...]:
    """Use one robust composition for each visible speaker in each source shot."""
    grouped: dict[tuple[int, int], list[tuple[float, float]]] = {}
    keys: list[tuple[int, int] | None] = []
    for window, track_id, target in zip(windows, assignments, targets, strict=True):
        if track_id is None or target is None:
            keys.append(None)
            continue
        shot_index = bisect_right(source_cuts, (window.start + window.end) / 2)
        key = (shot_index, track_id)
        grouped.setdefault(key, []).append(target)
        keys.append(key)
    representative = {
        key: (float(median(item[0] for item in values)), float(median(item[1] for item in values)))
        for key, values in grouped.items()
    }
    return tuple(representative.get(key) if key is not None else None for key in keys)


def _safe_shot_targets(
    windows: tuple[_SpeakerWindow, ...],
    targets: tuple[tuple[float, float] | None, ...],
    source_cuts: tuple[float, ...],
    fallback: tuple[float, float],
    clip_duration: float,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Fill detector gaps from the same shot, or reset an uncovered shot to center."""
    indexed = [
        (bisect_right(source_cuts, (window.start + window.end) / 2), index, target)
        for index, (window, target) in enumerate(zip(windows, targets, strict=True))
    ]
    observed_by_shot: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    for shot_index, index, target in indexed:
        if target is not None:
            observed_by_shot.setdefault(shot_index, []).append((index, target))

    safe: list[tuple[float, float]] = []
    fallback_indices: set[int] = set()
    for shot_index, index, target in indexed:
        if target is not None:
            safe.append(target)
            continue
        observed = observed_by_shot.get(shot_index, [])
        if observed:
            safe.append(min(observed, key=lambda item: abs(item[0] - index))[1])
            continue
        safe.append(fallback)
        fallback_indices.add(shot_index)

    boundaries = (0.0, *source_cuts, clip_duration)
    fallback_shots = tuple(
        (boundaries[index], boundaries[index + 1])
        for index in sorted(fallback_indices)
        if index + 1 < len(boundaries)
    )
    return tuple(safe), fallback_shots


def _speaker_score(
    track: _TrackState,
    window: _SpeakerWindow,
    *,
    sample_fps: float = 4.0,
    min_detection_coverage: float = 0.35,
) -> tuple[float, int]:
    observations = [item for item in track.observations if window.start <= item.time <= window.end]
    if not observations:
        return -1.0, 0
    expected_samples = max(1.0, (window.end - window.start) * max(1.0, sample_fps))
    coverage = min(1.0, len(observations) / expected_samples)
    if coverage < min_detection_coverage:
        return -1.0, len(observations)
    motions = sorted((item.mouth_motion for item in observations), reverse=True)
    active_count = max(1, (len(motions) + 1) // 2)
    active_motion = sum(motions[:active_count]) / active_count
    coverage_weight = 0.55 + 0.45 * coverage
    visibility_bonus = min(0.002, len(observations) * 0.00025)
    return active_motion * coverage_weight + visibility_bonus, len(observations)


def _choose_active_speaker(
    tracks: list[_TrackState],
    window: _SpeakerWindow,
    previous_track_id: int | None,
    *,
    switch_margin: float = 1.35,
    sample_fps: float = 4.0,
    min_detection_coverage: float = 0.35,
) -> int | None:
    scored = [
        (
            track.track_id,
            *_speaker_score(
                track,
                window,
                sample_fps=sample_fps,
                min_detection_coverage=min_detection_coverage,
            ),
        )
        for track in tracks
    ]
    visible = [item for item in scored if item[1] >= 0 and item[2] > 0]
    if not visible:
        return None
    if len(visible) == 1:
        return visible[0][0]
    visible.sort(key=lambda item: item[1], reverse=True)
    best_id, best_score, _ = visible[0]
    if previous_track_id is None:
        return best_id
    previous = next((item for item in visible if item[0] == previous_track_id), None)
    if previous is None:
        return best_id
    previous_score = previous[1]
    if best_id == previous_track_id:
        return previous_track_id
    required = max(previous_score * switch_margin, previous_score + 0.012)
    return best_id if best_score >= required else previous_track_id


def _speaker_window_crop(
    track: _TrackState,
    window: _SpeakerWindow,
    crop_width: int,
    crop_height: int,
    source_width: int,
    source_height: int,
) -> tuple[float, float] | None:
    """Return one stable crop using only observations inside this decision window."""
    observations = [item for item in track.observations if window.start <= item.time <= window.end]
    if not observations:
        return None
    center_x = float(median(item.center[0] for item in observations))
    center_y = float(median(item.center[1] for item in observations))
    max_x = max(0.0, source_width - crop_width)
    max_y = max(0.0, source_height - crop_height)
    return (
        _clamp(center_x - crop_width / 2, 0.0, max_x),
        _clamp(center_y - crop_height * FACE_VERTICAL_POSITION, 0.0, max_y),
    )


def _speaker_window_ready_time(track: _TrackState, window: _SpeakerWindow) -> float:
    """Delay a reframe until the selected face is actually visible in the source shot."""
    times = [item.time for item in track.observations if window.start <= item.time <= window.end]
    return min(times) if times else window.start


def _speaker_change_reverses_soon(
    index: int,
    current_id: int | None,
    windows: tuple[_SpeakerWindow, ...],
    assignments: tuple[int | None, ...],
    source_cuts: tuple[float, ...],
    guard_seconds: float,
) -> bool:
    candidate_id = assignments[index]
    if current_id is None or candidate_id is None or candidate_id == current_id:
        return False
    switch_time = windows[index].start
    for future_index in range(index + 1, len(windows)):
        future_time = windows[future_index].start
        if future_time - switch_time > guard_seconds:
            break
        future_id = assignments[future_index]
        if future_id is None or future_id == candidate_id:
            continue
        if any(switch_time < cut <= future_time for cut in source_cuts):
            return False
        return future_id == current_id
    return False


def _speaker_locked_anchors(
    clip_duration: float,
    windows: tuple[_SpeakerWindow, ...],
    assignments: tuple[int | None, ...],
    targets: tuple[tuple[float, float] | None, ...],
    ready_times: tuple[float, ...],
    fallback: tuple[float, float],
    *,
    crop_width: int,
    crop_height: int,
    source_cuts: tuple[float, ...] = (),
    min_reframe_seconds: float = 0.35,
    max_reframe_seconds: float = 0.9,
    seconds_per_crop: float = 0.75,
    speaker_hold_threshold: float = 0.28,
    speaker_reversal_guard_seconds: float = 2.0,
) -> tuple[
    tuple[FaceAnchor, ...],
    tuple[CameraTransition, ...],
    int,
    tuple[int | None, ...],
]:
    current = next((target for target in targets if target is not None), fallback)
    current_id = next(
        (
            track_id
            for track_id, target in zip(assignments, targets, strict=True)
            if target is not None
        ),
        None,
    )
    anchors = [FaceAnchor(0.0, *current)]
    transitions: list[CameraTransition] = []
    rendered_assignments: list[int | None] = []
    reframe_events = 0
    for index, (window, track_id, target, ready_time) in enumerate(
        zip(windows, assignments, targets, ready_times, strict=True)
    ):
        if target is None:
            rendered_assignments.append(current_id)
            continue
        speaker_changed = track_id is not None and current_id is not None and track_id != current_id
        dx_signed = target[0] - current[0]
        dy_signed = target[1] - current[1]
        distance = hypot(dx_signed, dy_signed)
        normalized = distance / max(float(crop_width), 1.0)
        switch_at = _clamp(max(window.start, ready_time), 0.0, clip_duration)
        nearby_cut = _nearest_source_cut(source_cuts, switch_at)
        reverses_soon = speaker_changed and _speaker_change_reverses_soon(
            index,
            current_id,
            windows,
            assignments,
            source_cuts,
            speaker_reversal_guard_seconds,
        )

        if reverses_soon:
            rendered_assignments.append(current_id)
            transitions.append(
                CameraTransition(
                    "speaker_change",
                    switch_at,
                    switch_at,
                    distance,
                    crop_width,
                    normalized,
                    "hold",
                    current[0],
                    current[1],
                    target[0],
                    target[1],
                    ready_time,
                )
            )
            continue

        if speaker_changed and nearby_cut is None and normalized <= speaker_hold_threshold:
            transitions.append(
                CameraTransition(
                    "speaker_change",
                    switch_at,
                    switch_at,
                    distance,
                    crop_width,
                    normalized,
                    "hold",
                    current[0],
                    current[1],
                    target[0],
                    target[1],
                    ready_time,
                )
            )
            current_id = track_id
            rendered_assignments.append(current_id)
            continue
        if not speaker_changed and nearby_cut is None:
            rendered_assignments.append(current_id)
            if distance > 0.0:
                transitions.append(
                    CameraTransition(
                        "no_change_required",
                        switch_at,
                        switch_at,
                        distance,
                        crop_width,
                        normalized,
                        "hold",
                        current[0],
                        current[1],
                        target[0],
                        target[1],
                        ready_time,
                    )
                )
            continue

        if nearby_cut is not None:
            transition_start = _clamp(nearby_cut, 0.0, clip_duration)
            transition_end = transition_start
            reason = "source_cut"
            mode = "hard_cut"
        elif speaker_changed:
            transition_start = switch_at
            transition_end = switch_at
            reason = "speaker_change"
            mode = "hard_cut"
        else:  # Defensive: same-speaker movement was held above.
            rendered_assignments.append(current_id)
            continue

        if anchors[-1].time < transition_start:
            anchors.append(FaceAnchor(transition_start, *current))
        anchors.append(FaceAnchor(transition_end, *target))
        transitions.append(
            CameraTransition(
                reason,
                transition_start,
                transition_end,
                distance,
                crop_width,
                normalized,
                mode,
                current[0],
                current[1],
                target[0],
                target[1],
                ready_time,
            )
        )
        current = target
        current_id = track_id if track_id is not None else current_id
        rendered_assignments.append(current_id)
        reframe_events += 1
    if anchors[-1].time < clip_duration:
        anchors.append(FaceAnchor(clip_duration, *current))
    return tuple(anchors), tuple(transitions), reframe_events, tuple(rendered_assignments)


def _piecewise_expression(anchors: tuple[FaceAnchor, ...], axis: str) -> str:
    values = [anchor.x if axis == "x" else anchor.y for anchor in anchors]
    if not anchors:
        return "(iw-ow)/2" if axis == "x" else "(ih-oh)/2"
    expression = f"{values[-1]:.3f}"
    for index in range(len(anchors) - 2, -1, -1):
        current = anchors[index]
        following = anchors[index + 1]
        current_value = values[index]
        following_value = values[index + 1]
        duration = max(0.001, following.time - current.time)
        slope = (following_value - current_value) / duration
        linear = f"{current_value:.3f}+({slope:.6f})*(t-{current.time:.3f})"
        expression = f"if(lt(t,{following.time:.3f}),{linear},{expression})"
    return expression


def _transition_expression(
    transitions: tuple[CameraTransition, ...], axis: str, initial: float
) -> str:
    active = [item for item in transitions if item.mode != "hold"]
    if not active:
        return f"{initial:.3f}"
    value_attr = "x" if axis == "x" else "y"
    expression = f"{getattr(active[-1], 'to_' + value_attr):.3f}"
    for item in reversed(active):
        before = getattr(item, "from_" + value_attr)
        after = getattr(item, "to_" + value_attr)
        if item.mode == "hard_cut" or item.end <= item.start:
            expression = f"if(lt(t,{item.start:.3f}),{before:.3f},{expression})"
            continue
        duration = item.end - item.start
        unit = f"((t-{item.start:.3f})/{duration:.3f})"
        smooth = f"({unit}*{unit}*(3-2*{unit}))"
        eased = f"{before:.3f}+({after - before:.3f})*{smooth}"
        expression = (
            f"if(lt(t,{item.start:.3f}),{before:.3f},if(lt(t,{item.end:.3f}),{eased},{expression}))"
        )
    return expression


def _crop_position_at(plan: TrackingPlan, at: float) -> tuple[float, float]:
    if plan.anchors:
        current = (plan.anchors[0].x, plan.anchors[0].y)
    else:
        current = (
            max(0.0, (plan.source_width - plan.crop_width) / 2),
            max(0.0, (plan.source_height - plan.crop_height) / 2),
        )
    active = sorted(
        (item for item in plan.transitions if item.mode != "hold"), key=lambda item: item.start
    )
    if active:
        for item in active:
            if at < item.start:
                return current
            target = (item.to_x, item.to_y)
            if item.mode == "hard_cut" or item.end <= item.start:
                current = target
                continue
            if at < item.end:
                unit = _clamp((at - item.start) / (item.end - item.start), 0.0, 1.0)
                smooth = unit * unit * (3 - 2 * unit)
                return (
                    item.from_x + (item.to_x - item.from_x) * smooth,
                    item.from_y + (item.to_y - item.from_y) * smooth,
                )
            current = target
        return current
    for previous, following in pairwise(plan.anchors):
        if at >= following.time:
            current = (following.x, following.y)
            continue
        if at <= previous.time or following.time <= previous.time:
            return current
        unit = (at - previous.time) / (following.time - previous.time)
        return (
            previous.x + (following.x - previous.x) * unit,
            previous.y + (following.y - previous.y) * unit,
        )
    return current


def _composition_evidence(plan: TrackingPlan) -> dict[str, object]:
    if plan.crop_width <= 0 or plan.crop_height <= 0 or not plan.selected_faces:
        return {
            "samples": [],
            "summary": {
                "sample_count": 0,
                "centered_sample_ratio": None,
                "fully_visible_sample_ratio": None,
                "max_horizontal_center_offset": None,
                "face_height_ratio_min": None,
                "face_height_ratio_max": None,
            },
        }
    samples: list[dict[str, object]] = []
    centered: list[bool] = []
    visible: list[bool] = []
    height_ratios: list[float] = []
    offsets: list[float] = []
    for face in plan.selected_faces:
        crop_x, crop_y = _crop_position_at(plan, face.time)
        center_x, center_y = face.center
        relative_x = (center_x - crop_x) / plan.crop_width
        relative_y = (center_y - crop_y) / plan.crop_height
        height_ratio = face.height / plan.crop_height
        center_offset = abs(relative_x - 0.5)
        fully_visible = bool(
            face.x >= crop_x
            and face.x + face.width <= crop_x + plan.crop_width
            and face.y >= crop_y
            and face.y + face.height <= crop_y + plan.crop_height
        )
        samples.append(
            {
                **face.to_dict(),
                "crop_x": round(crop_x, 3),
                "crop_y": round(crop_y, 3),
                "center_x_ratio": round(relative_x, 4),
                "center_y_ratio": round(relative_y, 4),
                "face_width_ratio": round(face.width / plan.crop_width, 4),
                "face_height_ratio": round(height_ratio, 4),
                "fully_visible": fully_visible,
            }
        )
        centered.append(center_offset <= 0.22)
        visible.append(fully_visible)
        height_ratios.append(height_ratio)
        offsets.append(center_offset)
    return {
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "centered_sample_ratio": sum(centered) / len(centered),
            "fully_visible_sample_ratio": sum(visible) / len(visible),
            "max_horizontal_center_offset": max(offsets),
            "face_height_ratio_min": min(height_ratios),
            "face_height_ratio_max": max(height_ratios),
        },
    }


def _shot_coverage_evidence(plan: TrackingPlan) -> list[dict[str, object]]:
    """Record every source shot, including intervals where no face was selected."""
    duration = max((anchor.time for anchor in plan.anchors), default=0.0)
    if duration <= 0:
        return []
    boundaries = (0.0, *sorted(cut for cut in plan.source_cuts if 0 < cut < duration), duration)
    evidence: list[dict[str, object]] = []
    for index, (start, end) in enumerate(pairwise(boundaries)):
        sample_count = sum(
            1
            for face in plan.selected_faces
            if start <= face.time and (face.time < end or index == len(boundaries) - 2)
        )
        safe_fallback = any(
            abs(start - fallback_start) <= 0.05 and abs(end - fallback_end) <= 0.05
            for fallback_start, fallback_end in plan.safe_fallback_shots
        )
        crop_x, crop_y = _crop_position_at(plan, start + (end - start) / 2)
        center_x = max(0.0, (plan.source_width - plan.crop_width) / 2)
        center_y = max(0.0, (plan.source_height - plan.crop_height) / 2)
        evidence.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "selected_face_samples": sample_count,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "safe_centered": abs(crop_x - center_x) <= 1.0 and abs(crop_y - center_y) <= 1.0,
                "status": (
                    "face_observed"
                    if sample_count
                    else "safe_center_fallback"
                    if safe_fallback
                    else "uncovered"
                ),
            }
        )
    return evidence


def _best_stable_axis_position(
    spans: list[tuple[float, float, float]],
    *,
    crop_extent: int,
    max_position: float,
    preferred: float,
    target_ratio: float,
    safe_region: tuple[float, float] | None = None,
) -> float:
    """Choose a fixed crop axis that contains and composes the most face samples."""
    if not spans or crop_extent <= 0:
        return _clamp(preferred, 0.0, max_position)
    candidates = {
        0.0,
        max_position,
        _clamp(preferred, 0.0, max_position),
    }
    for start, stop, center in spans:
        candidates.update(
            {
                _clamp(start, 0.0, max_position),
                _clamp(stop - crop_extent, 0.0, max_position),
                _clamp(center - crop_extent * target_ratio, 0.0, max_position),
            }
        )
        if safe_region is not None:
            candidates.add(_clamp(center - crop_extent * safe_region[0], 0.0, max_position))
            candidates.add(_clamp(center - crop_extent * safe_region[1], 0.0, max_position))

    def score(position: float) -> tuple[int, int, float, float]:
        visible = sum(
            1
            for start, stop, _center in spans
            if start >= position - 1e-6 and stop <= position + crop_extent + 1e-6
        )
        if safe_region is None:
            safe = visible
        else:
            safe = sum(
                1
                for _start, _stop, center in spans
                if safe_region[0] <= (center - position) / crop_extent <= safe_region[1]
            )
        center_error = sum(
            abs((center - position) / crop_extent - target_ratio) for _start, _stop, center in spans
        )
        return visible, safe, -center_error, -abs(position - preferred)

    return max(candidates, key=score)


def _face_stable_crop_position(
    faces: list[FaceObservation],
    *,
    crop_width: int,
    crop_height: int,
    source_width: int,
    source_height: int,
    preferred: tuple[float, float],
) -> tuple[float, float]:
    max_x = max(0.0, source_width - crop_width)
    max_y = max(0.0, source_height - crop_height)
    x_spans = [(face.x, face.x + face.width, face.center[0]) for face in faces]
    y_spans = [(face.y, face.y + face.height, face.center[1]) for face in faces]
    return (
        _best_stable_axis_position(
            x_spans,
            crop_extent=crop_width,
            max_position=max_x,
            preferred=preferred[0],
            target_ratio=0.5,
            safe_region=(0.28, 0.72),
        ),
        _best_stable_axis_position(
            y_spans,
            crop_extent=crop_height,
            max_position=max_y,
            preferred=preferred[1],
            target_ratio=FACE_VERTICAL_POSITION,
        ),
    )


def stable_portrait_fallback(plan: TrackingPlan, duration: float) -> TrackingPlan:
    """Replace unstable motion with face-aware, shot-stable portrait crops."""
    if plan.source_width <= 0 or plan.source_height <= 0:
        return TrackingPlan(
            1.0,
            plan.source_width,
            plan.source_height,
            target_aspect=plan.target_aspect,
            framing_mode="stable_portrait_fallback",
            background_fill="none",
            speaker_focus=False,
        )
    crop_width, crop_height = portrait_crop_dimensions(
        plan.source_width,
        plan.source_height,
        target_aspect=plan.target_aspect,
        zoom_factor=1.0,
    )
    end = max(0.001, duration)
    max_x = max(0.0, plan.source_width - crop_width)
    max_y = max(0.0, plan.source_height - crop_height)
    center = (max_x / 2, max_y / 2)
    cuts = tuple(sorted(cut for cut in plan.source_cuts if 0.0 < cut < end))
    speaker_changes = tuple(
        sorted(
            {
                item.start
                for item in plan.transitions
                if item.mode != "hold"
                and item.reason == "speaker_change"
                and 0.0 < item.start < end
            }
        )
    )
    boundaries = tuple(sorted({0.0, *cuts, *speaker_changes, end}))
    positions: list[tuple[float, float]] = []
    intervals = tuple(pairwise(boundaries))
    for interval_index, (start, stop) in enumerate(intervals):
        faces = [
            face
            for face in plan.selected_faces
            if start <= face.time and (face.time < stop or interval_index == len(intervals) - 1)
        ]
        sample_at = start + (stop - start) / 2
        preferred = _crop_position_at(plan, sample_at)
        if faces:
            positions.append(
                _face_stable_crop_position(
                    faces,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    source_width=plan.source_width,
                    source_height=plan.source_height,
                    preferred=preferred,
                )
            )
            continue
        safe_center = any(
            abs(start - fallback_start) <= 0.05 and abs(stop - fallback_stop) <= 0.05
            for fallback_start, fallback_stop in plan.safe_fallback_shots
        )
        if safe_center:
            positions.append(center)
            continue
        candidates = [anchor for anchor in plan.anchors if start <= anchor.time < stop]
        prior = [anchor for anchor in plan.anchors if anchor.time <= start]
        if prior:
            candidates.append(prior[-1])
        if not candidates:
            following = next((anchor for anchor in plan.anchors if anchor.time >= stop), None)
            if following is not None:
                candidates.append(following)
        x = float(median(anchor.x for anchor in candidates)) if candidates else center[0]
        y = float(median(anchor.y for anchor in candidates)) if candidates else center[1]
        positions.append((_clamp(x, 0.0, max_x), _clamp(y, 0.0, max_y)))

    anchors = [FaceAnchor(0.0, *positions[0])]
    transitions: list[CameraTransition] = []
    current = positions[0]
    for boundary, target in zip(boundaries[1:-1], positions[1:], strict=True):
        distance = hypot(target[0] - current[0], target[1] - current[1])
        if distance > 0.0:
            reason = (
                "source_cut"
                if any(abs(boundary - cut) <= 0.05 for cut in cuts)
                else "speaker_change"
            )
            anchors.append(FaceAnchor(boundary, *target))
            transitions.append(
                CameraTransition(
                    reason,
                    boundary,
                    boundary,
                    distance,
                    crop_width,
                    distance / max(float(crop_width), 1.0),
                    "hard_cut",
                    current[0],
                    current[1],
                    target[0],
                    target[1],
                    boundary,
                )
            )
            current = target
    anchors.append(FaceAnchor(end, *current))
    return TrackingPlan(
        1.0,
        plan.source_width,
        plan.source_height,
        tuple(anchors),
        face_detected=plan.face_detected,
        crop_width=crop_width,
        crop_height=crop_height,
        target_aspect=plan.target_aspect,
        framing_mode="stable_portrait_fallback",
        background_fill="none",
        speaker_focus=plan.face_detected,
        speaker_tracks=plan.speaker_tracks,
        speaker_switches=0,
        reframe_events=len(transitions),
        transitions=tuple(transitions),
        source_cuts=plan.source_cuts,
        selected_faces=plan.selected_faces,
        safe_fallback_shots=plan.safe_fallback_shots,
    )


def tracking_expressions(plan: TrackingPlan | None) -> tuple[str, str]:
    if plan is None:
        return "(iw-ow)/2", "(ih-oh)/2"
    if plan.transitions:
        initial_x = plan.anchors[0].x if plan.anchors else 0.0
        initial_y = plan.anchors[0].y if plan.anchors else 0.0
        return (
            _transition_expression(plan.transitions, "x", initial_x),
            _transition_expression(plan.transitions, "y", initial_y),
        )
    if not plan.anchors:
        return "(iw-ow)/2", "(ih-oh)/2"
    return _piecewise_expression(plan.anchors, "x"), _piecewise_expression(plan.anchors, "y")


def plan_speaker_crop(
    source_path: str | Path,
    clip: ClipCandidate,
    segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...],
    *,
    zoom_factor: float = 1.0,
    sample_fps: float = 4.0,
    switch_margin: float = 1.35,
    min_reframe_seconds: float = 0.35,
    max_reframe_seconds: float = 0.9,
    seconds_per_crop: float = 0.75,
    speaker_hold_threshold: float = 0.28,
    speaker_reversal_guard_seconds: float = 2.0,
    decision_window_seconds: float = 0.8,
    min_detection_coverage: float = 0.35,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
) -> TrackingPlan:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return TrackingPlan(zoom_factor, 0, 0, target_aspect=target_aspect)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if width <= 0 or height <= 0:
        capture.release()
        return TrackingPlan(zoom_factor, 0, 0, target_aspect=target_aspect)

    crop_width, crop_height = portrait_crop_dimensions(
        width, height, target_aspect=target_aspect, zoom_factor=zoom_factor
    )
    fallback = ((width - crop_width) / 2, (height - crop_height) / 2)
    cascade_dir = Path(str(cv2.__file__)).resolve().parent / "data"
    frontal_detector = cv2.CascadeClassifier(
        str(cascade_dir / "haarcascade_frontalface_default.xml")
    )
    profile_detector = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml"))
    if frontal_detector.empty() and profile_detector.empty():
        capture.release()
        return TrackingPlan(
            zoom_factor,
            width,
            height,
            (FaceAnchor(0.0, *fallback), FaceAnchor(clip.duration, *fallback)),
            crop_width=crop_width,
            crop_height=crop_height,
            target_aspect=target_aspect,
            speaker_focus=False,
            safe_fallback_shots=((0.0, clip.duration),),
        )

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, clip.start) * 1000)
    sample_every = max(1, round(fps / max(1.0, sample_fps)))
    tracks: list[_TrackState] = []
    source_cuts: list[float] = []
    previous_scene: Any | None = None
    frame_index = 0
    next_track_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            relative_time = frame_index / fps
            if relative_time > clip.duration:
                break
            scene = _scene_signature(frame)
            if (
                previous_scene is not None
                and _scene_change_score(previous_scene, scene) >= 0.18
                and (not source_cuts or relative_time - source_cuts[-1] >= 0.4)
            ):
                source_cuts.append(relative_time)
                for known_track in tracks:
                    known_track.last_box = None
                    known_track.last_mouth = None
            previous_scene = scene
            if frame_index % sample_every:
                frame_index += 1
                continue
            gray, scale, boxes = _detect_face_boxes_with_recovery(
                frame, frontal_detector, profile_detector
            )
            used: set[int] = set()
            for analysis_box in sorted(boxes, key=lambda item: item[0]):
                ax, ay, aw, ah = analysis_box
                source_box = (ax / scale, ay / scale, aw / scale, ah / scale)
                track = _match_track(source_box, tracks, used, width, height)
                if track is None:
                    track = _TrackState(next_track_id)
                    next_track_id += 1
                    tracks.append(track)
                mouth = _mouth_patch(gray, analysis_box)
                motion = _mouth_motion(track.last_mouth, mouth)
                track.last_mouth = mouth
                track.last_box = source_box
                sx, sy, sw, sh = source_box
                track.observations.append(
                    FaceObservation(track.track_id, relative_time, sx, sy, sw, sh, motion)
                )
                used.add(track.track_id)
            frame_index += 1
    finally:
        capture.release()

    minimum_observations = max(2, round(min(sample_fps, 4.0)))
    stable_tracks = [track for track in tracks if len(track.observations) >= minimum_observations]
    if not stable_tracks:
        return TrackingPlan(
            zoom_factor,
            width,
            height,
            (FaceAnchor(0.0, *fallback), FaceAnchor(clip.duration, *fallback)),
            crop_width=crop_width,
            crop_height=crop_height,
            target_aspect=target_aspect,
            speaker_focus=True,
            safe_fallback_shots=tuple(pairwise((0.0, *source_cuts, clip.duration))),
        )

    windows = _split_windows_at_source_cuts(
        _segment_windows(clip, segments, max_window_seconds=decision_window_seconds),
        tuple(source_cuts),
    )
    assignments: list[int | None] = []
    targets: list[tuple[float, float] | None] = []
    ready_times: list[float] = []
    previous_track_id: int | None = None
    speaker_tracks_by_shot: dict[tuple[int, str], int] = {}
    tracks_by_id = {track.track_id: track for track in stable_tracks}
    for window in windows:
        shot_index = bisect_right(source_cuts, (window.start + window.end) / 2)
        speaker_key = (shot_index, window.speaker_id) if window.speaker_id is not None else None
        preferred_track_id = (
            speaker_tracks_by_shot.get(speaker_key)
            if speaker_key is not None
            else previous_track_id
        )
        active = _choose_active_speaker(
            stable_tracks,
            window,
            preferred_track_id,
            switch_margin=switch_margin,
            sample_fps=sample_fps,
            min_detection_coverage=min_detection_coverage,
        )
        assignments.append(active)
        target = (
            _speaker_window_crop(
                tracks_by_id[active], window, crop_width, crop_height, width, height
            )
            if active is not None and active in tracks_by_id
            else None
        )
        targets.append(target)
        ready_times.append(
            _speaker_window_ready_time(tracks_by_id[active], window)
            if active is not None and active in tracks_by_id
            else window.start
        )
        if active is not None:
            previous_track_id = active
            if speaker_key is not None:
                speaker_tracks_by_shot[speaker_key] = active
    stable_targets = _shot_stable_targets(
        windows,
        tuple(assignments),
        tuple(targets),
        tuple(source_cuts),
    )
    safe_targets, safe_fallback_shots = _safe_shot_targets(
        windows,
        stable_targets,
        tuple(source_cuts),
        fallback,
        clip.duration,
    )
    anchors, transitions, reframe_events, rendered_assignments = _speaker_locked_anchors(
        clip.duration,
        windows,
        tuple(assignments),
        safe_targets,
        tuple(ready_times),
        fallback,
        crop_width=crop_width,
        crop_height=crop_height,
        source_cuts=tuple(source_cuts),
        min_reframe_seconds=min_reframe_seconds,
        max_reframe_seconds=max_reframe_seconds,
        seconds_per_crop=seconds_per_crop,
        speaker_hold_threshold=speaker_hold_threshold,
        speaker_reversal_guard_seconds=speaker_reversal_guard_seconds,
    )
    switches = sum(
        1
        for previous, current in pairwise(assignments)
        if previous is not None and current is not None and previous != current
    )
    selected_faces_by_key: dict[tuple[int, float], FaceObservation] = {}
    for window, track_id in zip(windows, rendered_assignments, strict=True):
        if track_id is None or track_id not in tracks_by_id:
            continue
        for observation in tracks_by_id[track_id].observations:
            if window.start <= observation.time <= window.end:
                selected_faces_by_key[(track_id, observation.time)] = observation
    return TrackingPlan(
        zoom_factor,
        width,
        height,
        anchors,
        True,
        crop_width=crop_width,
        crop_height=crop_height,
        target_aspect=target_aspect,
        speaker_focus=True,
        speaker_tracks=len(stable_tracks),
        speaker_switches=switches,
        reframe_events=reframe_events,
        transitions=transitions,
        source_cuts=tuple(source_cuts),
        selected_faces=tuple(
            selected_faces_by_key[key]
            for key in sorted(selected_faces_by_key, key=lambda item: item[1])
        ),
        safe_fallback_shots=safe_fallback_shots,
    )

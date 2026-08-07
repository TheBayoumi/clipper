from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from .models import ClipCandidate, TranscriptSegment

cv2: Any = importlib.import_module("cv2")

DEFAULT_TARGET_ASPECT = 9 / 16
FACE_VERTICAL_POSITION = 0.38


@dataclass(frozen=True, slots=True)
class FaceAnchor:
    time: float
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"time": self.time, "x": self.x, "y": self.y}


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

    def to_dict(self) -> dict[str, object]:
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
            "face_detected": self.face_detected,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
        }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _even_floor(value: float) -> int:
    rounded = max(2, int(value))
    return rounded if rounded % 2 == 0 else rounded - 1


def portrait_crop_dimensions(
    source_width: int,
    source_height: int,
    *,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    zoom_factor: float = 1.12,
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
    clip: ClipCandidate, segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...]
) -> tuple[_SpeakerWindow, ...]:
    windows: list[_SpeakerWindow] = []
    for segment in segments:
        start = max(clip.start, segment.start)
        end = min(clip.end, segment.end)
        if end - start >= 0.08:
            windows.append(_SpeakerWindow(start - clip.start, end - clip.start))
    if not windows:
        return (_SpeakerWindow(0.0, clip.duration),)
    return tuple(sorted(windows, key=lambda item: item.start))


def _speaker_score(track: _TrackState, window: _SpeakerWindow) -> tuple[float, int]:
    observations = [item for item in track.observations if window.start <= item.time <= window.end]
    if not observations:
        return -1.0, 0
    motions = sorted((item.mouth_motion for item in observations), reverse=True)
    active_count = max(1, (len(motions) + 1) // 2)
    active_motion = sum(motions[:active_count]) / active_count
    visibility_bonus = min(0.004, len(observations) * 0.0005)
    return active_motion + visibility_bonus, len(observations)


def _choose_active_speaker(
    tracks: list[_TrackState],
    window: _SpeakerWindow,
    previous_track_id: int | None,
    *,
    switch_margin: float = 1.35,
) -> int | None:
    scored = [(track.track_id, *_speaker_score(track, window)) for track in tracks]
    visible = [item for item in scored if item[2] > 0]
    if not visible:
        return previous_track_id
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


def _speaker_home_crop(
    track: _TrackState, crop_width: int, crop_height: int, source_width: int, source_height: int
) -> tuple[float, float]:
    centers = [item.center for item in track.observations]
    if not centers:
        return (source_width - crop_width) / 2, (source_height - crop_height) / 2
    center_x = float(median(item[0] for item in centers))
    center_y = float(median(item[1] for item in centers))
    max_x = max(0.0, source_width - crop_width)
    max_y = max(0.0, source_height - crop_height)
    return (
        _clamp(center_x - crop_width / 2, 0.0, max_x),
        _clamp(center_y - crop_height * FACE_VERTICAL_POSITION, 0.0, max_y),
    )


def _speaker_locked_anchors(
    clip_duration: float,
    windows: tuple[_SpeakerWindow, ...],
    assignments: tuple[int | None, ...],
    homes: dict[int, tuple[float, float]],
    fallback: tuple[float, float],
    *,
    transition_seconds: float = 0.22,
) -> tuple[FaceAnchor, ...]:
    first_id = next((track_id for track_id in assignments if track_id in homes), None)
    current = homes[first_id] if first_id is not None else fallback
    anchors = [FaceAnchor(0.0, *current)]
    current_id = first_id
    for window, track_id in zip(windows, assignments, strict=True):
        if track_id is None or track_id == current_id or track_id not in homes:
            continue
        new_home = homes[track_id]
        switch_at = _clamp(window.start, 0.0, clip_duration)
        if anchors[-1].time < switch_at:
            anchors.append(FaceAnchor(switch_at, *current))
        transition_end = min(clip_duration, switch_at + transition_seconds)
        anchors.append(FaceAnchor(transition_end, *new_home))
        current = new_home
        current_id = track_id
    if anchors[-1].time < clip_duration:
        anchors.append(FaceAnchor(clip_duration, *current))
    return tuple(anchors)


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


def tracking_expressions(plan: TrackingPlan | None) -> tuple[str, str]:
    if plan is None or not plan.anchors:
        return "(iw-ow)/2", "(ih-oh)/2"
    return _piecewise_expression(plan.anchors, "x"), _piecewise_expression(plan.anchors, "y")


def plan_speaker_crop(
    source_path: str | Path,
    clip: ClipCandidate,
    segments: list[TranscriptSegment] | tuple[TranscriptSegment, ...],
    *,
    zoom_factor: float = 1.12,
    sample_fps: float = 4.0,
    switch_margin: float = 1.35,
    transition_seconds: float = 0.22,
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
    cascade_path = (
        Path(str(cv2.__file__)).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
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
        )

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, clip.start) * 1000)
    sample_every = max(1, round(fps / max(1.0, sample_fps)))
    tracks: list[_TrackState] = []
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
            if frame_index % sample_every:
                frame_index += 1
                continue
            scale = min(1.0, 480 / width)
            analysis = (
                cv2.resize(frame, (round(width * scale), round(height * scale)))
                if scale < 1.0
                else frame
            )
            gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
            detected = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            boxes = [(int(face[0]), int(face[1]), int(face[2]), int(face[3])) for face in detected]
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
        )

    homes = {
        track.track_id: _speaker_home_crop(track, crop_width, crop_height, width, height)
        for track in stable_tracks
    }
    windows = _segment_windows(clip, segments)
    assignments: list[int | None] = []
    previous_track_id: int | None = None
    for window in windows:
        active = _choose_active_speaker(
            stable_tracks, window, previous_track_id, switch_margin=switch_margin
        )
        assignments.append(active)
        if active is not None:
            previous_track_id = active
    anchors = _speaker_locked_anchors(
        clip.duration,
        windows,
        tuple(assignments),
        homes,
        fallback,
        transition_seconds=transition_seconds,
    )
    switches = sum(
        1
        for previous, current in pairwise(assignments)
        if previous is not None and current is not None and previous != current
    )
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
    )

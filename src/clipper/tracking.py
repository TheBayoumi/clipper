from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .models import ClipCandidate


@dataclass(frozen=True, slots=True)
class FaceAnchor:
    time: float
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"time": self.time, "x": self.x, "y": self.y}


@dataclass(frozen=True, slots=True)
class TrackingPlan:
    zoom_factor: float
    source_width: int
    source_height: int
    anchors: tuple[FaceAnchor, ...] = ()
    face_detected: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "zoom_factor": self.zoom_factor,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "face_detected": self.face_detected,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
        }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _choose_face(
    faces: list[tuple[int, int, int, int]],
    previous: tuple[float, float] | None,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float] | None:
    if not faces:
        return None
    diagonal = max(1.0, (frame_width**2 + frame_height**2) ** 0.5)

    def score(face: tuple[int, int, int, int]) -> float:
        x, y, width, height = face
        center = (x + width / 2, y + height / 2)
        area_score = (width * height) / max(1.0, frame_width * frame_height)
        if previous is None:
            return area_score
        distance = ((center[0] - previous[0]) ** 2 + (center[1] - previous[1]) ** 2) ** 0.5
        return float(area_score - 0.35 * distance / diagonal)

    x, y, width, height = max(faces, key=score)
    return x + width / 2, y + height / 2


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


def track_face_crop(
    source_path: str | Path,
    clip: ClipCandidate,
    *,
    zoom_factor: float = 1.12,
    sample_fps: float = 4.0,
    smoothing: float = 0.24,
) -> TrackingPlan:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return TrackingPlan(zoom_factor, 0, 0)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if width <= 0 or height <= 0:
        capture.release()
        return TrackingPlan(zoom_factor, 0, 0)

    cascade_path = (
        Path(str(cv2.__file__)).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        capture.release()
        return TrackingPlan(zoom_factor, width, height)

    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, clip.start) * 1000)
    sample_every = max(1, round(fps / max(1.0, sample_fps)))
    crop_width = width / zoom_factor
    crop_height = height / zoom_factor
    max_x = max(0.0, width - crop_width)
    max_y = max(0.0, height - crop_height)
    previous_face: tuple[float, float] | None = None
    smoothed: tuple[float, float] | None = None
    anchors: list[FaceAnchor] = []
    face_detected = False
    missed_samples = 0
    frame_index = 0
    emitted_at = -1.0

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
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
            faces = [(int(face[0]), int(face[1]), int(face[2]), int(face[3])) for face in detected]
            previous_scaled = (
                (previous_face[0] * scale, previous_face[1] * scale)
                if previous_face is not None
                else None
            )
            chosen = _choose_face(
                faces,
                previous_scaled,
                analysis.shape[1],
                analysis.shape[0],
            )
            if chosen is not None:
                target = (chosen[0] / scale, chosen[1] / scale)
                previous_face = target
                missed_samples = 0
                face_detected = True
            elif previous_face is not None and missed_samples < max(1, round(sample_fps)):
                target = previous_face
                missed_samples += 1
            else:
                target = (width / 2, height / 2)
                missed_samples += 1

            if smoothed is None:
                smoothed = target
            else:
                smoothed = (
                    smoothed[0] + smoothing * (target[0] - smoothed[0]),
                    smoothed[1] + smoothing * (target[1] - smoothed[1]),
                )

            if relative_time - emitted_at >= 0.5 or not anchors:
                x = _clamp(smoothed[0] - crop_width / 2, 0.0, max_x)
                y = _clamp(smoothed[1] - crop_height / 2, 0.0, max_y)
                anchors.append(FaceAnchor(relative_time, x, y))
                emitted_at = relative_time
            frame_index += 1
    finally:
        capture.release()

    if face_detected and anchors and anchors[-1].time < clip.duration:
        anchors.append(FaceAnchor(clip.duration, anchors[-1].x, anchors[-1].y))
    return TrackingPlan(
        zoom_factor,
        width,
        height,
        tuple(anchors) if face_detected else (),
        face_detected,
    )

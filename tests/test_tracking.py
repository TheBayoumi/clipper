from pathlib import Path
from unittest.mock import patch

import numpy as np

from clipper.models import ClipCandidate
from clipper.tracking import (
    FaceAnchor,
    TrackingPlan,
    _choose_face,
    track_face_crop,
    tracking_expressions,
)


def test_choose_face_prefers_largest_then_temporally_nearest() -> None:
    faces = [(20, 20, 60, 60), (300, 40, 100, 100)]
    assert _choose_face(faces, None, 640, 360) == (350.0, 90.0)
    chosen = _choose_face(faces, (50.0, 50.0), 640, 360)
    assert chosen == (50.0, 50.0)
    assert _choose_face([], None, 640, 360) is None


def test_tracking_expressions_are_continuous_piecewise_linear() -> None:
    plan = TrackingPlan(
        1.12,
        640,
        360,
        (FaceAnchor(0.0, 10.0, 20.0), FaceAnchor(1.0, 30.0, 40.0)),
        True,
    )
    x, y = tracking_expressions(plan)
    assert "if(lt(t,1.000)" in x
    assert "20.000000" in x
    assert "20.000000" in y
    assert tracking_expressions(None) == ("(iw-ow)/2", "(ih-oh)/2")


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(10)]
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        values = {3: 640.0, 4: 360.0, 5: 4.0}
        return values.get(prop, 0.0)

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def empty(self) -> bool:
        return False

    def detectMultiScale(self, *_args, **_kwargs):
        return np.array([[80, 60, 100, 100]])


def test_track_face_crop_detects_and_smooths_face_path(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 2, "text", 1)
    capture = FakeCapture()
    with (
        patch("clipper.tracking.cv2.VideoCapture", return_value=capture),
        patch("clipper.tracking.cv2.CascadeClassifier", return_value=FakeDetector()),
    ):
        plan = track_face_crop(tmp_path / "video.mp4", clip, sample_fps=4)
    assert plan.face_detected is True
    assert plan.zoom_factor == 1.12
    assert len(plan.anchors) >= 2
    assert plan.anchors[0].x == 0.0
    assert capture.released is True
    payload = plan.to_dict()
    assert payload["face_detected"] is True


def test_track_face_crop_falls_back_when_video_cannot_open(tmp_path: Path) -> None:
    with patch("clipper.tracking.cv2.VideoCapture", return_value=FakeCapture(False)):
        plan = track_face_crop(tmp_path / "missing.mp4", ClipCandidate("v", 0, 2, "t", 1))
    assert plan.face_detected is False
    assert plan.anchors == ()


class EmptyDetector:
    def empty(self) -> bool:
        return True


class NoFaceDetector:
    def empty(self) -> bool:
        return False

    def detectMultiScale(self, *_args, **_kwargs):
        return np.empty((0, 4), dtype=int)


def test_track_face_crop_handles_invalid_dimensions_and_missing_detector(tmp_path: Path) -> None:
    invalid = FakeCapture()
    invalid.get = lambda _prop: 0.0  # type: ignore[method-assign]
    with patch("clipper.tracking.cv2.VideoCapture", return_value=invalid):
        plan = track_face_crop(tmp_path / "invalid.mp4", ClipCandidate("v", 0, 1, "t", 1))
    assert plan.source_width == 0
    assert invalid.released is True

    capture = FakeCapture()
    with (
        patch("clipper.tracking.cv2.VideoCapture", return_value=capture),
        patch("clipper.tracking.cv2.CascadeClassifier", return_value=EmptyDetector()),
    ):
        plan = track_face_crop(tmp_path / "no-detector.mp4", ClipCandidate("v", 0, 1, "t", 1))
    assert plan.source_width == 640
    assert plan.face_detected is False
    assert capture.released is True


def test_track_face_crop_uses_center_fallback_when_no_face_is_seen(tmp_path: Path) -> None:
    capture = FakeCapture()
    with (
        patch("clipper.tracking.cv2.VideoCapture", return_value=capture),
        patch("clipper.tracking.cv2.CascadeClassifier", return_value=NoFaceDetector()),
    ):
        plan = track_face_crop(
            tmp_path / "no-face.mp4",
            ClipCandidate("v", 0, 1, "t", 1),
            sample_fps=4,
        )
    assert plan.face_detected is False
    assert plan.anchors == ()
    assert capture.released is True

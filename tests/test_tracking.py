from pathlib import Path
from unittest.mock import patch

import numpy as np

from clipper.models import ClipCandidate, TranscriptSegment
from clipper.tracking import (
    CameraTransition,
    FaceAnchor,
    FaceObservation,
    TrackingPlan,
    _box_iou,
    _choose_active_speaker,
    _detect_face_boxes,
    _face_box_plausible,
    _match_track,
    _mouth_motion,
    _scene_change_score,
    _segment_windows,
    _speaker_locked_anchors,
    _speaker_window_crop,
    _SpeakerWindow,
    _TrackState,
    plan_speaker_crop,
    portrait_crop_dimensions,
    tracking_expressions,
)


def _track(track_id: int, *items: FaceObservation) -> _TrackState:
    state = _TrackState(track_id)
    state.observations.extend(items)
    if items:
        last = items[-1]
        state.last_box = (last.x, last.y, last.width, last.height)
    return state


def _obs(track_id: int, time: float, x: float, motion: float) -> FaceObservation:
    return FaceObservation(track_id, time, x, 60.0, 100.0, 100.0, motion)


def test_portrait_crop_dimensions_fill_vertical_frame() -> None:
    assert portrait_crop_dimensions(1280, 720) == (404, 718)
    assert portrait_crop_dimensions(640, 360) == (202, 358)
    width, height = portrait_crop_dimensions(1080, 1920)
    assert abs(width / height - 9 / 16) < 0.01
    assert portrait_crop_dimensions(0, 720) == (0, 0)


def test_portrait_crop_dimensions_reject_invalid_parameters() -> None:
    try:
        portrait_crop_dimensions(1280, 720, target_aspect=0)
    except ValueError as exc:
        assert "target_aspect" in str(exc)
    else:
        raise AssertionError("invalid target aspect was accepted")
    try:
        portrait_crop_dimensions(1280, 720, zoom_factor=0.9)
    except ValueError as exc:
        assert "zoom_factor" in str(exc)
    else:
        raise AssertionError("invalid zoom was accepted")


def test_box_matching_preserves_face_identity_instead_of_chasing_largest_detection() -> None:
    left = _track(0, _obs(0, 0, 50, 0.01))
    right = _track(1, _obs(1, 0, 360, 0.01))
    box = (70.0, 62.0, 100.0, 100.0)
    assert _match_track(box, [left, right], set(), 640, 360) is left
    assert _match_track(box, [left, right], {0}, 640, 360) is None
    assert _box_iou((0, 0, 10, 10), (5, 5, 10, 10)) > 0
    assert _box_iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0


def test_face_box_plausibility_rejects_small_top_edge_logo_false_positive() -> None:
    assert _face_box_plausible((424, 11, 34, 34), 480, 270) is False
    assert _face_box_plausible((205, 55, 69, 69), 480, 270) is True
    assert _face_box_plausible((190, 50, 77, 77), 480, 270) is True
    assert _face_box_plausible((80, 60, 100, 100), 640, 360) is True


def test_profile_detection_finds_both_wide_shot_sides_and_dedupes() -> None:
    gray = np.zeros((270, 480), dtype=np.uint8)

    class Empty:
        def empty(self) -> bool:
            return True

    class Profile:
        def empty(self) -> bool:
            return False

        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[353, 81, 47, 47]])

    boxes = _detect_face_boxes(gray, Empty(), Profile())
    assert len(boxes) == 2
    assert boxes[0][0] < 120
    assert boxes[1][0] > 340


def test_mouth_motion_uses_only_face_mouth_patch_signal() -> None:
    previous = np.zeros((16, 32), dtype=np.uint8)
    same = previous.copy()
    changed = np.full((16, 32), 255, dtype=np.uint8)
    assert _mouth_motion(None, changed) == 0
    assert _mouth_motion(previous, same) == 0
    assert _mouth_motion(previous, changed) == 1.0


def test_active_speaker_prefers_mouth_motion_and_hysteresis() -> None:
    window = _SpeakerWindow(0.0, 1.0)
    speaker = _track(0, _obs(0, 0.2, 40, 0.08), _obs(0, 0.7, 42, 0.07))
    gesturing_listener = _track(1, _obs(1, 0.2, 300, 0.005), _obs(1, 0.7, 500, 0.006))
    assert _choose_active_speaker([speaker, gesturing_listener], window, None) == 0

    near_tie = _track(1, _obs(1, 0.2, 300, 0.085), _obs(1, 0.7, 300, 0.08))
    assert _choose_active_speaker([speaker, near_tie], window, 0, switch_margin=1.35) == 0

    clear_new_speaker = _track(1, _obs(1, 0.2, 300, 0.20), _obs(1, 0.7, 300, 0.18))
    assert _choose_active_speaker([speaker, clear_new_speaker], window, 0) == 1
    assert _choose_active_speaker([], window, 0) == 0


def test_sparse_high_motion_detection_cannot_override_persistent_speaker() -> None:
    window = _SpeakerWindow(0.0, 0.8)
    persistent = _track(
        0,
        _obs(0, 0.1, 300, 0.07),
        _obs(0, 0.35, 302, 0.065),
        _obs(0, 0.6, 304, 0.07),
    )
    cut_artifact = _track(1, _obs(1, 0.2, 100, 0.4))
    assert _choose_active_speaker([persistent, cut_artifact], window, 0) == 0


def test_segment_windows_clip_absolute_transcript_times() -> None:
    clip = ClipCandidate("v", 10.0, 15.0, "text", 1)
    segments = [
        TranscriptSegment(9.0, 10.5, "a"),
        TranscriptSegment(10.5, 12.0, "b"),
        TranscriptSegment(14.8, 16.0, "c"),
    ]
    windows = _segment_windows(clip, segments)
    assert windows[0] == _SpeakerWindow(0.0, 0.5)
    assert windows[1].start == 0.5
    assert abs(windows[1].end - 1.3) < 1e-9
    assert abs(windows[2].start - 1.3) < 1e-9
    assert windows[2].end == 2.0
    assert abs(windows[-1].start - 4.8) < 1e-9
    assert windows[-1].end == 5.0
    assert _segment_windows(clip, []) == (_SpeakerWindow(0.0, 5.0),)


def test_speaker_window_crop_uses_local_camera_composition_not_global_track_home() -> None:
    track = _track(
        0,
        _obs(0, 0.2, 100, 0.1),
        _obs(0, 0.7, 110, 0.1),
        _obs(0, 5.2, 360, 0.1),
        _obs(0, 5.7, 370, 0.1),
    )
    early = _speaker_window_crop(track, _SpeakerWindow(0.0, 1.0), 180, 320, 640, 360)
    late = _speaker_window_crop(track, _SpeakerWindow(5.0, 6.0), 180, 320, 640, 360)
    assert early is not None and late is not None
    assert early[0] < 100
    assert late[0] > 300
    assert _speaker_window_crop(track, _SpeakerWindow(9.0, 10.0), 180, 320, 640, 360) is None


def test_speaker_locked_anchors_hold_crop_inside_dead_zone_and_ease_subject_motion() -> None:
    windows = (_SpeakerWindow(0, 2), _SpeakerWindow(2, 4), _SpeakerWindow(4, 6))
    anchors, transitions, reframes = _speaker_locked_anchors(
        6.0,
        windows,
        (0, 0, 0),
        ((100.0, 0.0), (120.0, 10.0), (300.0, 0.0)),
        (0.0, 2.0, 4.0),
        (230.0, 0.0),
        crop_width=202,
        crop_height=358,
    )
    assert reframes == 1
    assert transitions[0].mode == "hold"
    assert transitions[-1].reason == "subject_motion"
    assert transitions[-1].mode == "eased_reframe"
    assert 4.35 <= transitions[-1].end <= 4.9
    plan = TrackingPlan(1.0, 640, 360, anchors, True, transitions=transitions)
    x, _ = tracking_expressions(plan)
    assert "3-2*" in x
    assert "if(lt(t,4.000)" in x
    assert tracking_expressions(None) == ("(iw-ow)/2", "(ih-oh)/2")


def test_reframe_waits_until_target_face_is_observed_and_uses_hard_speaker_cut() -> None:
    windows = (_SpeakerWindow(0, 1), _SpeakerWindow(1, 2))
    anchors, transitions, reframes = _speaker_locked_anchors(
        2.0,
        windows,
        (0, 1),
        ((100.0, 0.0), (300.0, 0.0)),
        (0.0, 1.4),
        (230.0, 0.0),
        crop_width=202,
        crop_height=358,
    )
    assert reframes == 1
    transition = transitions[-1]
    assert transition.reason == "speaker_change"
    assert transition.mode == "hard_cut"
    assert transition.start == transition.end == 1.4
    assert transition.start >= transition.target_visible_at
    assert anchors[-1] == FaceAnchor(2.0, 300.0, 0.0)


def test_same_shot_small_speaker_change_holds_instead_of_panning() -> None:
    windows = (_SpeakerWindow(0, 2), _SpeakerWindow(2, 4))
    anchors, transitions, reframes = _speaker_locked_anchors(
        4.0,
        windows,
        (0, 1),
        ((100.0, 0.0), (130.0, 10.0)),
        (0.0, 2.0),
        (230.0, 0.0),
        crop_width=202,
        crop_height=358,
    )
    assert reframes == 0
    assert transitions[-1].reason == "speaker_change"
    assert transitions[-1].mode == "hold"
    assert anchors[0].x == anchors[-1].x == 100.0


def test_source_camera_cut_changes_crop_as_hard_cut_without_slide() -> None:
    windows = (_SpeakerWindow(0, 2), _SpeakerWindow(2, 4))
    anchors, transitions, reframes = _speaker_locked_anchors(
        4.0,
        windows,
        (0, 1),
        ((80.0, 0.0), (300.0, 0.0)),
        (0.0, 2.0),
        (230.0, 0.0),
        crop_width=202,
        crop_height=358,
        source_cuts=(2.04,),
    )
    assert reframes == 1
    transition = transitions[-1]
    assert transition.reason == "source_cut"
    assert transition.mode == "hard_cut"
    assert transition.start == transition.end == 2.04
    x, _ = tracking_expressions(TrackingPlan(1.0, 640, 360, anchors, True, transitions=transitions))
    assert "if(lt(t,2.040),80.000" in x
    assert "3-2*" not in x


def test_scene_change_score_distinguishes_cut_from_static_frame() -> None:
    black = np.zeros((54, 96), dtype=np.uint8)
    white = np.full((54, 96), 255, dtype=np.uint8)
    assert _scene_change_score(black, black) == 0.0
    assert _scene_change_score(black, white) == 1.0
    assert _scene_change_score(None, white) == 0.0


def test_tracking_plan_records_source_pixel_quality_evidence() -> None:
    transition = CameraTransition(
        "speaker_change",
        2.0,
        2.0,
        240.0,
        1214,
        0.198,
        "hard_cut",
        100.0,
        0.0,
        340.0,
        0.0,
        2.0,
    )
    payload = TrackingPlan(
        1.0,
        3840,
        2160,
        (FaceAnchor(0, 100, 0), FaceAnchor(4, 340, 0)),
        True,
        crop_width=1214,
        crop_height=2158,
        transitions=(transition,),
        source_cuts=(2.0,),
    ).to_dict()
    quality = payload["image_quality"]
    assert isinstance(quality, dict)
    assert quality["digital_zoom_used"] is False
    assert quality["effective_upscale_factor"] < 1.0
    assert payload["transitions"][0]["mode"] == "hard_cut"


class FakeCapture:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(16)]
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


class EmptyDetector:
    def empty(self) -> bool:
        return True


class NoFaceDetector:
    def empty(self) -> bool:
        return False

    def detectMultiScale(self, *_args, **_kwargs):
        return np.empty((0, 4), dtype=int)


def test_plan_speaker_crop_locks_single_speaker_without_per_frame_drift(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 3, "text", 1)
    segments = [TranscriptSegment(0, 1.5, "first"), TranscriptSegment(1.5, 3, "second")]
    capture = FakeCapture()
    with (
        patch("clipper.tracking.cv2.VideoCapture", return_value=capture),
        patch(
            "clipper.tracking.cv2.CascadeClassifier",
            side_effect=[FakeDetector(), NoFaceDetector()],
        ),
    ):
        plan = plan_speaker_crop(tmp_path / "video.mp4", clip, segments, sample_fps=4)
    assert plan.face_detected is True
    assert plan.speaker_focus is True
    assert plan.speaker_tracks == 1
    assert plan.speaker_switches == 0
    assert plan.framing_mode == "speaker_locked_portrait"
    assert plan.background_fill == "none"
    assert plan.crop_width == 202
    assert plan.crop_height == 358
    assert len(plan.anchors) == 2
    assert plan.anchors[0].x == plan.anchors[-1].x
    assert capture.released is True


def test_plan_speaker_crop_fallbacks_are_static_and_safe(tmp_path: Path) -> None:
    clip = ClipCandidate("v", 0, 2, "t", 1)
    with patch("clipper.tracking.cv2.VideoCapture", return_value=FakeCapture(False)):
        plan = plan_speaker_crop(tmp_path / "missing.mp4", clip, [])
    assert plan.source_width == 0

    invalid = FakeCapture()
    invalid.get = lambda _prop: 0.0  # type: ignore[method-assign]
    with patch("clipper.tracking.cv2.VideoCapture", return_value=invalid):
        plan = plan_speaker_crop(tmp_path / "invalid.mp4", clip, [])
    assert plan.source_width == 0
    assert invalid.released is True

    capture = FakeCapture()
    with (
        patch("clipper.tracking.cv2.VideoCapture", return_value=capture),
        patch("clipper.tracking.cv2.CascadeClassifier", return_value=EmptyDetector()),
    ):
        plan = plan_speaker_crop(tmp_path / "empty-detector.mp4", clip, [])
    assert plan.speaker_focus is False
    assert len(plan.anchors) == 2

    capture = FakeCapture()
    with (
        patch("clipper.tracking.cv2.VideoCapture", return_value=capture),
        patch("clipper.tracking.cv2.CascadeClassifier", return_value=NoFaceDetector()),
    ):
        plan = plan_speaker_crop(tmp_path / "no-face.mp4", clip, [], sample_fps=4)
    assert plan.face_detected is False
    assert plan.speaker_focus is True
    assert len(plan.anchors) == 2
    assert plan.anchors[0].x == plan.anchors[-1].x

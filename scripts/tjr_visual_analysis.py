"""Measured visual evidence for TJR candidate windows.

This pass samples the real source before rendering. It does not pretend to
understand charts or certify editorial integrity; it supplies objective face,
motion, and continuity evidence to the editorial rubric.
"""

from __future__ import annotations

from pathlib import Path

from clipper.models import ClipCandidate
from scripts.tjr_editorial import EditorialReview


def analyze_candidate_visuals(source: Path, candidate: ClipCandidate) -> EditorialReview:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "visual editorial analysis requires opencv-python-headless production extras"
        ) from exc

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("unable to open source for visual editorial analysis")
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    if cascade.empty():
        capture.release()
        raise RuntimeError("OpenCV frontal-face detector is unavailable")

    sample_count = 7
    faces_present = 0
    valid_frames = 0
    motion_values: list[float] = []
    previous_small = None
    try:
        for index in range(sample_count):
            fraction = (index + 1) / (sample_count + 1)
            timestamp = candidate.start + candidate.duration * fraction
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            valid_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            min_face = max(32, min(width, height) // 12)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=4,
                minSize=(min_face, min_face),
            )
            if len(faces):
                faces_present += 1
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if previous_small is not None:
                diff = cv2.absdiff(previous_small, small)
                motion_values.append(float(diff.mean()) / 255.0)
            previous_small = small
    finally:
        capture.release()

    if valid_frames < 4:
        raise RuntimeError("insufficient real source frames for visual editorial analysis")
    face_ratio = faces_present / valid_frames
    mean_motion = sum(motion_values) / len(motion_values) if motion_values else 0.0
    motion_signal = min(1.0, mean_motion / 0.055)
    # Face presence is useful for creator-centric footage, while visual activity
    # lets chart/screen moments remain viable even when no frontal face is visible.
    score = min(5.0, max(1.0, 2.4 + 1.35 * face_ratio + 1.25 * motion_signal))
    notes = (
        f"measured_source_frames={valid_frames}/{sample_count}; "
        f"face_presence_ratio={face_ratio:.2f}; "
        f"mean_frame_motion={mean_motion:.4f}; "
        "visual score is measured pre-render evidence, not integrity approval"
    )
    return EditorialReview(
        visual_score=round(score, 2), visual_notes=notes, visual_basis="machine_measured"
    )

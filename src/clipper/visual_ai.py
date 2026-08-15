from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .providers.base import ProviderResult, VisionProvider
from .visual import VisualEvent, VisualTimeline

ReviewDecision = Literal["PASS", "REPAIR"]
Severity = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True, slots=True)
class VisualReviewIssue:
    issue_type: str
    start: float
    end: float
    severity: Severity
    confidence: float
    repair_target: str
    description: str

    def __post_init__(self) -> None:
        if (
            not self.issue_type.strip()
            or not self.repair_target.strip()
            or not self.description.strip()
        ):
            raise ValueError("visual review issue fields cannot be empty")
        if self.start < 0 or self.end < self.start:
            raise ValueError("visual review issue timestamps are invalid")
        if self.severity not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("visual review severity is invalid")
        if not 0 <= self.confidence <= 1:
            raise ValueError("visual review confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class VisualReviewReport:
    decision: ReviewDecision
    summary: str
    overall_confidence: float
    issues: tuple[VisualReviewIssue, ...] = ()
    escalated: bool = False

    def __post_init__(self) -> None:
        if self.decision not in {"PASS", "REPAIR"}:
            raise ValueError("visual review decision is invalid")
        if not self.summary.strip():
            raise ValueError("visual review summary cannot be empty")
        if not 0 <= self.overall_confidence <= 1:
            raise ValueError("visual review confidence must be between 0 and 1")
        if self.decision == "PASS" and any(issue.severity == "HIGH" for issue in self.issues):
            raise ValueError("PASS visual review cannot contain a HIGH severity issue")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "summary": self.summary,
            "overall_confidence": self.overall_confidence,
            "issues": [asdict(issue) for issue in self.issues],
            "escalated": self.escalated,
        }


def _float(value: object, field: str) -> float:
    if not isinstance(value, int | float | str):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def parse_visual_review(payload: dict[str, Any]) -> VisualReviewReport:
    decision = str(payload.get("decision") or "").upper()
    summary = str(payload.get("summary") or "").strip()
    raw_issues = payload.get("issues", [])
    if not isinstance(raw_issues, list) or not all(isinstance(item, dict) for item in raw_issues):
        raise ValueError("visual review issues must be a list of objects")
    issues_list: list[VisualReviewIssue] = []
    for item in raw_issues:
        severity_text = str(item.get("severity") or "").upper()
        if severity_text not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("visual review severity is invalid")
        issues_list.append(
            VisualReviewIssue(
                issue_type=str(item.get("issue_type") or "").strip(),
                start=_float(item.get("start", 0.0), "issue start"),
                end=_float(item.get("end", item.get("start", 0.0)), "issue end"),
                severity=cast(Severity, severity_text),
                confidence=_float(item.get("confidence", 0.0), "issue confidence"),
                repair_target=str(item.get("repair_target") or "").strip(),
                description=str(item.get("description") or "").strip(),
            )
        )
    if decision not in {"PASS", "REPAIR"}:
        raise ValueError("visual review decision is invalid")
    issues = tuple(issues_list)
    return VisualReviewReport(
        decision=cast(ReviewDecision, decision),
        summary=summary,
        overall_confidence=_float(payload.get("overall_confidence", 0.0), "overall confidence"),
        issues=issues,
    )


def adaptive_sample_times(
    duration: float,
    *,
    scene_cuts: tuple[float, ...] = (),
    candidate_ranges: tuple[tuple[float, float], ...] = (),
    base_interval: float = 90.0,
) -> tuple[float, ...]:
    if duration <= 0:
        raise ValueError("visual sampling duration must be positive")
    if base_interval <= 0:
        raise ValueError("visual sampling interval must be positive")
    samples: set[float] = {0.0, max(0.0, duration - 0.05)}
    count = max(1, math.ceil(duration / base_interval))
    for index in range(count + 1):
        samples.add(min(duration - 0.05, index * base_interval))
    for cut in scene_cuts:
        for delta in (-0.2, 0.0, 0.2):
            samples.add(min(duration - 0.05, max(0.0, cut + delta)))
    first_region = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    for start, end in candidate_ranges:
        if end <= start:
            continue
        for delta in first_region:
            if start + delta < end:
                samples.add(min(duration - 0.05, max(0.0, start + delta)))
        samples.add(min(duration - 0.05, max(0.0, (start + end) / 2)))
        samples.add(min(duration - 0.05, max(0.0, end - 0.15)))
    return tuple(sorted(round(value, 3) for value in samples if 0 <= value < duration))


def media_duration_seconds(video_path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - base dependency in production image
        raise RuntimeError("opencv-python-headless is required for visual media probing") from exc
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video for duration probing: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"unable to determine media duration: {video_path}")
    return frames / fps


def extract_video_frames(
    video_path: Path,
    times: tuple[float, ...],
    output_dir: Path,
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - base dependency in production image
        raise RuntimeError(
            "opencv-python-headless is required for visual frame extraction"
        ) from exc
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video for visual sampling: {video_path}")
    frames: list[Path] = []
    try:
        for index, timestamp in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"unable to decode visual sample at {timestamp:.3f}s")
            path = output_dir / f"frame-{index:04d}-{timestamp:010.3f}.jpg"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"unable to write visual sample: {path}")
            frames.append(path)
    finally:
        capture.release()
    return frames


def parse_visual_timeline(
    payload: dict[str, Any], *, video_id: str, source_hash: str
) -> VisualTimeline:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not all(isinstance(item, dict) for item in raw_events):
        raise ValueError("visual scout must return an events list")
    events: list[VisualEvent] = []
    for item in raw_events:
        speakers = item.get("visible_speakers", [])
        labels = item.get("event_labels", [])
        if not isinstance(speakers, list) or not all(isinstance(value, str) for value in speakers):
            raise ValueError("visible_speakers must be strings")
        if not isinstance(labels, list) or not all(isinstance(value, str) for value in labels):
            raise ValueError("event_labels must be strings")
        events.append(
            VisualEvent(
                start=_float(item.get("start"), "visual event start"),
                end=_float(item.get("end"), "visual event end"),
                scene_id=str(item.get("scene_id") or "").strip(),
                summary=str(item.get("summary") or "").strip(),
                visible_speakers=tuple(value.strip() for value in speakers if value.strip()),
                event_labels=tuple(value.strip() for value in labels if value.strip()),
                confidence=_float(item.get("confidence", 0.0), "visual event confidence"),
            )
        )
    events.sort(key=lambda event: (event.start, event.end, event.scene_id))
    return VisualTimeline(video_id, source_hash, tuple(events))


def scout_visual_timeline(
    video_path: Path,
    provider: VisionProvider,
    *,
    video_id: str,
    source_hash: str,
    duration: float,
    output_dir: Path,
    scene_cuts: tuple[float, ...] = (),
    candidate_ranges: tuple[tuple[float, float], ...] = (),
) -> tuple[VisualTimeline, ProviderResult[dict[str, Any]]]:
    media_duration = media_duration_seconds(video_path)
    effective_duration = min(duration, media_duration)
    times = adaptive_sample_times(
        effective_duration,
        scene_cuts=scene_cuts,
        candidate_ranges=candidate_ranges,
    )
    frames = extract_video_frames(video_path, times, output_dir)
    result = provider.inspect(
        task="visual_timeline_scout",
        frames=frames,
        context={
            "video_id": video_id,
            "source_hash": source_hash,
            "frame_timestamps": list(times),
            "instruction": (
                "Describe only visible evidence. Do not retranscribe audio. Return semantic "
                "visual events with source timestamps, scene IDs, visible speakers, event "
                "labels, summaries, and confidence."
            ),
        },
    )
    return parse_visual_timeline(result.value, video_id=video_id, source_hash=source_hash), result


def _needs_escalation(report: VisualReviewReport, threshold: float) -> bool:
    if report.overall_confidence < threshold:
        return True
    return any(issue.severity == "HIGH" and issue.confidence < threshold for issue in report.issues)


def _compact_technical_qc(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload

    def selected(section: str, fields: tuple[str, ...]) -> dict[str, Any]:
        value = payload.get(section)
        if not isinstance(value, dict):
            return {}
        return {field: value[field] for field in fields if field in value}

    return {
        "status": payload.get("status"),
        "issues": payload.get("issues", []),
        "video": selected(
            "video",
            (
                "duration_seconds",
                "expected_duration_seconds",
                "width",
                "height",
                "fps",
                "video_codec",
                "audio_codec",
                "decode_pass",
            ),
        ),
        "audio": selected("audio", ("integrated_lufs", "true_peak_dbfs", "lra_lu")),
        "captions": selected(
            "captions",
            (
                "platform",
                "safe_region_pass",
                "timing_mode",
                "word_exact",
                "alignment",
                "partial_words_dropped",
            ),
        ),
        "framing": selected(
            "framing",
            (
                "framing_mode",
                "background_fill",
                "zoom_factor",
                "speaker_tracks",
                "speaker_switches",
                "reframe_events",
                "source_cuts",
                "no_filler_pass",
                "valid_crop_pass",
                "transition_qc_pass",
            ),
        ),
        "image_quality": payload.get("image_quality", {}),
        "watermark": payload.get("watermark", {}),
    }


def _review_context(
    *,
    duration: float,
    frame_times: tuple[float, ...],
    context: dict[str, Any],
) -> dict[str, Any]:
    compact_context = dict(context)
    if "technical_qc" in compact_context:
        compact_context["technical_qc"] = _compact_technical_qc(compact_context["technical_qc"])
    return {
        **compact_context,
        "duration": duration,
        "frame_timestamps": list(frame_times),
        "inspection_requirements": [
            "opening immediately makes sense",
            "hook resolves truthfully",
            "clip is self-contained",
            "correct speaker is framed",
            "no face is materially cropped",
            "transitions and reframes are visually coherent",
            "captions do not obstruct important visual content",
            "ending feels complete",
            "no visible quality degradation",
        ],
        "instruction": (
            "Return PASS or REPAIR with timestamped structured issues. "
            "Do not infer spoken words from pixels."
        ),
    }


def review_rendered_clip(
    video_path: Path,
    primary: VisionProvider,
    *,
    duration: float,
    output_dir: Path,
    context: dict[str, Any],
    transitions: tuple[float, ...] = (),
    escalation: VisionProvider | None = None,
    escalation_threshold: float = 0.75,
) -> tuple[VisualReviewReport, list[ProviderResult[dict[str, Any]]]]:
    frame_times = adaptive_sample_times(
        duration,
        scene_cuts=transitions,
        candidate_ranges=((0.0, duration),),
        base_interval=max(2.0, min(6.0, duration / 8)),
    )
    frames = extract_video_frames(video_path, frame_times, output_dir)
    request_context = _review_context(duration=duration, frame_times=frame_times, context=context)
    first = primary.inspect(task="rendered_clip_review", frames=frames, context=request_context)
    first_report = parse_visual_review(first.value)
    results = [first]
    if escalation is None or not _needs_escalation(first_report, escalation_threshold):
        return first_report, results
    second = escalation.inspect(
        task="rendered_clip_review_escalation",
        frames=frames,
        context={**request_context, "primary_review": first_report.to_dict()},
    )
    second_report = parse_visual_review(second.value)
    results.append(second)
    if second_report.decision == first_report.decision:
        return VisualReviewReport(
            second_report.decision,
            second_report.summary,
            second_report.overall_confidence,
            second_report.issues,
            escalated=True,
        ), results
    disagreement = VisualReviewIssue(
        "reviewer_disagreement",
        0.0,
        duration,
        "HIGH",
        max(first_report.overall_confidence, second_report.overall_confidence),
        "EDITORIAL_QC",
        "Primary and escalation visual reviewers disagree on release decision.",
    )
    issues = tuple(first_report.issues + second_report.issues + (disagreement,))
    return VisualReviewReport(
        "REPAIR",
        "Visual reviewers disagree; conservative repair/review is required.",
        max(first_report.overall_confidence, second_report.overall_confidence),
        issues,
        escalated=True,
    ), results


def repair_stage(issue_type: str) -> str:
    normalized = issue_type.casefold().replace("-", "_")
    if any(
        token in normalized for token in ("crop", "framing", "speaker", "reframe", "transition")
    ):
        return "TRACKING"
    if "caption" in normalized or "subtitle" in normalized:
        return "CAPTION"
    if (
        "ending" in normalized
        or "hook" in normalized
        or "context" in normalized
        or "continuity" in normalized
    ):
        return "EDIT_PLAN"
    if "source" in normalized or "quality" in normalized or "decode" in normalized:
        return "SOURCE"
    return "EDITORIAL_QC"

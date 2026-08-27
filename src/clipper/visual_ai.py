from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .cache import FileCache
from .providers.base import InferenceUsage, ModelIdentity, ProviderResult, VisionProvider
from .stage_contracts import content_fingerprint
from .visual import VisualEvent, VisualEvidenceScope, VisualEvidenceSpan, VisualTimeline

ReviewDecision = Literal["PASS", "REPAIR", "REJECT", "ESCALATE"]
Severity = Literal["LOW", "MEDIUM", "HIGH"]
VISUAL_SAMPLE_MAX_EDGE = 960
SOURCE_POLICY_SAMPLE_INTERVAL_SECONDS = 4.0
SOURCE_POLICY_SINGLE_FRAME_RETRIES = 2


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
        if self.decision not in {"PASS", "REPAIR", "REJECT", "ESCALATE"}:
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
    if decision not in {"PASS", "REPAIR", "REJECT", "ESCALATE"}:
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


def source_policy_sample_times(
    duration: float,
    *,
    scene_cuts: tuple[float, ...] = (),
    interval_seconds: float = SOURCE_POLICY_SAMPLE_INTERVAL_SECONDS,
) -> tuple[float, ...]:
    """Sample source-wide policy evidence independently from candidate/editorial review."""
    return adaptive_sample_times(
        duration,
        scene_cuts=scene_cuts,
        base_interval=interval_seconds,
    )


def visual_evidence_spans_from_samples(
    times: tuple[float, ...],
    duration: float,
    *,
    scope: VisualEvidenceScope = "source_policy",
) -> tuple[VisualEvidenceSpan, ...]:
    if duration <= 0:
        raise ValueError("visual evidence duration must be positive")
    ordered = tuple(sorted(dict.fromkeys(times)))
    if not ordered:
        raise ValueError("visual evidence requires at least one sample")
    if ordered[0] < 0 or ordered[-1] >= duration:
        raise ValueError("visual evidence sample lies outside source duration")
    if scope not in {"source_policy", "candidate_editorial"}:
        raise ValueError("visual evidence scope is invalid")

    spans: list[VisualEvidenceSpan] = []
    for index, sample in enumerate(ordered):
        left = 0.0 if index == 0 else (ordered[index - 1] + sample) / 2
        right = duration if index == len(ordered) - 1 else (sample + ordered[index + 1]) / 2
        spans.append(
            VisualEvidenceSpan(
                start=left,
                end=right,
                sample_time=sample,
                scope=scope,
            )
        )
    return tuple(spans)


def tracking_transition_sample_times(transitions: object) -> tuple[float, ...]:
    """Return start, midpoint, and end samples for every rendered camera move."""
    if not isinstance(transitions, (list, tuple)):
        return ()
    samples: set[float] = set()
    for item in transitions:
        if not isinstance(item, dict) or str(item.get("mode") or "") == "hold":
            continue
        raw_start = item.get("start")
        if raw_start is None:
            raw_start = item.get("start_time")
        raw_end = item.get("end")
        if raw_end is None:
            raw_end = item.get("end_time", raw_start)
        if not isinstance(raw_start, (int, float, str)) or not isinstance(
            raw_end, (int, float, str)
        ):
            continue
        try:
            start = float(raw_start)
            end = float(raw_end)
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        samples.add(start)
        samples.add(end)
        if end > start:
            samples.add((start + end) / 2)
    return tuple(sorted(round(value, 3) for value in samples))


def _opencv_media_duration_seconds(video_path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - base dependency in production image
        raise RuntimeError("opencv-python-headless is required for visual media fallback") from exc
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


def media_duration_seconds(video_path: Path) -> float:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    ffprobe_error: str | None = None
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            duration = float(completed.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(
                "unable to determine media duration from ffprobe output: "
                f"{completed.stdout.strip()!r}"
            ) from exc
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError(f"unable to determine media duration: {video_path}")
        return duration
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "ffprobe returned no diagnostic output").strip()
        ffprobe_error = f"unable to probe visual media duration: {detail[-4000:]}"
    except subprocess.TimeoutExpired:
        ffprobe_error = "visual media duration probe timed out"
    except OSError as exc:
        ffprobe_error = f"visual media duration probe unavailable: {exc}"
    except RuntimeError as exc:
        ffprobe_error = str(exc)
    try:
        return _opencv_media_duration_seconds(video_path)
    except Exception as exc:
        raise RuntimeError(f"{ffprobe_error}; OpenCV fallback failed: {exc}") from exc


def _opencv_extract_video_frames(
    video_path: Path,
    times: tuple[float, ...],
    output_dir: Path,
) -> list[Path]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - base dependency in production image
        raise RuntimeError("opencv-python-headless is required for visual frame fallback") from exc
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


def extract_video_frames(
    video_path: Path,
    times: tuple[float, ...],
    output_dir: Path,
) -> list[Path]:
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    scale_filter = (
        f"scale=w='min({VISUAL_SAMPLE_MAX_EDGE},iw)':"
        f"h='min({VISUAL_SAMPLE_MAX_EDGE},ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    ffmpeg_error: str | None = None
    try:
        for index, timestamp in enumerate(times):
            path = output_dir / f"frame-{index:04d}-{timestamp:010.3f}.jpg"
            path.unlink(missing_ok=True)
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
                str(video_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                scale_filter,
                "-q:v",
                "3",
                str(path),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
            except subprocess.CalledProcessError as exc:
                detail = (
                    exc.stderr or exc.stdout or "ffmpeg returned no diagnostic output"
                ).strip()
                raise RuntimeError(
                    f"unable to decode visual sample at {timestamp:.3f}s: {detail[-4000:]}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"visual sample extraction timed out at {timestamp:.3f}s"
                ) from exc
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"ffmpeg produced no visual sample at {timestamp:.3f}s")
            frames.append(path)
        return frames
    except OSError as exc:
        ffmpeg_error = f"ffmpeg visual sampling unavailable: {exc}"
    except RuntimeError as exc:
        ffmpeg_error = str(exc)
    finally:
        if ffmpeg_error is not None:
            for frame in frames:
                frame.unlink(missing_ok=True)
    try:
        return _opencv_extract_video_frames(video_path, times, output_dir)
    except Exception as exc:
        raise RuntimeError(f"{ffmpeg_error}; OpenCV fallback failed: {exc}") from exc


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


def _parse_source_policy_events(
    payload: dict[str, Any],
    *,
    frame_timestamps: tuple[float, ...],
    spans: tuple[VisualEvidenceSpan, ...],
) -> tuple[tuple[VisualEvent, ...], tuple[float, ...]]:
    raw = payload.get("observations")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("source-policy visual scout must return an observations list")
    expected_order = tuple(round(value, 3) for value in frame_timestamps)
    expected = set(expected_order)
    if len(expected) != len(expected_order):
        raise ValueError("source-policy frame timestamps must be unique")
    if len(raw) > len(expected):
        raise ValueError("source-policy visual scout returned more observations than frames")
    span_by_sample = {round(span.sample_time, 3): span for span in spans}
    seen: set[float] = set()
    events: list[VisualEvent] = []
    for item in raw:
        timestamp = round(_float(item.get("timestamp"), "source-policy observation timestamp"), 3)
        if timestamp not in expected or timestamp in seen:
            raise ValueError("source-policy observation references an unknown or duplicate frame")
        seen.add(timestamp)
        speakers = item.get("visible_speakers", [])
        labels = item.get("event_labels", [])
        if not isinstance(speakers, list) or not all(isinstance(value, str) for value in speakers):
            raise ValueError("visible_speakers must be strings")
        if not isinstance(labels, list) or not all(isinstance(value, str) for value in labels):
            raise ValueError("event_labels must be strings")
        span = span_by_sample[timestamp]
        events.append(
            VisualEvent(
                start=span.start,
                end=span.end,
                scene_id=str(item.get("scene_id") or "").strip(),
                summary=str(item.get("summary") or "").strip(),
                visible_speakers=tuple(value.strip() for value in speakers if value.strip()),
                event_labels=tuple(value.strip() for value in labels if value.strip()),
                confidence=_float(item.get("confidence", 0.0), "visual event confidence"),
            )
        )
    missing = tuple(value for value in frame_timestamps if round(value, 3) not in seen)
    return (
        tuple(sorted(events, key=lambda event: (event.start, event.end, event.scene_id))),
        missing,
    )


def _source_policy_events(
    payload: dict[str, Any],
    *,
    frame_timestamps: tuple[float, ...],
    spans: tuple[VisualEvidenceSpan, ...],
) -> tuple[VisualEvent, ...]:
    events, missing = _parse_source_policy_events(
        payload,
        frame_timestamps=frame_timestamps,
        spans=spans,
    )
    if missing:
        raise ValueError("source-policy visual scout must return one observation per frame")
    return events


def _source_policy_context(
    *,
    video_id: str,
    source_hash: str,
    frame_timestamps: tuple[float, ...],
    recovery_attempt: int,
) -> dict[str, object]:
    instruction = (
        "Inspect every supplied source frame independently for source-visible branding, "
        "logos, overlays, on-screen text, people, scenes, and policy-relevant hazards. "
        "Do not retranscribe audio. Return exactly one observation for every supplied "
        "frame timestamp; never infer an uninspected time range."
    )
    if recovery_attempt:
        instruction += (
            " This is a recovery inspection containing only frames whose previous response "
            "was missing or invalid. Return one observation for each supplied timestamp even "
            "when no policy-relevant label is visible."
        )
    return {
        "video_id": video_id,
        "source_hash": source_hash,
        "frame_timestamps": list(frame_timestamps),
        "inspection_scope": "source_policy",
        "source_policy_recovery_attempt": recovery_attempt,
        "instruction": instruction,
    }


def _inspect_source_policy_batch(
    provider: VisionProvider,
    *,
    video_id: str,
    source_hash: str,
    frame_timestamps: tuple[float, ...],
    frames: list[Path],
    spans: tuple[VisualEvidenceSpan, ...],
    on_observations: Callable[[tuple[VisualEvent, ...], tuple[float, ...], ModelIdentity], None]
    | None = None,
) -> tuple[tuple[VisualEvent, ...], list[ProviderResult[dict[str, Any]]]]:
    if not frame_timestamps:
        raise ValueError("source-policy inspection batch cannot be empty")
    if len(frame_timestamps) != len(frames) or len(frame_timestamps) != len(spans):
        raise ValueError("source-policy inspection batch evidence is inconsistent")

    items = tuple(zip(frame_timestamps, frames, spans, strict=True))
    accepted: list[VisualEvent] = []
    results: list[ProviderResult[dict[str, Any]]] = []

    def inspect_subset(
        subset: tuple[tuple[float, Path, VisualEvidenceSpan], ...],
        *,
        recovery_attempt: int = 0,
        single_retry: int = 0,
    ) -> None:
        subset_times = tuple(item[0] for item in subset)
        subset_frames = [item[1] for item in subset]
        subset_spans = tuple(item[2] for item in subset)
        result = provider.inspect(
            task="source_policy_visual_scout",
            frames=subset_frames,
            context=_source_policy_context(
                video_id=video_id,
                source_hash=source_hash,
                frame_timestamps=subset_times,
                recovery_attempt=recovery_attempt,
            ),
        )
        results.append(result)
        try:
            parsed, missing = _parse_source_policy_events(
                result.value,
                frame_timestamps=subset_times,
                spans=subset_spans,
            )
        except ValueError as exc:
            if len(subset) == 1:
                if single_retry < SOURCE_POLICY_SINGLE_FRAME_RETRIES:
                    inspect_subset(
                        subset,
                        recovery_attempt=recovery_attempt + 1,
                        single_retry=single_retry + 1,
                    )
                    return
                raise ValueError(
                    "source-policy visual scout must return one observation per frame "
                    "after single-frame recovery"
                ) from exc
            midpoint = len(subset) // 2
            inspect_subset(
                subset[:midpoint],
                recovery_attempt=recovery_attempt + 1,
            )
            inspect_subset(
                subset[midpoint:],
                recovery_attempt=recovery_attempt + 1,
            )
            return

        if parsed and on_observations is not None:
            missing_keys = {round(value, 3) for value in missing}
            observed_times = tuple(
                value for value in subset_times if round(value, 3) not in missing_keys
            )
            on_observations(parsed, observed_times, result.model)
        accepted.extend(parsed)
        if not missing:
            return

        missing_keys = {round(value, 3) for value in missing}
        missing_subset = tuple(item for item in subset if round(item[0], 3) in missing_keys)
        if not missing_subset:
            raise RuntimeError("source-policy recovery lost missing frame identity")
        if len(missing_subset) == 1:
            if len(subset) == 1 and single_retry >= SOURCE_POLICY_SINGLE_FRAME_RETRIES:
                raise ValueError(
                    "source-policy visual scout must return one observation per frame "
                    "after single-frame recovery"
                )
            inspect_subset(
                missing_subset,
                recovery_attempt=recovery_attempt + 1,
                single_retry=single_retry + 1 if len(subset) == 1 else 0,
            )
            return
        if len(missing_subset) == len(subset):
            midpoint = len(missing_subset) // 2
            inspect_subset(
                missing_subset[:midpoint],
                recovery_attempt=recovery_attempt + 1,
            )
            inspect_subset(
                missing_subset[midpoint:],
                recovery_attempt=recovery_attempt + 1,
            )
            return
        inspect_subset(
            missing_subset,
            recovery_attempt=recovery_attempt + 1,
        )

    inspect_subset(items)
    return (
        tuple(sorted(accepted, key=lambda event: (event.start, event.end, event.scene_id))),
        results,
    )


def _model_identity_from_payload(payload: object) -> ModelIdentity | None:
    if not isinstance(payload, dict):
        return None
    fields = (
        "model_id",
        "revision",
        "quantization",
        "inference_engine",
        "prompt_version",
        "schema_version",
    )
    if not all(payload.get(field) is not None for field in fields):
        return None
    return ModelIdentity(*(str(payload[field]) for field in fields))


def _source_policy_cache_namespace(
    *,
    source_hash: str,
    requested_identity: ModelIdentity,
) -> str:
    instruction = str(
        _source_policy_context(
            video_id="cache-contract",
            source_hash=source_hash,
            frame_timestamps=(),
            recovery_attempt=0,
        )["instruction"]
    )
    return content_fingerprint(
        {
            "stage": "source_policy_visual_frame",
            "source_hash": source_hash,
            "model_identity": requested_identity.to_dict(),
            "inspection_contract": instruction,
            "frame_contract": {"max_edge": VISUAL_SAMPLE_MAX_EDGE},
        }
    )


def _source_policy_frame_cache_key(namespace: str, timestamp: float) -> str:
    return content_fingerprint({"namespace": namespace, "timestamp": round(timestamp, 3)})


def _read_source_policy_checkpoint(
    cache: FileCache,
    *,
    namespace: str,
    timestamp: float,
    span: VisualEvidenceSpan,
) -> tuple[VisualEvent, ModelIdentity] | None:
    payload = cache.read(_source_policy_frame_cache_key(namespace, timestamp), "observation")
    if not isinstance(payload, dict):
        return None
    model = _model_identity_from_payload(payload.get("model"))
    observation = payload.get("observation")
    if model is None or not isinstance(observation, dict):
        return None
    try:
        events, missing = _parse_source_policy_events(
            {"observations": [observation]},
            frame_timestamps=(timestamp,),
            spans=(span,),
        )
    except (TypeError, ValueError):
        return None
    if missing or len(events) != 1:
        return None
    return events[0], model


def _write_source_policy_checkpoint(
    cache: FileCache,
    *,
    namespace: str,
    timestamp: float,
    event: VisualEvent,
    model: ModelIdentity,
) -> None:
    cache.write(
        _source_policy_frame_cache_key(namespace, timestamp),
        "observation",
        {
            "timestamp": round(timestamp, 3),
            "model": model.to_dict(),
            "observation": {
                "timestamp": round(timestamp, 3),
                "scene_id": event.scene_id,
                "summary": event.summary,
                "visible_speakers": list(event.visible_speakers),
                "event_labels": list(event.event_labels),
                "confidence": event.confidence,
            },
        },
    )


def _is_vision_capacity_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "outofmemoryerror",
            "out of memory",
            "exceeds model context",
            "request too large",
            "payload too large",
        )
    )


def _capacity_cache_key(requested_identity: ModelIdentity) -> str:
    return content_fingerprint(
        {
            "stage": "source_policy_visual_capacity",
            "model_identity": requested_identity.to_dict(),
            "frame_contract": {"max_edge": VISUAL_SAMPLE_MAX_EDGE},
        }
    )


def _load_capacity_state(
    cache: FileCache | None,
    *,
    requested_identity: ModelIdentity,
) -> tuple[str | None, int, int | None]:
    if cache is None:
        return None, 0, None
    key = _capacity_cache_key(requested_identity)
    payload = cache.read(key, "capacity")
    if not isinstance(payload, dict):
        return key, 0, None
    raw_good = payload.get("largest_good")
    raw_bad = payload.get("smallest_bad")
    good = int(raw_good) if isinstance(raw_good, int) and raw_good > 0 else 0
    bad = int(raw_bad) if isinstance(raw_bad, int) and raw_bad > good else None
    return key, good, bad


def _persist_capacity_state(
    cache: FileCache | None,
    key: str | None,
    *,
    largest_good: int,
    smallest_bad: int | None,
    checkpoint_commit: Callable[[], None] | None,
) -> None:
    if cache is None or key is None:
        return
    cache.write(
        key,
        "capacity",
        {
            "largest_good": largest_good,
            "smallest_bad": smallest_bad,
        },
    )
    if checkpoint_commit is not None:
        checkpoint_commit()


def _next_batch_after_success(
    current: int,
    *,
    largest_good: int,
    smallest_bad: int | None,
    remaining: int,
) -> int:
    if remaining <= 0:
        return 0
    if smallest_bad is None:
        return min(remaining, max(current + 1, current * 2))
    if largest_good + 1 < smallest_bad:
        return min(remaining, largest_good + (smallest_bad - largest_good) // 2)
    return min(remaining, max(1, largest_good))


def _next_batch_after_capacity_failure(
    current: int,
    *,
    largest_good: int,
    smallest_bad: int,
) -> int:
    if current <= 1:
        return 1
    if largest_good > 0 and largest_good + 1 < smallest_bad:
        candidate = largest_good + (smallest_bad - largest_good) // 2
    elif largest_good > 0:
        candidate = largest_good
    else:
        candidate = current // 2
    return max(1, min(current - 1, candidate))


def _aggregate_vision_results(
    results: list[ProviderResult[dict[str, Any]]],
    timeline: VisualTimeline,
    *,
    cached_model: ModelIdentity | None = None,
    cache_hits: int = 0,
    requested_frames: int = 0,
) -> ProviderResult[dict[str, Any]]:
    model = results[0].model if results else cached_model
    if model is None:
        raise ValueError("source-policy visual scout produced no inference or cached evidence")
    if cached_model is not None and model != cached_model:
        raise RuntimeError("source-policy cache model identity differs from active vision model")
    if any(result.model != model for result in results):
        raise RuntimeError("source-policy visual scout changed model identity between batches")

    usages = [result.usage for result in results]
    gpu_types = {usage.gpu_type for usage in usages if usage.gpu_type}
    peaks = [usage.peak_vram_mb for usage in usages if usage.peak_vram_mb is not None]
    lifecycle_loads: dict[str, int] = {}
    peak_by_device: dict[str, float] = {}
    for usage in usages:
        lifecycle_id = usage.runtime.get("worker_lifecycle_id")
        load_count = usage.runtime.get("model_load_count")
        if isinstance(lifecycle_id, str) and isinstance(load_count, int):
            lifecycle_loads[lifecycle_id] = max(lifecycle_loads.get(lifecycle_id, 0), load_count)
        raw_peaks = usage.runtime.get("peak_vram_mb_by_device")
        if isinstance(raw_peaks, dict):
            for device, value in raw_peaks.items():
                if isinstance(value, int | float):
                    peak_by_device[str(device)] = max(
                        peak_by_device.get(str(device), 0.0), float(value)
                    )
    runtime: dict[str, Any] = {
        "source_policy_cache_hits": cache_hits,
        "source_policy_requested_frames": requested_frames,
        "source_policy_provider_calls": len(results),
        "worker_lifecycle_model_loads": lifecycle_loads,
        "peak_vram_mb_by_device": peak_by_device,
    }
    usage = InferenceUsage(
        provider=usages[0].provider if usages else "cache",
        started_at=usages[0].started_at if usages else "cache",
        duration_seconds=sum(item.duration_seconds for item in usages),
        gpu_type=next(iter(gpu_types)) if len(gpu_types) == 1 else None,
        gpu_seconds=sum(item.gpu_seconds for item in usages),
        peak_vram_mb=max(peaks) if peaks else None,
        input_units=sum(item.input_units for item in usages),
        output_units=sum(item.output_units for item in usages),
        estimated_cost_usd=sum(item.estimated_cost_usd for item in usages),
        runtime=runtime,
    )
    return ProviderResult(
        timeline.to_dict(),
        model,
        usage,
        degraded=any(result.degraded for result in results),
    )


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
    checkpoint_dir: Path | None = None,
    checkpoint_commit: Callable[[], None] | None = None,
) -> tuple[VisualTimeline, ProviderResult[dict[str, Any]]]:
    """Build source-wide policy evidence with resumable, runtime-learned capacity."""
    media_duration = media_duration_seconds(video_path)
    effective_duration = min(duration, media_duration)
    times = source_policy_sample_times(effective_duration, scene_cuts=scene_cuts)
    if candidate_ranges:
        dense_candidate_times = adaptive_sample_times(
            effective_duration,
            candidate_ranges=candidate_ranges,
            base_interval=SOURCE_POLICY_SAMPLE_INTERVAL_SECONDS,
        )
        times = tuple(sorted(set(times) | set(dense_candidate_times)))
    spans = visual_evidence_spans_from_samples(times, effective_duration, scope="source_policy")
    spans_by_sample = {round(span.sample_time, 3): span for span in spans}
    requested_identity = provider.identity
    cache = FileCache(checkpoint_dir) if checkpoint_dir is not None else None
    namespace = _source_policy_cache_namespace(
        source_hash=source_hash,
        requested_identity=requested_identity,
    )

    events: list[VisualEvent] = []
    cached_model: ModelIdentity | None = None
    cached_times: set[float] = set()
    if cache is not None:
        for timestamp in times:
            hit = _read_source_policy_checkpoint(
                cache,
                namespace=namespace,
                timestamp=timestamp,
                span=spans_by_sample[round(timestamp, 3)],
            )
            if hit is None:
                continue
            event, model = hit
            if cached_model is not None and model != cached_model:
                continue
            cached_model = model
            events.append(event)
            cached_times.add(round(timestamp, 3))

    pending_times = tuple(
        timestamp for timestamp in times if round(timestamp, 3) not in cached_times
    )
    results: list[ProviderResult[dict[str, Any]]] = []
    if pending_times:
        prepared_dir = output_dir / "source-policy-pending"
        prepared_frames = extract_video_frames(video_path, pending_times, prepared_dir)
        if len(prepared_frames) != len(pending_times):
            raise RuntimeError("source-policy frame extraction returned incomplete prepared jobs")

        warm = getattr(provider, "warm", None)
        if callable(warm):
            warm()
        if cached_model is not None and provider.identity != cached_model:
            events.clear()
            cached_times.clear()
            cached_model = None
            pending_times = times
            prepared_dir = output_dir / "source-policy-revalidated"
            prepared_frames = extract_video_frames(video_path, pending_times, prepared_dir)
            if callable(warm):
                warm()

        work = [
            (timestamp, frame, spans_by_sample[round(timestamp, 3)])
            for timestamp, frame in zip(pending_times, prepared_frames, strict=True)
        ]

        def persist_observations(
            parsed: tuple[VisualEvent, ...],
            observed_times: tuple[float, ...],
            model: ModelIdentity,
        ) -> None:
            nonlocal cached_model
            if cache is None:
                return
            if len(parsed) != len(observed_times):
                raise RuntimeError("source-policy checkpoint lost observation identity")
            for timestamp, event in zip(observed_times, parsed, strict=True):
                _write_source_policy_checkpoint(
                    cache,
                    namespace=namespace,
                    timestamp=timestamp,
                    event=event,
                    model=model,
                )
            cached_model = model
            if checkpoint_commit is not None:
                checkpoint_commit()

        capacity_key, largest_good, smallest_bad = _load_capacity_state(
            cache, requested_identity=requested_identity
        )
        batch_size = largest_good if largest_good > 0 else 1

        while work:
            size = min(batch_size, len(work))
            subset = tuple(work[:size])
            subset_times = tuple(item[0] for item in subset)
            subset_frames = [item[1] for item in subset]
            subset_spans = tuple(item[2] for item in subset)
            try:
                batch_events, batch_results = _inspect_source_policy_batch(
                    provider,
                    video_id=video_id,
                    source_hash=source_hash,
                    frame_timestamps=subset_times,
                    frames=subset_frames,
                    spans=subset_spans,
                    on_observations=persist_observations,
                )
            except Exception as exc:
                if not _is_vision_capacity_error(exc):
                    raise

                if cache is not None:
                    retained: list[tuple[float, Path, VisualEvidenceSpan]] = []
                    completed_events: list[VisualEvent] = []
                    for item in subset:
                        hit = _read_source_policy_checkpoint(
                            cache,
                            namespace=namespace,
                            timestamp=item[0],
                            span=item[2],
                        )
                        if hit is None:
                            retained.append(item)
                        else:
                            completed_events.append(hit[0])
                    if completed_events:
                        events.extend(completed_events)
                        work = retained + work[size:]
                        if not retained:
                            batch_size = min(max(1, batch_size), len(work)) if work else 0
                            continue

                if size <= 1:
                    raise RuntimeError(
                        "vision capacity exhausted for an indivisible single-frame inspection"
                    ) from exc
                smallest_bad = size if smallest_bad is None else min(smallest_bad, size)
                batch_size = _next_batch_after_capacity_failure(
                    size,
                    largest_good=largest_good,
                    smallest_bad=smallest_bad,
                )
                _persist_capacity_state(
                    cache,
                    capacity_key,
                    largest_good=largest_good,
                    smallest_bad=smallest_bad,
                    checkpoint_commit=checkpoint_commit,
                )
                continue

            events.extend(batch_events)
            results.extend(batch_results)
            work = work[size:]
            largest_good = max(largest_good, size)
            batch_size = _next_batch_after_success(
                size,
                largest_good=largest_good,
                smallest_bad=smallest_bad,
                remaining=len(work),
            )
            _persist_capacity_state(
                cache,
                capacity_key,
                largest_good=largest_good,
                smallest_bad=smallest_bad,
                checkpoint_commit=checkpoint_commit,
            )

    timeline = VisualTimeline(
        video_id,
        source_hash,
        tuple(
            sorted(
                events,
                key=lambda event: (event.start, event.end, event.scene_id),
            )
        ),
        coverage_spans=spans,
        source_duration=effective_duration,
    )
    return timeline, _aggregate_vision_results(
        results,
        timeline,
        cached_model=cached_model,
        cache_hits=len(cached_times),
        requested_frames=len(times),
    )


def _needs_escalation(report: VisualReviewReport, threshold: float) -> bool:
    if report.decision == "ESCALATE":
        return True
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
                "composition",
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
            "canonical transcript proves a complete start and ending",
            "campaign source-segment and branding policy passes",
            "approved campaign overlay is not confused with source-visible foreign branding",
            "no visible quality degradation",
        ],
        "instruction": (
            "Return PASS, REPAIR, REJECT, or ESCALATE with timestamped structured issues. "
            "Use canonical transcript evidence for semantic judgment and frames for visual "
            "judgment. Do not infer spoken words from pixels. Missing or uncertain mandatory "
            "evidence must never become PASS."
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
        "ESCALATE",
        "Visual reviewers disagree; release decision requires further review.",
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

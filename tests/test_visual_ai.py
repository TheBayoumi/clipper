from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.visual_ai import (
    VisualReviewIssue,
    VisualReviewReport,
    _inspect_source_policy_batch,
    adaptive_sample_times,
    extract_video_frames,
    media_duration_seconds,
    parse_visual_review,
    parse_visual_timeline,
    repair_stage,
    review_rendered_clip,
    scout_visual_timeline,
    source_policy_sample_times,
    tracking_transition_sample_times,
    visual_evidence_spans_from_samples,
)


class FakeVision:
    identity = ModelIdentity("vision", "rev", "none", "test", "vision", "v1")

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, list[Path], dict[str, object]]] = []

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        self.calls.append((task, frames, context))
        return ProviderResult(
            self.payload,
            self.identity,
            InferenceUsage("test", "now", 0.01, input_units=len(frames)),
        )


class PolicyVision(FakeVision):
    def __init__(self) -> None:
        super().__init__({})

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        self.calls.append((task, frames, context))
        timestamps = context["frame_timestamps"]
        assert isinstance(timestamps, list)
        observations = [
            {
                "timestamp": timestamp,
                "scene_id": f"scene-{index}",
                "summary": "source frame inspected for policy evidence",
                "visible_speakers": [],
                "event_labels": [],
                "confidence": 0.95,
            }
            for index, timestamp in enumerate(timestamps)
        ]
        return ProviderResult(
            {"observations": observations},
            self.identity,
            InferenceUsage("test", "now", 0.01, input_units=len(frames)),
        )


class PartialPolicyVision(PolicyVision):
    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        result = super().inspect(task=task, frames=frames, context=context)
        observations = result.value["observations"]
        assert isinstance(observations, list)
        if len(frames) > 1:
            observations = observations[:-1]
        return ProviderResult(
            {"observations": observations},
            result.model,
            result.usage,
        )


class InvalidBatchPolicyVision(PolicyVision):
    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        if len(frames) == 1:
            return super().inspect(task=task, frames=frames, context=context)
        self.calls.append((task, frames, context))
        return ProviderResult(
            {
                "observations": [
                    {
                        "timestamp": -1.0,
                        "scene_id": "invalid-batch",
                        "summary": "invalid timestamp forces bounded batch recovery",
                        "visible_speakers": [],
                        "event_labels": [],
                        "confidence": 0.5,
                    }
                ]
            },
            self.identity,
            InferenceUsage("test", "now", 0.01, input_units=len(frames)),
        )


def _pass_payload(confidence: float = 0.95) -> dict[str, object]:
    return {
        "decision": "PASS",
        "summary": "The clip is visually coherent.",
        "overall_confidence": confidence,
        "issues": [],
    }


def _repair_payload(confidence: float = 0.95) -> dict[str, object]:
    return {
        "decision": "REPAIR",
        "summary": "The speaker framing reverses unnecessarily.",
        "overall_confidence": confidence,
        "issues": [
            {
                "issue_type": "crop_oscillation",
                "start": 1.0,
                "end": 1.8,
                "severity": "HIGH",
                "confidence": confidence,
                "repair_target": "TRACKING",
                "description": "The crop jumps away and back while the same speaker continues.",
            }
        ],
    }


def test_visual_review_models_validate_and_serialize() -> None:
    issue = VisualReviewIssue("crop", 1, 2, "HIGH", 0.9, "TRACKING", "bad crop")
    report = VisualReviewReport("REPAIR", "repair needed", 0.9, (issue,), escalated=True)
    assert report.to_dict()["issues"][0]["severity"] == "HIGH"
    assert report.to_dict()["escalated"] is True
    for args, match in [
        (("", 0, 1, "LOW", 0.5, "X", "x"), "cannot be empty"),
        (("x", 2, 1, "LOW", 0.5, "X", "x"), "timestamps"),
        (("x", 0, 1, "BAD", 0.5, "X", "x"), "severity"),
        (("x", 0, 1, "LOW", 2.0, "X", "x"), "confidence"),
    ]:
        with pytest.raises(ValueError, match=match):
            VisualReviewIssue(*args)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="HIGH severity"):
        VisualReviewReport("PASS", "bad pass", 0.9, (issue,))
    with pytest.raises(ValueError, match="decision"):
        VisualReviewReport("MAYBE", "x", 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="summary"):
        VisualReviewReport("PASS", "", 0.5)
    with pytest.raises(ValueError, match="confidence"):
        VisualReviewReport("PASS", "x", 2.0)


def test_parse_visual_review_validates_schema_and_numbers() -> None:
    report = parse_visual_review(_repair_payload())
    assert report.decision == "REPAIR"
    assert report.issues[0].repair_target == "TRACKING"
    with pytest.raises(ValueError, match="issues"):
        parse_visual_review(
            {"decision": "PASS", "summary": "x", "overall_confidence": 1, "issues": {}}
        )
    with pytest.raises(ValueError, match="severity"):
        parse_visual_review(
            {
                "decision": "REPAIR",
                "summary": "x",
                "overall_confidence": 1,
                "issues": [
                    {
                        "issue_type": "x",
                        "start": 0,
                        "end": 1,
                        "severity": "WEIRD",
                        "confidence": 1,
                        "repair_target": "X",
                        "description": "x",
                    }
                ],
            }
        )
    with pytest.raises(ValueError, match="numeric"):
        parse_visual_review({"decision": "PASS", "summary": "x", "overall_confidence": None})


@pytest.mark.parametrize("decision", ["REJECT", "ESCALATE"])
def test_visual_review_supports_fatal_and_uncertain_release_decisions(decision: str) -> None:
    report = parse_visual_review(
        {
            "decision": decision,
            "summary": "Campaign evidence does not permit PASS.",
            "overall_confidence": 0.9,
            "issues": [],
        }
    )
    assert report.decision == decision


def test_adaptive_sampling_covers_first_three_seconds_scene_cuts_and_episode() -> None:
    times = adaptive_sample_times(
        200,
        scene_cuts=(50.0,),
        candidate_ranges=((100.0, 110.0), (20.0, 20.0)),
        base_interval=90,
    )
    assert 0.0 in times and 199.95 in times
    assert {49.8, 50.0, 50.2} <= set(times)
    assert {100.0, 100.1, 100.25, 100.5, 101.0, 101.5, 102.0, 103.0} <= set(times)
    assert 105.0 in times and 109.85 in times
    with pytest.raises(ValueError, match="duration"):
        adaptive_sample_times(0)
    with pytest.raises(ValueError, match="interval"):
        adaptive_sample_times(1, base_interval=0)


def test_tracking_transition_samples_use_serialized_start_midpoint_and_end() -> None:
    assert tracking_transition_sample_times(
        [
            {"mode": "hold", "start": 1.0, "end": 1.0},
            {"mode": "eased_reframe", "start": 2.0, "end": 2.8},
            {"mode": "hard_cut", "start": 4.0, "end": 4.0},
            {"mode": "hard_cut", "start_time": 6.0, "end_time": 6.0},
            {"mode": "hard_cut", "start": "bad", "end": 7.0},
        ]
    ) == (2.0, 2.4, 2.8, 4.0, 6.0)
    assert tracking_transition_sample_times({}) == ()


def test_parse_visual_timeline_sorts_and_validates_model_output() -> None:
    payload = {
        "events": [
            {
                "start": 2,
                "end": 3,
                "scene_id": "s2",
                "summary": "reaction",
                "visible_speakers": ["B"],
                "event_labels": ["reaction"],
                "confidence": 0.8,
            },
            {
                "start": 0,
                "end": 1,
                "scene_id": "s1",
                "summary": "speaker explains",
                "visible_speakers": ["A"],
                "event_labels": [],
                "confidence": 0.9,
            },
        ]
    }
    timeline = parse_visual_timeline(payload, video_id="v", source_hash="h")
    assert [event.scene_id for event in timeline.events] == ["s1", "s2"]
    with pytest.raises(ValueError, match="events list"):
        parse_visual_timeline({}, video_id="v", source_hash="h")
    broken = {"events": [{**payload["events"][0], "visible_speakers": "A"}]}
    with pytest.raises(ValueError, match="visible_speakers"):
        parse_visual_timeline(broken, video_id="v", source_hash="h")
    broken = {"events": [{**payload["events"][0], "event_labels": "reaction"}]}
    with pytest.raises(ValueError, match="event_labels"):
        parse_visual_timeline(broken, video_id="v", source_hash="h")


def test_source_policy_sampling_builds_explicit_continuous_inspection_cells() -> None:
    times = source_policy_sample_times(20.0, interval_seconds=4.0)
    spans = visual_evidence_spans_from_samples(times, 20.0)
    assert times[0] == 0.0 and times[-1] == 19.95
    assert spans[0].start == 0.0
    assert spans[-1].end == 20.0
    assert all(left.end == pytest.approx(right.start) for left, right in pairwise(spans))
    assert sum(span.duration for span in spans) == pytest.approx(20.0)


def test_scout_visual_timeline_batches_source_policy_frames_and_records_coverage(
    tmp_path: Path,
) -> None:
    provider = PolicyVision()

    def frames_for_times(_source: Path, times: tuple[float, ...], output: Path) -> list[Path]:
        output.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, timestamp in enumerate(times):
            frame = output / f"{index:03d}-{timestamp:.3f}.jpg"
            frame.write_bytes(b"frame")
            frames.append(frame)
        return frames

    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=100.0),
        patch("clipper.visual_ai.extract_video_frames", side_effect=frames_for_times),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=100,
            output_dir=tmp_path / "frames",
        )

    summary = timeline.coverage_summary("source_policy")
    assert summary["coverage_fraction"] == pytest.approx(1.0)
    assert summary["max_sample_gap_seconds"] <= 4.0
    assert summary["sample_count"] > 1
    assert len(provider.calls) > 1
    assert all(call[0] == "source_policy_visual_scout" for call in provider.calls)
    call_sizes = [len(call[1]) for call in provider.calls]
    assert call_sizes[0] == 1
    assert max(call_sizes) > call_sizes[0]
    assert all(
        "Do not retranscribe audio" in str(call[2]["instruction"]) for call in provider.calls
    )
    assert result.usage.input_units == int(summary["sample_count"])
    assert len(timeline.events) == int(summary["sample_count"])


def test_source_policy_missing_observation_is_reinspected_without_fabricating_coverage(
    tmp_path: Path,
) -> None:
    provider = PartialPolicyVision()

    def frames_for_times(_source: Path, times: tuple[float, ...], output: Path) -> list[Path]:
        output.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, timestamp in enumerate(times):
            frame = output / f"{index:03d}-{timestamp:.3f}.jpg"
            frame.write_bytes(b"frame")
            frames.append(frame)
        return frames

    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=12.0),
        patch("clipper.visual_ai.extract_video_frames", side_effect=frames_for_times),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=12.0,
            output_dir=tmp_path / "frames",
        )

    summary = timeline.coverage_summary("source_policy")
    sample_count = int(summary["sample_count"])
    assert summary["coverage_fraction"] == pytest.approx(1.0)
    assert len(timeline.events) == sample_count
    recovery_calls = [
        call for call in provider.calls if call[2]["source_policy_recovery_attempt"] == 1
    ]
    assert recovery_calls
    assert any(len(call[1]) == 1 for call in recovery_calls)
    assert result.usage.input_units == sample_count + 1


def test_source_policy_malformed_multi_frame_response_splits_until_valid(
    tmp_path: Path,
) -> None:
    provider = InvalidBatchPolicyVision()

    def frames_for_times(_source: Path, times: tuple[float, ...], output: Path) -> list[Path]:
        output.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, timestamp in enumerate(times):
            frame = output / f"{index:03d}-{timestamp:.3f}.jpg"
            frame.write_bytes(b"frame")
            frames.append(frame)
        return frames

    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=12.0),
        patch("clipper.visual_ai.extract_video_frames", side_effect=frames_for_times),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=12.0,
            output_dir=tmp_path / "frames",
        )

    sample_count = int(timeline.coverage_summary("source_policy")["sample_count"])
    assert len(timeline.events) == sample_count
    assert len(provider.calls) > sample_count
    assert any(len(frames) > 1 for _, frames, _ in provider.calls)
    assert result.usage.input_units > sample_count


def test_source_policy_batch_rejects_empty_and_inconsistent_evidence() -> None:
    provider = PolicyVision()
    with pytest.raises(ValueError, match="cannot be empty"):
        _inspect_source_policy_batch(
            provider,
            video_id="v",
            source_hash="h",
            frame_timestamps=(),
            frames=[],
            spans=(),
        )
    with pytest.raises(ValueError, match="evidence is inconsistent"):
        _inspect_source_policy_batch(
            provider,
            video_id="v",
            source_hash="h",
            frame_timestamps=(0.0,),
            frames=[],
            spans=(),
        )


def test_source_policy_observation_omission_fails_closed(tmp_path: Path) -> None:
    provider = FakeVision({"observations": []})
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")

    def repeated_frames(_source: Path, times: tuple[float, ...], _output: Path) -> list[Path]:
        return [frame for _ in times]

    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=1.0),
        patch("clipper.visual_ai.extract_video_frames", side_effect=repeated_frames),
        pytest.raises(ValueError, match="one observation per frame"),
    ):
        scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=1.0,
            output_dir=tmp_path / "frames",
        )


def test_rendered_clip_review_pass_does_not_escalate(tmp_path: Path) -> None:
    provider = FakeVision(_pass_payload())
    escalation = FakeVision(_repair_payload())
    frames = [tmp_path / "f.jpg"]
    frames[0].write_bytes(b"frame")
    with patch("clipper.visual_ai.extract_video_frames", return_value=frames):
        report, results = review_rendered_clip(
            tmp_path / "clip.mp4",
            provider,
            duration=12,
            output_dir=tmp_path / "review",
            context={"plan_id": "p"},
            transitions=(4.0,),
            escalation=escalation,
        )
    assert report.decision == "PASS" and report.escalated is False
    assert len(results) == 1 and escalation.calls == []
    _, _, context = provider.calls[0]
    assert 0.25 in context["frame_timestamps"]
    assert 3.0 in context["frame_timestamps"]


def test_rendered_clip_review_compacts_technical_qc_tracking_details(tmp_path: Path) -> None:
    provider = FakeVision(_pass_payload())
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"frame")
    technical_qc = {
        "status": "PASS",
        "issues": [],
        "video": {"width": 1080, "height": 1920, "path": "private.mp4"},
        "framing": {
            "framing_mode": "speaker_locked_portrait",
            "speaker_switches": 4,
            "transitions": [{"start": index / 10} for index in range(100)],
        },
        "captions": {"alignment": "PASS", "audit_path": "private.json"},
    }
    with patch("clipper.visual_ai.extract_video_frames", return_value=[frame]):
        review_rendered_clip(
            tmp_path / "clip.mp4",
            provider,
            duration=12,
            output_dir=tmp_path / "review",
            context={"technical_qc": technical_qc},
        )
    _, _, context = provider.calls[0]
    compact = context["technical_qc"]
    assert compact["video"] == {"width": 1080, "height": 1920}
    assert compact["framing"]["speaker_switches"] == 4
    assert "transitions" not in compact["framing"]
    assert "audit_path" not in compact["captions"]


def test_rendered_clip_review_escalates_and_disagreement_is_conservative(tmp_path: Path) -> None:
    frames = [tmp_path / "f.jpg"]
    frames[0].write_bytes(b"frame")
    primary = FakeVision(_pass_payload(0.5))
    escalation = FakeVision(_pass_payload(0.95))
    with patch("clipper.visual_ai.extract_video_frames", return_value=frames):
        report, results = review_rendered_clip(
            tmp_path / "clip.mp4",
            primary,
            duration=8,
            output_dir=tmp_path / "same",
            context={},
            escalation=escalation,
            escalation_threshold=0.75,
        )
    assert report.decision == "PASS" and report.escalated is True and len(results) == 2

    disagreement_primary = FakeVision(_pass_payload(0.5))
    disagreement_large = FakeVision(_repair_payload(0.95))
    with patch("clipper.visual_ai.extract_video_frames", return_value=frames):
        report, _ = review_rendered_clip(
            tmp_path / "clip.mp4",
            disagreement_primary,
            duration=8,
            output_dir=tmp_path / "disagree",
            context={},
            escalation=disagreement_large,
        )
    assert report.decision == "ESCALATE" and report.escalated is True
    assert report.issues[-1].issue_type == "reviewer_disagreement"
    assert report.issues[-1].severity == "HIGH"


def test_high_ambiguous_issue_triggers_escalation(tmp_path: Path) -> None:
    payload = _repair_payload(0.95)
    payload["issues"][0]["confidence"] = 0.5  # type: ignore[index]
    primary = FakeVision(payload)
    escalation = FakeVision(_repair_payload(0.95))
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"frame")
    with patch("clipper.visual_ai.extract_video_frames", return_value=[frame]):
        report, results = review_rendered_clip(
            tmp_path / "clip.mp4",
            primary,
            duration=8,
            output_dir=tmp_path / "review",
            context={},
            escalation=escalation,
        )
    assert report.escalated is True and len(results) == 2


def test_repair_stage_routes_minimum_downstream_invalidation() -> None:
    assert repair_stage("crop_oscillation") == "TRACKING"
    assert repair_stage("caption_collision") == "CAPTION"
    assert repair_stage("incomplete_ending") == "EDIT_PLAN"
    assert repair_stage("source_quality") == "SOURCE"
    assert repair_stage("subjective_vibe") == "EDITORIAL_QC"


def test_extract_video_frames_handles_missing_open_decode_write_and_success(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_video_frames(tmp_path / "missing.mp4", (0.0,), tmp_path / "out")

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    closed_capture = Mock()
    closed_capture.isOpened.return_value = False
    fake_cv2 = SimpleNamespace(
        VideoCapture=Mock(return_value=closed_capture), CAP_PROP_POS_MSEC=0, imwrite=Mock()
    )
    with (
        patch.dict(sys.modules, {"cv2": fake_cv2}),
        pytest.raises(RuntimeError, match="unable to open"),
    ):
        extract_video_frames(video, (0.0,), tmp_path / "closed")

    decode_capture = Mock()
    decode_capture.isOpened.return_value = True
    decode_capture.read.return_value = (False, None)
    fake_cv2 = SimpleNamespace(
        VideoCapture=Mock(return_value=decode_capture), CAP_PROP_POS_MSEC=0, imwrite=Mock()
    )
    with (
        patch.dict(sys.modules, {"cv2": fake_cv2}),
        pytest.raises(RuntimeError, match="unable to decode"),
    ):
        extract_video_frames(video, (0.0,), tmp_path / "decode")
    decode_capture.release.assert_called_once()

    write_capture = Mock()
    write_capture.isOpened.return_value = True
    write_capture.read.return_value = (True, object())
    fake_cv2 = SimpleNamespace(
        VideoCapture=Mock(return_value=write_capture),
        CAP_PROP_POS_MSEC=0,
        imwrite=Mock(return_value=False),
    )
    with (
        patch.dict(sys.modules, {"cv2": fake_cv2}),
        pytest.raises(RuntimeError, match="unable to write"),
    ):
        extract_video_frames(video, (0.0,), tmp_path / "write")

    good_capture = Mock()
    good_capture.isOpened.return_value = True
    good_capture.read.return_value = (True, object())

    def write_file(path: str, _image: object) -> bool:
        Path(path).write_bytes(b"jpg")
        return True

    fake_cv2 = SimpleNamespace(
        VideoCapture=Mock(return_value=good_capture),
        CAP_PROP_POS_MSEC=0,
        imwrite=Mock(side_effect=write_file),
    )
    with patch.dict(sys.modules, {"cv2": fake_cv2}):
        frames = extract_video_frames(video, (0.0, 1.5), tmp_path / "good")
    assert len(frames) == 2 and all(path.is_file() for path in frames)
    assert good_capture.set.call_count == 2
    good_capture.release.assert_called_once()


def test_visual_scout_clamps_transcript_duration_to_real_media_eof(tmp_path: Path) -> None:
    provider = PolicyVision()
    requested: list[tuple[float, ...]] = []

    def frames_for_times(_source: Path, times: tuple[float, ...], output: Path) -> list[Path]:
        requested.append(times)
        output.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, timestamp in enumerate(times):
            frame = output / f"{index:03d}-{timestamp:.3f}.jpg"
            frame.write_bytes(b"frame")
            frames.append(frame)
        return frames

    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=2995.838),
        patch("clipper.visual_ai.extract_video_frames", side_effect=frames_for_times),
    ):
        scout_visual_timeline(
            tmp_path / "proxy.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=2997.599,
            output_dir=tmp_path / "frames",
        )
    times = tuple(timestamp for batch in requested for timestamp in batch)
    assert max(times) <= 2995.788
    assert 2997.549 not in times


def test_media_duration_probe_reads_frame_metadata_and_rejects_invalid_values(
    tmp_path: Path,
) -> None:
    video = tmp_path / "duration.mp4"
    video.write_bytes(b"video")
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.side_effect = [30.0, 300.0]
    fake_cv2 = SimpleNamespace(
        VideoCapture=Mock(return_value=capture),
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
    )
    with patch.dict(sys.modules, {"cv2": fake_cv2}):
        assert media_duration_seconds(video) == pytest.approx(10.0)
    capture.release.assert_called_once()

    bad = Mock()
    bad.isOpened.return_value = True
    bad.get.side_effect = [0.0, 300.0]
    fake_cv2.VideoCapture.return_value = bad
    with (
        patch.dict(sys.modules, {"cv2": fake_cv2}),
        pytest.raises(RuntimeError, match="determine media duration"),
    ):
        media_duration_seconds(video)
    bad.release.assert_called_once()


class CapacityPolicyVision(PolicyVision):
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity
        self.attempted_sizes: list[int] = []

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        self.attempted_sizes.append(len(frames))
        if len(frames) > self.capacity:
            raise RuntimeError("CUDA out of memory")
        return super().inspect(task=task, frames=frames, context=context)


class InterruptingPolicyVision(PolicyVision):
    def __init__(self, fail_after: int | None) -> None:
        super().__init__()
        self.fail_after = fail_after

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, object]
    ) -> ProviderResult[dict[str, object]]:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated external interruption")
        return super().inspect(task=task, frames=frames, context=context)


def _prepared_source_policy_frames(
    _source: Path, times: tuple[float, ...], output: Path
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, timestamp in enumerate(times):
        frame = output / f"{index:03d}-{timestamp:.3f}.jpg"
        frame.write_bytes(b"frame")
        frames.append(frame)
    return frames


def test_source_policy_capacity_is_learned_from_runtime_failures(
    tmp_path: Path,
) -> None:
    provider = CapacityPolicyVision(capacity=3)
    commits: list[None] = []
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=40.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=40.0,
            output_dir=tmp_path / "frames",
            checkpoint_dir=tmp_path / "cache",
            checkpoint_commit=lambda: commits.append(None),
        )
    sample_count = int(timeline.coverage_summary("source_policy")["sample_count"])
    assert len(timeline.events) == sample_count
    assert any(size > provider.capacity for size in provider.attempted_sizes)
    assert (
        max(size for size in provider.attempted_sizes if size <= provider.capacity)
        == provider.capacity
    )
    assert result.usage.input_units == sample_count
    assert commits


def test_source_policy_resume_reuses_durable_frame_checkpoints(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    first = InterruptingPolicyVision(fail_after=1)
    commits: list[None] = []
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=20.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
        pytest.raises(RuntimeError, match="simulated external interruption"),
    ):
        scout_visual_timeline(
            tmp_path / "source.mp4",
            first,
            video_id="v",
            source_hash="h",
            duration=20.0,
            output_dir=tmp_path / "first",
            checkpoint_dir=cache,
            checkpoint_commit=lambda: commits.append(None),
        )
    assert commits
    completed_timestamps = {
        round(float(timestamp), 3)
        for _, _, context in first.calls
        for timestamp in context["frame_timestamps"]  # type: ignore[index]
    }

    resumed = InterruptingPolicyVision(fail_after=None)
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=20.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            resumed,
            video_id="v",
            source_hash="h",
            duration=20.0,
            output_dir=tmp_path / "second",
            checkpoint_dir=cache,
            checkpoint_commit=lambda: commits.append(None),
        )
    resumed_timestamps = {
        round(float(timestamp), 3)
        for _, _, context in resumed.calls
        for timestamp in context["frame_timestamps"]  # type: ignore[index]
    }
    assert completed_timestamps
    assert completed_timestamps.isdisjoint(resumed_timestamps)
    assert len(timeline.events) == int(timeline.coverage_summary("source_policy")["sample_count"])
    assert result.usage.runtime["source_policy_cache_hits"] >= len(completed_timestamps)


def test_source_policy_fully_cached_resume_performs_no_inference(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    provider = PolicyVision()
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=12.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
    ):
        scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=12.0,
            output_dir=tmp_path / "first",
            checkpoint_dir=cache,
        )
    cached = PolicyVision()
    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=12.0),
        patch("clipper.visual_ai.extract_video_frames") as extract,
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            cached,
            video_id="v",
            source_hash="h",
            duration=12.0,
            output_dir=tmp_path / "second",
            checkpoint_dir=cache,
        )
    assert not cached.calls
    extract.assert_not_called()
    assert result.usage.provider == "cache"
    assert result.usage.runtime["source_policy_cache_hits"] == int(
        timeline.coverage_summary("source_policy")["sample_count"]
    )

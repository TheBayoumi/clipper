from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from clipper import visual_ai
from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.providers.modal import ModalRemoteError, ModalVisionProvider


def _identity() -> ModelIdentity:
    return ModelIdentity("vision-model", "rev", "none", "test", "prompt", "schema")


def _spy_module() -> Any:
    path = Path("scripts/modal_execution_spy.py")
    spec = importlib.util.spec_from_file_location("modal_execution_spy_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load modal execution spy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_vision_deadline_cancels_remote_call_and_surfaces_repartition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    monkeypatch.setenv("CLIPPER_VISION_CALL_DEADLINE_SECONDS", "12.5")
    provider = ModalVisionProvider(
        app_name="app",
        identity=_identity(),
        function_name="vision",
    )
    call = Mock()
    call.get.side_effect = [TimeoutError(), RuntimeError("cancelled")]
    function = Mock()
    function.spawn.return_value = call

    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(ModalRemoteError) as raised,
    ):
        provider.inspect(task="source_policy_visual_scout", frames=[frame], context={})

    assert raised.value.error_type == "VisionGenerationDeadlineError"
    assert raised.value.details["reason"] == "generation_runtime_deadline"
    assert raised.value.details["recovery_action"] == "REPARTITION"
    assert raised.value.details["frames"] == 1
    assert call.get.call_count == 2
    assert call.get.call_args_list[0].kwargs == {"timeout": 12.5}
    assert call.get.call_args_list[1].kwargs == {"timeout": 30.0}
    call.cancel.assert_called_once_with()
    assert raised.value.details["cancellation_confirmed"] is True


def test_vision_deadline_is_repartitionable_capacity_signal() -> None:
    error = ModalRemoteError(
        function_name="VisionModel.inspect",
        error_type="VisionGenerationDeadlineError",
        message="vision generation exceeded the bounded client deadline",
        details={"reason": "generation_runtime_deadline"},
    )
    assert visual_ai._is_vision_capacity_error(error)


def test_source_policy_dynamic_batches_never_exceed_safe_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()

    class Provider:
        def __init__(self) -> None:
            self.identity = identity
            self.batch_sizes: list[int] = []

        def warm(self) -> dict[str, object]:
            return {}

        def inspect(
            self,
            *,
            task: str,
            frames: list[Path],
            context: dict[str, Any],
        ) -> ProviderResult[dict[str, Any]]:
            assert task == "source_policy_visual_scout"
            self.batch_sizes.append(len(frames))
            timestamps = context["frame_timestamps"]
            assert isinstance(timestamps, list)
            observations = [
                {
                    "timestamp": float(timestamp),
                    "scene_id": f"scene-{index}",
                    "summary": "visible source frame",
                    "visible_speakers": [],
                    "event_labels": [],
                    "confidence": 0.9,
                }
                for index, timestamp in enumerate(timestamps)
            ]
            return ProviderResult(
                {"observations": observations},
                identity,
                InferenceUsage("test", "now", 0.0),
            )

    provider = Provider()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(visual_ai, "media_duration_seconds", lambda _path: 400.0)
    monkeypatch.setattr(
        visual_ai,
        "extract_video_frames",
        lambda _path, times, _output: [
            tmp_path / f"frame-{index}.jpg" for index, _ in enumerate(times)
        ],
    )

    visual_ai.scout_visual_timeline(
        source,
        provider,  # type: ignore[arg-type]
        video_id="video",
        source_hash="hash",
        duration=400.0,
        output_dir=tmp_path / "frames",
    )

    assert provider.batch_sizes
    assert max(provider.batch_sizes) == visual_ai.SOURCE_POLICY_MAX_BATCH_FRAMES
    assert max(provider.batch_sizes) <= 32


def test_source_policy_cache_namespace_ignores_runtime_deadline_and_batch_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = {
        "source_hash": "source-hash",
        "requested_identity": _identity(),
        "required_visual_entities": ("Lovable",),
    }
    before = visual_ai._source_policy_cache_namespace(**kwargs)
    monkeypatch.setenv("CLIPPER_VISION_CALL_DEADLINE_SECONDS", "1")
    monkeypatch.setenv("CLIPPER_MODAL_VISION_STALL_SECONDS", "2")
    after = visual_ai._source_policy_cache_namespace(**kwargs)
    assert before == after


def test_spy_aborts_stalled_vision_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _spy_module()
    clock = {"now": 100.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "spy.ndjson",
        execution_id="exec-1",
        vision_stall_seconds=10,
    )
    spy._record(
        "clipper-open-editor",
        '{"event":"vision_generation_start","execution_id":"exec-1",'
        '"worker_lifecycle_id":"worker-1","task":"source_policy_visual_scout",'
        '"attempt":1,"frames":32}',
    )
    assert spy.abort_reason is None
    assert len(spy.summary()["active_vision_generations"]) == 1

    clock["now"] = 111.0
    spy._check_stalled_vision_generations()

    assert spy.abort_reason is not None
    assert "vision generation" in spy.abort_reason
    assert spy.abort_event["worker_lifecycle_id"] == "worker-1"


def test_spy_clears_vision_generation_on_completion(
    tmp_path: Path,
) -> None:
    module = _spy_module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "spy.ndjson",
        execution_id="exec-1",
        vision_stall_seconds=10,
    )
    spy._record(
        "clipper-open-editor",
        '{"event":"vision_generation_start","execution_id":"exec-1",'
        '"worker_lifecycle_id":"worker-1","task":"source_policy_visual_scout",'
        '"attempt":1,"frames":8}',
    )
    spy._record(
        "clipper-open-editor",
        '{"event":"vision_generation_complete","execution_id":"exec-1",'
        '"worker_lifecycle_id":"worker-1","attempt":1,"frames":8,'
        '"generated_tokens":123,"duration_seconds":2.5}',
    )

    assert spy.abort_reason is None
    assert spy.summary()["active_vision_generations"] == []


def test_production_workflow_grants_spy_comment_permission_and_sets_vision_bounds() -> None:
    workflow = Path(".github/workflows/production-pipeline.yml").read_text(encoding="utf-8")
    assert "pull-requests: read" in workflow
    assert "CLIPPER_VISION_CALL_DEADLINE_SECONDS: 360" in workflow
    assert "CLIPPER_MODAL_VISION_STALL_SECONDS: 420" in workflow


def test_modal_vision_workers_have_no_hard_timeout() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    for class_name in ("VisionModel", "VisionModelLarge"):
        prefix = source.split(f"class {class_name}:", 1)[0]
        decorator = prefix.rsplit("@app.cls(", 1)[-1]
        assert "timeout=" not in decorator


def test_modal_vision_deadline_fails_closed_when_cancel_fails(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    provider = ModalVisionProvider(
        app_name="app",
        identity=_identity(),
        function_name="vision",
    )
    call = Mock()
    call.get.side_effect = TimeoutError
    call.cancel.side_effect = RuntimeError("cancel transport failed")
    function = Mock()
    function.spawn.return_value = call

    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(ModalRemoteError) as raised,
    ):
        provider.inspect(task="source_policy_visual_scout", frames=[frame], context={})

    assert raised.value.error_type == "VisionCancellationUnconfirmedError"
    assert raised.value.details["reason"] == "vision_cancellation_unconfirmed"
    assert "recovery_action" not in raised.value.details
    assert not visual_ai._is_vision_capacity_error(raised.value)


def test_modal_vision_deadline_fails_closed_when_terminal_confirmation_times_out(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    provider = ModalVisionProvider(
        app_name="app",
        identity=_identity(),
        function_name="vision",
    )
    call = Mock()
    call.get.side_effect = [TimeoutError(), TimeoutError()]
    function = Mock()
    function.spawn.return_value = call

    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(ModalRemoteError) as raised,
    ):
        provider.inspect(task="source_policy_visual_scout", frames=[frame], context={})

    assert raised.value.error_type == "VisionCancellationUnconfirmedError"
    assert raised.value.details["reason"] == "vision_cancellation_unconfirmed"
    assert "recovery_action" not in raised.value.details
    assert not visual_ai._is_vision_capacity_error(raised.value)
    call.cancel.assert_called_once_with()


def test_modal_vision_deadline_fails_closed_on_confirmation_transport_error(
    tmp_path: Path,
) -> None:
    class FakeModalError(Exception):
        pass

    class ServiceError(FakeModalError):
        pass

    class RemoteError(FakeModalError):
        pass

    exception_namespace = type(
        "ExceptionNamespace",
        (),
        {
            "Error": FakeModalError,
            "ServiceError": ServiceError,
            "RemoteError": RemoteError,
        },
    )
    fake_modal = type("FakeModal", (), {"exception": exception_namespace})()

    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    provider = ModalVisionProvider(
        app_name="app",
        identity=_identity(),
        function_name="vision",
    )
    call = Mock()
    call.get.side_effect = [TimeoutError(), ServiceError("control plane unavailable")]
    function = Mock()
    function.spawn.return_value = call

    with (
        patch.object(provider, "_function", return_value=function),
        patch.object(provider, "_modal", return_value=fake_modal),
        pytest.raises(ModalRemoteError) as raised,
    ):
        provider.inspect(task="source_policy_visual_scout", frames=[frame], context={})

    assert raised.value.error_type == "VisionCancellationUnconfirmedError"
    assert raised.value.details["reason"] == "vision_cancellation_unconfirmed"
    assert raised.value.details["confirmation_error_type"] == "ServiceError"
    assert "recovery_action" not in raised.value.details
    assert not visual_ai._is_vision_capacity_error(raised.value)
    assert call.get.call_count == 2
    call.cancel.assert_called_once_with()

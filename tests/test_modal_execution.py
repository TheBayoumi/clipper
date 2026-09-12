from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock, patch
from unittest.mock import call as mock_call

import pytest

from clipper.modal_execution import (
    ProductionBudgetExceeded,
    ProductionCallNotTerminated,
    ProductionCallSubmissionFailed,
    _acquire_remote_source,
    _BudgetLedger,
    _cancel_confirmation_seconds,
    _cancel_remote_call,
    _cancellation_confirmation_is_terminal,
    _class,
    _deploy,
    _explicit_candidates,
    _function,
    _invoke_remote_with_budget,
    _load_reviewed_resume_provenance,
    _local_git_sha,
    _materialize_remote_run,
    _positive_budget,
    _runtime_source_sha,
    _validate_model_access,
    _verify_deployed_runtime_sha,
    ensure_modal_runtime,
    run_modal_pipeline,
)
from clipper.modal_execution import (
    _spawn_recoverable_modal_call as _real_spawn_recoverable_modal_call,
)
from clipper.models import VideoCandidate


class NotFoundError(RuntimeError):
    pass


class ServiceError(RuntimeError):
    pass


class InputCancellation(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _synthetic_recoverable_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    def spawn(
        function: Any,
        request: dict[str, Any],
        **_kwargs: object,
    ) -> tuple[Any, float, None]:
        started = __import__("time").monotonic()
        return function.spawn(request), started, None

    monkeypatch.setattr("clipper.modal_execution._spawn_recoverable_modal_call", spawn)


def _fake_modal_submission_modules(
    *,
    map_response: object,
    put_response: object | None = None,
    put_error: Exception | None = None,
) -> tuple[object, object, object, object, object, list[str]]:
    events: list[str] = []
    stub = SimpleNamespace(
        FunctionMap=AsyncMock(return_value=map_response),
        FunctionPutInputs=AsyncMock(),
    )
    if put_error is not None:
        stub.FunctionPutInputs.side_effect = put_error
    else:
        stub.FunctionPutInputs.return_value = put_response
    internal_function = SimpleNamespace(
        object_id="fu-recoverable",
        client=SimpleNamespace(stub=stub),
        hydrate=AsyncMock(),
    )

    class Synchronizer:
        def _translate_in(self, _function: object) -> object:
            return internal_function

        def create_blocking(self, async_function: Any) -> Any:
            return lambda: asyncio.run(async_function())

    synchronizer = Synchronizer()
    async_utils = SimpleNamespace(synchronizer=synchronizer)
    function_utils = SimpleNamespace(_create_input=AsyncMock(return_value="serialized-input"))

    def map_request(**kwargs: object) -> object:
        return SimpleNamespace(**kwargs)

    def put_request(**kwargs: object) -> object:
        events.append("put-input")
        return SimpleNamespace(**kwargs)

    api_pb2 = SimpleNamespace(
        FUNCTION_CALL_TYPE_UNARY=1,
        FUNCTION_CALL_INVOCATION_TYPE_ASYNC=2,
        FunctionMapRequest=map_request,
        FunctionPutInputsRequest=put_request,
    )
    call = Mock()
    call.object_id = "fc-recoverable"

    def from_id(call_id: str) -> object:
        events.append("from-id")
        assert call_id == "fc-recoverable"
        return call

    modal = SimpleNamespace(FunctionCall=SimpleNamespace(from_id=from_id))
    return modal, async_utils, function_utils, api_pb2, stub, events


def test_recoverable_modal_submission_allocates_handle_before_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_response = SimpleNamespace(
        function_call_id="fc-recoverable",
        pipelined_inputs=[],
    )
    put_response = SimpleNamespace(inputs=[SimpleNamespace(input_id="in-1")])
    modal, async_utils, function_utils, api_pb2, stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=put_response,
    )
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: 12.5)
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )

    call, started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=_BudgetLedger(100.0, 100.0),
        gpu_count=1.0,
        estimated_usd_per_second=0.01,
    )

    assert call.object_id == "fc-recoverable"
    assert started == pytest.approx(12.5)
    assert error is None
    assert events == ["from-id", "put-input"]
    function_utils._create_input.assert_awaited_once()
    map_request = stub.FunctionMap.await_args.args[0]
    assert map_request.function_id == "fu-recoverable"
    assert not hasattr(map_request, "pipelined_inputs")
    put_request = stub.FunctionPutInputs.await_args.args[0]
    assert put_request.function_call_id == "fc-recoverable"
    assert put_request.function_id == "fu-recoverable"
    assert put_request.inputs == ["serialized-input"]


@pytest.mark.parametrize(
    ("map_response", "error_match"),
    [
        (
            SimpleNamespace(function_call_id="", pipelined_inputs=[]),
            "did not allocate a recoverable function_call_id",
        ),
        (
            SimpleNamespace(
                function_call_id="fc-recoverable",
                pipelined_inputs=[SimpleNamespace(input_id="unexpected")],
            ),
            "unexpectedly attached producer inputs",
        ),
    ],
)
def test_recoverable_modal_submission_rejects_unsafe_call_allocation(
    monkeypatch: pytest.MonkeyPatch,
    map_response: object,
    error_match: str,
) -> None:
    modal, async_utils, function_utils, api_pb2, stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=SimpleNamespace(inputs=[SimpleNamespace(input_id="in-1")]),
    )
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )

    with pytest.raises(RuntimeError, match=error_match):
        _real_spawn_recoverable_modal_call(
            object(),
            {"request": True},
            budget=_BudgetLedger(100.0, 100.0),
            gpu_count=1.0,
            estimated_usd_per_second=0.01,
        )

    stub.FunctionPutInputs.assert_not_awaited()
    function_utils._create_input.assert_not_awaited()
    assert events == []


def test_recoverable_modal_submission_reconciles_missing_input_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_response = SimpleNamespace(
        function_call_id="fc-recoverable",
        pipelined_inputs=[],
    )
    modal, async_utils, function_utils, api_pb2, _stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=SimpleNamespace(inputs=[]),
    )
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: 3.0)
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )

    call, started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=_BudgetLedger(100.0, 100.0),
        gpu_count=1.0,
        estimated_usd_per_second=0.01,
    )

    assert call.object_id == "fc-recoverable"
    assert started == pytest.approx(3.0)
    assert isinstance(error, ProductionCallSubmissionFailed)
    assert error.call_id == "fc-recoverable"
    assert "did not acknowledge exactly one producer input" in error.error
    assert events == ["from-id", "put-input"]


def test_recoverable_modal_submission_reconciles_serialization_failure_before_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    map_response = SimpleNamespace(
        function_call_id="fc-recoverable",
        pipelined_inputs=[],
    )
    modal, async_utils, function_utils, api_pb2, stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=SimpleNamespace(inputs=[SimpleNamespace(input_id="in-1")]),
    )

    async def fail_serialize(*_args: object, **_kwargs: object) -> object:
        clock["now"] = 0.5
        raise ServiceError("serialization failed")

    function_utils._create_input.side_effect = fail_serialize
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )
    budget = _BudgetLedger(10.0, 100.0)

    call, started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=budget,
        gpu_count=1.0,
        estimated_usd_per_second=0.01,
    )

    assert call.object_id == "fc-recoverable"
    assert started == pytest.approx(0.5)
    assert isinstance(error, ProductionCallSubmissionFailed)
    assert error.call_id == "fc-recoverable"
    assert error.error_type == "ServiceError"
    assert "serialization failed" in error.error
    stub.FunctionPutInputs.assert_not_awaited()
    assert events == ["from-id"]
    assert budget.gpu_seconds == pytest.approx(0.5)
    assert budget.estimated_usd == pytest.approx(0.005)


def test_recoverable_modal_submission_rejects_before_input_when_serialization_exhausts_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    map_response = SimpleNamespace(
        function_call_id="fc-recoverable",
        pipelined_inputs=[],
    )
    modal, async_utils, function_utils, api_pb2, stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=SimpleNamespace(inputs=[SimpleNamespace(input_id="in-1")]),
    )

    async def slow_serialize(*_args: object, **_kwargs: object) -> object:
        clock["now"] = 1.0
        return "serialized-input"

    function_utils._create_input.side_effect = slow_serialize
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )
    budget = _BudgetLedger(1.0, 100.0)

    call, started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=budget,
        gpu_count=1.0,
        estimated_usd_per_second=0.0,
    )

    assert call.object_id == "fc-recoverable"
    assert started == pytest.approx(1.0)
    assert isinstance(error, ProductionBudgetExceeded)
    assert "before Modal producer input attachment" in str(error)
    stub.FunctionPutInputs.assert_not_awaited()
    assert events == ["from-id"]
    assert budget.gpu_seconds == pytest.approx(1.0)


def test_recoverable_modal_submission_rechecks_budget_at_attachment_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_response = SimpleNamespace(
        function_call_id="fc-recoverable",
        pipelined_inputs=[],
    )
    modal, async_utils, function_utils, api_pb2, stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=SimpleNamespace(inputs=[SimpleNamespace(input_id="in-1")]),
    )
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    clock = {"now": 0.0}
    create_blocking = async_utils.synchronizer.create_blocking

    def delayed_create_blocking(async_function: Any) -> Any:
        blocking = create_blocking(async_function)
        if async_function.__name__ != "submit_input":
            return blocking

        def delayed_submit() -> object:
            clock["now"] = 1.0
            return blocking()

        return delayed_submit

    async_utils.synchronizer.create_blocking = delayed_create_blocking
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )
    budget = _BudgetLedger(1.0, 100.0)

    call, started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=budget,
        gpu_count=1.0,
        estimated_usd_per_second=0.0,
    )

    assert call.object_id == "fc-recoverable"
    assert started == pytest.approx(1.0)
    assert isinstance(error, ProductionBudgetExceeded)
    assert "input attachment boundary" in str(error)
    stub.FunctionPutInputs.assert_not_awaited()
    assert events == ["from-id"]
    assert budget.gpu_seconds == pytest.approx(1.0)


def test_recoverable_modal_submission_keeps_exact_call_on_lost_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    map_response = SimpleNamespace(
        function_call_id="fc-recoverable",
        pipelined_inputs=[],
    )
    modal, async_utils, function_utils, api_pb2, _stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_error=ServiceError("input acknowledgement lost"),
    )
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: 4.0)
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )

    call, started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=_BudgetLedger(100.0, 100.0),
        gpu_count=1.0,
        estimated_usd_per_second=0.01,
    )

    assert call.object_id == "fc-recoverable"
    assert started == pytest.approx(4.0)
    assert isinstance(error, ProductionCallSubmissionFailed)
    assert error.call_id == "fc-recoverable"
    assert error.error_type == "ServiceError"
    assert "acknowledgement lost" in error.error
    assert events == ["from-id", "put-input"]


def _write_brief(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign",
                "title": "Podcast",
                "objective": "Find worthwhile complete clips",
                "targets": {
                    "mode": "explicit",
                    "videos": [
                        {
                            "video_id": "v1",
                            "url": "https://www.youtube.com/watch?v=v1",
                            "channel_id": "UC1",
                        }
                    ],
                },
                "rights": {"confirmed": True, "authorized_channels": ["UC1"]},
                "content_constraints": {"min_clip_seconds": 20, "max_clip_seconds": 45},
            }
        ),
        encoding="utf-8",
    )


def test_function_hydrates_deployed_handle() -> None:
    handle = Mock()
    from_name = Mock(return_value=handle)
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=from_name))
    with patch("clipper.modal_execution.importlib.import_module", return_value=modal):
        assert _function("app", "worker") is handle
    from_name.assert_called_once_with("app", "worker")
    handle.hydrate.assert_called_once_with()


def test_class_hydrates_deployed_handle() -> None:
    handle = Mock()
    from_name = Mock(return_value=handle)
    modal = SimpleNamespace(Cls=SimpleNamespace(from_name=from_name))
    with patch("clipper.modal_execution.importlib.import_module", return_value=modal):
        assert _class("app", "Worker") is handle
    from_name.assert_called_once_with("app", "Worker")
    handle.hydrate.assert_called_once_with()


def test_function_retries_transient_service_errors() -> None:
    handle = Mock()
    handle.hydrate.side_effect = [ServiceError("unavailable"), ServiceError("unavailable"), None]
    from_name = Mock(return_value=handle)
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=from_name))
    with (
        patch("clipper.modal_execution.time.sleep") as sleep,
        patch("clipper.modal_execution.importlib.import_module", return_value=modal),
    ):
        assert _function("app", "worker") is handle
    assert from_name.call_count == 3
    assert [item.args[0] for item in sleep.call_args_list] == [2.0, 5.0]


def test_function_does_not_retry_nontransient_hydration_failure() -> None:
    handle = Mock()
    handle.hydrate.side_effect = ValueError("bad handle")
    from_name = Mock(return_value=handle)
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=from_name))
    with (
        patch("clipper.modal_execution.importlib.import_module", return_value=modal),
        pytest.raises(ValueError, match="bad handle"),
    ):
        _function("app", "worker")
    from_name.assert_called_once_with("app", "worker")


def test_deploy_requires_modal_cli_and_existing_source(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    with (
        patch("clipper.modal_execution.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="Modal CLI"),
    ):
        _deploy(script)

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        pytest.raises(RuntimeError, match="deployment source"),
    ):
        _deploy(script)

    script.write_text("# worker\n", encoding="utf-8")
    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution._repo_root", return_value=tmp_path),
        patch("clipper.modal_execution._local_git_sha", return_value="a" * 40),
        patch("clipper.modal_execution.subprocess.run") as run,
    ):
        _deploy(script)
    run.assert_called_once_with(
        ["modal", "deploy", str(script)],
        check=True,
        timeout=1800,
        cwd=tmp_path,
        env=run.call_args.kwargs["env"],
    )
    assert run.call_args.kwargs["env"]["CLIPPER_DEPLOYED_GIT_SHA"] == "a" * 40


def test_deploy_retries_cli_failure(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text("# worker\n", encoding="utf-8")
    failure = subprocess.CalledProcessError(1, ["modal", "deploy", str(script)])
    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution._repo_root", return_value=tmp_path),
        patch("clipper.modal_execution._local_git_sha", return_value="a" * 40),
        patch("clipper.modal_execution.subprocess.run", side_effect=[failure, None]) as run,
        patch("clipper.modal_execution.time.sleep") as sleep,
    ):
        _deploy(script)
    assert run.call_count == 2
    sleep.assert_called_once_with(2.0)


def test_deploy_raises_after_final_cli_failure(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text("# worker\n", encoding="utf-8")
    failure = subprocess.CalledProcessError(1, ["modal", "deploy", str(script)])
    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution._repo_root", return_value=tmp_path),
        patch("clipper.modal_execution._local_git_sha", return_value="a" * 40),
        patch("clipper.modal_execution.subprocess.run", side_effect=failure) as run,
        patch("clipper.modal_execution.time.sleep"),
        pytest.raises(subprocess.CalledProcessError),
    ):
        _deploy(script)
    assert run.call_count == 3


def test_ensure_modal_runtime_attaches_without_deploying_when_apps_exist() -> None:
    with (
        patch("clipper.modal_execution._function", return_value=Mock()) as function,
        patch("clipper.modal_execution._class", return_value=Mock()) as cls,
        patch("clipper.modal_execution._deploy") as deploy,
        patch("clipper.modal_execution._repo_script", side_effect=lambda name: Path(name)),
        patch("clipper.modal_execution._validate_model_access") as validate,
    ):
        ensure_modal_runtime()
    assert function.call_count == 9
    assert cls.call_count == 3
    deploy.assert_not_called()
    validate.assert_called_once_with("clipper-open-editor")


def test_ensure_modal_runtime_repairs_missing_model_without_redeploying_pipeline() -> None:
    calls: list[tuple[str, str]] = []
    model_failed = False

    def fake_function(app: str, name: str) -> Mock:
        nonlocal model_failed
        calls.append((app, name))
        if name == "transcribe" and not model_failed:
            model_failed = True
            raise NotFoundError("missing model")
        return Mock()

    with (
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch("clipper.modal_execution._class", return_value=Mock()),
        patch("clipper.modal_execution._deploy") as deploy,
        patch("clipper.modal_execution._repo_script", side_effect=lambda name: Path(name)),
        patch("clipper.modal_execution._validate_model_access") as validate,
    ):
        ensure_modal_runtime()

    assert model_failed is True
    assert [call.args[0].name for call in deploy.call_args_list] == ["modal_open_models.py"]
    assert ("clipper-open-editor", "transcribe") in calls
    assert ("clipper-open-editor", "hf_access_smoke") in calls
    assert ("clipper-production-pipeline", "run_full_cycle") in calls
    validate.assert_called_once_with("clipper-open-editor")


def test_ensure_modal_runtime_repairs_missing_pipeline_only() -> None:
    calls: list[tuple[str, str]] = []
    pipeline_failed = False

    def fake_function(app: str, name: str) -> Mock:
        nonlocal pipeline_failed
        calls.append((app, name))
        if (
            app == "clipper-production-pipeline"
            and name == "acquire_source"
            and not pipeline_failed
        ):
            pipeline_failed = True
            raise NotFoundError("missing pipeline")
        return Mock()

    with (
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch("clipper.modal_execution._class", return_value=Mock()),
        patch("clipper.modal_execution._deploy") as deploy,
        patch("clipper.modal_execution._repo_script", side_effect=lambda name: Path(name)),
        patch("clipper.modal_execution._validate_model_access") as validate,
    ):
        ensure_modal_runtime()

    assert pipeline_failed is True
    deploy.assert_called_once_with(Path("modal_pipeline.py"))
    assert ("clipper-open-editor", "editorial_schema_smoke") in calls
    assert ("clipper-production-pipeline", "run_full_cycle") in calls
    validate.assert_called_once_with("clipper-open-editor")


def test_ensure_modal_runtime_does_not_redeploy_on_connectivity_failure() -> None:
    with (
        patch("clipper.modal_execution._function", side_effect=ServiceError("unavailable")),
        patch("clipper.modal_execution._class", return_value=Mock()),
        patch("clipper.modal_execution._deploy") as deploy,
        pytest.raises(RuntimeError, match="control-plane validation failed"),
    ):
        ensure_modal_runtime()
    deploy.assert_not_called()


def test_ensure_modal_runtime_fails_closed_after_unsuccessful_redeploy() -> None:
    with (
        patch(
            "clipper.modal_execution._function",
            side_effect=[NotFoundError("missing"), RuntimeError("still missing")],
        ),
        patch("clipper.modal_execution._class", return_value=Mock()),
        patch("clipper.modal_execution._deploy"),
        pytest.raises(RuntimeError, match="unavailable after runtime repair"),
    ):
        ensure_modal_runtime()


def test_validate_model_access_requires_successful_remote_smoke() -> None:
    smoke = Mock()
    smoke.remote.return_value = {
        "ok": True,
        "model_id": "pyannote/speaker-diarization-community-1",
        "revision": "revision",
    }
    with patch("clipper.modal_execution._function", return_value=smoke) as function:
        _validate_model_access("clipper-open-editor")
    function.assert_called_once_with("clipper-open-editor", "hf_access_smoke")

    smoke.remote.return_value = {"ok": False}
    with (
        patch("clipper.modal_execution._function", return_value=smoke),
        pytest.raises(RuntimeError, match="invalid result"),
    ):
        _validate_model_access("clipper-open-editor")

    smoke.remote.side_effect = RuntimeError("denied")
    with (
        patch("clipper.modal_execution._function", return_value=smoke),
        pytest.raises(RuntimeError, match="Hugging Face access preflight failed"),
    ):
        _validate_model_access("clipper-open-editor")


def test_explicit_candidates_resolve_only_campaign_targets(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidates = _explicit_candidates(brief_path)
    assert [item.video_id for item in candidates] == ["v1"]
    assert candidates[0].url == "https://www.youtube.com/watch?v=v1"
    assert candidates[0].channel_id == "UC1"


def test_explicit_candidate_keeps_youtube_url_when_supplemental_media_url_exists(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    payload = json.loads(brief_path.read_text(encoding="utf-8"))
    payload["targets"]["videos"][0]["media_url"] = (
        "https://drive.google.com/file/d/supplemental/view"
    )
    brief_path.write_text(json.dumps(payload), encoding="utf-8")

    candidates = _explicit_candidates(brief_path)
    assert candidates[0].url == "https://www.youtube.com/watch?v=v1"


def test_acquire_remote_source_uses_modal_egress_and_validates_quality() -> None:
    function = Mock()
    call = Mock()
    call.get.return_value = {
        "video_id": "v1",
        "channel_id": "UC1",
        "quality_policy": "highest_available_no_transcode",
        "bytes": 123,
        "sha256": "abc",
    }
    function.with_options.return_value.spawn.return_value = call
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    result = _acquire_remote_source(
        function, candidate, expected_git_sha="a" * 40, budget=_BudgetLedger(100.0, 1.0)
    )
    assert result["sha256"] == "abc"
    function.with_options.assert_called_once_with(cloud="gcp", timeout=1800)
    function.with_options.return_value.spawn.assert_called_once_with(
        {
            "video_id": "v1",
            "channel_id": "UC1",
            "video_url": "https://youtu.be/v1",
            "expected_git_sha": "a" * 40,
        }
    )


def test_acquire_remote_source_exhausts_invalid_and_failed_egress() -> None:
    class FailedCall:
        def __init__(self) -> None:
            self.cancelled = False

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            assert timeout > 0
            if self.cancelled:
                raise InputCancellation("cancelled")
            raise RuntimeError("blocked")

        def cancel(self, *, terminate_containers: bool) -> None:
            assert terminate_containers is False
            self.cancelled = True

    class AlwaysBad:
        def with_options(self, **_kwargs: object) -> AlwaysBad:
            return self

        def spawn(self, _payload: dict[str, object]) -> FailedCall:
            return FailedCall()

    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(
            AlwaysBad(),
            candidate,
            expected_git_sha="a" * 40,
            budget=_BudgetLedger(100.0, 1.0),
        )

    function = Mock()
    call = Mock()
    call.get.return_value = {
        "video_id": "v1",
        "channel_id": "UC1",
        "quality_policy": "downgraded",
    }
    function.with_options.return_value.spawn.return_value = call
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(
            function,
            candidate,
            expected_git_sha="a" * 40,
            budget=_BudgetLedger(100.0, 1.0),
        )


def test_acquire_remote_source_skips_invalid_response_and_uses_default_budget() -> None:
    invalid_call = Mock()
    invalid_call.get.return_value = "invalid"
    success_call = Mock()
    success_call.get.return_value = {
        "video_id": "v1",
        "channel_id": "UC1",
        "quality_policy": "highest_available_no_transcode",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    first_variant = SimpleNamespace(spawn=Mock(return_value=invalid_call))
    second_variant = SimpleNamespace(spawn=Mock(return_value=success_call))
    function = Mock()
    function.with_options.side_effect = [first_variant, second_variant]

    result = _acquire_remote_source(
        function,
        VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1"),
        expected_git_sha="a" * 40,
    )

    assert result["sha256"] == "a" * 64
    assert function.with_options.call_args_list == [
        mock_call(cloud="gcp", timeout=1800),
        mock_call(cloud="aws", timeout=1800),
    ]


def test_acquire_remote_source_rejects_wrong_resolved_identity() -> None:
    function = Mock()
    call = Mock()
    call.get.return_value = {
        "video_id": "v1",
        "channel_id": "UC-other",
        "quality_policy": "highest_available_no_transcode",
    }
    function.with_options.return_value.spawn.return_value = call
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with pytest.raises(RuntimeError, match="source acquisition failed"):
        _acquire_remote_source(
            function,
            candidate,
            expected_git_sha="a" * 40,
            budget=_BudgetLedger(100.0, 1.0),
        )


def test_source_acquisition_cost_is_deducted_before_root_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])

    class Call:
        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            assert timeout > 0
            clock["now"] = 10.0
            return {
                "video_id": "v1",
                "channel_id": "UC1",
                "quality_policy": "highest_available_no_transcode",
                "bytes": 123,
                "sha256": "a" * 64,
            }

        def cancel(self, *, terminate_containers: bool) -> None:
            raise AssertionError(terminate_containers)

    function = Mock()
    function.with_options.return_value.spawn.return_value = Call()
    budget = _BudgetLedger(100.0, 1.0)
    _acquire_remote_source(
        function,
        VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1"),
        expected_git_sha="a" * 40,
        budget=budget,
    )
    assert budget.gpu_seconds == 0.0
    assert budget.estimated_usd > 0.0
    assert budget.remaining_budgets()[1] < 1.0


def test_budget_ledger_enforces_rates_and_remaining_capacity() -> None:
    budget = _BudgetLedger(10.0, 1.0)
    budget.charge(2.0, gpu_count=1.0, estimated_usd_per_second=0.1)
    assert budget.projected_usage(
        1.0,
        gpu_count=1.0,
        estimated_usd_per_second=0.1,
    ) == pytest.approx((3.0, 0.3))
    assert budget.remaining_wall_seconds(
        0.0,
        gpu_count=1.0,
        estimated_usd_per_second=0.1,
    ) == pytest.approx(8.0)
    assert budget.remaining_budgets() == pytest.approx((8.0, 0.8))
    assert budget.to_dict()["remaining_estimated_usd"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="gpu_count"):
        budget.projected_usage(
            0.0,
            gpu_count=-1.0,
            estimated_usd_per_second=0.0,
        )

    gpu_exhausted = _BudgetLedger(1.0, 1.0, gpu_seconds=2.0)
    assert (
        gpu_exhausted.remaining_wall_seconds(
            0.0,
            gpu_count=0.0,
            estimated_usd_per_second=0.0,
        )
        == 0.0
    )
    cost_exhausted = _BudgetLedger(10.0, 1.0, estimated_usd=2.0)
    assert (
        cost_exhausted.remaining_wall_seconds(
            0.0,
            gpu_count=0.0,
            estimated_usd_per_second=0.0,
        )
        == 0.0
    )
    idle = _BudgetLedger(10.0, 1.0)
    assert idle.remaining_wall_seconds(
        0.0,
        gpu_count=0.0,
        estimated_usd_per_second=0.0,
    ) == float("inf")


def test_runtime_source_sha_fails_closed_without_or_with_conflicting_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    monkeypatch.delenv("CLIPPER_SOURCE_SHA", raising=False)
    with (
        patch(
            "clipper.modal_execution.subprocess.run", side_effect=FileNotFoundError("git missing")
        ),
        pytest.raises(RuntimeError, match="runtime source SHA is unavailable"),
    ):
        _runtime_source_sha()

    monkeypatch.setenv("CLIPPER_SOURCE_SHA", "not-a-sha")
    with pytest.raises(RuntimeError, match="full immutable source SHA"):
        _runtime_source_sha()

    monkeypatch.setenv("CLIPPER_SOURCE_SHA", "a" * 40)
    with (
        patch("clipper.modal_execution._local_git_sha", return_value="b" * 40),
        pytest.raises(RuntimeError, match="runtime source SHA mismatch"),
    ):
        _runtime_source_sha()


def test_remote_invocation_requires_complete_budget() -> None:
    function = SimpleNamespace(spawn=Mock())
    with pytest.raises(ValueError, match="complete compute budget"):
        _invoke_remote_with_budget(function, {}, max_gpu_seconds=1.0)
    function.spawn.assert_not_called()


def test_remote_invocation_charges_terminal_usage_before_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter((0.0, 0.49, 0.49, 0.51))
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: next(timestamps))
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5")
    call = Mock()
    call.get.return_value = {"status": "PASS"}
    function = SimpleNamespace(spawn=Mock(return_value=call))
    budget = _BudgetLedger(1.0, 10.0)

    with pytest.raises(ProductionBudgetExceeded, match="final poll"):
        _invoke_remote_with_budget(
            function,
            {},
            budget=budget,
            gpu_count=2.0,
            estimated_usd_per_second=0.0,
        )

    assert budget.gpu_seconds == pytest.approx(1.02)
    call.cancel.assert_not_called()


def test_source_acquisition_evidences_configuration_and_semantic_failures() -> None:
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    invalid_call = Mock()
    invalid_call.get.return_value = "invalid"
    identity_call = Mock()
    identity_call.get.return_value = {
        "video_id": "v1",
        "channel_id": "wrong",
        "quality_policy": "highest_available_no_transcode",
    }
    quality_call = Mock()
    quality_call.get.return_value = {
        "video_id": "v1",
        "channel_id": "UC1",
        "quality_policy": "downgraded",
    }
    success_call = Mock()
    success_call.get.return_value = {
        "video_id": "v1",
        "channel_id": "UC1",
        "quality_policy": "highest_available_no_transcode",
        "sha256": "a" * 64,
    }
    function = Mock()
    function.with_options.side_effect = [
        RuntimeError("gcp option unavailable"),
        SimpleNamespace(spawn=Mock(return_value=invalid_call)),
        SimpleNamespace(spawn=Mock(return_value=identity_call)),
        SimpleNamespace(spawn=Mock(return_value=quality_call)),
        SimpleNamespace(spawn=Mock(return_value=success_call)),
    ]
    attempts: list[dict[str, object]] = []

    result = _acquire_remote_source(
        function,
        candidate,
        expected_git_sha="a" * 40,
        budget=_BudgetLedger(100.0, 1.0),
        attempt_evidence=attempts,
    )

    assert result["sha256"] == "a" * 64
    assert [attempt["status"] for attempt in attempts] == [
        "FAIL",
        "FAIL",
        "FAIL",
        "FAIL",
        "PASS",
    ]
    assert [attempt.get("phase") for attempt in attempts] == [
        "configuration",
        "validation",
        "validation",
        "validation",
        "complete",
    ]
    assert [attempt.get("error_type") for attempt in attempts[:-1]] == [
        "RuntimeError",
        "InvalidResponse",
        "SourceIdentityError",
        "QualityPolicyError",
    ]
    assert attempts[0]["estimated_usd"] == pytest.approx(0.0)
    assert attempts[0]["gpu_seconds"] == pytest.approx(0.0)


def test_source_acquisition_records_failed_clouds_before_region_success() -> None:
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    failed_calls = []
    variants = []
    for label in ("gcp", "aws", "oci"):
        call = Mock()
        call.get.side_effect = [RuntimeError(f"{label} blocked"), InputCancellation("cancelled")]
        failed_calls.append(call)
        variants.append(SimpleNamespace(spawn=Mock(return_value=call)))
    success_call = Mock()
    success_call.get.return_value = {
        "video_id": "v1",
        "channel_id": "UC1",
        "quality_policy": "highest_available_no_transcode",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    variants.append(SimpleNamespace(spawn=Mock(return_value=success_call)))
    function = Mock()
    function.with_options.side_effect = variants
    attempts: list[dict[str, object]] = []

    result = _acquire_remote_source(
        function,
        candidate,
        expected_git_sha="a" * 40,
        budget=_BudgetLedger(100.0, 1.0),
        attempt_evidence=attempts,
        execution_id="e" * 32,
    )

    assert result["sha256"] == "a" * 64
    assert [item["status"] for item in attempts] == ["FAIL", "FAIL", "FAIL", "PASS"]
    assert function.with_options.call_args_list == [
        mock_call(cloud="gcp", timeout=1800),
        mock_call(cloud="aws", timeout=1800),
        mock_call(cloud="oci", timeout=1800),
        mock_call(region="eu", timeout=1800),
    ]
    for call in failed_calls:
        call.cancel.assert_called_once_with(terminate_containers=False)
    payload = variants[-1].spawn.call_args.args[0]
    assert payload["execution_id"] == "e" * 32


def test_source_acquisition_budget_exhaustion_is_evidenced_and_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5")

    class SlowCall:
        def __init__(self) -> None:
            self.cancelled = False

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            if self.cancelled:
                raise InputCancellation("cancelled")
            assert 0.0 < timeout < 1.0
            clock["now"] = 1.0
            raise TimeoutError

        def cancel(self, *, terminate_containers: bool) -> None:
            assert terminate_containers is False
            self.cancelled = True

    call = SlowCall()
    function = Mock()
    function.with_options.return_value.spawn.return_value = call
    attempts: list[dict[str, object]] = []

    with pytest.raises(RuntimeError, match="compute budget"):
        _acquire_remote_source(
            function,
            VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1"),
            expected_git_sha="a" * 40,
            budget=_BudgetLedger(100.0, 0.00001),
            attempt_evidence=attempts,
        )

    assert attempts[0]["status"] == "BUDGET_EXCEEDED"
    assert call.cancelled is True


def test_materialize_remote_run_downloads_only_artifact_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"

    def fake_run(command: list[str], **_kwargs: object) -> None:
        staging = Path(command[-1])
        downloaded = staging / "campaign-run"
        downloaded.mkdir(parents=True)
        (downloaded / "manifest.json").write_text("{}", encoding="utf-8")
        (downloaded / "clip.mp4").write_bytes(b"clip")

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution.subprocess.run", side_effect=fake_run) as run,
    ):
        result = _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="artifacts-volume",
            remote_run_path="/campaign-run",
        )

    assert result == artifact_root / "campaign-run"
    assert (result / "manifest.json").is_file()
    assert (result / "clip.mp4").read_bytes() == b"clip"
    command = run.call_args.args[0]
    assert command[:5] == ["modal", "volume", "get", "--force", "artifacts-volume"]
    child_env = run.call_args.kwargs["env"]
    assert child_env["PYTHONUTF8"] == "1"
    assert child_env["PYTHONIOENCODING"] == "utf-8"


def test_materialize_remote_run_handles_flat_download_and_bad_targets(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    with (
        patch("clipper.modal_execution.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="Modal CLI"),
    ):
        _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="volume",
            remote_run_path="/run",
        )

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        pytest.raises(RuntimeError, match="invalid remote run path"),
    ):
        _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="volume",
            remote_run_path="/",
        )

    (artifact_root / "run").mkdir(parents=True)
    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        pytest.raises(RuntimeError, match="overwrite"),
    ):
        _materialize_remote_run(
            artifact_root=artifact_root,
            volume_name="volume",
            remote_run_path="/run",
        )

    flat_root = tmp_path / "flat-artifacts"

    def flat_run(command: list[str], **_kwargs: object) -> None:
        staging = Path(command[-1])
        (staging / "manifest.json").write_text("{}", encoding="utf-8")
        (staging / "evidence.json").write_text("{}", encoding="utf-8")

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution.subprocess.run", side_effect=flat_run),
    ):
        result = _materialize_remote_run(
            artifact_root=flat_root,
            volume_name="volume",
            remote_run_path="/flat-run",
        )
    assert (result / "manifest.json").is_file()
    assert (result / "evidence.json").is_file()


def test_materialize_remote_run_rejects_ambiguous_download(tmp_path: Path) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> None:
        staging = Path(command[-1])
        (staging / "a").mkdir()
        (staging / "b").mkdir()
        (staging / "a" / "manifest.json").write_text("{}", encoding="utf-8")
        (staging / "b" / "manifest.json").write_text("{}", encoding="utf-8")

    with (
        patch("clipper.modal_execution.shutil.which", return_value="modal"),
        patch("clipper.modal_execution.subprocess.run", side_effect=fake_run),
        pytest.raises(RuntimeError, match="expected one manifest"),
    ):
        _materialize_remote_run(
            artifact_root=tmp_path / "artifacts",
            volume_name="volume",
            remote_run_path="/run",
        )


def test_run_modal_pipeline_acquires_in_modal_runs_remote_and_materializes(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()
    remote_result = {
        "execution_id": "e" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "/campaign-run",
        "run_volume": "clipper-production-artifacts",
    }
    materialized = tmp_path / "artifacts" / "campaign-run"

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    with (
        patch("clipper.modal_execution.ensure_modal_runtime") as ensure,
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ) as acquire_remote,
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=remote_result,
        ) as invoke,
        patch(
            "clipper.modal_execution._materialize_remote_run",
            return_value=materialized,
        ) as download,
    ):
        result = run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    assert result == materialized
    ensure.assert_called_once_with()
    acquire_remote.assert_called_once_with(
        acquire,
        candidate,
        expected_git_sha="a" * 40,
        budget=ANY,
        execution_id="e" * 32,
    )
    assert invoke.call_args.args[0] is runner
    payload = invoke.call_args.args[1]
    ledger = invoke.call_args.kwargs["budget"]
    assert isinstance(ledger, _BudgetLedger)
    assert ledger.max_gpu_seconds == 21600.0
    assert ledger.max_estimated_usd == 10.0
    assert payload["resume_from_run_id"] is None
    assert payload["resume_provenance"] is None
    assert payload["render"] is True
    assert payload["fresh_inference"] is False
    assert payload["git_sha"] == "a" * 40
    assert payload["execution_id"] == "e" * 32
    assert payload["max_gpu_seconds"] == 21600.0
    assert payload["max_estimated_usd"] == 10.0
    download.assert_called_once_with(
        artifact_root=tmp_path / "artifacts",
        volume_name="clipper-production-artifacts",
        remote_run_path="/campaign-run",
    )


def test_run_modal_pipeline_stops_when_acquisition_exhausts_budget(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    def exhaust_budget(*_args: object, **kwargs: object) -> dict[str, object]:
        budget = kwargs["budget"]
        assert isinstance(budget, _BudgetLedger)
        budget.estimated_usd = budget.max_estimated_usd
        return {"quality_policy": "highest_available_no_transcode"}

    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch("clipper.modal_execution._acquire_remote_source", side_effect=exhaust_budget),
        patch("clipper.modal_execution._invoke_remote_with_budget") as invoke,
        pytest.raises(ProductionBudgetExceeded, match="source acquisition exhausted"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=False,
            fresh_inference=False,
            max_gpu_seconds=100.0,
            max_estimated_usd=1.0,
        )
    invoke.assert_not_called()


def test_run_modal_pipeline_fails_closed_for_runtime_empty_targets_and_bad_runner(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")

    with (
        patch(
            "clipper.modal_execution.ensure_modal_runtime",
            side_effect=RuntimeError("model access denied"),
        ),
        patch("clipper.modal_execution._function") as function,
        pytest.raises(RuntimeError, match="model access denied"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )
    function.assert_not_called()

    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._function", return_value=Mock()),
        patch("clipper.modal_execution._explicit_candidates", return_value=[]),
        pytest.raises(RuntimeError, match="no explicit authorized targets"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts-empty",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    acquire = Mock()
    runner = Mock()
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch(
            "clipper.modal_execution._function",
            side_effect=lambda _app, name: acquire if name == "acquire_source" else runner,
        ),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode"},
        ),
        patch("clipper.modal_execution._invoke_remote_with_budget", return_value="invalid"),
        pytest.raises(RuntimeError, match="invalid response"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts-invalid",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    no_path_result = {
        "execution_id": "e" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "",
    }
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch(
            "clipper.modal_execution._function",
            side_effect=lambda _app, name: acquire if name == "acquire_source" else runner,
        ),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode"},
        ),
        patch("clipper.modal_execution._invoke_remote_with_budget", return_value=no_path_result),
        pytest.raises(RuntimeError, match="no run path"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts-no-path",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )


def test_verify_deployed_runtime_sha_requires_both_apps_match_local_checkout() -> None:
    expected = "a" * 40
    model = Mock()
    model.remote.return_value = {"deployed_git_sha": expected}
    pipeline = Mock()
    pipeline.remote.return_value = {"deployed_git_sha": expected}

    def function(app: str, name: str) -> Mock:
        assert name == "deployment_identity"
        return model if app == "model-app" else pipeline

    with (
        patch("clipper.modal_execution._local_git_sha", return_value=expected),
        patch("clipper.modal_execution._function", side_effect=function),
    ):
        assert (
            _verify_deployed_runtime_sha(model_app="model-app", pipeline_app="pipeline-app")
            == expected
        )

    pipeline.remote.return_value = {"deployed_git_sha": "b" * 40}
    with (
        patch("clipper.modal_execution._local_git_sha", return_value=expected),
        patch("clipper.modal_execution._function", side_effect=function),
        pytest.raises(RuntimeError, match="pipeline deployed SHA mismatch"),
    ):
        _verify_deployed_runtime_sha(model_app="model-app", pipeline_app="pipeline-app")


def test_source_acquisition_rejects_missing_exact_sha_before_remote() -> None:
    function = Mock()
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with pytest.raises(ValueError, match="full expected_git_sha"):
        _acquire_remote_source(function, candidate, expected_git_sha="")
    function.with_options.assert_not_called()
    function.remote.assert_not_called()


def test_local_git_sha_requires_full_hex_checkout() -> None:
    completed = SimpleNamespace(stdout="A" * 40 + "\n")
    with (
        patch("clipper.modal_execution._repo_root", return_value=Path("/repo")),
        patch("clipper.modal_execution.subprocess.run", return_value=completed) as run,
    ):
        assert _local_git_sha() == "a" * 40
    run.assert_called_once_with(
        ["git", "rev-parse", "HEAD"],
        cwd=Path("/repo"),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    with (
        patch("clipper.modal_execution._repo_root", return_value=Path("/repo")),
        patch(
            "clipper.modal_execution.subprocess.run",
            return_value=SimpleNamespace(stdout="not-a-sha\n"),
        ),
        pytest.raises(RuntimeError, match="full git SHA"),
    ):
        _local_git_sha()


def test_verify_deployed_runtime_sha_rejects_invalid_identity_object() -> None:
    handle = Mock()
    handle.remote.return_value = "invalid"
    with (
        patch("clipper.modal_execution._local_git_sha", return_value="a" * 40),
        patch("clipper.modal_execution._function", return_value=handle),
        pytest.raises(RuntimeError, match="deployment identity is not an object"),
    ):
        _verify_deployed_runtime_sha(model_app="model-app", pipeline_app="pipeline-app")


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_positive_budget_rejects_nonfinite_or_nonpositive_values(value: float) -> None:
    assert _positive_budget(1.5, name="budget") == 1.5
    with pytest.raises(ValueError, match="budget must be finite and positive"):
        _positive_budget(value, name="budget")


def test_run_modal_pipeline_rejects_mismatched_execution_or_sha_before_download(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    wrong_execution_result = {
        "execution_id": "f" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "/run",
    }
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ),
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=wrong_execution_result,
        ),
        patch("clipper.modal_execution._materialize_remote_run") as download,
        pytest.raises(RuntimeError, match="mismatched execution ID"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts-execution",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )
    download.assert_not_called()

    wrong_sha_result = {
        "execution_id": "e" * 32,
        "deployed_git_sha": "b" * 40,
        "run_path": "/run",
    }
    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ),
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=wrong_sha_result,
        ),
        patch("clipper.modal_execution._materialize_remote_run") as download,
        pytest.raises(RuntimeError, match="mismatched deployed SHA"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts-sha",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )
    download.assert_not_called()


def test_invoke_remote_with_budget_cancels_exact_call_while_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "0.1")

    class Call:
        def __init__(self) -> None:
            self.cancel_args: list[bool] = []

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            if self.cancel_args:
                raise InputCancellation("cancelled")
            assert timeout == 0.1
            clock["now"] = 1.0
            raise TimeoutError

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_args.append(terminate_containers)

    call = Call()
    function = SimpleNamespace(spawn=Mock(return_value=call))
    budget = _BudgetLedger(1.0, 100.0)

    with pytest.raises(
        ProductionBudgetExceeded,
        match="through cancellation acknowledgement",
    ):
        _invoke_remote_with_budget(
            function,
            {"request": True},
            budget=budget,
        )

    function.spawn.assert_called_once_with({"request": True})
    assert call.cancel_args == [False]
    assert budget.gpu_seconds == pytest.approx(2.0)
    assert budget.remaining_budgets()[0] == pytest.approx(0.0)


def test_invoke_remote_with_budget_caps_poll_to_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5")

    class Call:
        def __init__(self) -> None:
            self.timeouts: list[float] = []
            self.cancel_args: list[bool] = []

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            if self.cancel_args:
                raise InputCancellation("cancelled")
            self.timeouts.append(timeout)
            clock["now"] = timeout
            raise TimeoutError

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_args.append(terminate_containers)

    call = Call()
    function = SimpleNamespace(spawn=Mock(return_value=call))

    with pytest.raises(RuntimeError, match="in-flight compute budget"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=1.0,
            max_estimated_usd=100.0,
        )

    assert call.timeouts == [pytest.approx(0.5)]
    assert call.cancel_args == [False]


def test_invoke_remote_with_budget_cancels_if_hydration_fails() -> None:
    call = Mock()
    call.hydrate.side_effect = ServiceError("hydrate failed")
    function = SimpleNamespace(spawn=Mock(return_value=call))

    with pytest.raises(ServiceError, match="hydrate failed"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )

    call.cancel.assert_called_once_with(terminate_containers=False)


def test_invoke_remote_with_budget_cancels_non_timeout_poll_failure() -> None:
    call = Mock()
    call.get.side_effect = [ServiceError("poll failed"), InputCancellation("cancelled")]
    function = SimpleNamespace(spawn=Mock(return_value=call))

    with pytest.raises(ServiceError, match="poll failed"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )

    call.cancel.assert_called_once_with(terminate_containers=False)


def test_invoke_remote_with_budget_rejects_exhausted_shared_budget_before_call_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = Mock()
    budget = _BudgetLedger(1.0, 1.0)
    budget.gpu_seconds = 1.0
    allocate = Mock()
    monkeypatch.setattr("clipper.modal_execution._spawn_recoverable_modal_call", allocate)

    with pytest.raises(
        ProductionBudgetExceeded,
        match="before recoverable Modal call allocation",
    ):
        _invoke_remote_with_budget(
            function,
            {},
            budget=budget,
            gpu_count=1.0,
            estimated_usd_per_second=0.0,
        )

    allocate.assert_not_called()


def test_invoke_remote_with_budget_retries_cancellation_then_preserves_poll_failure() -> None:
    call_handle = Mock()
    call_handle.object_id = "fc-retry"
    call_handle.get.side_effect = [ServiceError("poll failed"), InputCancellation("cancelled")]
    call_handle.cancel.side_effect = [RuntimeError("cancel unavailable"), None]
    function = SimpleNamespace(spawn=Mock(return_value=call_handle))

    with (
        patch("clipper.modal_execution.time.sleep") as sleep,
        pytest.raises(ServiceError, match="poll failed"),
    ):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )

    assert call_handle.cancel.call_count == 2
    sleep.assert_called_once_with(2.0)


def test_invoke_remote_with_budget_charges_delayed_spawn_before_call_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])

    call_handle = Mock()
    call_handle.object_id = "fc-delayed-spawn"
    call_handle.get.side_effect = InputCancellation("cancelled")

    def delayed_spawn(_request: dict[str, object]) -> object:
        clock["now"] = 2.0
        return call_handle

    function = SimpleNamespace(spawn=Mock(side_effect=delayed_spawn))
    budget = _BudgetLedger(1.0, 100.0)

    with pytest.raises(
        ProductionBudgetExceeded,
        match="through cancellation acknowledgement",
    ):
        _invoke_remote_with_budget(
            function,
            {"request": True},
            budget=budget,
            gpu_count=1.0,
            estimated_usd_per_second=0.0,
        )

    call_handle.get.assert_called_once_with(timeout=30.0)
    call_handle.cancel.assert_called_once_with(terminate_containers=False)
    assert budget.gpu_seconds == pytest.approx(2.0)
    assert budget.remaining_budgets()[0] == pytest.approx(0.0)


def test_invoke_remote_with_budget_reconciles_lost_input_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])

    call = Mock()
    call.object_id = "fc-recovered"
    function = Mock()
    budget = _BudgetLedger(100.0, 1.0)

    def recovered_spawn(
        _function: Any,
        _request: dict[str, Any],
        **_kwargs: object,
    ) -> tuple[Any, float, ProductionCallSubmissionFailed]:
        clock["now"] = 0.25
        return (
            call,
            0.0,
            ProductionCallSubmissionFailed(
                "fc-recovered",
                "ServiceError",
                "input acknowledgement lost after acceptance",
            ),
        )

    monkeypatch.setattr(
        "clipper.modal_execution._spawn_recoverable_modal_call",
        recovered_spawn,
    )

    with pytest.raises(ProductionCallSubmissionFailed, match="fc-recovered"):
        _invoke_remote_with_budget(
            function,
            {"request": True},
            budget=budget,
            gpu_count=1.0,
            estimated_usd_per_second=0.01,
        )

    call.cancel.assert_called_once_with(terminate_containers=False)
    assert budget.gpu_seconds == pytest.approx(0.25)
    assert budget.estimated_usd == pytest.approx(0.0025)


def test_invoke_remote_with_budget_charges_until_cancellation_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clipper.modal_execution.time.sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )

    call_handle = Mock()
    call_handle.object_id = "fc-budget-retry"

    def fail_poll(*, timeout: float) -> object:
        assert timeout > 0
        if call_handle.cancel.call_count:
            raise InputCancellation("cancelled")
        clock["now"] = 0.1
        raise ServiceError("poll failed")

    call_handle.get.side_effect = fail_poll
    call_handle.cancel.side_effect = [RuntimeError("cancel unavailable"), None]
    function = SimpleNamespace(spawn=Mock(return_value=call_handle))
    budget = _BudgetLedger(1.0, 100.0)

    with pytest.raises(
        ProductionBudgetExceeded,
        match="through cancellation acknowledgement",
    ):
        _invoke_remote_with_budget(
            function,
            {},
            budget=budget,
            gpu_count=1.0,
            estimated_usd_per_second=0.0,
        )

    assert call_handle.cancel.call_count == 2
    assert budget.gpu_seconds == pytest.approx(2.1)
    assert budget.remaining_budgets()[0] == pytest.approx(0.0)


def test_invoke_remote_with_budget_fails_closed_when_cancellation_is_unconfirmed() -> None:
    call_handle = Mock()
    call_handle.object_id = "fc-stuck"
    call_handle.get.side_effect = ServiceError("poll failed")
    call_handle.cancel.side_effect = RuntimeError("cancel unavailable")
    function = SimpleNamespace(spawn=Mock(return_value=call_handle))

    with (
        patch("clipper.modal_execution.time.sleep") as sleep,
        pytest.raises(ProductionCallNotTerminated, match="fc-stuck") as caught,
    ):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )

    assert caught.value.call_id == "fc-stuck"
    assert call_handle.cancel.call_count == 3
    assert sleep.call_args_list == [mock_call(2.0), mock_call(5.0)]


def test_source_acquisition_does_not_fallback_when_cancellation_is_unconfirmed() -> None:
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    stuck = Mock()
    stuck.object_id = "fc-acquire-stuck"
    stuck.get.side_effect = ServiceError("poll failed")
    stuck.cancel.side_effect = RuntimeError("cancel unavailable")
    function = Mock()
    function.with_options.return_value.spawn.return_value = stuck
    attempts: list[dict[str, object]] = []

    with (
        patch("clipper.modal_execution.time.sleep"),
        pytest.raises(ProductionCallNotTerminated, match="fc-acquire-stuck"),
    ):
        _acquire_remote_source(
            function,
            candidate,
            expected_git_sha="a" * 40,
            budget=_BudgetLedger(100.0, 1.0),
            attempt_evidence=attempts,
        )

    function.with_options.assert_called_once_with(cloud="gcp", timeout=1800)
    assert attempts == [
        {
            "egress": "cloud:gcp",
            "status": "NONTERMINAL_CALL",
            "phase": "invoke",
            "error_type": "ProductionCallNotTerminated",
            "error": attempts[0]["error"],
            "call_id": "fc-acquire-stuck",
            "estimated_usd": attempts[0]["estimated_usd"],
            "gpu_seconds": attempts[0]["gpu_seconds"],
        }
    ]


def test_invoke_remote_with_budget_rechecks_successful_final_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "0.1")

    class Call:
        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            assert timeout == 0.1
            clock["now"] = 1.0
            return {"status": "PASS"}

        def cancel(self, *, terminate_containers: bool) -> None:
            raise AssertionError(f"completed call must not be cancelled: {terminate_containers}")

    function = SimpleNamespace(spawn=Mock(return_value=Call()))
    with pytest.raises(RuntimeError, match="final poll"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=1.0,
            max_estimated_usd=100.0,
        )


def test_invoke_remote_with_budget_rejects_nonfinite_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    function = SimpleNamespace(spawn=Mock())
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "nan")
    with pytest.raises(ValueError, match="finite and positive"):
        _invoke_remote_with_budget(
            function,
            {},
            max_gpu_seconds=10.0,
            max_estimated_usd=1.0,
        )
    function.spawn.assert_not_called()


def test_failed_runner_response_is_authenticated_and_materialized(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()
    failed_local = tmp_path / "artifacts" / "failed-run"
    failed_result = {
        "status": "FAIL",
        "execution_mode": "content-addressed-resume",
        "execution_id": "e" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "/failed-run",
        "run_volume": "clipper-production-artifacts",
    }

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    with (
        patch("clipper.modal_execution.ensure_modal_runtime"),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={"quality_policy": "highest_available_no_transcode", "sha256": "abc"},
        ),
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=failed_result,
        ),
        patch(
            "clipper.modal_execution._materialize_remote_run",
            return_value=failed_local,
        ) as materialize,
    ):
        result = run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id=None,
            render=True,
            fresh_inference=False,
        )

    assert result == failed_local
    materialize.assert_called_once_with(
        artifact_root=tmp_path / "artifacts",
        volume_name="clipper-production-artifacts",
        remote_run_path="/failed-run",
    )


def test_runtime_source_sha_uses_embedded_image_identity_without_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_SOURCE_SHA", "a" * 40)
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    with patch(
        "clipper.modal_execution.subprocess.run",
        side_effect=FileNotFoundError("git missing"),
    ):
        assert _runtime_source_sha() == "a" * 40

    (tmp_path / ".clipper-source-sha").write_text("b" * 40 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="disagrees"):
        _runtime_source_sha()


def _write_reviewed_resume_registry(root: Path, brief_path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "clipper-resume-provenance-v1",
        "workflow_run_id": "123",
        "artifact_run_path": "/prior-run",
        "artifact_origin_workflow_run_id": "122",
        "artifact_origin_head_sha": "a" * 40,
        "campaign_id": "campaign",
        "campaign_brief_sha256": __import__("hashlib").sha256(brief_path.read_bytes()).hexdigest(),
        "target_video_id": "v1",
        "source_hashes": {"v1": "b" * 64},
        "cache_root": "/artifacts/_cache",
    }
    acceptance = root / "acceptance"
    acceptance.mkdir(parents=True, exist_ok=True)
    (acceptance / "resume-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "clipper-resume-provenance-registry-v1",
                "records": {"123": record},
            }
        ),
        encoding="utf-8",
    )
    return record


def test_run_modal_pipeline_forwards_reviewed_resume_provenance(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    repo_root = tmp_path / "repo"
    record = _write_reviewed_resume_registry(repo_root, brief_path)
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    acquire = Mock()
    runner = Mock()
    materialized = tmp_path / "artifacts" / "campaign-run"
    remote_result = {
        "execution_id": "e" * 32,
        "deployed_git_sha": "a" * 40,
        "run_path": "/campaign-run",
        "run_volume": "clipper-production-artifacts",
    }

    def fake_function(_app: str, name: str) -> Mock:
        return acquire if name == "acquire_source" else runner

    with (
        patch("clipper.modal_execution._repo_root", return_value=repo_root),
        patch("clipper.modal_execution.ensure_modal_runtime") as ensure,
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution._verify_deployed_runtime_sha", return_value="a" * 40),
        patch("clipper.modal_execution.uuid.uuid4", return_value=SimpleNamespace(hex="e" * 32)),
        patch("clipper.modal_execution._function", side_effect=fake_function),
        patch(
            "clipper.modal_execution._acquire_remote_source",
            return_value={
                "quality_policy": "highest_available_no_transcode",
                "sha256": "b" * 64,
            },
        ),
        patch(
            "clipper.modal_execution._invoke_remote_with_budget",
            return_value=remote_result,
        ) as invoke,
        patch(
            "clipper.modal_execution._materialize_remote_run",
            return_value=materialized,
        ),
    ):
        result = run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id="123",
            render=True,
            fresh_inference=False,
        )

    assert result == materialized
    ensure.assert_called_once_with()
    payload = invoke.call_args.args[1]
    assert payload["resume_from_run_id"] == "123"
    assert payload["resume_provenance"] == record


def test_run_modal_pipeline_rejects_missing_resume_provenance_before_modal_work(
    tmp_path: Path,
) -> None:
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path)
    repo_root = tmp_path / "repo"
    acceptance = repo_root / "acceptance"
    acceptance.mkdir(parents=True)
    (acceptance / "resume-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "clipper-resume-provenance-registry-v1",
                "records": {},
            }
        ),
        encoding="utf-8",
    )
    candidate = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with (
        patch("clipper.modal_execution._repo_root", return_value=repo_root),
        patch("clipper.modal_execution._explicit_candidates", return_value=[candidate]),
        patch("clipper.modal_execution.ensure_modal_runtime") as ensure,
        patch("clipper.modal_execution._function") as function,
        pytest.raises(RuntimeError, match="no reviewed compatible artifact provenance"),
    ):
        run_modal_pipeline(
            brief_path,
            artifact_root=tmp_path / "artifacts",
            resume_from_run_id="missing",
            render=True,
            fresh_inference=False,
        )
    ensure.assert_not_called()
    function.assert_not_called()


def test_cancel_remote_call_requires_exact_terminal_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "7")
    call = Mock()
    call.object_id = "fc-confirmed"
    call.get.side_effect = InputCancellation("cancelled")
    _cancel_remote_call(call)
    call.cancel.assert_called_once_with(terminate_containers=False)
    call.get.assert_called_once_with(timeout=7.0)


def test_cancel_remote_call_rejects_timeout_and_transport_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "2")
    for error in (TimeoutError(), ServiceError("control plane unavailable")):
        call = Mock()
        call.object_id = "fc-unconfirmed"
        call.get.side_effect = error
        with pytest.raises(ProductionCallNotTerminated, match="remained nonterminal"):
            _cancel_remote_call(call)
        call.cancel.assert_called_once_with(terminate_containers=False)


def test_cancel_remote_call_retries_cancel_request_and_validates_confirmation_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "3")
    call = Mock()
    call.object_id = "fc-retry"
    call.cancel.side_effect = [ServiceError("retry"), None]
    call.get.side_effect = InputCancellation("cancelled")
    with patch("clipper.modal_execution.time.sleep") as sleep:
        _cancel_remote_call(call)
    assert call.cancel.call_count == 2
    sleep.assert_called_once_with(2.0)
    monkeypatch.setenv("CLIPPER_MODAL_CANCEL_CONFIRM_SECONDS", "nan")
    with pytest.raises(ValueError, match="finite and positive"):
        _cancel_confirmation_seconds()


def _valid_resume_registry(tmp_path: Path) -> tuple[Path, dict[str, object], list[VideoCandidate]]:
    brief_path = tmp_path / "brief.yaml"
    brief_path.write_text("campaign: reviewed\n", encoding="utf-8")
    digest = __import__("hashlib").sha256(brief_path.read_bytes()).hexdigest()
    record: dict[str, object] = {
        "schema_version": "clipper-resume-provenance-v1",
        "workflow_run_id": "123",
        "campaign_id": "campaign",
        "campaign_brief_sha256": digest,
        "artifact_run_path": "/run-123",
        "artifact_origin_workflow_run_id": "122",
        "artifact_origin_head_sha": "a" * 40,
        "cache_root": "/artifacts/_cache",
        "source_hashes": {"v1": "B" * 64},
        "target_video_id": "v1",
    }
    registry: dict[str, object] = {
        "schema_version": "clipper-resume-provenance-registry-v1",
        "records": {"123": record},
    }
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    (acceptance / "resume-provenance.json").write_text(json.dumps(registry), encoding="utf-8")
    candidates = [VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")]
    return brief_path, registry, candidates


def _write_resume_registry(tmp_path: Path, registry: dict[str, object]) -> None:
    (tmp_path / "acceptance" / "resume-provenance.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )


def test_reviewed_resume_provenance_normalizes_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path, _registry, candidates = _valid_resume_registry(tmp_path)
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    result = _load_reviewed_resume_provenance(
        requested_run_id="123", brief_path=brief_path, campaign_id="campaign", candidates=candidates
    )
    assert result is not None
    assert result["source_hashes"] == {"v1": "b" * 64}


def test_reviewed_resume_provenance_rejects_missing_or_malformed_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path = tmp_path / "brief.yaml"
    brief_path.write_text("campaign: reviewed\n", encoding="utf-8")
    candidates = [VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")]
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="valid reviewed provenance registry"):
        _load_reviewed_resume_provenance(
            requested_run_id="123",
            brief_path=brief_path,
            campaign_id="campaign",
            candidates=candidates,
        )
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    (acceptance / "resume-provenance.json").write_text("{bad-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="valid reviewed provenance registry"):
        _load_reviewed_resume_provenance(
            requested_run_id="123",
            brief_path=brief_path,
            campaign_id="campaign",
            candidates=candidates,
        )


def test_reviewed_resume_provenance_rejects_identity_and_compatibility_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path, base, candidates = _valid_resume_registry(tmp_path)
    monkeypatch.setattr("clipper.modal_execution._repo_root", lambda: tmp_path)
    cases = [
        (("schema_version",), "bad", "registry schema"),
        (("records",), [], "records must be an object"),
        (("records",), {}, "no reviewed compatible"),
        (("record", "schema_version"), "bad", "record schema"),
        (("record", "workflow_run_id"), "other", "does not match workflow run ID"),
        (("record", "campaign_id"), "other", "campaign does not match"),
        (("record", "campaign_brief_sha256"), "0" * 64, "brief digest"),
        (("record", "artifact_run_path"), "run-123", "artifact path"),
        (("record", "artifact_origin_workflow_run_id"), "", "origin workflow run ID"),
        (("record", "artifact_origin_head_sha"), "bad", "origin SHA"),
        (("record", "cache_root"), "/wrong", "cache root"),
        (("record", "source_hashes"), [], "source hashes must be an object"),
        (("record", "source_hashes"), {"other": "b" * 64}, "source identities"),
        (("record", "source_hashes"), {"v1": "bad"}, "invalid source hash"),
        (("record", "target_video_id"), "other", "target does not match"),
    ]
    for path, value, message in cases:
        registry = __import__("json").loads(__import__("json").dumps(base))
        if path[0] == "record":
            registry["records"]["123"][path[1]] = value
        else:
            registry[path[0]] = value
        _write_resume_registry(tmp_path, registry)
        with pytest.raises(RuntimeError, match=message):
            _load_reviewed_resume_provenance(
                requested_run_id="123",
                brief_path=brief_path,
                campaign_id="campaign",
                candidates=candidates,
            )


def test_cancellation_confirmation_unknown_error_is_not_terminal() -> None:
    assert _cancellation_confirmation_is_terminal(RuntimeError("unknown")) is False


def test_hilp_advisory_budget_opt_out_allows_exhausted_recoverable_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    map_response = SimpleNamespace(
        function_call_id="fc-recoverable",
        pipelined_inputs=[],
    )
    put_response = SimpleNamespace(inputs=[SimpleNamespace(input_id="in-1")])
    modal, async_utils, function_utils, api_pb2, stub, events = _fake_modal_submission_modules(
        map_response=map_response,
        put_response=put_response,
    )

    async def slow_serialize(*_args: object, **_kwargs: object) -> object:
        clock["now"] = 2.0
        return "serialized-input"

    function_utils._create_input.side_effect = slow_serialize
    modules = {
        "modal": modal,
        "modal._utils.async_utils": async_utils,
        "modal._utils.function_utils": function_utils,
        "modal_proto.api_pb2": api_pb2,
    }
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        "clipper.modal_execution.importlib.import_module",
        lambda name: modules[name],
    )
    budget = _BudgetLedger(1.0, 100.0)

    call, _started, error = _real_spawn_recoverable_modal_call(
        object(),
        {"request": True},
        budget=budget,
        gpu_count=1.0,
        estimated_usd_per_second=0.0,
        enforce_budget=False,
    )

    assert call.object_id == "fc-recoverable"
    assert error is None
    assert events == ["from-id", "put-input"]
    stub.FunctionPutInputs.assert_awaited_once()
    assert budget.gpu_seconds > budget.max_gpu_seconds


def test_hilp_advisory_budget_opt_out_does_not_cap_or_cancel_remote_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr("clipper.modal_execution.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5")

    class AdvisoryCall:
        def __init__(self) -> None:
            self.timeouts: list[float] = []
            self.cancel_args: list[bool] = []

        def hydrate(self) -> None:
            return None

        def get(self, *, timeout: float) -> object:
            self.timeouts.append(timeout)
            if len(self.timeouts) == 1:
                clock["now"] = 3.0
                raise TimeoutError
            clock["now"] = 4.0
            return {"ok": True}

        def cancel(self, *, terminate_containers: bool) -> None:
            self.cancel_args.append(terminate_containers)

    call = AdvisoryCall()
    function = SimpleNamespace(spawn=Mock(return_value=call))
    budget = _BudgetLedger(1.0, 1.0)
    budget.gpu_seconds = 1.0

    result = _invoke_remote_with_budget(
        function,
        {"request": True},
        budget=budget,
        gpu_count=1.0,
        estimated_usd_per_second=0.0,
        enforce_budget=False,
    )

    assert result == {"ok": True}
    assert call.timeouts == [pytest.approx(5.0), pytest.approx(5.0)]
    assert call.cancel_args == []
    assert budget.gpu_seconds > budget.max_gpu_seconds


def test_source_acquisition_forwards_advisory_budget_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = VideoCandidate(
        "video",
        "title",
        "channel",
        "channel-title",
        "https://www.youtube.com/watch?v=video",
    )
    function = Mock()
    variant = Mock()
    function.with_options.return_value = variant
    observed: list[bool] = []

    def invoke(_function: object, _payload: object, **kwargs: object) -> object:
        observed.append(bool(kwargs.get("enforce_budget")))
        return {
            "video_id": "video",
            "channel_id": "channel",
            "quality_policy": "highest_available_no_transcode",
            "bytes": 1,
            "sha256": "a" * 64,
            "volume_path": "/inputs/video/master.mp4",
        }

    monkeypatch.setattr("clipper.modal_execution._invoke_remote_with_budget", invoke)
    result = _acquire_remote_source(
        function,
        candidate,
        expected_git_sha="b" * 40,
        budget=_BudgetLedger(1.0, 1.0),
        enforce_budget=False,
    )

    assert result["video_id"] == "video"
    assert observed == [False]

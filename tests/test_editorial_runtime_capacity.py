from __future__ import annotations

import json

import pytest

from clipper.editorial_capacity import capacity_target_input_tokens
from clipper.providers.base import EditorialCapacityError, ModelIdentity
from clipper.providers.local import ProviderUnavailable
from clipper.providers.modal import (
    EditorialInvocation,
    ModalEditorialProvider,
    invoke_editorial_capacity_probe,
)


def _identity() -> ModelIdentity:
    return ModelIdentity(
        "test-model",
        "test-revision",
        "test-quantization",
        "test-engine",
        "test-prompt",
        "test-schema",
    )


def test_runtime_safe_target_participates_in_capacity_repartition() -> None:
    assert (
        capacity_target_input_tokens(
            {
                "reason": "runtime_input_guard",
                "input_tokens": 126_398,
                "context_limit_tokens": 262_144,
                "runtime_safe_input_tokens": 65_536,
            }
        )
        == 65_536
    )


def test_modal_function_timeout_becomes_editorial_capacity_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = ModalEditorialProvider(
        app_name="test-app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity(),
    )

    class FunctionTimeoutError(Exception):
        pass

    exception_namespace = type(
        "_ExceptionNamespace",
        (),
        {"FunctionTimeoutError": FunctionTimeoutError},
    )

    class _ModalModule:
        exception = exception_namespace()

    def timeout(_request):
        raise FunctionTimeoutError("timed out")

    monkeypatch.setattr(provider, "invoke", timeout)
    monkeypatch.setattr(provider, "_modal", lambda: _ModalModule())
    monkeypatch.setenv("CLIPPER_EDITORIAL_EXECUTION_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("CLIPPER_EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS", "65536")
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "execution-1")

    with pytest.raises(EditorialCapacityError) as captured:
        provider.complete_json(
            task="source_hazards:w0-w9",
            payload={"capacity_repartitionable": True},
        )

    details = captured.value.details
    assert details["reason"] == "execution_timeout"
    assert details["remote_error_type"] == "FunctionTimeoutError"
    assert details["runtime_safe_input_tokens"] == 65_536
    assert details["timeout_seconds"] == 900
    assert details["recovery_action"] == "REPARTITION"
    assert provider._instance_handle is None

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    start = next(event for event in events if event.get("event") == "editorial_remote_call_start")
    terminal = next(
        event for event in events if event.get("event") == "editorial_remote_call_terminal"
    )
    timeout_event = next(
        event for event in events if event.get("event") == "editorial_execution_timeout"
    )
    assert start["invocation_id"] == terminal["invocation_id"] == timeout_event["invocation_id"]
    assert terminal["status"] == "TIMEOUT"


def test_modal_provider_propagates_execution_and_expected_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ModalEditorialProvider(
        app_name="test-app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity(),
    )

    class Remote:
        def __init__(self) -> None:
            self.payload = None

        def remote(self, payload):
            self.payload = payload
            return {
                "value": {"segments": []},
                "model": {"model_id": "test-model", "revision": "test-revision"},
                "usage": {},
                "runtime": {},
            }

    remote = Remote()
    monkeypatch.setattr(provider, "_function", lambda: remote)
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-abc")
    monkeypatch.setenv("CLIPPER_ACCEPTANCE_SHA", "a" * 40)

    result = provider.invoke({"task": "source_hazards:x", "payload": {}})

    assert result.value == {"segments": []}
    assert remote.payload["execution_id"] == "exec-abc"
    assert remote.payload["expected_git_sha"] == "a" * 40


def test_modal_editorial_emits_closed_pipeline_call_pair(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = ModalEditorialProvider(
        app_name="test-app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity(),
    )

    class Remote:
        def __init__(self) -> None:
            self.payload = None

        def remote(self, payload):
            self.payload = payload
            return {
                "value": {"segments": []},
                "model": {"model_id": "test-model", "revision": "test-revision"},
                "usage": {},
                "runtime": {
                    "editorial_capacity": {
                        "input_tokens": 12_345,
                        "context_limit_tokens": 262_144,
                        "available_output_tokens": 249_799,
                        "generation_budget_tokens": 512,
                        "runtime_safe_input_tokens": 65_536,
                        "capacity_repartitionable": True,
                    }
                },
            }

    remote = Remote()
    monkeypatch.setattr(provider, "_function", lambda: remote)
    monkeypatch.setenv("CLIPPER_EXECUTION_ID", "exec-barrier")
    monkeypatch.setenv("CLIPPER_ACCEPTANCE_SHA", "b" * 40)

    result = provider.complete_json(task="source_hazards:x", payload={})

    assert result.value == {"segments": []}
    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    starts = [event for event in events if event.get("event") == "editorial_remote_call_start"]
    terminals = [
        event for event in events if event.get("event") == "editorial_remote_call_terminal"
    ]
    assert len(starts) == len(terminals) == 1
    assert starts[0]["execution_id"] == terminals[0]["execution_id"] == "exec-barrier"
    assert starts[0]["invocation_id"] == terminals[0]["invocation_id"]
    assert terminals[0]["status"] == "COMPLETE"
    assert terminals[0]["input_tokens"] == 12_345
    assert terminals[0]["runtime_safe_input_tokens"] == 65_536
    assert terminals[0]["capacity_repartitionable"] is True
    assert remote.payload["editorial_invocation_id"] == starts[0]["invocation_id"]
    assert remote.payload["execution_id"] == "exec-barrier"
    assert remote.payload["expected_git_sha"] == "b" * 40


def test_modal_editorial_emits_error_terminal_when_modal_runtime_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = ModalEditorialProvider(
        app_name="test-app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity(),
    )

    original = RuntimeError("synthetic provider failure")

    def fail(_request):
        raise original

    def unavailable():
        raise ProviderUnavailable("modal unavailable")

    monkeypatch.setattr(provider, "invoke", fail)
    monkeypatch.setattr(provider, "_modal", unavailable)

    with pytest.raises(RuntimeError, match="synthetic provider failure") as captured:
        provider.complete_json(task="semantic_cores:x", payload={})

    assert captured.value is original
    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    terminal = next(
        event for event in events if event.get("event") == "editorial_remote_call_terminal"
    )
    assert terminal["status"] == "ERROR"
    assert terminal["error_type"] == "RuntimeError"
    assert terminal["reason"] == "local_provider_error"


def test_modal_editorial_emits_error_terminal_for_non_timeout_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = ModalEditorialProvider(
        app_name="test-app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity(),
    )

    class FunctionTimeoutError(Exception):
        pass

    exception_namespace = type(
        "_ExceptionNamespace",
        (),
        {"FunctionTimeoutError": FunctionTimeoutError},
    )

    class _ModalModule:
        exception = exception_namespace()

    def fail(_request):
        raise ValueError("synthetic non-timeout failure")

    monkeypatch.setattr(provider, "invoke", fail)
    monkeypatch.setattr(provider, "_modal", lambda: _ModalModule())

    with pytest.raises(ValueError, match="synthetic non-timeout failure"):
        provider.complete_json(task="semantic_cores:x", payload={})

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    terminal = next(
        event for event in events if event.get("event") == "editorial_remote_call_terminal"
    )
    assert terminal["status"] == "ERROR"
    assert terminal["error_type"] == "ValueError"
    assert terminal["reason"] == "provider_error"


def test_capacity_probe_uses_closed_producer_invocation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def remote(request: dict[str, object]) -> dict[str, object]:
        captured.update(request)
        return {
            "value": {
                "status": "CAPACITY_REJECTED",
                "input_tokens": 300_000,
                "context_limit_tokens": 262_144,
                "runtime_safe_input_tokens": 65_536,
                "capacity_repartitionable": True,
                "serialized_request_bytes": 900_000,
            }
        }

    response = invoke_editorial_capacity_probe(
        remote,
        task="source_hazards:acceptance_probe:video",
        payload={"capacity_repartitionable": True},
        execution_id="exec-probe",
        expected_git_sha="c" * 40,
    )

    assert response["value"]["status"] == "CAPACITY_REJECTED"
    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    start = next(event for event in events if event.get("event") == "editorial_remote_call_start")
    terminal = next(
        event for event in events if event.get("event") == "editorial_remote_call_terminal"
    )
    assert start["execution_id"] == terminal["execution_id"] == "exec-probe"
    assert start["invocation_id"] == terminal["invocation_id"]
    assert captured["execution_id"] == "exec-probe"
    assert captured["editorial_invocation_id"] == start["invocation_id"]
    assert captured["expected_git_sha"] == "c" * 40
    assert terminal["status"] == "CAPACITY_REJECTED"
    assert terminal["input_tokens"] == 300_000


def test_capacity_probe_fit_closes_as_complete(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def remote(_request: dict[str, object]) -> dict[str, object]:
        return {
            "value": {
                "status": "FIT",
                "input_tokens": 32_000,
                "context_limit_tokens": 262_144,
                "generation_budget_tokens": 1_024,
                "runtime_safe_input_tokens": 65_536,
                "capacity_repartitionable": True,
            }
        }

    invoke_editorial_capacity_probe(
        remote,
        task="source_hazards:acceptance_probe:fit",
        payload={"capacity_repartitionable": True},
        execution_id="exec-fit",
        expected_git_sha="d" * 40,
    )
    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    terminal = next(
        event for event in events if event.get("event") == "editorial_remote_call_terminal"
    )
    assert terminal["status"] == "COMPLETE"
    assert terminal["generation_budget_tokens"] == 1_024


def test_capacity_probe_remote_timeout_closes_fail_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FunctionTimeoutError(Exception):
        pass

    def remote(_request: dict[str, object]) -> dict[str, object]:
        raise FunctionTimeoutError("synthetic probe timeout")

    with pytest.raises(FunctionTimeoutError, match="synthetic probe timeout"):
        invoke_editorial_capacity_probe(
            remote,
            task="source_hazards:acceptance_probe:timeout",
            payload={"capacity_repartitionable": True},
            execution_id="exec-timeout",
            expected_git_sha="e" * 40,
        )

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    terminal = next(
        event for event in events if event.get("event") == "editorial_remote_call_terminal"
    )
    assert terminal["status"] == "TIMEOUT"
    assert terminal["error_type"] == "FunctionTimeoutError"


def test_editorial_invocation_rejects_second_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    invocation = EditorialInvocation.start(task="semantic_cores:x", execution_id="exec-once")
    invocation.terminal("COMPLETE", details={"input_tokens": 1})
    with pytest.raises(RuntimeError, match="already emitted a terminal event"):
        invocation.terminal("ERROR")

    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert [event["event"] for event in events] == [
        "editorial_remote_call_start",
        "editorial_remote_call_terminal",
    ]

from __future__ import annotations

import pytest

from clipper.editorial_capacity import capacity_target_input_tokens
from clipper.providers.base import EditorialCapacityError, ModelIdentity
from clipper.providers.modal import ModalEditorialProvider


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
) -> None:
    provider = ModalEditorialProvider(
        app_name="test-app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity(),
    )

    class FunctionTimeoutError(Exception):
        pass

    class _ExceptionNamespace:
        FunctionTimeoutError = FunctionTimeoutError

    class _ModalModule:
        exception = _ExceptionNamespace()

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

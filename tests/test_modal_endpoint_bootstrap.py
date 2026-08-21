from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from clipper.providers.base import ModelIdentity
from clipper.providers.modal import ModalEditorialProvider


def _module() -> ModuleType:
    path = Path("scripts/modal_endpoint_bootstrap.py")
    spec = importlib.util.spec_from_file_location("modal_endpoint_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _editorial_provider() -> ModalEditorialProvider:
    return ModalEditorialProvider(
        app_name="clipper-open-editor",
        function_name="editorial",
        identity=ModelIdentity("qwen", "revision", "4bit", "modal"),
    )


def test_proxy_token_parses_modal_1_5_json_schema() -> None:
    module = _module()
    assert module._proxy_token({"Modal-Key": "wk-example", "Modal-Secret": "ws-example"}) == (
        "wk-example",
        "ws-example",
    )


def test_modal_editorial_recovers_from_malformed_json_generation() -> None:
    provider = _editorial_provider()
    function = Mock()
    function.remote.side_effect = [
        {"error": {"type": "JSONDecodeError", "message": "Expecting ',' delimiter"}},
        {"value": {"cores": []}, "usage": {}},
    ]
    payload = {"source": "canonical words"}

    with patch.object(provider, "_function", return_value=function):
        result = provider.complete_json(task="semantic_cores:3", payload=payload)

    assert result.value == {"cores": []}
    assert function.remote.call_count == 2
    first_request, second_request = [call.args[0] for call in function.remote.call_args_list]
    assert first_request == {"task": "semantic_cores:3", "payload": payload}
    assert second_request["task"] == "semantic_cores:3"
    assert second_request["payload"] is payload
    assert second_request["generation_recovery_attempt"] == 2
    assert "strict JSON object" in second_request["generation_recovery_instruction"]


def test_modal_editorial_recovery_is_bounded_and_does_not_retry_runtime_failures() -> None:
    provider = _editorial_provider()
    function = Mock()
    malformed = {"error": {"type": "JSONDecodeError", "message": "invalid JSON"}}
    function.remote.side_effect = [malformed, malformed, malformed]

    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(RuntimeError, match="JSONDecodeError"),
    ):
        provider.complete_json(task="semantic_cores:3", payload={})
    assert function.remote.call_count == 3

    function.reset_mock()
    function.remote.side_effect = None
    function.remote.return_value = {
        "error": {"type": "OutOfMemoryError", "message": "CUDA exhausted"}
    }
    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(RuntimeError, match="OutOfMemoryError"),
    ):
        provider.complete_json(task="semantic_cores:3", payload={})
    assert function.remote.call_count == 1

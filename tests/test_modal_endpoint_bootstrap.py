from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from clipper.providers.base import EditorialCapacityError, ModelIdentity
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


def test_modal_editorial_recovers_when_remote_reports_more_output_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    provider = _editorial_provider()
    function = Mock()
    function.remote.side_effect = [
        {
            "error": {
                "type": "EditorialOutputTruncated",
                "message": "runtime output capacity exhausted",
                "details": {
                    "generation_budget_tokens": 100,
                    "next_output_budget_tokens": 200,
                    "generated_sha256": "first",
                },
            }
        },
        {"value": {"cores": []}, "usage": {}},
    ]
    payload = {"source": "canonical words"}

    with patch.object(provider, "_function", return_value=function):
        result = provider.complete_json(task="semantic_cores:3", payload=payload)

    assert result.value == {"cores": []}
    assert function.remote.call_count == 2
    first_request, second_request = [call.args[0] for call in function.remote.call_args_list]
    assert first_request["task"] == "semantic_cores:3"
    assert first_request["payload"] is payload
    assert first_request["expected_git_sha"] == "a" * 40
    first_invocation = first_request["editorial_invocation_id"]
    second_invocation = second_request["editorial_invocation_id"]
    assert isinstance(first_invocation, str) and len(first_invocation) == 32
    assert isinstance(second_invocation, str) and len(second_invocation) == 32
    assert first_invocation != second_invocation
    assert second_request["task"] == "semantic_cores:3"
    assert second_request["payload"] is payload
    assert second_request["generation_minimum_output_tokens"] == 200
    assert "runtime-derived output capacity" in second_request["generation_recovery_instruction"]


def test_modal_editorial_fails_closed_without_capacity_expansion_and_maps_oom() -> None:
    provider = _editorial_provider()
    function = Mock()
    malformed = {"error": {"type": "JSONDecodeError", "message": "invalid JSON"}}
    function.remote.return_value = malformed

    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(RuntimeError, match="JSONDecodeError"),
    ):
        provider.complete_json(task="semantic_cores:3", payload={})
    assert function.remote.call_count == 1

    function.reset_mock()
    function.remote.return_value = {
        "error": {
            "type": "OutOfMemoryError",
            "message": "CUDA exhausted",
            "details": {"input_tokens": 1234},
        }
    }
    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(EditorialCapacityError, match="CUDA exhausted") as caught,
    ):
        provider.complete_json(task="semantic_cores:3", payload={})
    assert caught.value.details["input_tokens"] == 1234
    assert function.remote.call_count == 1

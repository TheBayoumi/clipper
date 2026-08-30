from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper.providers.base import InferenceUsage, ModelIdentity, ProviderResult
from clipper.providers.modal import (
    ModalEditorialProvider,
    ModalJSONProvider,
    ModalRemoteError,
    ModalVisionProvider,
)


def _identity() -> ModelIdentity:
    return ModelIdentity("model", "rev", "none", "test", "prompt", "schema")


def test_modal_json_provider_constructor_and_function_resolution_guards() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ModalJSONProvider(app_name="app", identity=_identity())
    with pytest.raises(ValueError, match="exactly one"):
        ModalJSONProvider(
            app_name="app",
            identity=_identity(),
            function_name="fn",
            class_name="Cls",
            method_name="call",
        )
    with pytest.raises(ValueError, match="requires method_name"):
        ModalJSONProvider(app_name="app", identity=_identity(), class_name="Cls")

    provider = ModalJSONProvider(
        app_name="app",
        identity=_identity(),
        function_name="fn",
    )
    with pytest.raises(RuntimeError, match="no class instance"):
        provider._class_instance()
    assert provider.warm() == {}

    handle = object()
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=lambda *_args: handle))
    with patch.object(provider, "_modal", return_value=modal):
        assert provider._function() is handle


def test_modal_json_provider_warm_and_invoke_fail_closed_on_invalid_remote_shapes() -> None:
    provider = ModalJSONProvider(
        app_name="app",
        identity=_identity(),
        class_name="Cls",
        method_name="call",
    )
    with patch.object(provider, "_class_instance", return_value=object()):
        assert provider.warm() == {}

    ready = Mock()
    ready.remote.return_value = []
    with (
        patch.object(provider, "_class_instance", return_value=SimpleNamespace(ready=ready)),
        pytest.raises(ValueError, match="warmup returned an invalid response"),
    ):
        provider.warm()

    function = Mock()
    function.remote.return_value = []
    direct = ModalJSONProvider(app_name="app", identity=_identity(), function_name="fn")
    with (
        patch.object(direct, "_function", return_value=function),
        pytest.raises(ValueError, match="invalid response"),
    ):
        direct.invoke({})

    function.remote.return_value = {"value": []}
    with (
        patch.object(direct, "_function", return_value=function),
        pytest.raises(ValueError, match="invalid response"),
    ):
        direct.invoke({})


def test_modal_editorial_stops_on_repeated_capacity_recovery_signature() -> None:
    provider = ModalEditorialProvider(
        app_name="app",
        identity=_identity(),
        function_name="editorial",
    )
    error = {
        "error": {
            "type": "EditorialOutputTruncated",
            "message": "truncated",
            "details": {
                "generation_budget_tokens": 100,
                "next_output_budget_tokens": 200,
                "generated_sha256": "same",
            },
        }
    }
    function = Mock()
    function.remote.side_effect = [error, error]
    with (
        patch.object(provider, "_function", return_value=function),
        pytest.raises(ModalRemoteError, match="truncated"),
    ):
        provider.complete_json(task="semantic_cores:range", payload={})
    assert function.remote.call_count == 2


def test_modal_vision_provider_encodes_frames_and_forwards_context(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame-bytes")
    provider = ModalVisionProvider(
        app_name="app",
        identity=_identity(),
        function_name="vision",
    )
    result = ProviderResult(
        {"observations": []},
        _identity(),
        InferenceUsage("modal", "now", 0.0),
    )
    with patch.object(provider, "invoke", return_value=result) as invoked:
        assert (
            provider.inspect(task="source_policy_visual_scout", frames=[frame], context={"x": 1})
            is result
        )
    request = invoked.call_args.args[0]
    assert request["task"] == "source_policy_visual_scout"
    assert request["context"] == {"x": 1}
    assert request["frames_base64"]

from __future__ import annotations

import ast
import json
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _tree() -> ast.Module:
    return ast.parse(Path("scripts/modal_open_models.py").read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )


def test_transport_error_emits_editorial_invocation_id() -> None:
    tree = _tree()
    function = _function(tree, "_transport_error")

    keyword_only = {argument.arg for argument in function.args.kwonlyargs}
    assert "execution_id" in keyword_only
    assert "invocation_id" in keyword_only

    event_dicts = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "event"
            and isinstance(value, ast.Constant)
            and value.value == "application_result"
            for key, value in zip(node.keys, node.values, strict=True)
        )
    ]
    assert len(event_dicts) == 1
    event = event_dicts[0]
    mapping = {
        key.value: value
        for key, value in zip(event.keys, event.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert isinstance(mapping["execution_id"], ast.Name)
    assert mapping["execution_id"].id == "execution_id"
    assert isinstance(mapping["invocation_id"], ast.Name)
    assert mapping["invocation_id"].id == "invocation_id"


def test_editorial_model_forwards_invocation_id_to_transport_errors() -> None:
    tree = _tree()
    editorial = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "EditorialModel"
    )

    for method_name in ("capacity_probe", "complete"):
        method = next(
            item
            for item in editorial.body
            if isinstance(item, ast.FunctionDef) and item.name == method_name
        )
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_transport_error"
        ]
        assert len(calls) == 1
        keywords = {item.arg: item.value for item in calls[0].keywords}
        assert "invocation_id" in keywords
        expression = ast.unparse(keywords["invocation_id"])
        assert "editorial_invocation_id" in expression


def _load_transport_error() -> Callable[..., dict[str, Any]]:
    tree = _tree()
    function = _function(tree, "_transport_error")
    isolated = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {
        "Any": Any,
        "json": json,
        "traceback": traceback,
    }
    exec(compile(isolated, "scripts/modal_open_models.py", "exec"), namespace)  # noqa: S102
    loaded = namespace["_transport_error"]
    assert callable(loaded)
    return loaded


class EditorialOutputTruncated(Exception):
    def __init__(self) -> None:
        super().__init__("synthetic truncated output")
        self.details = {
            "generation_budget_tokens": 100,
            "next_output_budget_tokens": 200,
        }


class EditorialCapacityError(Exception):
    def __init__(self) -> None:
        super().__init__("synthetic capacity rejection")
        self.details = {"reason": "runtime_input_guard", "input_tokens": 70_000}


def test_transport_error_preserves_invocation_id_in_event_and_returned_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport_error = _load_transport_error()

    result = transport_error(
        EditorialOutputTruncated(),
        context="task=semantic_cores:x",
        execution_id="exec-1",
        invocation_id="invocation-1",
    )

    event = json.loads(capsys.readouterr().out.strip())
    assert event["event"] == "application_result"
    assert event["execution_id"] == "exec-1"
    assert event["invocation_id"] == "invocation-1"
    assert event["error_type"] == "EditorialOutputTruncated"
    assert event["application_status"] == "FAILED"

    details = result["error"]["details"]
    assert details["editorial_invocation_id"] == "invocation-1"
    assert details["generation_budget_tokens"] == 100
    assert details["next_output_budget_tokens"] == 200


def test_capacity_error_preserves_same_invocation_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport_error = _load_transport_error()

    result = transport_error(
        EditorialCapacityError(),
        execution_id="exec-2",
        invocation_id="invocation-2",
    )

    event = json.loads(capsys.readouterr().out.strip())
    assert event["invocation_id"] == "invocation-2"
    assert event["application_status"] == "CAPACITY_REJECTED"
    assert event["recovery_action"] == "REPARTITION"
    assert result["error"]["details"]["editorial_invocation_id"] == "invocation-2"


def test_capacity_probe_diagnostic_carries_editorial_invocation_id() -> None:
    tree = _tree()
    function = _function(tree, "_editorial_capacity_probe")
    event_dicts = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "event"
            and isinstance(value, ast.Constant)
            and value.value == "editorial_capacity_probe"
            for key, value in zip(node.keys, node.values, strict=True)
        )
    ]
    assert len(event_dicts) == 2
    for event in event_dicts:
        mapping = {
            key.value: value
            for key, value in zip(event.keys, event.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert isinstance(mapping["invocation_id"], ast.Name)
        assert mapping["invocation_id"].id == "invocation_id"

    rendered = ast.unparse(function)
    assert "payload.get('editorial_invocation_id')" in rendered



def _load_editorial_output_template() -> Callable[[dict[str, Any]], dict[str, Any]]:
    tree = _tree()
    names = {
        "_editorial_payload",
        "_editorial_discourse_units",
        "_editorial_output_template",
    }
    functions = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    isolated = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(isolated, "scripts/modal_open_models.py", "exec"), namespace)  # noqa: S102
    loaded = namespace["_editorial_output_template"]
    assert callable(loaded)
    return loaded


def test_source_hazard_structural_floor_is_bounded_and_does_not_echo_source_prose() -> None:
    output_template = _load_editorial_output_template()
    marker = "VERY_LONG_SOURCE_PROSE_SHOULD_NOT_BE_IN_OUTPUT_FLOOR"
    words = [
        {
            "word_ref": f"w{index:04d}",
            "text": f"{marker}-{index}.",
            "speaker_id": "speaker",
        }
        for index in range(100)
    ]

    template = output_template(
        {
            "task": "source_hazards:test",
            "payload": {"words": words},
        }
    )

    segments = template["segments"]
    assert len(segments) == 64
    assert all(segment["evidence"] == ["source evidence"] for segment in segments)
    assert marker not in json.dumps(template)


def test_editorial_generation_has_internal_deadline_before_modal_timeout() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")

    assert 'CLIPPER_EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS", "32768"' in source
    assert 'CLIPPER_EDITORIAL_GENERATION_DEADLINE_SECONDS", "300"' in source
    assert '"max_time": EDITORIAL_GENERATION_DEADLINE_SECONDS' in source
    assert '"reason": "generation_runtime_deadline"' in source
    assert '"event": "editorial_generation_deadline"' in source

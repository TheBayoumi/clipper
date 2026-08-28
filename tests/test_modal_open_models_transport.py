from __future__ import annotations

import ast
import json
import traceback
from pathlib import Path
from typing import Any


def _transport_error_function():
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_transport_error"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "json": json,
        "traceback": traceback,
    }
    exec(compile(module, "scripts/modal_open_models.py", "exec"), namespace)
    return namespace["_transport_error"]


def test_transport_error_preserves_editorial_invocation_id(
    capsys,
) -> None:
    transport_error = _transport_error_function()

    class EditorialOutputTruncated(Exception):
        details = {"generation_budget_tokens": 100}

    result = transport_error(
        EditorialOutputTruncated("truncated"),
        context="task=semantic_cores:x",
        execution_id="exec-123",
        invocation_id="inv-456",
    )

    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event["event"] == "application_result"
    assert event["execution_id"] == "exec-123"
    assert event["invocation_id"] == "inv-456"
    assert result["error"]["type"] == "EditorialOutputTruncated"


def test_editorial_model_forwards_invocation_id_to_transport_errors() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
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

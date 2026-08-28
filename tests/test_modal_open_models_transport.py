from __future__ import annotations

import ast
from pathlib import Path


def _tree() -> ast.Module:
    return ast.parse(Path("scripts/modal_open_models.py").read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
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

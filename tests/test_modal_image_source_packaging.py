from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path("scripts/modal_open_models.py")


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _assignment(tree: ast.Module, name: str) -> ast.Assign:
    for item in tree.body:
        if not isinstance(item, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in item.targets):
            return item
    raise AssertionError(f"missing assignment: {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    raise AssertionError(f"missing function: {name}")


def _function_image(function: ast.FunctionDef) -> ast.AST:
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if not isinstance(decorator.func, ast.Attribute) or decorator.func.attr != "function":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "image":
                return keyword.value
    raise AssertionError(f"{function.name} has no @app.function image")


def _adds_clipper_source(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "add_local_python_source"
        and any(isinstance(arg, ast.Constant) and arg.value == "clipper" for arg in item.args)
        for item in ast.walk(node)
    )


def test_all_custom_modal_images_package_clipper_source() -> None:
    tree = _tree()

    for image_name in ("media_image", "state_image", "text_image", "speech_image"):
        assert _adds_clipper_source(_assignment(tree, image_name).value), image_name

    for function_name in ("deployment_identity", "credential_smoke"):
        image = _function_image(_function(tree, function_name))
        assert isinstance(image, ast.Name), function_name
        assert image.id == "state_image", function_name

    assert _adds_clipper_source(_function_image(_function(tree, "hf_access_smoke")))

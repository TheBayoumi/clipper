"""retry_feedback forwards a HybridStop message to the model, so no message may carry model text.

Every HybridStop and _stop call in the package must take a string literal, a name bound to a
string literal in the same module or function, or the message of another HybridStop.
"""

from __future__ import annotations

import ast
from pathlib import Path

import clipper

_ALLOWED_WRAPPERS = {"str(first)", "str(error)"}


def _constant_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def test_hybrid_stop_messages_are_never_built_from_runtime_values() -> None:
    root = Path(clipper.__file__).parent
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        constants = _constant_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name not in {"HybridStop", "_stop"} or not node.args:
                continue
            arg = node.args[0]
            text = ast.unparse(arg)
            ok = (
                (isinstance(arg, ast.Constant) and isinstance(arg.value, str))
                or (isinstance(arg, ast.Name) and (arg.id in constants or arg.id == "message"))
                or text in _ALLOWED_WRAPPERS
            )
            if not ok:
                offenders.append(f"{path.name}:{node.lineno}: {text}")
    assert offenders == []

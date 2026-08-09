from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path("scripts/modal_endpoint_bootstrap.py")
    spec = importlib.util.spec_from_file_location("modal_endpoint_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_token_parses_modal_1_5_json_schema() -> None:
    module = _module()
    assert module._proxy_token({"Modal-Key": "wk-example", "Modal-Secret": "ws-example"}) == (
        "wk-example",
        "ws-example",
    )

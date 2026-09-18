from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_IMPL: ModuleType = importlib.import_module("clipper_mw4.mw4_v3_1_source_qa")


def __getattr__(name: str) -> Any:
    return getattr(_IMPL, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_IMPL)))


def _main() -> None:
    entrypoint = getattr(_IMPL, "main", None)
    if not callable(entrypoint):
        raise RuntimeError("installed MW4 compatibility target has no main()")
    entrypoint()


if __name__ == "__main__":
    _main()

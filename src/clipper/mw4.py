from __future__ import annotations

import argparse
import importlib
from typing import Protocol, cast


class _MW4Runner(Protocol):
    def __call__(self, args: argparse.Namespace) -> int: ...


def _load_runner() -> _MW4Runner:
    module = importlib.import_module("clipper_mw4.orchestrator")
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise RuntimeError("installed MW4 runtime does not expose a callable orchestrator")
    return cast(_MW4Runner, runner)


def run(args: argparse.Namespace) -> int:
    return _load_runner()(args)

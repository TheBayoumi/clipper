"""Modal is an optional bounded compute backend, never a pipeline authority."""

from __future__ import annotations

import importlib
from typing import Any

from .vision import VisionSettings


class ModalVisionProvider:
    def __init__(self, settings: VisionSettings, *, worker: Any | None = None) -> None:
        self.settings = settings
        self.worker = worker

    def inspect(self, request: dict[str, Any]) -> dict[str, Any]:
        worker = self.worker
        if worker is None:
            try:
                modal = importlib.import_module("modal")
            except ImportError as exc:
                raise RuntimeError(
                    "Modal visual inference requires the clipper[modal] extra"
                ) from exc
            worker = modal.Cls.from_name(self.settings.app_name, "VisionModel")()
        call = worker.inspect.spawn(request)
        try:
            result: Any = call.get(timeout=self.settings.deadline_seconds)
        except TimeoutError as exc:
            # A timed-out model invocation may still be running and incurring GPU charges.
            # Never retry until the producer has provably reached its terminal state.
            try:
                call.cancel()
                call.get(timeout=30)
            except TimeoutError as error:
                raise RuntimeError(
                    "Modal vision cancellation terminal state unconfirmed"
                ) from error
            except Exception as terminal_error:
                if type(terminal_error).__name__ not in {
                    "InputCancellation",
                    "FunctionTimeoutError",
                    "RemoteError",
                    "ExecutionError",
                    "InternalFailure",
                }:
                    raise RuntimeError(
                        "Modal vision cancellation terminal state unconfirmed"
                    ) from terminal_error
            raise RuntimeError("Modal vision call exceeded its bounded inference deadline") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Modal vision worker returned a non-object result")
        return result

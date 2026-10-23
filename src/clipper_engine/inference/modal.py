"""Modal is an optional bounded compute backend, never a pipeline authority."""

from __future__ import annotations

import importlib
import sys
from typing import Any

from .vision import VISION_MODEL_ID, VisionSettings


def bounded_modal_readiness(
    worker: Any, *, stage: str, deadline_seconds: float = 900.0
) -> dict[str, Any]:
    """Bound model cold-start, preserving strict remote cancellation evidence."""
    if not 0 < deadline_seconds <= 900:
        raise ValueError("Modal readiness deadline must be between 0 and 900 seconds")
    print(
        f"MODAL_READINESS_STARTED stage={stage} deadline_seconds={deadline_seconds}",
        flush=True,
    )
    call = worker.ready.spawn()
    try:
        ready = call.get(timeout=deadline_seconds)
    except TimeoutError as exc:
        try:
            call.cancel(terminate_containers=True)
            call.get(timeout=30)
        except TimeoutError as terminal_error:
            raise RuntimeError(
                f"Modal {stage} readiness cancellation terminal state unconfirmed"
            ) from terminal_error
        except Exception as terminal_error:
            if type(terminal_error).__name__ not in {
                "InputCancellation",
                "FunctionTimeoutError",
                "RemoteError",
                "ExecutionError",
                "InternalFailure",
            }:
                raise RuntimeError(
                    f"Modal {stage} readiness cancellation terminal state unconfirmed"
                ) from terminal_error
        raise RuntimeError(f"Modal {stage} readiness exceeded {deadline_seconds:g}s") from exc
    if not isinstance(ready, dict):
        raise RuntimeError(f"Modal {stage} readiness returned non-object evidence")
    print(
        f"MODAL_READINESS_COMPLETED stage={stage} "
        f"cuda={ready.get('cuda_available')} gpu_count={ready.get('gpu_count')}",
        flush=True,
    )
    return ready


class ModalVisionProvider:
    def __init__(self, settings: VisionSettings, *, worker: Any | None = None) -> None:
        self.settings = settings
        self.worker = worker

    def __enter__(self) -> ModalVisionProvider:
        """Keep one ephemeral exact-SHA GPU worker alive for the full source.

        CI remains validation-only: no persistent Modal app is deployed, and
        another gameplay profile cannot take over an existing shared worker.
        """
        if self.worker is not None:
            return self
        worker_module = importlib.import_module("clipper_engine.inference.modal_worker")
        if (
            self.settings.deployed_sha != worker_module.DEPLOYED_GIT_SHA
            or self.settings.app_name != worker_module.APP_NAME
        ):
            raise RuntimeError("ephemeral Modal visual worker SHA or app identity mismatch")
        self._app_scope = worker_module.app.run()
        self._app_scope.__enter__()
        try:
            self.worker = worker_module.VisionModel()
            ready = bounded_modal_readiness(
                self.worker,
                stage="production_source",
                deadline_seconds=min(900.0, self.settings.deadline_seconds),
            )
            if (
                ready.get("deployed_git_sha") != self.settings.deployed_sha
                or ready.get("model_id") != VISION_MODEL_ID
                or ready.get("cuda_available") is not True
                or ready.get("gpu_count", 0) < 2
            ):
                raise RuntimeError("Modal visual source worker model/CUDA/SHA readiness failed")
        except BaseException:
            self._app_scope.__exit__(*sys.exc_info())
            self.worker = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if hasattr(self, "_app_scope"):
            try:
                self._app_scope.__exit__(exc_type, exc, traceback)
            finally:
                self.worker = None

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
                # Ask Modal to terminate the GPU container as well as its call.
                # A timeout must never leave a billable worker running silently.
                call.cancel(terminate_containers=True)
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

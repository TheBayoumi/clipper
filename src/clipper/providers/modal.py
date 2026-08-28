from __future__ import annotations

import base64
import importlib
import json
import os
from pathlib import Path
from typing import Any

from .base import EditorialCapacityError, InferenceUsage, ModelIdentity, ProviderResult
from .local import ProviderUnavailable


class ModalRemoteError(RuntimeError):
    """Structured error returned by a remote Modal inference function."""

    def __init__(
        self,
        *,
        function_name: str,
        error_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.function_name = function_name
        self.error_type = error_type
        self.remote_message = message
        self.details = dict(details or {})
        super().__init__(f"Modal {function_name} failed: {error_type}: {message}")


class ModalJSONProvider:
    def __init__(
        self,
        *,
        app_name: str,
        identity: ModelIdentity,
        function_name: str | None = None,
        class_name: str | None = None,
        method_name: str | None = None,
        class_parameters: dict[str, Any] | None = None,
    ) -> None:
        if (function_name is None) == (class_name is None):
            raise ValueError("Modal provider requires exactly one function_name or class_name")
        if class_name is not None and not method_name:
            raise ValueError("class-backed Modal provider requires method_name")
        self.app_name = app_name
        self.class_name = class_name
        self.method_name = method_name
        self.class_parameters = dict(class_parameters or {})
        self.function_name = function_name or f"{class_name}.{method_name}"
        self.identity = identity
        self._instance_handle: Any | None = None

    def _resolved_identity(self, response: dict[str, Any]) -> ModelIdentity:
        raw = response.get("model")
        if not isinstance(raw, dict):
            return self.identity
        return ModelIdentity(
            str(raw.get("model_id") or self.identity.model_id),
            str(raw.get("revision") or self.identity.revision),
            self.identity.quantization,
            self.identity.inference_engine,
            self.identity.prompt_version,
            self.identity.schema_version,
        )

    def _modal(self) -> Any:
        try:
            return importlib.import_module("modal")
        except ImportError as exc:
            raise ProviderUnavailable("install clipper[modal]") from exc

    def _class_instance(self) -> Any:
        if self.class_name is None:
            raise RuntimeError("function-backed provider has no class instance")
        if self._instance_handle is None:
            cls = self._modal().Cls.from_name(self.app_name, self.class_name)
            self._instance_handle = cls(**self.class_parameters)
        return self._instance_handle

    def _function(self) -> Any:
        if self.class_name is None:
            return self._modal().Function.from_name(self.app_name, self.function_name)
        return getattr(self._class_instance(), str(self.method_name))

    def warm(self) -> dict[str, Any]:
        if self.class_name is None:
            return {}
        ready = getattr(self._class_instance(), "ready", None)
        if ready is None:
            return {}
        response = ready.remote()
        if not isinstance(response, dict):
            raise ValueError("Modal class warmup returned an invalid response")
        self._raise_remote_error(response)
        self.identity = self._resolved_identity(response)
        raw_runtime = response.get("runtime")
        return dict(raw_runtime) if isinstance(raw_runtime, dict) else {}

    def _raise_remote_error(self, response: dict[str, Any]) -> None:
        raw_error = response.get("error")
        if not isinstance(raw_error, dict):
            return
        error_type = str(raw_error.get("type") or "RemoteError")
        message = str(raw_error.get("message") or "remote inference failed")
        raw_details = raw_error.get("details")
        details = dict(raw_details) if isinstance(raw_details, dict) else {}
        raise ModalRemoteError(
            function_name=self.function_name,
            error_type=error_type,
            message=message,
            details=details,
        )

    def invoke(self, payload: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
        request = dict(payload)
        execution_id = os.getenv("CLIPPER_EXECUTION_ID", "").strip()
        expected_git_sha = (
            os.getenv("CLIPPER_ACCEPTANCE_SHA", "").strip() or os.getenv("GITHUB_SHA", "").strip()
        )
        if execution_id:
            request.setdefault("execution_id", execution_id)
        if expected_git_sha:
            request.setdefault("expected_git_sha", expected_git_sha.lower())
        response = self._function().remote(request)
        if not isinstance(response, dict):
            raise ValueError("Modal provider returned an invalid response")
        self._raise_remote_error(response)
        if not isinstance(response.get("value"), dict):
            raise ValueError("Modal provider returned an invalid response")
        raw_usage = response.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        raw_runtime = response.get("runtime")
        runtime: dict[str, Any] = dict(raw_runtime) if isinstance(raw_runtime, dict) else {}
        peak_by_device = usage.get("peak_vram_mb_by_device")
        if isinstance(peak_by_device, dict):
            runtime["peak_vram_mb_by_device"] = dict(peak_by_device)
        return ProviderResult(
            response["value"],
            self._resolved_identity(response),
            InferenceUsage(
                provider="modal",
                started_at=str(usage.get("started_at") or "unknown"),
                duration_seconds=float(usage.get("duration_seconds") or 0.0),
                gpu_type=str(usage["gpu_type"]) if usage.get("gpu_type") else None,
                gpu_seconds=float(usage.get("gpu_seconds") or 0.0),
                peak_vram_mb=float(usage["peak_vram_mb"])
                if usage.get("peak_vram_mb") is not None
                else None,
                input_units=int(usage.get("input_units") or 0),
                output_units=int(usage.get("output_units") or 0),
                estimated_cost_usd=float(usage.get("estimated_cost_usd") or 0.0),
                runtime=runtime,
            ),
        )


class ModalEditorialProvider(ModalJSONProvider):
    @staticmethod
    def _is_output_contract_error(exc: ModalRemoteError) -> bool:
        if exc.error_type in {"JSONDecodeError", "EditorialOutputTruncated"}:
            return True
        return (
            exc.error_type == "ValueError"
            and "model output must be a JSON object" in exc.remote_message
        )

    @staticmethod
    def _is_capacity_error(exc: ModalRemoteError) -> bool:
        return exc.error_type in {"OutOfMemoryError", "EditorialCapacityError"}

    def _invoke_with_timeout_capacity(
        self,
        request: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        try:
            return self.invoke(request)
        except Exception as exc:
            try:
                modal_module = self._modal()
            except ProviderUnavailable:
                raise exc from None
            exception_namespace = getattr(modal_module, "exception", None)
            timeout_type = getattr(exception_namespace, "FunctionTimeoutError", None)
            if not isinstance(timeout_type, type) or not isinstance(exc, timeout_type):
                raise
            self._instance_handle = None
            timeout_seconds = int(os.getenv("CLIPPER_EDITORIAL_EXECUTION_TIMEOUT_SECONDS", "900"))
            runtime_safe_input_tokens = int(
                os.getenv("CLIPPER_EDITORIAL_RUNTIME_SAFE_INPUT_TOKENS", "65536")
            )
            execution_id = os.getenv("CLIPPER_EXECUTION_ID", "").strip()
            event = {
                "event": "editorial_execution_timeout",
                "task": str(request.get("task") or ""),
                "execution_id": execution_id,
                "timeout_seconds": timeout_seconds,
                "recovery_action": "REPARTITION",
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            raise EditorialCapacityError(
                f"Modal {self.function_name} exceeded its execution timeout",
                details={
                    "reason": "execution_timeout",
                    "function_name": self.function_name,
                    "remote_error_type": type(exc).__name__,
                    "runtime_safe_input_tokens": runtime_safe_input_tokens,
                    "timeout_seconds": timeout_seconds,
                    "recovery_action": "REPARTITION",
                },
            ) from exc

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        request: dict[str, Any] = {"task": task, "payload": payload}
        seen_recovery_signatures: set[tuple[object, ...]] = set()
        while True:
            try:
                return self._invoke_with_timeout_capacity(request)
            except ModalRemoteError as exc:
                if self._is_capacity_error(exc):
                    raise EditorialCapacityError(
                        str(exc),
                        details={
                            "function_name": exc.function_name,
                            "remote_error_type": exc.error_type,
                            **exc.details,
                        },
                    ) from exc
                if not self._is_output_contract_error(exc):
                    raise

                next_budget = exc.details.get("next_output_budget_tokens")
                current_budget = exc.details.get("generation_budget_tokens")
                signature = (
                    exc.error_type,
                    exc.details.get("generated_sha256"),
                    current_budget,
                    next_budget,
                )
                if signature in seen_recovery_signatures:
                    raise
                seen_recovery_signatures.add(signature)
                if (
                    not isinstance(next_budget, int)
                    or isinstance(next_budget, bool)
                    or next_budget <= 0
                    or (
                        isinstance(current_budget, int)
                        and not isinstance(current_budget, bool)
                        and next_budget <= current_budget
                    )
                ):
                    raise
                request = {
                    "task": task,
                    "payload": payload,
                    "generation_minimum_output_tokens": next_budget,
                    "generation_recovery_instruction": (
                        "Regenerate the complete strict JSON object from the original source "
                        "evidence using the expanded runtime-derived output capacity. Do not add "
                        "Markdown or prose outside the object."
                    ),
                }


class ModalVisionProvider(ModalJSONProvider):
    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        encoded_frames = [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in frames]
        return self.invoke({"task": task, "frames_base64": encoded_frames, "context": context})

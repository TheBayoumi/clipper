from __future__ import annotations

import base64
import importlib
from pathlib import Path
from typing import Any

from .base import InferenceUsage, ModelIdentity, ProviderResult
from .local import ProviderUnavailable


class ModalRemoteError(RuntimeError):
    """Structured error returned by a remote Modal inference function."""

    def __init__(self, *, function_name: str, error_type: str, message: str) -> None:
        self.function_name = function_name
        self.error_type = error_type
        self.remote_message = message
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
        raise ModalRemoteError(
            function_name=self.function_name,
            error_type=error_type,
            message=message,
        )

    def invoke(self, payload: dict[str, Any]) -> ProviderResult[dict[str, Any]]:
        response = self._function().remote(payload)
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
    _MAX_OUTPUT_RECOVERY_ATTEMPTS = 3

    @staticmethod
    def _is_output_contract_error(exc: ModalRemoteError) -> bool:
        if exc.error_type in {"JSONDecodeError", "EditorialOutputTruncated"}:
            return True
        return (
            exc.error_type == "ValueError"
            and "model output must be a JSON object" in exc.remote_message
        )

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        request: dict[str, Any] = {"task": task, "payload": payload}
        for attempt in range(1, self._MAX_OUTPUT_RECOVERY_ATTEMPTS + 1):
            try:
                result = self.invoke(request)
            except ModalRemoteError as exc:
                if (
                    not self._is_output_contract_error(exc)
                    or attempt >= self._MAX_OUTPUT_RECOVERY_ATTEMPTS
                ):
                    raise
                request = {
                    "task": task,
                    "payload": payload,
                    "generation_recovery_attempt": attempt + 1,
                    "generation_recovery_instruction": (
                        "The previous generation violated the JSON output contract. Regenerate the "
                        "complete answer from the original task as exactly one strict JSON object. "
                        "Use valid JSON syntax with double-quoted keys and strings, no Markdown, "
                        "no comments, no prose outside the object, and no truncated fields."
                    ),
                }
                continue

            return result
        raise AssertionError("unreachable editorial recovery loop")


class ModalVisionProvider(ModalJSONProvider):
    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        encoded_frames = [base64.b64encode(frame.read_bytes()).decode("ascii") for frame in frames]
        return self.invoke({"task": task, "frames_base64": encoded_frames, "context": context})

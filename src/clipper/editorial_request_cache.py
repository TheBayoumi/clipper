"""Exact-request persistence for the Clipper editorial reviewer.

The package owns replay integrity, checkpoints and raw traces. Model-specific
completions and their stage fingerprints are injected by the caller so moving
this cache does not silently invalidate compatible saved model responses.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast


class EditorialRequestCache:
    """Resume exact review calls, including valid negative evidence, without weights."""

    def __init__(
        self,
        path: Path,
        factory: Callable[[], Any],
        identity: dict[str, Any],
        *,
        draft_completion: Callable[[Any, str, dict[str, Any], int], str],
        json_implementation: Callable[..., Any],
        stage_fingerprint: Callable[..., str],
        reuse_path: Path | None = None,
        replay_only: bool = False,
        recorded_runtime: str | None = None,
        sampling_parameters: dict[str, Any] | None = None,
    ) -> None:
        if (recorded_runtime is not None) != replay_only:
            raise ValueError("a recorded runtime is permitted only for replay without inference")
        try:
            runtime = version("llama-cpp-python")
        except PackageNotFoundError:
            runtime = "not-installed"
        self.path = path
        self.factory = factory
        self.replay_only = replay_only
        self.draft_completion = draft_completion
        self.json_implementation = json_implementation
        self.stage_fingerprint = stage_fingerprint
        self.identity = {
            **identity,
            "runtime": recorded_runtime if replay_only else runtime,
            "seed": 0,
            "temperature": 0,
            **(sampling_parameters or {}),
        }
        self.records: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {"cache_hits": 0, "model_calls": 0, "model_seconds": 0.0}
        source = reuse_path if reuse_path and reuse_path.is_file() else path
        if source.is_file():
            saved = json.loads(source.read_text())
            if saved.get("identity") == self.identity and isinstance(saved.get("records"), dict):
                self.records = saved["records"]

    def _invoke(
        self, method: str, prompt: str, payload: dict[str, Any], properties: Any, tokens: int
    ) -> Any:
        request = dict(
            method=method, prompt=prompt, payload=payload, properties=properties, tokens=tokens
        )
        implementation = self.stage_fingerprint(
            self.draft_completion if method == "draft" else self.json_implementation
        )
        serialized = json.dumps(
            {"identity": self.identity, "implementation": implementation, "request": request},
            sort_keys=True,
        )
        key = hashlib.sha256(serialized.encode()).hexdigest()
        saved = self.records.get(key, {})
        cached_response = json.dumps(saved.get("response"), sort_keys=True)
        if (
            saved.get("request") == request
            and saved.get("response_sha256") != hashlib.sha256(cached_response.encode()).hexdigest()
        ):
            # Recover a model-owned response if a prior consumer enriched its object.
            for raw in saved.get("raw_calls", []):
                try:
                    choice = raw["response"]["choices"][0]
                    if choice["finish_reason"] != "stop":
                        continue
                    content = choice["message"]["content"]
                    recovered = content if method == "draft" else json.loads(content)
                    serialized_response = json.dumps(recovered, sort_keys=True)
                    if isinstance(recovered, str if method == "draft" else dict) and hashlib.sha256(
                        serialized_response.encode()
                    ).hexdigest() == saved.get("response_sha256"):
                        saved["response"] = recovered
                        cached_response = serialized_response
                        self.metrics["recovered_responses"] = (
                            self.metrics.get("recovered_responses", 0) + 1
                        )
                        break
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
        hit = (
            saved.get("request") == request
            and saved.get("response_sha256") == hashlib.sha256(cached_response.encode()).hexdigest()
            and isinstance(saved.get("response"), str if method == "draft" else dict)
        )
        began = time.monotonic()
        if hit:
            self.metrics["cache_hits"] += 1
            response = json.loads(cached_response)
        else:
            if self.replay_only:
                raise RuntimeError(
                    "recorded-response replay has a cache miss; inference is forbidden"
                )
            editor = self.factory()
            raw_calls: list[dict[str, Any]] = []
            model = getattr(editor, "model", None)
            original = getattr(model, "create_chat_completion", None)

            if original is not None:

                def traced(**kwargs: Any) -> Any:
                    result = original(**kwargs)
                    raw_calls.append({"request": kwargs, "response": result})
                    return result

                cast(Any, model).create_chat_completion = traced
            self.metrics["model_calls"] += 1
            try:
                response = (
                    self.draft_completion(editor, prompt, payload, tokens)
                    if method == "draft"
                    else editor._review_completion(prompt, payload, properties, tokens)
                )
            except Exception as error:
                self.calls.append(
                    {
                        "request": request,
                        "error": f"{type(error).__name__}: {error}",
                        "cache_hit": False,
                        "raw_calls": raw_calls,
                        "seconds": round(time.monotonic() - began, 3),
                    }
                )
                raise
            finally:
                self.metrics["model_seconds"] += time.monotonic() - began
                if original is not None:
                    cast(Any, model).create_chat_completion = original
            self.records[key] = {
                "request": request,
                "response": response,
                "raw_calls": raw_calls,
                "response_sha256": hashlib.sha256(
                    json.dumps(response, sort_keys=True).encode()
                ).hexdigest(),
            }
        self.calls.append(
            {
                "request": request,
                "response": response,
                "cache_hit": hit,
                "raw_calls": self.records[key].get("raw_calls", []),
                "seconds": round(time.monotonic() - began, 3),
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".partial")
        partial.write_text(
            json.dumps({"identity": self.identity, "records": self.records}, indent=2) + "\n"
        )
        partial.replace(self.path)
        return json.loads(json.dumps(response))

    def _review_completion(
        self, prompt: str, payload: dict[str, Any], properties: dict[str, Any], tokens: int
    ) -> dict[str, Any]:
        return cast(dict[str, Any], self._invoke("json", prompt, payload, properties, tokens))

    def semantic_draft(self, prompt: str, payload: dict[str, Any], tokens: int) -> str:
        return cast(str, self._invoke("draft", prompt, payload, None, tokens))

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .base import InferenceUsage, ModelIdentity, ProviderResult
from .editorial_prompt import editorial_contract, editorial_output_budget
from .local import ProviderUnavailable


class ModalEndpointEditorialProvider:
    def __init__(
        self,
        *,
        endpoint_url: str,
        proxy_token_id: str,
        proxy_token_secret: str,
        identity: ModelIdentity,
        timeout_seconds: float = 300.0,
    ) -> None:
        if not endpoint_url:
            raise ProviderUnavailable("CLIPPER_MODAL_EDITORIAL_ENDPOINT_URL is required")
        if urlparse(endpoint_url).scheme != "https":
            raise ProviderUnavailable("Modal editorial endpoint must use https")
        if not proxy_token_id or not proxy_token_secret:
            raise ProviderUnavailable("Modal endpoint proxy token is required")
        self.endpoint_url = endpoint_url.rstrip("/")
        self.proxy_token_id = proxy_token_id
        self.proxy_token_secret = proxy_token_secret
        self.identity = identity
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        value = text.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines).strip()
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("managed editorial endpoint must return a JSON object")
        return parsed

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]:
        started = time.perf_counter()
        started_at = datetime.now(UTC).isoformat()
        body = {
            "model": self.identity.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a source-grounded podcast editor. "
                        "Never invent spoken words or IDs. " + editorial_contract(task)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"task": task, "payload": payload}, ensure_ascii=False),
                },
            ],
            "temperature": 0,
            "max_tokens": editorial_output_budget({"task": task}),
        }
        request = Request(  # noqa: S310
            f"{self.endpoint_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.proxy_token_id}.{self.proxy_token_secret}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        raw: Any = None
        for attempt in range(6):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 502, 503, 504} and attempt < 5:
                    time.sleep(min(30.0, 2.0**attempt))
                    continue
                raise RuntimeError(f"Modal editorial endpoint HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                if attempt < 5:
                    time.sleep(min(30.0, 2.0**attempt))
                    continue
                raise RuntimeError(
                    f"Modal editorial endpoint request failed: {exc.reason}"
                ) from exc
        if not isinstance(raw, dict):
            raise ValueError("managed editorial endpoint returned an invalid response")
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("managed editorial endpoint returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("managed editorial endpoint returned no message content")
        usage_raw = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
        return ProviderResult(
            self._json_object(message["content"]),
            self.identity,
            InferenceUsage(
                provider="modal-endpoint",
                started_at=started_at,
                duration_seconds=max(0.0, time.perf_counter() - started),
                input_units=int(usage.get("prompt_tokens") or 0),
                output_units=int(usage.get("completion_tokens") or 0),
            ),
        )

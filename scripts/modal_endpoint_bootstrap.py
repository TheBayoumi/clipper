from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import time
from collections.abc import Iterable
from typing import Any

from clipper.providers.editorial_model_contract import (
    EDITORIAL_MODEL_ID,
    EDITORIAL_MODEL_REVISION,
)


def _run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}"
            + (f"\n{detail}" if detail else "")
        )
    return result.stdout.strip()


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _find_endpoint(value: Any, name: str) -> tuple[str, str] | None:
    for item in _walk(value):
        item_name = str(item.get("name") or item.get("endpoint_name") or "")
        if item_name != name:
            continue
        url = str(item.get("url") or item.get("endpoint_url") or item.get("web_url") or "")
        status = str(item.get("status") or item.get("state") or "").lower()
        if url.startswith("http"):
            return url.rstrip("/"), status
    return None


def _assert_endpoint_identity(payload: Any, name: str) -> None:
    """Refuse to trust an endpoint without verifiable pinned model metadata."""
    for item in _walk(payload):
        if str(item.get("name") or item.get("endpoint_name") or "") != name:
            continue
        raw_model = item.get("model_id") or item.get("model")
        model_id = raw_model
        revision = item.get("model_revision") or item.get("revision")
        if isinstance(raw_model, dict):
            model_id = raw_model.get("model_id") or raw_model.get("id")
            revision = revision or raw_model.get("revision")
        if model_id != EDITORIAL_MODEL_ID or revision != EDITORIAL_MODEL_REVISION:
            raise RuntimeError(
                f"endpoint {name} model identity is missing or differs from pinned editorial "
                "contract; refuse to reuse an unverifiable endpoint"
            )
        return
    raise RuntimeError(f"endpoint {name} identity metadata is missing")


def _proxy_token(value: Any) -> tuple[str, str]:
    if isinstance(value, dict):
        modal_key = str(value.get("Modal-Key") or "")
        modal_secret = str(value.get("Modal-Secret") or "")
        if modal_key.startswith("wk-") and modal_secret.startswith("ws-"):
            return modal_key, modal_secret

    token_id = ""
    token_secret = ""
    for item in _walk(value):
        for key, raw in item.items():
            text = str(raw)
            lowered = str(key).lower()
            if text.startswith("wk-") and ("id" in lowered or "token" in lowered):
                token_id = text
            elif text.startswith("ws-") and ("secret" in lowered or "token" in lowered):
                token_secret = text
    if not token_id or not token_secret:
        raise RuntimeError("Modal proxy token JSON did not contain wk-/ws- credentials")
    return token_id, token_secret


def ensure_endpoint(name: str, model: str, *, timeout: float = 900.0) -> str:
    if model != EDITORIAL_MODEL_ID:
        raise ValueError("endpoint model disagrees with pinned editorial contract")

    def listing() -> Any:
        return json.loads(_run("modal", "endpoint", "list", "--json") or "[]")

    endpoints = listing()
    found = _find_endpoint(endpoints, name)
    if found is None:
        _run("modal", "endpoint", "create", "--name", name, "--model", model)
    else:
        _assert_endpoint_identity(endpoints, name)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        endpoints = listing()
        found = _find_endpoint(endpoints, name)
        if found is not None:
            _assert_endpoint_identity(endpoints, name)
            url, status = found
            if status not in {"failed", "error", "stopped"}:
                return url
        time.sleep(5)
    raise TimeoutError(f"Modal endpoint {name} did not become discoverable within {timeout}s")


def create_proxy_token() -> tuple[str, str]:
    raw = json.loads(_run("modal", "workspace", "proxy-tokens", "create", "--json"))
    return _proxy_token(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="clipper-editorial")
    parser.add_argument("--model", default=EDITORIAL_MODEL_ID)
    args = parser.parse_args()
    if args.model != EDITORIAL_MODEL_ID:
        parser.error("--model must match the pinned editorial model contract")
    token_id, token_secret = create_proxy_token()
    try:
        endpoint_url = ensure_endpoint(args.name, args.model)
    except Exception:
        with contextlib.suppress(Exception):
            _run("modal", "workspace", "proxy-tokens", "delete", "-y", token_id)
        raise
    print(
        json.dumps(
            {
                "endpoint_url": endpoint_url,
                "model_id": args.model,
                "proxy_token_id": token_id,
                "proxy_token_secret": token_secret,
            }
        )
    )


if __name__ == "__main__":
    main()

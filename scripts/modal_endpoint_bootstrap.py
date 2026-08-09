from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import time
from collections.abc import Iterable
from typing import Any


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


def _proxy_token(value: Any) -> tuple[str, str]:
    token_id = ""
    token_secret = ""

    def inspect(raw: Any) -> None:
        nonlocal token_id, token_secret
        if isinstance(raw, dict):
            for child in raw.values():
                inspect(child)
            return
        if isinstance(raw, list):
            for child in raw:
                inspect(child)
            return
        text = str(raw)
        id_match = re.search(r"(?:^|[=:\\s\"'])((?:wk)-[^\s\"',}]+)", text)
        secret_match = re.search(r"(?:^|[=:\\s\"'])((?:ws)-[^\s\"',}]+)", text)
        if id_match:
            token_id = id_match.group(1)
        if secret_match:
            token_secret = secret_match.group(1)

    inspect(value)
    if not token_id or not token_secret:
        raise RuntimeError("Modal proxy token JSON did not contain wk-/ws- credentials")
    return token_id, token_secret


def ensure_endpoint(name: str, model: str, *, timeout: float = 900.0) -> str:
    def listing() -> Any:
        return json.loads(_run("modal", "endpoint", "list", "--json") or "[]")

    found = _find_endpoint(listing(), name)
    if found is None:
        _run("modal", "endpoint", "create", "--name", name, "--model", model)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = _find_endpoint(listing(), name)
        if found is not None:
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
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    args = parser.parse_args()
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

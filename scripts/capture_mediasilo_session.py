from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

SENSITIVE = {"authorization", "cookie", "set-cookie", "x-api-key"}


def safe_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("<redacted>" if k.lower() in SENSITIVE else v) for k, v in headers.items()}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "item"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--folder", default="Gameplay Edit")
    parser.add_argument("--output-dir", type=Path, default=Path("mediasilo-session"))
    args = parser.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme != "https" or not parsed.netloc.endswith("mediasilo.com"):
        parser.error("expected a MediaSilo HTTPS URL")

    from playwright.sync_api import sync_playwright

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    captured: list[dict[str, object]] = []
    bodies: dict[str, object] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        def on_response(response: object) -> None:
            url = str(getattr(response, "url", ""))
            if "api.mediasilo.com" not in url or "/v3/quicklinks/" not in url:
                return
            request = getattr(response, "request")
            entry = {
                "url": url,
                "status": int(getattr(response, "status", 0)),
                "request_headers": safe_headers(dict(getattr(request, "headers", {}))),
                "response_headers": safe_headers(dict(getattr(response, "headers", {}))),
            }
            captured.append(entry)
            try:
                content_type = str(entry["response_headers"].get("content-type", ""))
                if "json" in content_type.lower():
                    bodies[url] = getattr(response, "json")()
            except Exception as exc:
                bodies[url] = {"capture_error": repr(exc)}

        page.on("response", on_response)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)

        folder = page.locator(f'[aria-label="{args.folder}"]')
        folder.wait_for(state="visible", timeout=40_000)
        page.wait_for_timeout(1_500)
        page.screenshot(path=str(out / "root-loaded.png"), full_page=True)
        folder.click(force=True)

        page.wait_for_timeout(8_000)
        page.screenshot(path=str(out / f"{safe_name(args.folder)}.png"), full_page=True)
        (out / f"{safe_name(args.folder)}.html").write_text(page.content(), encoding="utf-8")
        (out / f"{safe_name(args.folder)}.txt").write_text(
            page.locator("body").inner_text(), encoding="utf-8"
        )

        # Exercise the QuickLink Download control after the target folder is loaded.
        download_control = page.locator('[aria-label="Download"]')
        if download_control.count():
            try:
                download_control.first.click(force=True, timeout=3_000)
                page.wait_for_timeout(3_000)
                page.screenshot(path=str(out / "download-dialog.png"), full_page=True)
                (out / "download-dialog.txt").write_text(
                    page.locator("body").inner_text(), encoding="utf-8"
                )
            except Exception as exc:
                (out / "download-error.txt").write_text(repr(exc), encoding="utf-8")

        page.wait_for_timeout(2_000)
        browser.close()

    (out / "requests.json").write_text(json.dumps(captured, indent=2), encoding="utf-8")
    (out / "bodies.json").write_text(json.dumps(bodies, indent=2), encoding="utf-8")
    print(json.dumps({"responses": len(captured), "json_bodies": len(bodies)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

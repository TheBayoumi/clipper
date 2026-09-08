from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

SENSITIVE = {"authorization", "cookie", "set-cookie", "x-api-key"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)


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
    state: dict[str, object] = {"folder_visible": False, "folder_clicked": False}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=USER_AGENT,
            locale="en-US",
        )
        page = context.new_page()

        def on_response(response: object) -> None:
            url = str(getattr(response, "url", ""))
            if "api.mediasilo.com" not in url or "/v3/quicklinks/" not in url:
                return
            request = getattr(response, "request")
            try:
                request_headers = dict(request.all_headers())
            except Exception:
                request_headers = dict(getattr(request, "headers", {}))
            try:
                response_headers = dict(getattr(response, "all_headers")())
            except Exception:
                response_headers = dict(getattr(response, "headers", {}))
            entry = {
                "url": url,
                "status": int(getattr(response, "status", 0)),
                "request_headers": safe_headers(request_headers),
                "response_headers": safe_headers(response_headers),
            }
            captured.append(entry)
            try:
                content_type = str(response_headers.get("content-type", ""))
                if "json" in content_type.lower():
                    bodies[url] = getattr(response, "json")()
            except Exception as exc:
                bodies[url] = {"capture_error": repr(exc)}

        page.on("response", on_response)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)

        # Match the successful public QuickLink acquisition path: allow the
        # application to hydrate before touching its React controls.
        page.wait_for_timeout(12_000)
        page.screenshot(path=str(out / "root-after-12s.png"), full_page=True)
        (out / "root-after-12s.html").write_text(page.content(), encoding="utf-8")
        (out / "root-after-12s.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")

        folder = page.locator(f'[aria-label="{args.folder}"]')
        if folder.count() and folder.first.is_visible():
            state["folder_visible"] = True
            try:
                folder.first.click(force=True, timeout=3_000)
                state["folder_clicked"] = True
                page.wait_for_timeout(8_000)
                page.screenshot(path=str(out / f"{safe_name(args.folder)}.png"), full_page=True)
                (out / f"{safe_name(args.folder)}.html").write_text(page.content(), encoding="utf-8")
                (out / f"{safe_name(args.folder)}.txt").write_text(
                    page.locator("body").inner_text(), encoding="utf-8"
                )

                download_control = page.locator('[aria-label="Download"]')
                if download_control.count():
                    try:
                        download_control.first.click(force=True, timeout=3_000)
                        page.wait_for_timeout(4_000)
                        page.screenshot(path=str(out / "download-dialog.png"), full_page=True)
                        (out / "download-dialog.txt").write_text(
                            page.locator("body").inner_text(), encoding="utf-8"
                        )
                    except Exception as exc:
                        (out / "download-error.txt").write_text(repr(exc), encoding="utf-8")
            except Exception as exc:
                state["folder_click_error"] = repr(exc)

        page.wait_for_timeout(2_000)
        browser.close()

    (out / "requests.json").write_text(json.dumps(captured, indent=2), encoding="utf-8")
    (out / "bodies.json").write_text(json.dumps(bodies, indent=2), encoding="utf-8")
    (out / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"responses": len(captured), "json_bodies": len(bodies), **state}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

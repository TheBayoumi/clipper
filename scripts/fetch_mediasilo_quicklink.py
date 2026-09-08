from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


MIN_FILE_BYTES = 10_000


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "download"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download an authorized public MediaSilo QuickLink.")
    parser.add_argument("url")
    parser.add_argument("--output-dir", type=Path, default=Path("mediasilo-output"))
    args = parser.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme != "https" or not parsed.netloc.endswith("mediasilo.com"):
        parser.error("url must be an HTTPS mediasilo.com URL")

    from playwright.sync_api import sync_playwright

    output_dir = args.output_dir.resolve()
    downloads_dir = output_dir / "downloads"
    diagnostics_dir = output_dir / "diagnostics"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    saved: list[dict[str, object]] = []
    seen_names: dict[str, int] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 900})
        page = context.new_page()

        current_scope = "root"

        def on_download(download: object) -> None:
            suggested = Path(str(getattr(download, "suggested_filename", "download.bin"))).name
            stem = _safe_name(Path(suggested).stem)
            suffix = Path(suggested).suffix
            base = f"{_safe_name(current_scope)}__{stem}{suffix}"
            index = seen_names.get(base, 0)
            seen_names[base] = index + 1
            filename = base if index == 0 else f"{Path(base).stem}_{index}{Path(base).suffix}"
            destination = downloads_dir / filename
            getattr(download, "save_as")(str(destination))
            saved.append(
                {
                    "scope": current_scope,
                    "file": destination.name,
                    "bytes": destination.stat().st_size,
                }
            )

        page.on("download", on_download)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(7_000)

        for label in ("Dismiss", "Accept", "Accept All", "I Agree", "Got it"):
            locator = page.get_by_text(re.compile(f"^{re.escape(label)}$", re.I))
            if locator.count():
                try:
                    locator.first.click(timeout=2_000)
                except Exception:
                    pass

        root_html = page.content()
        (diagnostics_dir / "root.html").write_text(root_html, encoding="utf-8")
        page.screenshot(path=str(diagnostics_dir / "root.png"), full_page=True)

        folder_names: list[str] = []
        folder_labels = page.locator("#FOLDER_TILE_CONTAINER [aria-label]")
        for index in range(folder_labels.count()):
            label = folder_labels.nth(index).get_attribute("aria-label")
            if label and label not in folder_names:
                folder_names.append(label)

        def trigger_download(scope: str) -> None:
            nonlocal current_scope
            current_scope = scope
            before = len(saved)
            action = page.locator('[aria-label="Download"]')
            if not action.count():
                return
            try:
                action.first.click(force=True, timeout=3_000)
            except Exception:
                return
            page.wait_for_timeout(1_000)
            (diagnostics_dir / f"{_safe_name(scope)}_after_download.txt").write_text(
                page.locator("body").inner_text(), encoding="utf-8"
            )
            page.screenshot(
                path=str(diagnostics_dir / f"{_safe_name(scope)}_after_download.png"),
                full_page=True,
            )

            option_patterns = (
                r"download originals?",
                r"originals?",
                r"download all",
                r"download",
            )
            for pattern in option_patterns:
                candidates = page.get_by_text(re.compile(f"^{pattern}$", re.I))
                for idx in range(candidates.count()):
                    candidate = candidates.nth(idx)
                    try:
                        if not candidate.is_visible():
                            continue
                        candidate.click(force=True, timeout=2_000)
                        page.wait_for_timeout(12_000)
                        if len(saved) > before:
                            return
                    except Exception:
                        continue
            page.wait_for_timeout(8_000)

        trigger_download("root")

        for folder_name in folder_names:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(5_000)
            tile = page.locator("#FOLDER_TILE_CONTAINER").filter(
                has=page.locator(f'[aria-label="{folder_name}"]')
            )
            if not tile.count():
                continue
            try:
                tile.first.click(force=True, timeout=3_000)
                page.wait_for_timeout(6_000)
            except Exception:
                continue
            scope = _safe_name(folder_name)
            (diagnostics_dir / f"{scope}.html").write_text(page.content(), encoding="utf-8")
            page.screenshot(path=str(diagnostics_dir / f"{scope}.png"), full_page=True)
            trigger_download(scope)

        browser.close()

    valid = [item for item in saved if int(item["bytes"]) >= MIN_FILE_BYTES]
    result = {
        "source_review_url": args.url,
        "folders_discovered": folder_names,
        "downloads": valid,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    if not valid:
        print("No downloadable campaign files were captured.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

OFFICIAL_FILES = {
    "r1": "MW4_BetaTopPlays_Stringout_16X9_R1.mp4",
    "batch2": "BestOfBeta_Stringout_Batch2.mp4",
    "week2": "BestOfBeta_Week2_Stringout_1_8.28.mp4",
}
FOLDER_ASSET_PATH = "/folders/3b199368-5974-4eb1-a450-327a4e84c260/assets"


def _select_original_source(assets: list[dict[str, Any]], source_key: str) -> dict[str, Any]:
    wanted = OFFICIAL_FILES[source_key]
    for asset in assets:
        if asset.get("title") != wanted:
            continue
        derivatives = list(asset.get("derivatives") or [])
        candidates: list[tuple[int, dict[str, Any], str]] = []
        for item in derivatives:
            if str(item.get("type") or "").lower() != "source":
                continue
            url = item.get("url") or (item.get("properties") or {}).get("url")
            if not url:
                continue
            try:
                file_size = int(item.get("fileSize") or 0)
            except (TypeError, ValueError):
                file_size = 0
            candidates.append((file_size, item, str(url)))
        if not candidates:
            available = sorted({str(item.get("type")) for item in derivatives})
            raise RuntimeError(
                f"Original MediaSilo source master unavailable for {wanted}; "
                f"proxy fallback is forbidden. available derivative types={available}"
            )
        _, derivative, url = max(candidates, key=lambda item: item[0])
        if str(derivative.get("type") or "").lower() != "source":
            raise RuntimeError("internal invariant failure: selected derivative is not type=source")
        return {
            "title": asset.get("title"),
            "file_name": asset.get("fileName"),
            "url": url,
            "derivative_type": "source",
            "file_size": derivative.get("fileSize"),
            "width": derivative.get("width"),
            "height": derivative.get("height"),
            "duration_ms": derivative.get("duration"),
        }
    raise RuntimeError(f"official source not found: {wanted}")


def resolve(source_key: str, review_url: str, output: Path) -> dict[str, Any]:
    if source_key not in OFFICIAL_FILES:
        raise RuntimeError(f"unknown official source key: {source_key}")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required only for MediaSilo source resolution") from exc

    holder: list[list[dict[str, Any]] | None] = [None]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            for attempt in range(1, 4):
                page = browser.new_page(
                    viewport={"width": 1440, "height": 1000},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/139.0.0.0 Safari/537.36"
                    ),
                )

                def handle(response: Any) -> None:
                    if FOLDER_ASSET_PATH not in response.url or response.status != 200:
                        return
                    try:
                        data = response.json()
                    except Exception:
                        return
                    if isinstance(data, list) and data:
                        holder[0] = data

                page.on("response", handle)
                try:
                    page.goto(review_url, wait_until="domcontentloaded", timeout=90000)
                    page.wait_for_timeout(30000)
                finally:
                    page.close()
                if holder[0]:
                    print(f"MediaSilo assets resolved on attempt {attempt}")
                    break
        finally:
            browser.close()

    if not holder[0]:
        raise RuntimeError("MediaSilo assets were not resolved after 3 analysis-stage attempts")
    selected = _select_original_source(holder[0], source_key)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"official MediaSilo source selected: {selected['title']} type=source")
    return selected


def download(resolved_path: Path, target: Path) -> None:
    selected = json.loads(resolved_path.read_text(encoding="utf-8"))
    if str(selected.get("derivative_type") or "").lower() != "source":
        raise RuntimeError("refusing non-source MediaSilo derivative before download")
    url = str(selected.get("url") or "")
    if not url:
        raise RuntimeError("resolved MediaSilo source URL is missing")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=300) as source, target.open("wb") as output:
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
    if target.stat().st_size <= 0:
        raise RuntimeError(f"downloaded MediaSilo source is empty: {target}")
    print(f"downloaded original source bytes={target.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    resolve_parser = sub.add_parser("resolve")
    resolve_parser.add_argument("--source-key", required=True)
    resolve_parser.add_argument("--review-url", required=True)
    resolve_parser.add_argument("--output", type=Path, required=True)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("--resolved", type=Path, required=True)
    download_parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "resolve":
        resolve(args.source_key, args.review_url, args.output)
    else:
        download(args.resolved, args.target)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

OFFICIAL_FILES = {
    "r1": "MW4_BetaTopPlays_Stringout_16X9_R1.mp4",
    "batch2": "BestOfBeta_Stringout_Batch2.mp4",
    "week2": "BestOfBeta_Week2_Stringout_1_8.28.mp4",
    "bestofbeta_week2_stringout_2_8_31": "BestOfBeta_Week2_Stringout_2.8.31.mp4",
}


def _review_identifiers(review_url: str) -> tuple[str, str | None]:
    path_parts = [part for part in urllib.parse.urlparse(review_url).path.split("/") if part]
    try:
        review_index = path_parts.index("review")
        review_id = path_parts[review_index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"invalid MediaSilo review URL: {review_url}") from exc

    folder_id: str | None = None
    if len(path_parts) > review_index + 3 and path_parts[review_index + 2] == "f":
        folder_id = path_parts[review_index + 3]
    return review_id, folder_id


def _asset_response_matches(
    url: str,
    review_id: str,
    folder_id: str | None,
) -> bool:
    path = urllib.parse.urlparse(url).path.rstrip("/")
    expected = [f"/quicklinks/{review_id}/assets"]
    if folder_id:
        expected.append(f"/folders/{folder_id}/assets")
    return any(path.endswith(candidate) for candidate in expected)


def _asset_list(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        items = [dict(item) for item in payload if isinstance(item, dict)]
        if items and any(
            (item.get("title") or item.get("fileName"))
            and isinstance(item.get("derivatives"), list)
            for item in items
        ):
            return items
        return None

    if isinstance(payload, dict):
        for key in ("assets", "items", "results", "data"):
            assets = _asset_list(payload.get(key))
            if assets:
                return assets
    return None


def capture_assets(review_url: str) -> list[dict[str, Any]]:
    """Capture source-catalog assets from the public MediaSilo review session.

    MediaSilo review pages have used both folder-assets and QuickLink-assets REST
    routes. Both are provider-native representations of the same review asset set.
    The response is accepted only when it contains real asset objects with derivative
    metadata; downstream selection still requires an original type=source derivative.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for MediaSilo source discovery") from exc

    review_id, folder_id = _review_identifiers(review_url)
    holder: list[list[dict[str, Any]] | None] = [None]
    resolved_route: list[str | None] = [None]

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
                    if response.status != 200 or not _asset_response_matches(
                        response.url,
                        review_id,
                        folder_id,
                    ):
                        return
                    try:
                        assets = _asset_list(response.json())
                    except Exception:
                        return
                    if assets:
                        holder[0] = assets
                        resolved_route[0] = urllib.parse.urlparse(response.url).path

                page.on("response", handle)
                try:
                    page.goto(review_url, wait_until="domcontentloaded", timeout=90000)
                    page.wait_for_timeout(30000)
                finally:
                    page.close()
                if holder[0]:
                    print(
                        f"MediaSilo assets resolved on attempt {attempt} "
                        f"via {resolved_route[0]}"
                    )
                    break
        finally:
            browser.close()

    if not holder[0]:
        expected_routes = [f"/quicklinks/{review_id}/assets"]
        if folder_id:
            expected_routes.append(f"/folders/{folder_id}/assets")
        raise RuntimeError(
            "MediaSilo assets were not resolved after 3 attempts; "
            f"expected provider asset routes={expected_routes}"
        )
    return holder[0]


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
    selected = _select_original_source(capture_assets(review_url), source_key)
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
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme.lower() != "https" or not parsed_url.hostname:
        raise RuntimeError("refusing non-HTTPS MediaSilo source URL")
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(  # noqa: S310 - HTTPS validated above.
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with (
        urllib.request.urlopen(  # noqa: S310 - HTTPS validated above.
            request, timeout=300
        ) as source,
        target.open("wb") as output,
    ):
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
    if target.stat().st_size <= 0:
        raise RuntimeError(f"downloaded MediaSilo source is empty: {target}")
    print(f"downloaded original source bytes={target.stat().st_size}")


def self_test() -> None:
    review_url = (
        "https://app.mediasilo.com/review/"
        "6a88a6c15a183a21eeeae9e6/f/3b199368-5974-4eb1-a450-327a4e84c260"
    )
    review_id, folder_id = _review_identifiers(review_url)
    if review_id != "6a88a6c15a183a21eeeae9e6":
        raise AssertionError("MediaSilo review identifier parsing failed")
    if folder_id != "3b199368-5974-4eb1-a450-327a4e84c260":
        raise AssertionError("MediaSilo folder identifier parsing failed")

    quicklink_url = f"https://api.mediasilo.com/v3/quicklinks/{review_id}/assets"
    folder_url = f"https://api.mediasilo.com/v3/folders/{folder_id}/assets?_page=0"
    if not _asset_response_matches(quicklink_url, review_id, folder_id):
        raise AssertionError("MediaSilo QuickLink asset route is not recognized")
    if not _asset_response_matches(folder_url, review_id, folder_id):
        raise AssertionError("MediaSilo folder asset route is not recognized")

    asset = {
        "title": OFFICIAL_FILES["r1"],
        "fileName": OFFICIAL_FILES["r1"],
        "derivatives": [{"type": "source", "url": "https://example.invalid/source.mp4"}],
    }
    if _asset_list([asset]) != [asset]:
        raise AssertionError("MediaSilo list asset payload was not recognized")
    if _asset_list({"data": {"assets": [asset]}}) != [asset]:
        raise AssertionError("MediaSilo wrapped asset payload was not recognized")
    print("MediaSilo canonical review asset resolver self-test: PASS")


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

from __future__ import annotations

import argparse
import json
import os
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
        expected.extend(
            [
                f"/quicklinks/{review_id}/folders/{folder_id}/assets",
                f"/folders/{folder_id}/assets",
            ]
        )
    return any(path.endswith(candidate) for candidate in expected)


def _is_asset_payload(item: dict[str, Any]) -> bool:
    return bool(
        (item.get("title") or item.get("fileName")) and isinstance(item.get("derivatives"), list)
    )


def _asset_list(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        items = [dict(item) for item in payload if isinstance(item, dict)]
        assets = [item for item in items if _is_asset_payload(item)]
        if assets:
            return assets
        for item in items:
            nested = _asset_list(item)
            if nested:
                return nested
        return None

    if isinstance(payload, dict):
        if _is_asset_payload(payload):
            return [dict(payload)]
        preferred_keys = ("assets", "items", "results", "data", "content", "children")
        for key in preferred_keys:
            nested = payload.get(key)
            assets = _asset_list(nested)
            if assets:
                return assets
        for key, nested in payload.items():
            if key in preferred_keys:
                continue
            if isinstance(nested, (dict, list)):
                assets = _asset_list(nested)
                if assets:
                    return assets
    return None


def _is_provider_url(url: str) -> bool:
    hostname = (urllib.parse.urlparse(url).hostname or "").lower()
    return (
        hostname == "mediasilo.com"
        or hostname.endswith(".mediasilo.com")
        or hostname == "shift.io"
        or hostname.endswith(".shift.io")
    )


def _asset_identity(asset: dict[str, Any]) -> str:
    for key in ("id", "uuid", "assetId", "assetUuid"):
        value = asset.get(key)
        if value:
            return f"{key}:{value}"
    return "title:" + str(asset.get("title") or asset.get("fileName") or "")


def _sanitized_route(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return f"{parsed.hostname or 'unknown'}{parsed.path}"


def _forward_request_headers(request: Any) -> dict[str, str]:
    """Return only ordinary HTTP field headers safe for APIRequestContext.

    Browser requests can expose HTTP/2 pseudo-headers such as :authority. Those
    are transport metadata, not regular HTTP field names, and Playwright rejects
    them when supplied through the headers argument of APIRequestContext.
    """
    headers = dict(request.all_headers())
    blocked = {"host", "content-length", "connection"}
    token_chars = frozenset(
        "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    forwarded: dict[str, str] = {}
    for key, value in headers.items():
        normalized = str(key)
        if normalized.lower() in blocked:
            continue
        if not normalized or any(char not in token_chars for char in normalized):
            continue
        forwarded[normalized] = str(value)
    return forwarded


def _payload_shape(payload: Any) -> str:
    if isinstance(payload, dict):
        return "dict_keys=" + ",".join(sorted(str(key) for key in payload)[:24])
    if isinstance(payload, list):
        return f"list_length={len(payload)}"
    return type(payload).__name__


def _api_credentials() -> tuple[str, str] | None:
    key = os.environ.get("MEDIASILO_API_KEY", "").strip()
    secret = os.environ.get("MEDIASILO_API_SECRET", "").strip()
    if bool(key) != bool(secret):
        raise RuntimeError(
            "MediaSilo API credentials are incomplete; configure both "
            "MEDIASILO_API_KEY and MEDIASILO_API_SECRET"
        )
    if not key:
        return None
    return key, secret


def capture_assets(
    review_url: str,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
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
    canonical_assets: dict[str, dict[str, Any]] = {}
    fallback_assets: dict[str, dict[str, Any]] = {}
    observed_json_routes: list[str] = []
    observed_data_routes: list[str] = []
    quicklink_metadata_diagnostics: list[str] = []
    final_navigation: list[str] = []
    quicklink_request_context: list[tuple[str, dict[str, str]] | None] = [None]
    direct_probe_diagnostics: list[str] = []
    api_credentials = _api_credentials()

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
                    resource_type = str(getattr(response.request, "resource_type", "") or "")
                    route = _sanitized_route(response.url)
                    if resource_type in {"xhr", "fetch"}:
                        diagnostic = f"{route}:status={response.status}"
                        if (
                            diagnostic not in observed_data_routes
                            and len(observed_data_routes) < 80
                        ):
                            observed_data_routes.append(diagnostic)

                    if response.status != 200 or not _is_provider_url(response.url):
                        return
                    content_type = str(response.headers.get("content-type") or "").lower()
                    if "json" not in content_type:
                        return
                    parsed_response_url = urllib.parse.urlparse(response.url)
                    if route not in observed_json_routes and len(observed_json_routes) < 40:
                        observed_json_routes.append(route)
                    try:
                        payload = response.json()
                    except Exception:
                        return
                    if parsed_response_url.path.rstrip("/") == f"/v3/quicklinks/{review_id}":
                        origin = f"{parsed_response_url.scheme}://{parsed_response_url.netloc}"
                        quicklink_request_context[0] = (
                            origin,
                            _forward_request_headers(response.request),
                        )
                        asset_ids = payload.get("assetIds") if isinstance(payload, dict) else None
                        asset_id_count = len(asset_ids) if isinstance(asset_ids, list) else 0
                        header_names = sorted(_forward_request_headers(response.request).keys())
                        public_flags = {
                            key: payload.get(key)
                            for key in ("public", "private", "password", "signed")
                            if isinstance(payload, dict)
                        }
                        diagnostic = (
                            f"shape={_payload_shape(payload)}:"
                            f"asset_id_count={asset_id_count}:"
                            f"access_flags={public_flags}:"
                            f"request_header_names={header_names}"
                        )
                        if diagnostic not in quicklink_metadata_diagnostics:
                            quicklink_metadata_diagnostics.append(diagnostic)
                    assets = _asset_list(payload)
                    if not assets:
                        return
                    target = (
                        canonical_assets
                        if _asset_response_matches(response.url, review_id, folder_id)
                        else fallback_assets
                    )
                    for asset in assets:
                        target[_asset_identity(asset)] = asset

                page.on("response", handle)
                try:
                    navigation = page.goto(
                        review_url,
                        wait_until="domcontentloaded",
                        timeout=90000,
                    )
                    page.wait_for_timeout(30000)
                    final_navigation[:] = [
                        f"status={navigation.status if navigation else 'none'} "
                        f"url={_sanitized_route(page.url)}"
                    ]

                    captured = canonical_assets or fallback_assets
                    needs_direct_probe = not captured or (
                        expected_count is not None and len(captured) < expected_count
                    )
                    if needs_direct_probe and quicklink_request_context[0] is not None:
                        origin, browser_headers = quicklink_request_context[0]
                        request_headers = browser_headers
                        credential_mode = "public-review-session"
                        provider_paths = [f"/v3/quicklinks/{review_id}/assets"]
                        if api_credentials is not None:
                            api_key, api_secret = api_credentials
                            request_headers = {
                                "accept": "application/json",
                                "x-key": api_key,
                                "x-secret": api_secret,
                            }
                            credential_mode = "official-api"
                            if folder_id:
                                provider_paths = [
                                    f"/v3/folders/{folder_id}/assets",
                                    f"/v3/quicklinks/{review_id}/folders/{folder_id}/assets",
                                    *provider_paths,
                                ]
                        elif folder_id:
                            provider_paths = [
                                f"/v3/quicklinks/{review_id}/folders/{folder_id}/assets",
                                *provider_paths,
                                f"/v3/folders/{folder_id}/assets",
                            ]
                        for provider_path in provider_paths:
                            api_response = page.context.request.get(
                                origin + provider_path,
                                headers=request_headers,
                                timeout=30000,
                                fail_on_status_code=False,
                            )
                            diagnostic = (
                                f"{provider_path}:status={api_response.status}:"
                                f"auth={credential_mode}"
                            )
                            if api_response.status == 200:
                                try:
                                    payload = api_response.json()
                                except Exception:
                                    payload = None
                                diagnostic += f":shape={_payload_shape(payload)}"
                                assets = _asset_list(payload)
                                if assets:
                                    for asset in assets:
                                        canonical_assets[_asset_identity(asset)] = asset
                            if diagnostic not in direct_probe_diagnostics:
                                direct_probe_diagnostics.append(diagnostic)
                            captured = canonical_assets or fallback_assets
                            if captured and (
                                expected_count is None or len(captured) >= expected_count
                            ):
                                break
                finally:
                    page.close()

                captured = canonical_assets or fallback_assets
                if captured and (expected_count is None or len(captured) >= expected_count):
                    route_mode = "canonical-route" if canonical_assets else "schema-fallback"
                    print(
                        f"MediaSilo assets resolved on attempt {attempt} "
                        f"via {route_mode} count={len(captured)}"
                    )
                    break
        finally:
            browser.close()

    captured = canonical_assets or fallback_assets
    if not captured:
        expected_routes = [f"/quicklinks/{review_id}/assets"]
        if folder_id:
            expected_routes.extend(
                [
                    f"/quicklinks/{review_id}/folders/{folder_id}/assets",
                    f"/folders/{folder_id}/assets",
                ]
            )
        raise RuntimeError(
            "MediaSilo assets were not resolved after 3 attempts; "
            f"navigation={final_navigation or ['unavailable']}; "
            f"expected_provider_asset_routes={expected_routes}; "
            f"observed_provider_json_routes={observed_json_routes}; "
            f"observed_browser_data_routes={observed_data_routes}; "
            f"quicklink_metadata={quicklink_metadata_diagnostics}; "
            f"api_credentials_configured={api_credentials is not None}; "
            f"direct_asset_probe={direct_probe_diagnostics}"
        )
    if expected_count is not None and len(captured) < expected_count:
        raise RuntimeError(
            "MediaSilo review exposed an incomplete asset set; "
            f"captured={len(captured)} expected_at_least={expected_count}; "
            f"navigation={final_navigation or ['unavailable']}; "
            f"observed_provider_json_routes={observed_json_routes}; "
            f"observed_browser_data_routes={observed_data_routes}; "
            f"quicklink_metadata={quicklink_metadata_diagnostics}; "
            f"api_credentials_configured={api_credentials is not None}; "
            f"direct_asset_probe={direct_probe_diagnostics}"
        )
    return list(captured.values())


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
    selected = _select_original_source(
        capture_assets(review_url, expected_count=len(OFFICIAL_FILES)),
        source_key,
    )
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
    review_folder_url = (
        f"https://api.mediasilo.com/v3/quicklinks/{review_id}/folders/{folder_id}/assets"
    )
    folder_url = f"https://api.mediasilo.com/v3/folders/{folder_id}/assets?_page=0"
    if not _asset_response_matches(quicklink_url, review_id, folder_id):
        raise AssertionError("MediaSilo QuickLink asset route is not recognized")
    if not _asset_response_matches(review_folder_url, review_id, folder_id):
        raise AssertionError("MediaSilo QuickLink folder asset route is not recognized")
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
    if not _is_provider_url("https://api.mediasilo.com/v3/quicklinks/example/assets"):
        raise AssertionError("MediaSilo provider URL recognition failed")
    if _is_provider_url("https://example.invalid/assets"):
        raise AssertionError("non-MediaSilo provider URL was accepted")
    if _payload_shape({"assetIds": ["a", "b"]}) != "dict_keys=assetIds":
        raise AssertionError("MediaSilo diagnostic payload shape is unstable")

    class _SyntheticRequest:
        @staticmethod
        def all_headers() -> dict[str, str]:
            return {
                ":authority": "api.mediasilo.com",
                ":method": "GET",
                "authorization": "Bearer synthetic",
                "user-agent": "synthetic-agent",
                "host": "api.mediasilo.com",
                "content-length": "0",
                "x-invalid header": "discard-me",
            }

    forwarded = _forward_request_headers(_SyntheticRequest())
    if forwarded != {
        "authorization": "Bearer synthetic",
        "user-agent": "synthetic-agent",
    }:
        raise AssertionError(
            "MediaSilo browser request headers were not sanitized for APIRequestContext"
        )
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

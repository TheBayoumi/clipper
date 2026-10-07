from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from . import mediasilo

KNOWN_TITLES = {
    "MW4_BetaTopPlays_Stringout_16X9_R1.mp4": "r1",
    "BestOfBeta_Stringout_Batch2.mp4": "batch2",
    "BestOfBeta_Week2_Stringout_1_8.28.mp4": "week2",
}
KNOWN_ORDER = ("r1", "batch2", "week2")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mxf", ".m4v"}


def _requested_derivative(source_profile: dict[str, Any] | None) -> str:
    value = str((source_profile or {}).get("analysis_derivative") or "source").lower()
    if value not in {"source", "proxy"}:
        raise RuntimeError(f"unsupported MediaSilo analysis derivative: {value}")
    return value


def _derivative(asset: dict[str, Any], derivative_type: str) -> tuple[dict[str, Any], str] | None:
    candidates: list[tuple[int, dict[str, Any], str]] = []
    for item in asset.get("derivatives") or []:
        if str(item.get("type") or "").lower() != derivative_type:
            continue
        url = item.get("url") or (item.get("properties") or {}).get("url")
        if not url:
            continue
        try:
            size = int(item.get("fileSize") or 0)
        except (TypeError, ValueError):
            size = 0
        candidates.append((size, item, str(url)))
    if not candidates:
        return None
    _, selected, url = max(candidates, key=lambda item: item[0])
    return selected, url


def _source_derivative(asset: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Backward-compatible strict original-master selector."""
    return _derivative(asset, "source")


def _is_video(asset: dict[str, Any], derivative: dict[str, Any]) -> bool:
    name = str(asset.get("fileName") or asset.get("title") or "")
    if Path(name).suffix.lower() in VIDEO_EXTENSIONS:
        return True
    return bool(derivative.get("width") and derivative.get("height") and derivative.get("duration"))


def _slug(text: str) -> str:
    stem = Path(text).stem.lower()
    value = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return value or "mediasilo_source"


def _asset_specs(source_profile: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in (source_profile or {}).get("assets") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "")
        if not title:
            raise RuntimeError("source_profile.assets entries require exact title")
        result[title] = dict(item)
    return result


def _assign_keys(
    items: list[dict[str, Any]], source_profile: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    specs = _asset_specs(source_profile)
    used: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        title = str(item["title"])
        spec = specs.get(title, {})
        key = str(spec.get("source_key") or KNOWN_TITLES.get(title) or "")
        if not key:
            base = _slug(title)
            key = base
            suffix = 2
            while key in used or key in KNOWN_ORDER:
                key = f"{base}_{suffix}"
                suffix += 1
        if key in used:
            raise RuntimeError(f"duplicate configured MediaSilo source key: {key}")
        used.add(key)
        copy = dict(item)
        copy["source_key"] = key
        if spec.get("content_type"):
            copy["content_type"] = str(spec["content_type"])
        result.append(copy)
    order = {key: pos for pos, key in enumerate(KNOWN_ORDER)}
    result.sort(
        key=lambda item: (order.get(str(item["source_key"]), 999), str(item["title"]).lower())
    )
    return result


def _capture_assets(review_url: str, expected_count: int | None = None) -> list[dict[str, Any]]:
    return mediasilo.capture_assets(review_url, expected_count=expected_count)


def discover(
    review_url: str,
    output: Path,
    github_output: Path | None,
    expected_count: int | None,
    source_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    derivative_type = _requested_derivative(source_profile)
    specs = _asset_specs(source_profile)
    discovered: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    for asset in _capture_assets(review_url, expected_count):
        title = str(asset.get("title") or asset.get("fileName") or "untitled")
        if specs and title not in specs:
            continue
        selected = _derivative(asset, derivative_type)
        if selected is None:
            unavailable.append(
                {
                    "title": title,
                    "requested_derivative": derivative_type,
                    "available_derivatives": sorted(
                        {str(item.get("type") or "") for item in asset.get("derivatives") or []}
                    ),
                }
            )
            continue
        derivative, url = selected
        if not _is_video(asset, derivative):
            continue
        discovered.append(
            {
                "title": title,
                "file_name": asset.get("fileName"),
                "url": url,
                "derivative_type": derivative_type,
                "file_size": derivative.get("fileSize"),
                "width": derivative.get("width"),
                "height": derivative.get("height"),
                "duration_ms": derivative.get("duration") or asset.get("duration"),
                "provider_asset_id": asset.get("id"),
                "provider_presentation_id": asset.get("_presentation_id"),
                "provider_playlist_id": asset.get("_playlist_id"),
                "original_source_declared": any(
                    str(item.get("type") or "").lower() == "source"
                    for item in asset.get("derivatives") or []
                ),
            }
        )

    sources = _assign_keys(discovered, source_profile)
    if expected_count is not None and len(sources) != expected_count:
        raise RuntimeError(
            f"MediaSilo source catalog contains {len(sources)} selected video assets; "
            f"expected {expected_count}; titles={[item['title'] for item in sources]}; "
            f"unavailable={unavailable}"
        )
    payload = {
        "source_count": len(sources),
        "analysis_derivative": derivative_type,
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if github_output is not None:
        matrix = {
            "include": [
                {
                    "source": item["source_key"],
                    "runner": "ubuntu-22.04" if item["source_key"] == "batch2" else "ubuntu-24.04",
                }
                for item in sources
            ]
        }
        with github_output.open("a", encoding="utf-8") as handle:
            handle.write("source_matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")
            handle.write(f"source_count={len(sources)}\n")
    print(
        json.dumps(
            {
                "source_count": len(sources),
                "analysis_derivative": derivative_type,
                "sources": [
                    (item["source_key"], item["title"], item.get("content_type"))
                    for item in sources
                ],
            }
        )
    )
    return payload


def select(catalog: Path, source_key: str, output: Path) -> dict[str, Any]:
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    for item in payload.get("sources") or []:
        if str(item.get("source_key")) == source_key:
            selected = {key: value for key, value in item.items() if key != "source_key"}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
            print(
                f"MediaSilo asset selected: key={source_key} title={selected['title']} "
                f"type={selected['derivative_type']}"
            )
            return selected
    raise RuntimeError(f"source key not found in MediaSilo catalog: {source_key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--review-url", required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--github-output", type=Path)
    d.add_argument("--expected-count", type=int)
    s = sub.add_parser("select")
    s.add_argument("--catalog", type=Path, required=True)
    s.add_argument("--source-key", required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "discover":
        discover(args.review_url, args.output, args.github_output, args.expected_count)
    else:
        select(args.catalog, args.source_key, args.output)


if __name__ == "__main__":
    main()

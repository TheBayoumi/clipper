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


def _source_derivative(asset: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    candidates: list[tuple[int, dict[str, Any], str]] = []
    for item in asset.get("derivatives") or []:
        if str(item.get("type") or "").lower() != "source":
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
    _, derivative, url = max(candidates, key=lambda item: item[0])
    return derivative, url


def _is_video(asset: dict[str, Any], derivative: dict[str, Any]) -> bool:
    name = str(asset.get("fileName") or asset.get("title") or "")
    if Path(name).suffix.lower() in VIDEO_EXTENSIONS:
        return True
    return bool(derivative.get("width") and derivative.get("height") and derivative.get("duration"))


def _slug(text: str) -> str:
    stem = Path(text).stem.lower()
    value = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return value or "mediasilo_source"


def _assign_keys(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        title = str(item["title"])
        key = KNOWN_TITLES.get(title)
        if not key:
            base = _slug(title)
            key = base
            suffix = 2
            while key in used or key in KNOWN_ORDER:
                key = f"{base}_{suffix}"
                suffix += 1
        used.add(key)
        copy = dict(item)
        copy["source_key"] = key
        result.append(copy)
    order = {key: pos for pos, key in enumerate(KNOWN_ORDER)}
    result.sort(
        key=lambda item: (order.get(str(item["source_key"]), 999), str(item["title"]).lower())
    )
    return result


def _capture_assets(review_url: str) -> list[dict[str, Any]]:
    return mediasilo.capture_assets(review_url)


def discover(
    review_url: str, output: Path, github_output: Path | None, expected_count: int | None
) -> dict[str, Any]:
    discovered: list[dict[str, Any]] = []
    for asset in _capture_assets(review_url):
        selected = _source_derivative(asset)
        if selected is None:
            continue
        derivative, url = selected
        if not _is_video(asset, derivative):
            continue
        discovered.append(
            {
                "title": str(asset.get("title") or asset.get("fileName") or "untitled"),
                "file_name": asset.get("fileName"),
                "url": url,
                "derivative_type": "source",
                "file_size": derivative.get("fileSize"),
                "width": derivative.get("width"),
                "height": derivative.get("height"),
                "duration_ms": derivative.get("duration"),
            }
        )
    sources = _assign_keys(discovered)
    if expected_count is not None and len(sources) != expected_count:
        raise RuntimeError(
            f"MediaSilo source catalog contains {len(sources)} source video masters; "
            f"expected {expected_count}; "
            f"titles={[item['title'] for item in sources]}"
        )
    payload = {"source_count": len(sources), "sources": sources}
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
                "sources": [(i["source_key"], i["title"]) for i in sources],
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
                f"MediaSilo source selected: key={source_key} title={selected['title']} type=source"
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

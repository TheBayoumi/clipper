from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://api.mediasilo.com/v3"
ID_KEYS = ("id", "uuid", "folderId", "folderUuid", "assetId", "assetUuid")
NAME_KEYS = ("name", "title", "filename", "fileName")


def fetch_json(url: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://app.mediasilo.com",
            "Referer": "https://app.mediasilo.com/",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("items", "results", "folders", "assets", "data", "children"):
            child = value.get(key)
            if isinstance(child, list):
                return [item for item in child if isinstance(item, dict)]
        if any(key in value for key in ID_KEYS):
            return [value]
    return []


def pick_id(item: dict[str, Any]) -> str | None:
    for key in ID_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def pick_name(item: dict[str, Any]) -> str:
    for key in NAME_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return "unnamed"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "unnamed"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("quicklink_id")
    parser.add_argument("root_folder_id")
    parser.add_argument("--output-dir", type=Path, default=Path("mediasilo-api"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    quicklink_url = f"{API}/quicklinks/{args.quicklink_id}"
    quicklink = fetch_json(quicklink_url)
    dump(out / "quicklink.json", quicklink)

    queue: list[tuple[str, str]] = [(args.root_folder_id, "root")]
    seen: set[str] = set()
    inventory: dict[str, Any] = {"folders": [], "assets": []}

    while queue:
        folder_id, folder_path = queue.pop(0)
        if folder_id in seen:
            continue
        seen.add(folder_id)
        folder_dir = out / "folders" / safe_name(folder_path)

        sub_url = f"{API}/quicklinks/{args.quicklink_id}/folders/{folder_id}/subfolders"
        asset_url = f"{API}/quicklinks/{args.quicklink_id}/folders/{folder_id}/assets"
        try:
            subs = fetch_json(sub_url)
        except urllib.error.HTTPError as exc:
            subs = {"http_error": exc.code, "url": sub_url}
        try:
            assets = fetch_json(asset_url)
        except urllib.error.HTTPError as exc:
            assets = {"http_error": exc.code, "url": asset_url}
        dump(folder_dir / "subfolders.json", subs)
        dump(folder_dir / "assets.json", assets)

        for sub in records(subs):
            sub_id = pick_id(sub)
            if not sub_id:
                continue
            name = pick_name(sub)
            child_path = f"{folder_path}/{name}"
            inventory["folders"].append({"id": sub_id, "name": name, "path": child_path})
            queue.append((sub_id, child_path))

        for asset in records(assets):
            asset_id = pick_id(asset)
            name = pick_name(asset)
            entry: dict[str, Any] = {
                "id": asset_id,
                "name": name,
                "folder_path": folder_path,
                "folder_id": folder_id,
                "raw": asset,
            }
            if asset_id:
                detail_urls = {
                    "detail": f"{API}/quicklinks/{args.quicklink_id}/assets/{asset_id}",
                    "versions": f"{API}/quicklinks/{args.quicklink_id}/assets/{asset_id}/versions",
                }
                for label, url in detail_urls.items():
                    try:
                        payload = fetch_json(url)
                    except urllib.error.HTTPError as exc:
                        payload = {"http_error": exc.code, "url": url}
                    entry[label] = payload
                    dump(
                        out / "assets" / safe_name(folder_path) / f"{safe_name(name)}__{label}.json",
                        payload,
                    )
            inventory["assets"].append(entry)

    dump(out / "inventory.json", inventory)
    print(
        json.dumps(
            {
                "folders": len(inventory["folders"]),
                "assets": len(inventory["assets"]),
                "output": str(out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

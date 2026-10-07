from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .. import media_contract as media


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _declared_size_bounds(resolved: dict[str, Any]) -> tuple[int, int] | None:
    declared_kib = int(resolved.get("file_size") or 0)
    if declared_kib <= 0:
        return None
    floor = declared_kib * 1024
    return floor, floor + 1024


def certify(
    source: Path,
    source_key: str,
    resolved_path: Path,
    output: Path,
    *,
    allow_proxy: bool = False,
) -> dict[str, Any]:
    if not source.is_file():
        raise RuntimeError(f"downloaded source is missing: {source}")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    derivative_type = str(resolved.get("derivative_type") or "").lower()
    if derivative_type != "source" and not (allow_proxy and derivative_type == "proxy"):
        raise RuntimeError(
            f"refusing MediaSilo derivative type={derivative_type!r}; "
            f"allow_proxy_for_analysis={allow_proxy}"
        )

    actual_size = source.stat().st_size
    bounds = _declared_size_bounds(resolved)
    if bounds is not None and not (bounds[0] <= actual_size < bounds[1]):
        label = "original master" if derivative_type == "source" else "analysis proxy"
        raise RuntimeError(
            f"{label} size mismatch: downloaded={actual_size} "
            f"allowed_bytes=[{bounds[0]},{bounds[1]})"
        )

    contract = media.inspect_source(source, verify_timeline=True)
    declared_width = resolved.get("width")
    declared_height = resolved.get("height")
    if declared_width and int(declared_width) != contract.video.width:
        raise RuntimeError(
            f"MediaSilo {derivative_type} metadata width={declared_width} differs from file "
            f"width={contract.video.width}"
        )
    if declared_height and int(declared_height) != contract.video.height:
        raise RuntimeError(
            f"MediaSilo {derivative_type} metadata height={declared_height} differs from file "
            f"height={contract.video.height}"
        )

    payload = {
        "source_key": source_key,
        "title": resolved.get("title"),
        "file_name": resolved.get("file_name"),
        "content_type": resolved.get("content_type"),
        "derivative_type": derivative_type,
        "proxy_fallback_allowed": bool(allow_proxy),
        "production_original_required": True,
        "downloaded_bytes": actual_size,
        "declared_source_size_kib": int(resolved.get("file_size") or 0) or None,
        "declared_source_floor_bytes": bounds[0] if bounds else None,
        "sha256": _sha256(source),
        "media_contract": contract.to_json(),
        "video_profile": media.video_profile(source),
        "audio_profile": media.audio_profile(source),
        "provider_asset_id": resolved.get("provider_asset_id"),
        "provider_presentation_id": resolved.get("provider_presentation_id"),
        "provider_playlist_id": resolved.get("provider_playlist_id"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    timing = contract.video.timing
    policy = (
        "ORIGINAL SOURCE MASTER VERIFIED"
        if derivative_type == "source"
        else "ANALYSIS PROXY VERIFIED"
    )
    print(
        f"{policy} type={derivative_type} bytes={actual_size} "
        f"geometry={contract.video.width}x{contract.video.height} "
        f"rate={media.fraction_text(timing.nominal_rate)} "
        f"time_base={media.fraction_text(timing.time_base)} "
        f"pts_step={timing.ticks_per_frame} timescale={timing.track_timescale} "
        f"sha256={payload['sha256'][:16]}..."
    )
    return payload


def verify(
    source: Path,
    manifest_path: Path,
    *,
    require_original: bool = True,
) -> dict[str, Any]:
    if not source.is_file():
        raise RuntimeError(f"staged source artifact missing: {source}")
    if not manifest_path.is_file():
        raise RuntimeError(f"source QA manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    derivative_type = str(manifest.get("derivative_type") or "")
    if require_original and derivative_type != "source":
        raise RuntimeError(
            "production render requires a certified original MediaSilo type=source master; "
            f"staged derivative={derivative_type!r}"
        )
    if derivative_type not in {"source", "proxy"}:
        raise RuntimeError(f"staged reel has unsupported derivative type={derivative_type!r}")
    if source.stat().st_size != int(manifest.get("downloaded_bytes") or 0):
        raise RuntimeError("staged source byte-size differs from source QA manifest")
    digest = _sha256(source)
    if digest != manifest.get("sha256"):
        raise RuntimeError("staged source SHA256 differs from analysis-stage QA manifest")

    contract = media.inspect_source(source, verify_timeline=True)
    actual_contract = contract.to_json()
    if actual_contract != manifest.get("media_contract"):
        raise RuntimeError(
            "staged source media contract differs from analysis-stage certification: "
            f"expected={manifest.get('media_contract')} actual={actual_contract}"
        )
    timing = contract.video.timing
    label = "ORIGINAL source" if derivative_type == "source" else "analysis proxy"
    print(
        f"staged {label} verified bytes={source.stat().st_size} "
        f"geometry={contract.video.width}x{contract.video.height} "
        f"rate={media.fraction_text(timing.nominal_rate)} "
        f"time_base={media.fraction_text(timing.time_base)} "
        f"pts_step={timing.ticks_per_frame} sha256={digest[:16]}..."
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    certify_parser = sub.add_parser("certify")
    certify_parser.add_argument("--source", type=Path, required=True)
    certify_parser.add_argument("--source-key", required=True)
    certify_parser.add_argument("--resolved", type=Path, required=True)
    certify_parser.add_argument("--output", type=Path, required=True)
    certify_parser.add_argument("--allow-proxy", action="store_true")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--source", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--allow-proxy", action="store_true")
    args = parser.parse_args()
    if args.command == "certify":
        certify(
            args.source,
            args.source_key,
            args.resolved,
            args.output,
            allow_proxy=args.allow_proxy,
        )
    else:
        verify(args.source, args.manifest, require_original=not args.allow_proxy)


if __name__ == "__main__":
    main()

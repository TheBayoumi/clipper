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


def certify(source: Path, source_key: str, resolved_path: Path, output: Path) -> dict[str, Any]:
    if not source.is_file():
        raise RuntimeError(f"downloaded source is missing: {source}")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if str(resolved.get("derivative_type") or "").lower() != "source":
        raise RuntimeError("refusing non-source MediaSilo derivative")

    actual_size = source.stat().st_size
    bounds = _declared_size_bounds(resolved)
    if bounds is not None and not (bounds[0] <= actual_size < bounds[1]):
        raise RuntimeError(
            f"original master size mismatch: downloaded={actual_size} "
            f"allowed_bytes=[{bounds[0]},{bounds[1]})"
        )

    contract = media.inspect_source(source, verify_timeline=True)
    declared_width = resolved.get("width")
    declared_height = resolved.get("height")
    if declared_width and int(declared_width) != contract.video.width:
        raise RuntimeError(
            f"MediaSilo source metadata width={declared_width} differs from file "
            f"width={contract.video.width}"
        )
    if declared_height and int(declared_height) != contract.video.height:
        raise RuntimeError(
            f"MediaSilo source metadata height={declared_height} differs from file "
            f"height={contract.video.height}"
        )

    payload = {
        "source_key": source_key,
        "review_url": resolved.get("review_url"),
        "title": resolved.get("title"),
        "file_name": resolved.get("file_name"),
        "derivative_type": "source",
        "proxy_fallback_allowed": False,
        "downloaded_bytes": actual_size,
        "declared_source_size_kib": int(resolved.get("file_size") or 0) or None,
        "declared_source_floor_bytes": bounds[0] if bounds else None,
        "sha256": _sha256(source),
        "media_contract": contract.to_json(),
        "video_profile": media.video_profile(source),
        "audio_profile": media.audio_profile(source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    timing = contract.video.timing
    print(
        "ORIGINAL SOURCE MASTER VERIFIED "
        f"type=source bytes={actual_size} geometry={contract.video.width}x{contract.video.height} "
        f"rate={media.fraction_text(timing.nominal_rate)} "
        f"time_base={media.fraction_text(timing.time_base)} "
        f"pts_step={timing.ticks_per_frame} timescale={timing.track_timescale} "
        f"sha256={payload['sha256'][:16]}..."
    )
    return payload


def verify(source: Path, manifest_path: Path) -> dict[str, Any]:
    if not source.is_file():
        raise RuntimeError(f"staged source artifact missing: {source}")
    if not manifest_path.is_file():
        raise RuntimeError(f"original-source QA manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("derivative_type") != "source":
        raise RuntimeError("staged reel was not certified as MediaSilo type=source")
    if manifest.get("proxy_fallback_allowed") is not False:
        raise RuntimeError("staged reel source policy allows proxy fallback")
    if source.stat().st_size != int(manifest.get("downloaded_bytes") or 0):
        raise RuntimeError("staged source byte-size differs from original-source QA manifest")
    digest = _sha256(source)
    if digest != manifest.get("sha256"):
        raise RuntimeError("staged source SHA256 differs from original-source QA manifest")

    contract = media.inspect_source(source, verify_timeline=True)
    actual_contract = contract.to_json()
    if actual_contract != manifest.get("media_contract"):
        raise RuntimeError(
            "staged source media contract differs from analysis-stage certification: "
            f"expected={manifest.get('media_contract')} actual={actual_contract}"
        )
    timing = contract.video.timing
    print(
        "staged ORIGINAL source verified "
        f"bytes={source.stat().st_size} geometry={contract.video.width}x{contract.video.height} "
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
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--source", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "certify":
        certify(args.source, args.source_key, args.resolved, args.output)
    else:
        verify(args.source, args.manifest)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
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


def _duration_seconds(source: Path) -> float:
    completed = media.run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(source),
        ]
    )
    duration = float(completed.stdout.strip())
    if duration <= 0.0:
        raise RuntimeError(f"source duration is invalid: {source}")
    return duration


def _nominal_rate(manifest: dict[str, Any]) -> Fraction:
    timing = manifest.get("media_contract", {}).get("video", {}).get("timing", {})
    value = str(timing.get("nominal_rate") or "")
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"source QA manifest has invalid nominal_rate={value!r}") from exc
    if rate <= 0:
        raise RuntimeError(f"source QA manifest has non-positive nominal_rate={value!r}")
    return rate


def certify_analysis_alignment(
    analysis_manifest_path: Path,
    original_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    analysis = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    checks = {
        "analysis_is_proxy": analysis.get("derivative_type") == "proxy",
        "production_is_original_source": original.get("derivative_type") == "source",
        "same_source_key": analysis.get("source_key") == original.get("source_key"),
        "same_title": analysis.get("title") == original.get("title"),
        "same_file_name": analysis.get("file_name") == original.get("file_name"),
        "same_provider_asset": bool(analysis.get("provider_asset_id"))
        and analysis.get("provider_asset_id") == original.get("provider_asset_id"),
        "same_provider_presentation": bool(analysis.get("provider_presentation_id"))
        and analysis.get("provider_presentation_id") == original.get("provider_presentation_id"),
        "same_provider_playlist": bool(analysis.get("provider_playlist_id"))
        and analysis.get("provider_playlist_id") == original.get("provider_playlist_id"),
    }

    analysis_declared = int(analysis.get("declared_duration_ms") or 0)
    original_declared = int(original.get("declared_duration_ms") or 0)
    checks["same_provider_declared_duration"] = (
        analysis_declared > 0 and original_declared > 0 and analysis_declared == original_declared
    )

    analysis_duration = float(analysis.get("duration_seconds") or 0.0)
    original_duration = float(original.get("duration_seconds") or 0.0)
    slower_rate = min(_nominal_rate(analysis), _nominal_rate(original))
    maximum_delta = max(0.05, 2.0 / float(slower_rate))
    duration_delta = abs(analysis_duration - original_duration)
    checks["duration_alignment_within_two_frames"] = (
        analysis_duration > 0.0
        and original_duration > 0.0
        and duration_delta <= maximum_delta + 1e-9
    )

    payload = {
        "schema_version": 1,
        "analysis_derivative": "proxy",
        "production_derivative": "source",
        "provider_asset_id": original.get("provider_asset_id"),
        "analysis_sha256": analysis.get("sha256"),
        "production_sha256": original.get("sha256"),
        "analysis_duration_seconds": analysis_duration,
        "production_duration_seconds": original_duration,
        "duration_delta_seconds": round(duration_delta, 9),
        "maximum_duration_delta_seconds": round(maximum_delta, 9),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    if not all(checks.values()):
        raise RuntimeError(f"analysis proxy/original source temporal alignment failed: {payload}")

    original["analysis_alignment"] = payload
    original_manifest_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "ANALYSIS PROXY -> ORIGINAL SOURCE ALIGNMENT VERIFIED "
        f"asset={payload['provider_asset_id']} delta={duration_delta:.6f}s "
        f"limit={maximum_delta:.6f}s"
    )
    return payload


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
        "declared_duration_ms": int(resolved.get("duration_ms") or 0) or None,
        "duration_seconds": round(_duration_seconds(source), 9),
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
    require_analysis_alignment: bool = False,
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
    if require_analysis_alignment:
        alignment = dict(manifest.get("analysis_alignment") or {})
        checks = dict(alignment.get("checks") or {})
        if alignment.get("status") != "PASS" or not checks or not all(checks.values()):
            raise RuntimeError(
                "production source is missing verified analysis-proxy/original temporal alignment"
            )
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

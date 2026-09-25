"""Profile-driven Clipper campaign stages; editorial and rendering logic live in Clipper."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from . import montage
from .profiles import CampaignProfile, load_profile
from .rendering import montage as montage_renderer
from .sources import catalog, mediasilo
from .sources import qa as source_qa


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _profile(args: argparse.Namespace) -> CampaignProfile:
    profile = load_profile(args.profile, getattr(args, "config", None))
    if profile.config.get("kind") != "announcement":
        raise ValueError("campaign CLI currently supports announcement campaigns")
    if not profile.source_review_url:
        raise ValueError("campaign has no authorized source review")
    return profile


def discover(profile: CampaignProfile, output: Path) -> dict[str, Any]:
    payload = catalog.discover(profile.source_review_url, output, None, None)
    minimum = int(profile.config["source_profile"]["minimum_count"])
    if payload["source_count"] < minimum:
        raise ValueError(
            f"source discovery found {payload['source_count']} originals; expected >= {minimum}"
        )
    if any(item.get("review_url") != profile.source_review_url for item in payload["sources"]):
        raise ValueError("MediaSilo catalog mixes authorized and unauthorized reviews")
    return payload


def acquire(profile: CampaignProfile, source_key: str, output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="clipper-campaign-discovery-") as temp_dir:
        workspace = Path(temp_dir)
        catalog_path = workspace / "catalog.json"
        selected_path = workspace / "resolved.json"
        discover(profile, catalog_path)
        resolved = catalog.select(catalog_path, source_key, selected_path)
        if resolved.get("review_url") != profile.source_review_url:
            raise ValueError("selected asset is outside the configured campaign review")
        # The display title may omit its extension; file_name is the original media identity.
        original_name = str(resolved.get("file_name") or resolved["title"])
        suffix = Path(original_name).suffix.lower()
        if suffix not in catalog.VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported original source extension: {original_name}")
        source_path = output_dir / f"{source_key}{suffix}"
        qa_path = output_dir / f"{source_key}.source_qa.json"
        mediasilo.download(selected_path, source_path)
        certificate = source_qa.certify(source_path, source_key, selected_path, qa_path)
    return {"source": str(source_path), "source_qa": str(qa_path), "certificate": certificate}


def _certification(
    source: Path, source_qa_path: Path | None, profile: CampaignProfile
) -> dict[str, Any] | None:
    if source_qa_path is None:
        return None
    manifest = source_qa.verify(source, source_qa_path)
    if manifest.get("review_url") != profile.source_review_url:
        raise montage.MontageRejection(
            "source_outside_campaign",
            "certified original does not belong to the configured MediaSilo review",
        )
    return manifest


def plan(
    profile: CampaignProfile,
    source: Path,
    source_qa_path: Path | None,
    output: Path,
) -> dict[str, Any]:
    certificate = _certification(source, source_qa_path, profile)
    result = montage.build_plan(source, profile, certificate)
    _write(output, result)
    return result


def render(
    profile: CampaignProfile,
    source: Path,
    source_qa_path: Path | None,
    plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    certificate = _certification(source, source_qa_path, profile)
    approved = json.loads(plan_path.read_text(encoding="utf-8"))
    if approved.get("status") == "PLANNED" and certificate is None:
        raise montage.MontageRejection(
            "source_certificate_missing",
            "campaign-qualified plans require the original source certificate",
        )
    if approved.get("status") == "PREVIEW_ONLY" and certificate is not None:
        raise montage.MontageRejection(
            "preview_not_qualified", "regenerate the plan using the certified source"
        )
    if approved.get("source", {}).get("certified") is not (certificate is not None):
        raise montage.MontageRejection(
            "certification_changed", "plan and acquisition have different certification state"
        )
    return montage_renderer.render(source, profile, approved, output_dir)


def qualify(profile: CampaignProfile, manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = montage.profile_sha256(profile)
    if manifest.get("profile") != profile.name or manifest.get("profile_sha256") != expected_hash:
        raise montage.MontageRejection(
            "profile_mismatch", "render was not produced by this profile"
        )
    if manifest.get("status") != "PASS" or not manifest.get("source", {}).get("certified"):
        raise montage.MontageRejection(
            "uncertified_preview", "a local preview cannot pass production qualification"
        )
    if manifest["source"].get("review_url") != profile.source_review_url:
        raise montage.MontageRejection("source_outside_campaign", "incorrect MediaSilo review")
    path = Path(str(manifest["file"]))
    if not path.is_file() or montage.sha256(path) != manifest.get("sha256"):
        raise montage.MontageRejection("delivery_hash", "delivery is missing or has changed")
    if not all(manifest["qa"]["checks"].values()):
        raise montage.MontageRejection("technical_qa", "render manifest reports failed checks")
    # Check the encoded delivery rather than relying on an old JSON assertion.
    from . import media_contract as media

    probe = json.loads(
        media.run_capture(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
        ).stdout
    )
    duration = float(probe["format"]["duration"])
    editorial = profile.config["editorial"]
    if (
        not float(editorial["minimum_output_seconds"])
        <= duration
        <= float(editorial["maximum_output_seconds"])
    ):
        raise montage.MontageRejection("encoded_duration", f"actual duration {duration:.6f}s")
    result = {
        "profile": profile.name,
        "status": "PASS",
        "certified_review": profile.source_review_url,
        "video_sha256": manifest["sha256"],
        "encoded_duration_seconds": duration,
        "encoded_frames": manifest["qa"]["encoded_video_frames"],
        "allowed_on_screen_copy": manifest["plan"]["montage"]["approved_on_screen_text"],
        "audio_policy": "source_audio_only",
    }
    _write(output, result)
    return result


def run_campaign(args: argparse.Namespace) -> int:
    profile = _profile(args)
    command = args.campaign_command
    if command == "discover":
        result = discover(profile, args.output)
    elif command == "acquire":
        result = acquire(profile, args.source_key, args.output_dir)
    elif command == "plan":
        result = plan(profile, args.source, args.source_qa, args.output)
    elif command == "render":
        result = render(profile, args.source, args.source_qa, args.plan, args.output_dir)
    elif command == "qualify":
        result = qualify(profile, args.render_manifest, args.output)
    else:
        raise ValueError(f"unknown campaign stage: {command}")
    print(json.dumps(result, indent=2))
    return 0

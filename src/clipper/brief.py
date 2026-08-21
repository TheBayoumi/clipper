from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import BriefValidationError, CampaignBrief


@dataclass(frozen=True, slots=True)
class ExplicitTargetSpec:
    """Validated source identity from an explicit-target campaign brief."""

    video_id: str
    url: str
    channel_id: str
    media_url: str | None = None


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BriefValidationError("brief root must be an object")
    if not all(isinstance(key, str) for key in value):
        raise BriefValidationError("brief keys must be strings")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise BriefValidationError(
            "YAML brief requires PyYAML; install the project dependencies"
        ) from exc
    return _mapping(yaml.safe_load(text))


def _load_json(text: str) -> dict[str, Any]:
    return _mapping(json.loads(text))


def _load_brief_data(path: str | Path) -> dict[str, Any]:
    brief_path = Path(path)
    if not brief_path.is_file():
        raise FileNotFoundError(f"brief not found: {brief_path}")
    text = brief_path.read_text(encoding="utf-8")
    suffix = brief_path.suffix.lower()
    if suffix == ".json":
        return _load_json(text)
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(text)
    try:
        return _load_json(text)
    except json.JSONDecodeError:
        return _load_yaml(text)


def _contains_placeholder(value: str) -> bool:
    normalized = value.strip().upper()
    return "REPLACE_WITH" in normalized or normalized.startswith("UC_REPLACE")


def _reject_example_source_placeholders(data: dict[str, Any]) -> None:
    targets = data.get("targets")
    if isinstance(targets, dict):
        videos = targets.get("videos")
        if isinstance(videos, list):
            for item in videos:
                if not isinstance(item, dict):
                    continue
                for key in ("video_id", "channel_id", "url", "media_url"):
                    value = item.get(key)
                    if isinstance(value, str) and _contains_placeholder(value):
                        raise BriefValidationError(
                            f"targets.videos.{key} contains an example placeholder; "
                            "replace it with a real target"
                        )

    rights = data.get("rights")
    if isinstance(rights, dict):
        channels = rights.get("authorized_channels")
        if isinstance(channels, list):
            for value in channels:
                if isinstance(value, str) and _contains_placeholder(value):
                    raise BriefValidationError(
                        "rights.authorized_channels contains an example placeholder"
                    )


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BriefValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _https_url(value: object, field: str) -> str:
    url = _string(value, field)
    if not url.startswith("https://"):
        raise BriefValidationError(f"{field} must use https")
    return url


def _explicit_target_specs_from_data(data: dict[str, Any]) -> tuple[ExplicitTargetSpec, ...]:
    targets = data.get("targets")
    if not isinstance(targets, dict):
        raise BriefValidationError("production brief requires targets.mode=explicit")
    unknown = set(targets) - {"mode", "videos"}
    if unknown:
        raise BriefValidationError(f"unsupported targets rule: {sorted(unknown)[0]}")
    if str(targets.get("mode") or "").strip().lower() != "explicit":
        raise BriefValidationError("targets.mode must be explicit for production runs")
    videos = targets.get("videos")
    if not isinstance(videos, list) or not videos:
        raise BriefValidationError("targets.videos must contain at least one explicit video")

    specs: list[ExplicitTargetSpec] = []
    for index, item in enumerate(videos):
        if not isinstance(item, dict):
            raise BriefValidationError(f"targets.videos[{index}] must be an object")
        unknown_video_fields = set(item) - {"video_id", "url", "channel_id", "media_url"}
        if unknown_video_fields:
            raise BriefValidationError(
                f"unsupported targets.videos rule: {sorted(unknown_video_fields)[0]}"
            )
        video_id = _string(item.get("video_id"), f"targets.videos[{index}].video_id")
        url = _https_url(item.get("url"), f"targets.videos[{index}].url")
        channel_id = _string(item.get("channel_id"), f"targets.videos[{index}].channel_id")
        media_value = item.get("media_url")
        media_url = (
            _https_url(media_value, f"targets.videos[{index}].media_url")
            if media_value is not None
            else None
        )
        specs.append(ExplicitTargetSpec(video_id, url, channel_id, media_url))

    ids = [item.video_id for item in specs]
    if len(set(ids)) != len(ids):
        raise BriefValidationError("targets.videos contains duplicate video IDs")

    rights = data.get("rights")
    if not isinstance(rights, dict):
        raise BriefValidationError("production brief requires a rights object")
    raw_channels = rights.get("authorized_channels", [])
    if not isinstance(raw_channels, list) or not all(isinstance(item, str) for item in raw_channels):
        raise BriefValidationError("rights.authorized_channels must be a list of strings")
    authorized = {item.strip() for item in raw_channels if item.strip()}
    unauthorized = sorted({item.channel_id for item in specs if item.channel_id not in authorized})
    if unauthorized:
        raise BriefValidationError(
            "targets.videos contains channel IDs outside rights.authorized_channels: "
            + ", ".join(unauthorized)
        )
    return tuple(specs)


def load_explicit_targets(path: str | Path) -> tuple[ExplicitTargetSpec, ...]:
    data = _load_brief_data(path)
    _reject_example_source_placeholders(data)
    return _explicit_target_specs_from_data(data)


def _normalize_explicit_targets(normalized: dict[str, Any]) -> None:
    specs = _explicit_target_specs_from_data(normalized)
    normalized["allowed_video_ids"] = [item.video_id for item in specs]
    # Authorized channels are rights constraints only; they are never discovery inputs.
    normalized["source_channel_ids"] = []
    media_urls = {item.video_id: item.media_url for item in specs if item.media_url is not None}
    if media_urls:
        normalized["source_media_urls"] = media_urls


def _normalize_rights(normalized: dict[str, Any]) -> None:
    rights = normalized.get("rights")
    if not isinstance(rights, dict):
        raise BriefValidationError("rights must be an object")
    unknown = set(rights) - {"confirmed", "authorized_channels"}
    if unknown:
        raise BriefValidationError(f"unsupported rights rule: {sorted(unknown)[0]}")
    confirmed = rights.get("confirmed")
    if not isinstance(confirmed, bool):
        raise BriefValidationError("rights.confirmed must be true or false")
    channels = rights.get("authorized_channels", [])
    if not isinstance(channels, list) or not all(isinstance(item, str) for item in channels):
        raise BriefValidationError("rights.authorized_channels must be a list of strings")
    normalized["rights_confirmed"] = confirmed


def _normalize_content_constraints(normalized: dict[str, Any]) -> None:
    constraints = normalized.get("content_constraints")
    if constraints is None:
        return
    if not isinstance(constraints, dict):
        raise BriefValidationError("content_constraints must be an object")
    unknown = set(constraints) - {"min_clip_seconds", "max_clip_seconds"}
    if unknown:
        raise BriefValidationError(f"unsupported content_constraints rule: {sorted(unknown)[0]}")
    if "min_clip_seconds" in constraints:
        normalized["min_clip_seconds"] = constraints["min_clip_seconds"]
    if "max_clip_seconds" in constraints:
        normalized["max_clip_seconds"] = constraints["max_clip_seconds"]


def _normalize_generated_media_policy(normalized: dict[str, Any]) -> None:
    policy = normalized.get("acceptance_policy")
    if not isinstance(policy, dict):
        return
    generated = policy.get("generated_media")
    if not isinstance(generated, dict) or "synthetic_visuals" not in generated:
        return
    unknown = set(generated) - {"synthetic_visuals"}
    if unknown:
        raise BriefValidationError(
            f"unsupported acceptance_policy.generated_media rule: {sorted(unknown)[0]}"
        )
    migrated = dict(policy)
    migrated["generated_media"] = {"ai_generated_source_video": generated["synthetic_visuals"]}
    normalized["acceptance_policy"] = migrated


def _reject_obsolete_fields(data: dict[str, Any]) -> None:
    obsolete = {
        "keywords",
        "negative_keywords",
        "required_phrases",
        "clip_count",
        "max_clips_per_source",
        "source_limit",
        "published_after",
        "production",
        "diversity",
        "hooks",
        "editorial",
    }
    present = sorted(obsolete & set(data))
    if present:
        raise BriefValidationError(
            "production campaign briefs cannot configure legacy editorial/discovery behavior: "
            + ", ".join(present)
        )


def _normalize_semantic_brief(data: dict[str, Any]) -> dict[str, Any]:
    """Map campaign source/rights/policy only; editorial behavior remains autonomous."""
    normalized = dict(data)
    _reject_obsolete_fields(normalized)
    _normalize_explicit_targets(normalized)
    _normalize_rights(normalized)
    _normalize_content_constraints(normalized)
    _normalize_generated_media_policy(normalized)
    return normalized


def load_brief(path: str | Path) -> CampaignBrief:
    data = _load_brief_data(path)
    _reject_example_source_placeholders(data)
    return CampaignBrief.from_dict(_normalize_semantic_brief(data))

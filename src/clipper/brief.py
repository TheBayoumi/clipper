from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BriefValidationError, CampaignBrief


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


def _contains_placeholder(value: str) -> bool:
    normalized = value.strip().upper()
    return "REPLACE_WITH" in normalized or normalized.startswith("UC_REPLACE")


def _reject_example_source_placeholders(data: dict[str, Any]) -> None:
    for key in ("source_channel_ids", "allowed_video_ids"):
        values = data.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and _contains_placeholder(value):
                raise BriefValidationError(
                    f"{key} contains an example placeholder; replace it with a real source ID"
                )

    targets = data.get("targets")
    if isinstance(targets, dict):
        videos = targets.get("videos")
        if isinstance(videos, list):
            for item in videos:
                if not isinstance(item, dict):
                    continue
                for key in ("video_id", "channel_id", "url"):
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


def _normalize_explicit_targets(normalized: dict[str, Any]) -> None:
    targets = normalized.get("targets")
    if targets is None:
        return
    if not isinstance(targets, dict):
        raise BriefValidationError("targets must be an object")
    unknown = set(targets) - {"mode", "videos"}
    if unknown:
        raise BriefValidationError(f"unsupported targets rule: {sorted(unknown)[0]}")
    if str(targets.get("mode") or "").strip().lower() != "explicit":
        raise BriefValidationError("targets.mode must be explicit for production runs")
    videos = targets.get("videos")
    if not isinstance(videos, list) or not videos:
        raise BriefValidationError("targets.videos must contain at least one explicit video")

    ids: list[str] = []
    for index, item in enumerate(videos):
        if not isinstance(item, dict):
            raise BriefValidationError(f"targets.videos[{index}] must be an object")
        unknown_video_fields = set(item) - {"video_id", "url", "channel_id", "media_url"}
        if unknown_video_fields:
            raise BriefValidationError(
                f"unsupported targets.videos rule: {sorted(unknown_video_fields)[0]}"
            )
        video_id = _string(item.get("video_id"), f"targets.videos[{index}].video_id")
        url = _string(item.get("url"), f"targets.videos[{index}].url")
        if not url.startswith("https://"):
            raise BriefValidationError(f"targets.videos[{index}].url must use https")
        ids.append(video_id)

    if len(set(ids)) != len(ids):
        raise BriefValidationError("targets.videos contains duplicate video IDs")

    # The legacy CampaignBrief still exposes discovery fields internally. Explicit-target
    # production populates only allowed_video_ids; authorized channels remain rights data and
    # must never become implicit discovery inputs.
    normalized["allowed_video_ids"] = ids
    normalized["source_channel_ids"] = []
    normalized["source_limit"] = len(ids)


def _normalize_rights(normalized: dict[str, Any]) -> None:
    rights = normalized.get("rights")
    if rights is None:
        return
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


def _reject_modern_editor_quotas(normalized: dict[str, Any]) -> None:
    if "targets" not in normalized:
        return
    forbidden = {
        "clip_count",
        "max_clips_per_source",
        "source_limit",
        "published_after",
        "production",
        "diversity",
        "hooks",
    }
    present = sorted(forbidden & set(normalized))
    if present:
        raise BriefValidationError(
            "explicit-target campaign briefs cannot configure editorial/output quotas: "
            + ", ".join(present)
        )


def _normalize_semantic_brief(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize the modern campaign contract onto the current runtime boundary.

    Modern production briefs are explicit-target and quota-free. This adapter maps only source,
    rights, duration, and policy fields required by the current CampaignBrief. It intentionally
    does not inject campaign-specific cache identities, output counts, concept budgets, render
    budgets, hook counts, or diversity quotas.
    """

    normalized = dict(data)
    _reject_modern_editor_quotas(normalized)
    _normalize_explicit_targets(normalized)
    _normalize_rights(normalized)
    _normalize_content_constraints(normalized)
    _normalize_generated_media_policy(normalized)

    keywords = normalized.get("keywords")
    if not isinstance(keywords, list) or not any(
        isinstance(item, str) and item.strip() for item in keywords
    ):
        objective = str(
            normalized.get("objective") or normalized.get("title") or "campaign"
        ).strip()
        normalized["keywords"] = [objective]
    return normalized


def load_brief(path: str | Path) -> CampaignBrief:
    brief_path = Path(path)
    if not brief_path.is_file():
        raise FileNotFoundError(f"brief not found: {brief_path}")
    text = brief_path.read_text(encoding="utf-8")
    suffix = brief_path.suffix.lower()
    if suffix == ".json":
        data = _load_json(text)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(text)
    else:
        try:
            data = _load_json(text)
        except json.JSONDecodeError:
            data = _load_yaml(text)
    _reject_example_source_placeholders(data)
    return CampaignBrief.from_dict(_normalize_semantic_brief(data))

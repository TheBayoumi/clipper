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


def _reject_example_source_placeholders(data: dict[str, Any]) -> None:
    for key in ("source_channel_ids", "allowed_video_ids"):
        values = data.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = value.strip().upper()
            if "REPLACE_WITH" in normalized or normalized.startswith("UC_REPLACE"):
                raise BriefValidationError(
                    f"{key} contains an example placeholder; replace it with a real source ID"
                )


def _normalize_semantic_brief(data: dict[str, Any]) -> dict[str, Any]:
    """Bridge old schema requirements without requiring topic-word hardcoding.

    CampaignBrief still carries a legacy ``keywords`` field for backwards compatibility.
    The V10 open-weight planner does not consume it. When a modern brief omits the field,
    use the campaign objective as neutral discovery context for old adapters only.
    """
    normalized = dict(data)
    keywords = normalized.get("keywords")
    if not isinstance(keywords, list) or not any(
        isinstance(item, str) and item.strip() for item in keywords
    ):
        objective = str(normalized.get("objective") or normalized.get("title") or "campaign").strip()
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

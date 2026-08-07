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
    return CampaignBrief.from_dict(data)

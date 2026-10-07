from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GameplayProfile:
    name: str
    config: dict[str, Any]

    @property
    def source_url(self) -> str:
        source = self.config.get("source_profile", {})
        return str(source.get("presentation_url") or source.get("review_url") or "")

    @property
    def source_review_url(self) -> str:
        """Backward-compatible alias for older callers."""
        return self.source_url

    @property
    def expected_source_count(self) -> int:
        return int(self.config.get("source_profile", {}).get("expected_count", 0))

    def capability(self, name: str) -> dict[str, Any]:
        return dict(self.config.get("capabilities", {}).get(name) or {})


def _packaged_profile(name: str) -> dict[str, Any]:
    resource = resources.files(__package__).joinpath(f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_profile(name: str, override: Path | None = None) -> GameplayProfile:
    if name != "mw4":
        raise ValueError(f"unknown gameplay profile: {name}")

    packaged = _packaged_profile(name)
    if override is None:
        config = packaged
    else:
        campaign = json.loads(override.read_text(encoding="utf-8"))
        profile_name = str(campaign.get("profile", name))
        if profile_name != name:
            raise ValueError(f"profile file declares {profile_name!r}, expected {name!r}")
        config = _deep_merge(packaged, campaign)

    profile_name = str(config.get("profile", name))
    if profile_name != name:
        raise ValueError(f"profile file declares {profile_name!r}, expected {name!r}")
    return GameplayProfile(name=name, config=config)

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CampaignProfile:
    name: str
    config: dict[str, Any]

    @property
    def source_review_url(self) -> str:
        return str(self.config.get("source_profile", {}).get("review_url", ""))

    @property
    def expected_source_count(self) -> int:
        return int(self.config.get("source_profile", {}).get("expected_count", 0))

    def capability(self, name: str) -> dict[str, Any]:
        return dict(self.config.get("capabilities", {}).get(name) or {})


def _packaged_profile(name: str) -> dict[str, Any]:
    resource = resources.files(__package__).joinpath(f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_profile(name: str, override: Path | None = None) -> CampaignProfile:
    if name not in {"mw4", "warzone-operator-toggle"}:
        raise ValueError(f"unknown campaign profile: {name}")
    config = (
        json.loads(override.read_text(encoding="utf-8"))
        if override is not None
        else _packaged_profile(name)
    )
    profile_name = str(config.get("profile", name))
    if profile_name != name:
        raise ValueError(f"profile file declares {profile_name!r}, expected {name!r}")
    return CampaignProfile(name=name, config=config)

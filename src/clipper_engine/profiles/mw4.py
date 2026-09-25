from __future__ import annotations

from pathlib import Path

from . import CampaignProfile, load_profile


def profile(override: Path | None = None) -> CampaignProfile:
    return load_profile("mw4", override)

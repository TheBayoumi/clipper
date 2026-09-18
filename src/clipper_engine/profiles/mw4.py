from __future__ import annotations

from pathlib import Path

from . import GameplayProfile, load_profile


def profile(override: Path | None = None) -> GameplayProfile:
    return load_profile("mw4", override)

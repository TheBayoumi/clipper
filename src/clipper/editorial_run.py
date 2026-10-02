"""Validated run inputs for the existing contextual podcast editorial pipeline.

Campaign rights and creative policy live in the brief. This file contains only
per-run source selection, cache provenance, rendering, and output locations.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper.brief import load_brief
from clipper.editorial import MAX_RENDERABLE_CLIPS
from clipper.rights import assert_campaign_authorized

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_CHANNEL_ID = re.compile(r"UC[A-Za-z0-9_-]{22}\Z")


@dataclass(frozen=True)
class EditorialRunConfig:
    brief: Path
    artifact_root: Path
    source_video_id: str = ""
    target_channel_id: str = ""
    browser_capture_file: Path | None = None
    require_staged_original: bool = False
    transcript_cache_root: Path | None = None
    editorial_cache_root: Path | None = None
    render_cache_root: Path | None = None
    transcript_source_run_id: str = ""
    compare_prior_audio: bool = False
    render_safety_limit: int = MAX_RENDERABLE_CLIPS
    caption_style: str = "B2"

    @classmethod
    def from_legacy_environment(cls, brief: Path, artifact_root: Path) -> EditorialRunConfig:
        """Preserve existing workflow inputs while callers migrate to run files."""

        def path(name: str) -> Path | None:
            value = os.getenv(name, "").strip()
            return Path(value) if value else None

        return cls(
            brief=brief,
            artifact_root=artifact_root,
            source_video_id=os.getenv("TJR_SOURCE_VIDEO_ID", "").strip(),
            target_channel_id=os.getenv("TJR_TARGET_CHANNEL_ID", "").strip(),
            browser_capture_file=path("TJR_BROWSER_CAPTURE_FILE"),
            require_staged_original=os.getenv("TJR_REQUIRE_STAGED_ORIGINAL") == "1",
            transcript_cache_root=path("TJR_TRANSCRIPT_CACHE_ROOT"),
            editorial_cache_root=path("TJR_EDITORIAL_CACHE_ROOT"),
            render_cache_root=path("TJR_RENDER_CACHE_ROOT"),
            transcript_source_run_id=os.getenv("TJR_TRANSCRIPT_SOURCE_RUN_ID", "").strip(),
            compare_prior_audio=os.getenv("TJR_COMPARE_PRIOR_AUDIO") == "1",
            render_safety_limit=int(
                os.getenv("TJR_RENDER_SAFETY_LIMIT", str(MAX_RENDERABLE_CLIPS))
            ),
            caption_style=os.getenv("TJR_CAPTION_STYLE", "").strip().upper(),
        )

    def validate(self) -> None:
        brief = load_brief(self.brief)
        assert_campaign_authorized(brief)
        if self.source_video_id and not _VIDEO_ID.fullmatch(self.source_video_id):
            raise ValueError("source_video_id must be an exact YouTube video ID")
        if self.target_channel_id and (
            not _CHANNEL_ID.fullmatch(self.target_channel_id)
            or self.target_channel_id not in brief.source_channel_ids
        ):
            raise ValueError("target_channel_id must belong to the authorized campaign")
        if self.require_staged_original and self.browser_capture_file is None:
            raise ValueError("required staged original needs browser_capture_file")
        if not 1 <= self.render_safety_limit <= MAX_RENDERABLE_CLIPS:
            raise ValueError("render_safety_limit exceeds the pipeline safety bound")
        if self.caption_style != "B2":
            raise ValueError("contextual podcast production requires B2 captions")
        if self.transcript_source_run_id and not self.transcript_source_run_id.isdecimal():
            raise ValueError("transcript_source_run_id must be a decimal Actions run ID")


def load_editorial_run_config(path: str | Path) -> EditorialRunConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        raw = json.loads(text)
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(text)
    else:
        raise ValueError("editorial run config must be YAML or JSON")
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError("editorial run config must be an object with string keys")
    required = {"brief", "artifact_root"}
    optional = {
        "source_video_id",
        "target_channel_id",
        "browser_capture_file",
        "require_staged_original",
        "transcript_cache_root",
        "editorial_cache_root",
        "render_cache_root",
        "transcript_source_run_id",
        "compare_prior_audio",
        "render_safety_limit",
        "caption_style",
    }
    if not required <= set(raw) or set(raw) - required - optional:
        raise ValueError("editorial run config has missing or unknown fields")
    base = config_path.resolve().parent

    def location(key: str, *, required_path: bool = False) -> Path | None:
        value: Any = raw.get(key)
        if value is None and not required_path:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path")
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base / candidate).resolve()

    def string(key: str, default: str = "") -> str:
        value: Any = raw.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value.strip()

    def flag(key: str) -> bool:
        value: Any = raw.get(key, False)
        if type(value) is not bool:
            raise ValueError(f"{key} must be a boolean")
        return value

    limit: Any = raw.get("render_safety_limit", MAX_RENDERABLE_CLIPS)
    if type(limit) is not int:
        raise ValueError("render_safety_limit must be an integer")
    brief_path = location("brief", required_path=True)
    artifact_root = location("artifact_root", required_path=True)
    if brief_path is None or artifact_root is None:
        raise ValueError("editorial run requires brief and artifact_root")
    config = EditorialRunConfig(
        brief=brief_path,
        artifact_root=artifact_root,
        source_video_id=string("source_video_id"),
        target_channel_id=string("target_channel_id"),
        browser_capture_file=location("browser_capture_file"),
        require_staged_original=flag("require_staged_original"),
        transcript_cache_root=location("transcript_cache_root"),
        editorial_cache_root=location("editorial_cache_root"),
        render_cache_root=location("render_cache_root"),
        transcript_source_run_id=string("transcript_source_run_id"),
        compare_prior_audio=flag("compare_prior_audio"),
        render_safety_limit=limit,
        caption_style=string("caption_style", "B2"),
    )
    config.validate()
    return config

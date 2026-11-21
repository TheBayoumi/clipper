"""Committed Atomic Habits golden script bytes across canonical and legacy entrypoints."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

from clipper.books import scripting

ROOT = Path(__file__).resolve().parents[1]


def test_committed_atomic_habits_produces_identical_script_cache_and_audit(
    tmp_path: Path,
) -> None:
    config = scripting.NicheConfig.load(ROOT / "niches/book-lessons.yaml")
    episode = json.loads(
        (ROOT / "niches/atomic-habits-episode-02.json").read_text(encoding="utf-8")
    )["script"]
    expected = scripting.Episode.from_json(episode)
    completion = json.dumps(episode)
    canonical = Mock(return_value=completion)

    spec = importlib.util.spec_from_file_location(
        "niche_video_golden", ROOT / "scripts/niche_video.py"
    )
    assert spec is not None and spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    legacy_model = Mock(return_value=completion)
    legacy._local_script_completion = legacy_model

    canonical_result, canonical_key = scripting.generate_episode(
        config, tmp_path / "canonical", completion=canonical
    )
    legacy_result, legacy_key = legacy.generate_episode(config, tmp_path / "legacy")
    canonical.assert_called_once()
    legacy_model.assert_called_once()
    assert canonical_result == legacy_result == expected
    assert canonical_key == legacy_key

    request = scripting._script_request(config)
    assert request["model"] == scripting.SCRIPT_MODEL
    assert request["revision"] == scripting.SCRIPT_REVISION
    canonical_request = json.dumps(
        {"contract": scripting.BOOK_EDIT_CONTRACT, "request": request},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert canonical_key == hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    expected_bytes = json.dumps(expected.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    for relative in (
        Path("scripts") / f"{canonical_key}.json",
        Path("script_diagnostics") / f"{canonical_key}.json",
    ):
        canonical_bytes = (tmp_path / "canonical" / relative).read_bytes()
        legacy_bytes = (tmp_path / "legacy" / relative).read_bytes()
        assert canonical_bytes == legacy_bytes
        if relative.parent.name == "scripts":
            assert canonical_bytes == expected_bytes

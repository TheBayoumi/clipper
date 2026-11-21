"""Regression: overlapping pipeline invocations cannot exchange visual context."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from clipper import pipeline


class _CapturedVisualCall(Exception):
    pass


def test_overlapping_visual_scopes_are_isolated_and_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    both_scopes_ready = Barrier(2)
    recorded: dict[str, dict[str, Any]] = {}

    def capture(*_args: object, **kwargs: Any) -> None:
        recorded[str(kwargs["video_id"])] = kwargs
        raise _CapturedVisualCall

    monkeypatch.setattr(pipeline, "scout_visual_timeline", capture)

    def invoke(name: str) -> None:
        checkpoint = Path("checkpoints") / name
        commit = Mock(name=f"commit_{name}")
        tokens = (
            pipeline._VISUAL_CHECKPOINT_DIR.set(checkpoint),
            pipeline._VISUAL_CHECKPOINT_COMMIT.set(commit),
            pipeline._REQUIRED_VISUAL_ENTITIES.set((name,)),
        )
        try:
            both_scopes_ready.wait(timeout=5)
            video = SimpleNamespace(video_id=name, duration_seconds=2.0)
            timeline = SimpleNamespace(words=(object(),), end=2.0, source_hash=name)
            with pytest.raises(_CapturedVisualCall):
                pipeline._visual_timeline(Path(f"{name}.mp4"), video, timeline, Mock(), Path("out"))
        finally:
            pipeline._REQUIRED_VISUAL_ENTITIES.reset(tokens[2])
            pipeline._VISUAL_CHECKPOINT_COMMIT.reset(tokens[1])
            pipeline._VISUAL_CHECKPOINT_DIR.reset(tokens[0])
        assert pipeline._VISUAL_CHECKPOINT_DIR.get() is None
        assert pipeline._VISUAL_CHECKPOINT_COMMIT.get() is None
        assert pipeline._REQUIRED_VISUAL_ENTITIES.get() == ()

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(invoke, "left")
        right = pool.submit(invoke, "right")
        left.result(timeout=10)
        right.result(timeout=10)

    assert set(recorded) == {"left", "right"}
    for name in ("left", "right"):
        context = recorded[name]
        assert context["checkpoint_dir"] == Path("checkpoints") / name
        assert context["checkpoint_commit"]._mock_name == f"commit_{name}"
        assert context["required_visual_entities"] == (name,)

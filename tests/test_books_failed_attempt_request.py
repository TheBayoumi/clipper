from __future__ import annotations

from pathlib import Path

import pytest
from test_books_chapter import _feedback_validator, _plan

from clipper.books import chapter, scripting


def test_gpt1_15_failed_attempt2_keeps_request_evidence(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    def provider(request: dict[str, object]) -> str:
        seen.append(request)
        if len(seen) == 1:
            return '{"ok": false}'
        raise scripting.IncompleteConstrainedResponse('{"partial": ')

    def builder(feedback: dict[str, object]) -> tuple[dict[str, object], str]:
        return {**seen[0], "retry_feedback": feedback}, "c" * 64

    root = tmp_path / "root"
    with pytest.raises(chapter.ChapterStop, match="without EOS"):
        chapter._stage(
            root,
            "brief",
            {"scope": "synthetic"},
            _plan(max_attempts_per_stage=2),
            _feedback_validator,
            provider,
            retry_builder=builder,
        )
    directory = next((root / "brief").iterdir())
    blob = b"".join(f.read_bytes() for f in directory.iterdir())
    # retry request digest ("c"*64) must be retained somewhere for attempt 2
    assert b"c" * 64 in blob

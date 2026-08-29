from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_gate() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "codex_review_gate.py"
    spec = importlib.util.spec_from_file_location("codex_review_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_requires_exact_codex_head() -> None:
    gate = _load_gate()
    head = "a" * 40
    assert gate._review_matches_head(
        {
            "user": {"login": gate.CODEX_LOGIN},
            "commit_id": head,
            "body": "Reviewed commit: `aaaaaaaaaa`",
        },
        head,
    )
    assert not gate._review_matches_head(
        {"user": {"login": gate.CODEX_LOGIN}, "commit_id": "b" * 40}, head
    )
    assert not gate._review_matches_head(
        {"user": {"login": "someone-else"}, "commit_id": head}, head
    )


def test_unresolved_codex_p0_p1_p2_blocks_but_p3_does_not() -> None:
    gate = _load_gate()
    def thread(body: str, resolved: bool = False) -> dict[str, object]:
        return {
            "isResolved": resolved,
            "comments": {"nodes": [{"author": {"login": gate.CODEX_LOGIN}, "body": body}]},
        }

    assert gate._is_blocking_codex_thread(thread("P0 production corruption"))
    assert gate._is_blocking_codex_thread(thread("P1 NEEDS_RUNTIME_EVIDENCE"))
    assert gate._is_blocking_codex_thread(thread("P2 race"))
    assert not gate._is_blocking_codex_thread(thread("P3 cleanup"))
    assert not gate._is_blocking_codex_thread(thread("P1 fixed", resolved=True))


def test_clean_codex_reaction_requires_exact_head_request() -> None:
    gate = _load_gate()
    head = "c" * 40
    comment = {"body": f"@codex review\nReview exact head `{head}`"}
    reactions = [{"user": {"login": gate.CODEX_LOGIN}, "content": "+1"}]
    assert gate._clean_reaction_matches_head(comment, reactions, head)
    assert not gate._clean_reaction_matches_head(comment, reactions, "d" * 40)
    assert not gate._clean_reaction_matches_head(
        comment, [{"user": {"login": "someone-else"}, "content": "+1"}], head
    )

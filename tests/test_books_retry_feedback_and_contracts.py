"""LANE-6 verifier probes (synthetic text only)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from test_books_chapter import _plan
from test_books_chapter_hybrid import _argument, _audit_outputs, _ledger

from clipper.books import chapter
from clipper.books import chapter_hybrid as hybrid

IDENT = {"schema": "runtime-v1", "runtime": "CPU", "producer": "a" * 64}


def _stage_calls(tmp_path: Path, first: str) -> list[dict[str, object]]:
    ledger = _ledger()
    good = json.dumps(_argument(ledger))
    seen: list[dict[str, object]] = []

    def provider(request: dict[str, object]) -> str:
        seen.append(request)
        return first if len(seen) == 1 else good

    def validator(raw: object) -> dict:
        return hybrid.validate_argument(raw, ledger)

    def builder(feedback: dict[str, object]) -> tuple[dict[str, object], str]:
        return {**seen[0], "retry_feedback": feedback}, "c" * 64

    with contextlib.suppress(chapter.ChapterStop):
        chapter._stage(
            tmp_path / "root",
            "brief",
            {"scope": "synthetic"},
            _plan(max_attempts_per_stage=2),
            validator,
            provider,
            retry_builder=builder,
        )
    return seen


@pytest.mark.parametrize(
    "first",
    [
        '{"incorrect": "schema"}',
        '{"hypothesis": "x y z", "questions": ["q one?"], "source_links": [{"id": "zz", "role": "ASSERTION"}]}',
    ],
)
def test_gpt1_30_structural_failure_gets_second_producer_call(tmp_path: Path, first: str) -> None:
    assert len(_stage_calls(tmp_path, first)) == 2


def _six_assertions_messages() -> list[dict[str, str]]:
    ledger = _ledger()
    links = [{"id": f"s{n:04d}", "role": "ASSERTION"} for n in range(2, 8)]
    with pytest.raises(hybrid.HybridStop) as caught:
        hybrid.validate_argument({**_argument(ledger), "source_links": links}, ledger)
    feedback = hybrid.retry_feedback(caught.value)
    assert feedback is not None
    payload = {"schema": hybrid.ARGUMENT_SCHEMA, "original_source": {}, "retry_feedback": feedback}
    return hybrid._messages("argument", payload)


def test_gpt1_31_degeneracy_feedback_explains_role_concentration() -> None:
    user = _six_assertions_messages()[1]["content"]
    tail = user[user.index("previous answer was rejected") :]
    assert "concentrat" in tail.lower() and "role" in tail.lower().split("degenerate_links")[-1]


def test_gpt1_32_script_contract_names_required_scene_fields() -> None:
    import ast

    tree = ast.parse(Path(chapter.__file__).read_text())
    found = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.startswith("Existing chapter connected multi-scene script JSON")
    ]
    assert len(found) == 1
    contract = found[0]
    messages = hybrid._messages(
        "chapter_script", {"schema": "x", "brief": {}, "response_contract": contract}
    )
    user = messages[1]["content"]
    tail = user[user.index("TASK:") :]
    missing = [
        f
        for f in (
            "scene_id",
            "previous_id",
            "next_id",
            "purpose",
            "action",
            "environment_id",
            "narration",
            "dialogue",
            "beat_ids",
            "illustration_label",
            "transition",
            "consequence",
            "speaker_id",
            "fiction_label",
        )
        if f not in tail
    ]
    assert not missing, missing


def test_cdx1_04_invalid_audit_decision_id_gets_feedback() -> None:
    ledger = _ledger()
    windows = hybrid.audit_windows(ledger)
    audit = _audit_outputs(ledger)[0]
    audit["decisions"][0]["id"] = "s9999"
    with pytest.raises(hybrid.HybridStop) as caught:
        hybrid.validate_audit(audit, ledger, windows[0])
    assert hybrid.retry_feedback(caught.value) is not None

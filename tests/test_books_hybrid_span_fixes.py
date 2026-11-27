from __future__ import annotations

import copy

import pytest
from test_books_chapter_hybrid import _argument, _audit_outputs, _ledger, _reconciliation

from clipper.books import chapter_hybrid as hybrid


def _entries(text: str) -> dict[str, dict[str, object]]:
    return {"s0001": {"kind": "CANONICAL_WORK_UNIT", "text": text, "source_id": "synthetic-1"}}


def test_gpt1_03_short_unit_citable() -> None:
    span = hybrid._span({"id": "s0001", "role": "QUALIFICATION"}, _entries("No"))
    assert span["quote"] == "No"


def test_gpt1_04_whitespace_span_rejected() -> None:
    with pytest.raises(hybrid.HybridStop):
        hybrid._span({"id": "s0001", "start": 1, "end": 4, "role": "ASSERTION"}, _entries("A   B"))


def test_gpt1_07_mixed_format_order_independent() -> None:
    ref = {"id": "s0001", "role": "ASSERTION"}
    legacy = {**ref, "start": 0, "end": 5}
    e = _entries("Alpha")
    a = hybrid._canonical_spans([ref, legacy], e, "source_links")
    b = hybrid._canonical_spans([legacy, ref], e, "source_links")
    assert a == b


def test_gpt1_08_duplicate_legacy_spans_preserved() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    # audits are model outputs; reuse as in existing tests
    raw = _reconciliation(ledger)
    span = raw["beats"][0]["spans"][0]  # type: ignore[index]
    rest = raw["beats"][0]["spans"][1:]  # type: ignore[index]
    raw["beats"][0]["spans"] = [copy.deepcopy(span), copy.deepcopy(span), *rest]  # type: ignore[index]
    brief = hybrid.reconcile(raw, ledger, _argument(ledger), audits)
    assert len(brief["beats"][0]["spans"]) == len(raw["beats"][0]["spans"])  # type: ignore[index]

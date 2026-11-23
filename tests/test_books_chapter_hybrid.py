"""Model-free adversarial checks for hybrid chapter source/decision contracts."""

from __future__ import annotations

import copy
import hashlib

import pytest

from clipper.books import chapter_hybrid as hybrid


def _original(count: int = 101) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    units: list[dict[str, object]] = []
    for index in range(count):
        text = f"Unit {index}: an example and an assertion about choices."
        if index == 2:
            text = "She chose it. But this claim only holds under a narrow condition."
        if index == 3:
            text = "That condition was not universal; a contrary example disproves it."
        if index == 5:
            text = "A repeated term, term, can mean something different in context."
        if index == 6:
            text = "العربية: تُفسَّر الدلالة هنا، و naïve — 🧭 demonstrates Unicode offsets."
        units.append(
            {
                "source_id": f"unit-parent-{index:04d}",
                "kind": "CANONICAL_WORK_UNIT",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "unit_fingerprint": "f" * 64,
                "anchor": f"section-{index // 20}",
                "origin": "source",
                "review_status": "unreviewed",
                "research_link_status": "none",
                "section_id": f"s-{index // 20}",
                "source_path": f"section-{index // 20}.html",
                "original_parent_ordinal": index,
            }
        )
    gaps: list[dict[str, object]] = [
        {
            "source_id": "gap-unresolved",
            "kind": "UNRECOVERED_OR_EMPTY_SOURCE_MARKER",
            "text": "",
            "anchor": "unrecovered",
            "origin": "unrecovered_image",
        }
    ]
    return units, gaps, [str(u["source_id"]) for u in units] + ["gap-unresolved"]


def _ledger(count: int = 101) -> dict[str, object]:
    units, gaps, ordered = _original(count)
    return hybrid.freeze(units, gaps, ordered, {"chapter": 1, "parent": "a" * 64})


def _audit_outputs(ledger: dict[str, object]) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []
    for window in hybrid.audit_windows(ledger):
        decisions = []
        for sid in window["core_ids"]:
            source = {
                entry["id"]: entry for entry in hybrid.validate_ledger(ledger)
            }[sid]
            if source["kind"] == "CANONICAL_WORK_UNIT":
                evidence = [
                    {
                        "id": sid,
                        "start": 0,
                        "end": len(source["text"]),
                        "role": "ASSERTION",
                    }
                ]
                decision = "keep"
            else:
                evidence = []
                decision = "omit"
            decisions.append(
                {
                    "id": sid,
                    "decision": decision,
                    "reason": "Original argument or explicit unresolved source marker",
                    "evidence": evidence,
                }
            )
        outputs.append(
            {
                "audit_id": window["audit_id"],
                "core_ids": window["core_ids"],
                "decisions": decisions,
                "disagreements": [],
            }
        )
    return outputs


def _reconciliation(ledger: dict[str, object]) -> dict[str, object]:
    entries = [e for e in hybrid.validate_ledger(ledger) if e["text"]]
    beats = []
    for offset in range(0, len(entries), 20):
        chunk = entries[offset : offset + 20]
        beats.append(
            {
                "beat_id": f"beat-{len(beats) + 1:03d}",
                "spans": [
                    {
                        "id": entry["id"],
                        "start": 0,
                        "end": len(entry["text"]),
                        "role": "ASSERTION",
                    }
                    for entry in chunk
                ],
                "meaning": "A connected argument which remains subject to review.",
                "qualification": "Scope and counterexamples need human attestation.",
                "causal_link": "The following cases narrow the opening hypothesis.",
                "essential": True,
            }
        )
    return {
        "argument": "A revisable, grounded throughline with a counterexample.",
        "amendments": [],
        "resolutions": [],
        "beats": beats,
    }


def _argument(ledger: dict[str, object]) -> dict[str, object]:
    return {
        "hypothesis": "Provisional interpretation, not a settled claim.",
        "questions": ["Does the middle qualifier contradict this interpretation?"],
        "source_links": [
            {"id": "s0003", "start": 0, "end": 13, "role": "ASSERTION"}
        ],
    }


def test_101_units_and_gap_preserve_bytes_order_reversible_ids() -> None:
    original, gap, ordered = _original()
    ledger = _ledger()
    entries = hybrid.validate_ledger(ledger)
    assert len(entries) == 102
    assert [e["source_id"] for e in entries] == ordered
    assert [e["text"].encode() for e in entries] == [
        u["text"].encode() for u in [*original, *gap]
    ]
    assert entries[0]["id"] == "s0001"
    assert entries[-1]["id"] == "s0102"
    assert ledger["identity"]["parent"] == "a" * 64
    assert len(hybrid.audit_windows(ledger)) == 9
    assert [
        sid for window in hybrid.audit_windows(ledger) for sid in window["core_ids"]
    ] == [entry["id"] for entry in entries]


def test_real_unicode_offsets_and_many_to_one_beats_without_forced_omissions() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    initial = hybrid.validate_argument(_argument(ledger), ledger)
    brief = hybrid.reconcile(_reconciliation(ledger), ledger, initial, audits)
    assert len(brief["beats"]) == 6
    assert sum(x["decision"] != "omit" for x in brief["coverage"]) == 101
    assert brief["coverage"][-1]["decision"] == "omit"
    unicode_span = brief["beats"][0]["spans"][6]
    assert unicode_span["quote"] == hybrid.validate_ledger(ledger)[6]["text"]
    assert unicode_span["quote"].startswith("العربية:")
    packet = hybrid.human_review_packet(brief, ledger)
    assert packet["units"][2]["next_text"].startswith("That condition")
    assert packet["review_status"] == "PENDING_ACTUAL_HUMAN_ATTESTATION"


def test_middle_qualifier_and_contrary_example_are_not_auto_reconciled() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    audits[0]["disagreements"] = [
        {
            "id": "a001-d01",
            "issue": "Counterexample changes the opening claim.",
            "evidence": [{"id": "s0004", "start": 0, "end": 20, "role": "COUNTEREXAMPLE"}],
        }
    ]
    with pytest.raises(hybrid.HybridStop, match="unresolved"):
        hybrid.reconcile(_reconciliation(ledger), ledger, _argument(ledger), audits)
    response = _reconciliation(ledger)
    response["resolutions"] = [
        {
            "id": "a001-d01",
            "resolution": "qualified",
            "reason": "A narrower argument survives the contrary example.",
            "evidence": [{"id": "s0004", "start": 0, "end": 20, "role": "COUNTEREXAMPLE"}],
        }
    ]
    assert len(hybrid.reconcile(response, ledger, _argument(ledger), audits)["resolutions"]) == 1


def test_missing_duplicate_and_conflicting_decisions_fail_closed() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    audits[0]["decisions"].pop()
    with pytest.raises(hybrid.HybridStop, match="omitted"):
        hybrid.reconcile(_reconciliation(ledger), ledger, _argument(ledger), audits)
    audits = _audit_outputs(ledger)
    audits[0]["decisions"][0]["id"] = audits[0]["decisions"][1]["id"]
    with pytest.raises(hybrid.HybridStop, match="duplicated"):
        hybrid.reconcile(_reconciliation(ledger), ledger, _argument(ledger), audits)


def test_explicit_amendment_requires_spans_and_final_beat_reconciliation() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    response = _reconciliation(ledger)
    response["amendments"] = [
        {
            "id": "s0002",
            "from": "keep",
            "to": "omit",
            "reason": "Superseded by an explicit counterexample.",
            "evidence": [{"id": "s0004", "start": 0, "end": 20, "role": "COUNTEREXAMPLE"}],
        }
    ]
    with pytest.raises(hybrid.HybridStop, match="omitted source"):
        hybrid.reconcile(response, ledger, _argument(ledger), audits)
    response["beats"][0]["spans"] = [
        span for span in response["beats"][0]["spans"] if span["id"] != "s0002"
    ]
    brief = hybrid.reconcile(response, ledger, _argument(ledger), audits)
    assert brief["coverage"][1]["decision"] == "omit"
    assert brief["coverage"][1]["reason"] == "Superseded by an explicit counterexample."


def test_gap_cannot_be_cited_or_retained() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    audits[-1]["decisions"][-1]["decision"] = "keep"
    with pytest.raises(hybrid.HybridStop, match="gap"):
        hybrid.reconcile(_reconciliation(ledger), ledger, _argument(ledger), audits)
    audits = _audit_outputs(ledger)
    response = _reconciliation(ledger)
    response["beats"][0]["spans"][0]["id"] = "s0102"
    with pytest.raises(hybrid.HybridStop, match="omitted source"):
        hybrid.reconcile(response, ledger, _argument(ledger), audits)


def test_immutable_source_change_map_and_reordering_invalidate() -> None:
    ledger = _ledger()
    for field, value in (
        ("text", "forged text"),
        ("id", "s0002"),
        ("source_id", "unit-forged"),
    ):
        forged = copy.deepcopy(ledger)
        forged["entries"][0][field] = value
        with pytest.raises(hybrid.HybridStop):
            hybrid.validate_ledger(forged)
    swapped = copy.deepcopy(ledger)
    swapped["entries"][0], swapped["entries"][1] = (
        swapped["entries"][1],
        swapped["entries"][0],
    )
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_ledger(swapped)


def test_pinned_context_capacity_and_cache_identity_changes() -> None:
    payload = {"source": hybrid.projection(hybrid.validate_ledger(_ledger()))}
    identity = {"schema": "runtime-v1", "runtime": "CPU", "producer": "a" * 64}
    req, key, count = hybrid.request(
        "argument",
        payload,
        output_tokens=4096,
        token_count=lambda _messages: 7968,
        producer_identity=identity,
    )
    assert count == 7968
    assert req["revision"] == hybrid.scripting.SCRIPT_REVISION
    changed = {**identity, "producer": "b" * 64}
    assert hybrid.request(
        "argument", payload, output_tokens=4096,
        token_count=lambda _messages: 7968, producer_identity=changed,
    )[1] != key
    assert hybrid.request(
        "argument", {"source": [*payload["source"], ["new"]]},
        output_tokens=4096, token_count=lambda _messages: 7968,
        producer_identity=identity,
    )[1] != key
    with pytest.raises(hybrid.HybridStop, match="safe context"):
        hybrid.request(
            "argument", payload, output_tokens=4096,
            token_count=lambda _messages: 28700, producer_identity=identity,
        )
    with pytest.raises(hybrid.HybridStop, match="unavailable"):
        hybrid.request(
            "argument", payload, output_tokens=4096,
            token_count=lambda _messages: (_ for _ in ()).throw(RuntimeError()),
            producer_identity=identity,
        )


def test_exact_quotes_cannot_prove_meaning_or_keyword_faithfulness() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    response = _reconciliation(ledger)
    response["beats"][0]["meaning"] = "Untrue semantic inversion despite exact quotes"
    response["beats"][0]["qualification"] = "A faithful paraphrase without a source keyword"
    brief = hybrid.reconcile(response, ledger, _argument(ledger), audits)
    assert "Untrue semantic inversion" in brief["beats"][0]["meaning"]
    assert "faithful paraphrase" in brief["beats"][0]["qualification"]
    with pytest.raises(hybrid.HybridStop, match="not individually reviewed"):
        hybrid.validate_human_review(
            {
                "schema": hybrid.REVIEW_SCHEMA,
                "brief_sha256": brief["brief_sha256"],
                "reviewer": "Human",
                "unit_decisions": [],
                "beat_decisions": [],
            },
            brief,
            ledger,
        )


def test_human_review_requires_exact_brief_and_each_unit_and_beat() -> None:
    ledger = _ledger(3)
    audits = _audit_outputs(ledger)
    response = _reconciliation(ledger)
    brief = hybrid.reconcile(response, ledger, _argument(ledger), audits)
    approval = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "reviewer": "Independent human",
        "unit_decisions": [
            {"id": e["id"], "review": "accept", "reason": "Read original context"}
            for e in hybrid.validate_ledger(ledger)
        ],
        "beat_decisions": [
            {
                "beat_id": b["beat_id"],
                "review": "support_attested",
                "reason": "Compared against original argument and qualifiers",
            }
            for b in brief["beats"]
        ],
    }
    assert hybrid.validate_human_review(approval, brief, ledger) == approval
    approval["brief_sha256"] = "f" * 64
    with pytest.raises(hybrid.HybridStop, match="exact coherent brief"):
        hybrid.validate_human_review(approval, brief, ledger)

"""Model-free adversarial checks for hybrid chapter source/decision contracts."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from public_book_fixture_support import (
    _bundle,
    _hybrid_creative_output,
    _hybrid_wav,
    _injected_hybrid_response,
    _selected_hybrid_source,
    ingested,
)
from test_books_chapter_host import _call, _locked, _private_setup

from clipper.books import chapter, chapter_host, evaluation, scripting, selected_scope, whole_book
from clipper.books import chapter_hybrid as hybrid
from clipper.books.orchestrator import BookOrchestrator


def _original(
    count: int = 101,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
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
            source = {entry["id"]: entry for entry in hybrid.validate_ledger(ledger)}[sid]
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
        "source_links": [{"id": "s0003", "start": 0, "end": 13, "role": "ASSERTION"}],
    }


def test_101_units_and_gap_preserve_bytes_order_reversible_ids() -> None:
    original, gap, ordered = _original()
    ledger = _ledger()
    entries = hybrid.validate_ledger(ledger)
    assert len(entries) == 102
    assert [e["source_id"] for e in entries] == ordered
    assert [e["text"].encode() for e in entries] == [u["text"].encode() for u in [*original, *gap]]
    assert entries[0]["id"] == "s0001"
    assert entries[-1]["id"] == "s0102"
    assert ledger["identity"]["parent"] == "a" * 64
    assert len(hybrid.audit_windows(ledger)) == 9
    assert [sid for window in hybrid.audit_windows(ledger) for sid in window["core_ids"]] == [
        entry["id"] for entry in entries
    ]


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
            "proposed_change": "Qualify the initial chapter argument.",
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
    with pytest.raises(hybrid.HybridStop, match="character span"):
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
    assert (
        hybrid.request(
            "argument",
            payload,
            output_tokens=4096,
            token_count=lambda _messages: 7968,
            producer_identity=changed,
        )[1]
        != key
    )
    assert (
        hybrid.request(
            "argument",
            {"source": [*payload["source"], ["new"]]},
            output_tokens=4096,
            token_count=lambda _messages: 7968,
            producer_identity=identity,
        )[1]
        != key
    )
    with pytest.raises(hybrid.HybridStop, match="safe context"):
        hybrid.request(
            "argument",
            payload,
            output_tokens=4096,
            token_count=lambda _messages: 28700,
            producer_identity=identity,
        )
    with pytest.raises(hybrid.HybridStop, match="unavailable"):
        hybrid.request(
            "argument",
            payload,
            output_tokens=4096,
            token_count=lambda _messages: (_ for _ in ()).throw(RuntimeError()),
            producer_identity=identity,
        )


def test_actual_pinned_public_tokenizer_template_and_same_count_tokens_bind_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public-only CPU preflight: no tokenizer downloads or model weights."""
    from clipper.books import constrained

    home = tmp_path / "local-huggingface"
    snapshot = home / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots" / scripting.SCRIPT_REVISION
    snapshot.mkdir(parents=True)
    tokenizer_json = b"synthetic pinned public tokenizer JSON (not production weights)"
    (snapshot / "tokenizer.json").write_bytes(tokenizer_json)
    config = snapshot / "tokenizer_config.json"
    config.write_bytes(b'{"chat_template":"synthetic template A"}')
    monkeypatch.setattr(
        constrained,
        "PINNED_TOKENIZER_SHA256",
        hashlib.sha256(tokenizer_json).hexdigest(),
    )
    monkeypatch.setenv("HF_HOME", str(home))

    class PublicTokenizer:
        name_or_path = str(snapshot)
        chat_template = "synthetic template A"

        def __init__(self) -> None:
            self.ids = [71, 72, 73, 74]

        def apply_chat_template(self, *_args: object, **kwargs: object) -> list[int]:
            assert kwargs["enable_thinking"] is False
            assert kwargs["tokenize"] is True
            return list(self.ids)

    fake = PublicTokenizer()
    monkeypatch.setattr(scripting, "_public_book_context", lambda: (fake, 40960))
    legacy_a = whole_book.local_producer_material()
    producer_a = hybrid.producer_identity(
        ledger_sha256="a" * 64,
        chapter_scope_sha256="b" * 64,
        selected_bundle_sha256="c" * 64,
        plan_fingerprint="d" * 64,
        model_producer=whole_book.hybrid_chapter_producer_material(),
    )
    material_a = scripting._cached_chapter_tokenizer_material()
    assert material_a["status"] == "PINNED_PUBLIC_TOKENIZER"
    assert material_a["tokenizer_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    payload = {"source": hybrid.projection(hybrid.validate_ledger(_ledger(1)))}
    identity = {"schema": "synthetic", "runtime": producer_a}
    request_a, key_a, count = hybrid.request(
        "argument",
        payload,
        output_tokens=1024,
        token_count=scripting._chapter_prompt_encoding,
        producer_identity=identity,
    )
    assert count == 4
    assert request_a["verified_input_encoding"] == scripting._chapter_encoding_receipt(
        scripting._chapter_prompt_encoding(request_a["messages"])
    )
    assert scripting._chapter_context_tokens(request_a) == 4
    assert (
        hybrid.request(
            "argument",
            payload,
            output_tokens=1024,
            token_count=scripting._chapter_prompt_encoding,
            producer_identity=identity,
        )[1]
        == key_a
    )

    for change in ("tokens_same_count", "effective_template", "tokenizer_config"):
        if change == "tokens_same_count":
            fake.ids = [71, 72, 73, 75]
        elif change == "effective_template":
            fake.chat_template = "synthetic template B"
        else:
            config.write_bytes(b'{"chat_template":"synthetic template B"}')
        request_b, key_b, next_count = hybrid.request(
            "argument",
            payload,
            output_tokens=1024,
            token_count=scripting._chapter_prompt_encoding,
            producer_identity=identity,
        )
        assert next_count == count
        assert key_b != key_a
        assert request_b["verified_input_encoding"] != request_a["verified_input_encoding"]
        with pytest.raises(scripting.NicheVideoError, match="tokens or template"):
            scripting._chapter_context_tokens(request_a)
        if change == "tokenizer_config":
            assert whole_book.local_producer_material() == legacy_a
            producer_b = hybrid.producer_identity(
                ledger_sha256="a" * 64,
                chapter_scope_sha256="b" * 64,
                selected_bundle_sha256="c" * 64,
                plan_fingerprint="d" * 64,
                model_producer=whole_book.hybrid_chapter_producer_material(),
            )
            assert producer_b != producer_a
            assert material_a != scripting._cached_chapter_tokenizer_material()
        request_a, key_a = request_b, key_b

    (snapshot / "tokenizer.json").write_bytes(b"wrong pinned tokenizer")
    assert scripting._cached_chapter_tokenizer_material()["status"] == "TOKENIZER_JSON_UNVERIFIED"
    with pytest.raises(scripting.PinnedQwenSnapshotError, match="tokenizer JSON"):
        scripting._chapter_prompt_encoding(request_a["messages"])


def test_effective_public_tokenizer_provenance_and_input_guards_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No weights: actual provenance rejects missing, altered and oversized files."""
    from clipper.books import constrained

    snapshot = (
        tmp_path / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots" / scripting.SCRIPT_REVISION
    )
    snapshot.mkdir(parents=True)
    config = snapshot / "tokenizer_config.json"
    token_file = snapshot / "tokenizer.json"
    token_file.write_bytes(b"public-synthetic-tokenizer")
    config.write_bytes(b'{"chat_template":"synthetic valid"}')
    monkeypatch.setattr(
        constrained, "PINNED_TOKENIZER_SHA256", hashlib.sha256(token_file.read_bytes()).hexdigest()
    )

    class FakeTokenizer:
        name_or_path = str(snapshot)
        chat_template: object = "synthetic valid"

        def __init__(self) -> None:
            self.ids: object = [13, 14, 15]

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> object:
            return self.ids

    fake = FakeTokenizer()
    messages = [{"role": "user", "content": "synthetic, no book contents"}]
    original = scripting._chapter_prompt_encoding(messages, tokenizer=fake)
    assert original["token_ids"] == [13, 14, 15]
    assert scripting._chapter_tokenizer_material(fake)["tokenizer_config_sha256"]

    fake.name_or_path = str(tmp_path)
    with pytest.raises(scripting.PinnedQwenSnapshotError, match="snapshot identity"):
        scripting._chapter_tokenizer_material(fake)
    fake.name_or_path = str(snapshot)
    config.unlink()
    with pytest.raises(scripting.PinnedQwenSnapshotError, match="provenance unavailable"):
        scripting._chapter_tokenizer_material(fake)
    config.write_bytes(b"")
    with pytest.raises(scripting.PinnedQwenSnapshotError, match="config absent or oversized"):
        scripting._chapter_tokenizer_material(fake)
    config.write_bytes(b"{" + b"x" * 1_000_001 + b"}")
    with pytest.raises(scripting.PinnedQwenSnapshotError, match="config absent or oversized"):
        scripting._chapter_tokenizer_material(fake)
    config.write_bytes(b'{"chat_template":"synthetic valid"}')
    token_file.write_bytes(b"wrong token bytes")
    with pytest.raises(scripting.PinnedQwenSnapshotError, match="tokenizer JSON"):
        scripting._chapter_tokenizer_material(fake)
    token_file.write_bytes(b"public-synthetic-tokenizer")
    fake.chat_template = None
    with pytest.raises(scripting.PinnedQwenSnapshotError, match="chat template absent"):
        scripting._chapter_tokenizer_material(fake)
    fake.chat_template = "synthetic valid"

    for ids in (None, [], [0, -2], [True, 2], [1] * (scripting.MAX_CHAPTER_CONTEXT + 1)):
        fake.ids = ids
        with pytest.raises(scripting.PinnedQwenSnapshotError, match="rendered input"):
            scripting._chapter_prompt_encoding(messages, tokenizer=fake)
    fake.ids = [13, 14, 15]
    for ids in ([], [False], ["invalid"], [-1]):
        with pytest.raises(scripting.PinnedQwenSnapshotError, match="verified token"):
            scripting._chapter_encoding_receipt({"token_ids": ids})
    assert scripting._chapter_encoding_receipt(original)["input_token_ids_sha256"]


def test_effective_tokenizer_cache_status_and_cpu_context_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify missing/mutated public tokenizer config cannot masquerade as pinned."""
    from clipper.books import constrained

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert scripting._cached_chapter_tokenizer_material()["status"] == "TOKENIZER_NOT_CACHED"
    snapshot = (
        tmp_path / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots" / scripting.SCRIPT_REVISION
    )
    snapshot.mkdir(parents=True)
    config = snapshot / "tokenizer_config.json"
    tokenizer = snapshot / "tokenizer.json"
    tokenizer.write_bytes(b"synthetic-public")
    config.write_bytes(b"")
    assert scripting._cached_chapter_tokenizer_material()["status"] == "TOKENIZER_CONFIG_UNVERIFIED"
    config.write_bytes(b'{"chat_template":"pinned"}')
    assert scripting._cached_chapter_tokenizer_material()["status"] == "TOKENIZER_JSON_UNVERIFIED"
    monkeypatch.setattr(
        constrained, "PINNED_TOKENIZER_SHA256", hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    )
    assert scripting._cached_chapter_tokenizer_material()["status"] == "PINNED_PUBLIC_TOKENIZER"
    read = Path.read_bytes

    def unreadable(path: Path) -> bytes:
        if path == config:
            raise OSError("private synthetic permission error")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    assert scripting._cached_chapter_tokenizer_material()["status"] == "TOKENIZER_CONFIG_UNVERIFIED"
    monkeypatch.setattr(Path, "read_bytes", read)

    class FakePublicTokenizer:
        def __init__(self) -> None:
            self.ids: list[int] = [7]

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> list[int]:
            return self.ids

    public = FakePublicTokenizer()
    monkeypatch.setattr(scripting, "_public_book_context", lambda: (public, 1500))
    request = {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 1024,
        "messages": [{"role": "user", "content": "synthetic only"}],
    }
    assert scripting._chapter_context_tokens(request) == 1
    public.ids = []
    with pytest.raises(scripting.NicheVideoError, match="token context"):
        scripting._chapter_context_tokens(request)
    public.ids = [8] * 600
    with pytest.raises(scripting.NicheVideoError, match="token context"):
        scripting._chapter_context_tokens(request)
    with pytest.raises(scripting.NicheVideoError, match="output budget mismatch"):
        scripting._chapter_context_tokens({**request, "max_new_tokens": 16385})
    request["verified_input_encoding"] = {"incorrect": "reserved digest"}
    # An absent actual tokenizer identity fails before CPU invocation or raw claim.
    with pytest.raises(scripting.NicheVideoError, match="snapshot identity"):
        scripting._chapter_context_tokens(request)
    request.pop("verified_input_encoding")
    fake_encoder = Mock(side_effect=ValueError("model-free synthetic tokenizer malfunction"))
    monkeypatch.setattr(public, "apply_chat_template", fake_encoder)
    with pytest.raises(scripting.NicheVideoError, match="pinned tokenizer preflight failed"):
        scripting._chapter_context_tokens(request)
    fake_encoder.assert_called_once()


@pytest.mark.parametrize(
    "bad_encoding",
    [
        {"token_ids": "not-a-list"},
        {"token_ids": []},
        {"token_ids": [-1]},
        {"token_ids": [True]},
        {"token_ids": [1], "chat_template_sha256": "g" * 64},
        {"token_ids": [1], "tokenizer_config_sha256": "1" * 63},
        {"token_ids": [1], "tokenizer_json_sha256": "0" * 64},
    ],
)
def test_hybrid_request_rejects_unverifiable_effective_input_tokens(
    bad_encoding: dict[str, object],
) -> None:
    good: dict[str, object] = {
        "token_ids": [1, 2],
        "tokenizer_config_sha256": "a" * 64,
        "chat_template_sha256": "b" * 64,
        "tokenizer_json_sha256": hybrid.constrained.PINNED_TOKENIZER_SHA256,
    }
    candidate = {**good, **bad_encoding}
    with pytest.raises(hybrid.HybridStop, match="effective token encoding"):
        hybrid.request(
            "argument",
            {"source": hybrid.projection(hybrid.validate_ledger(_ledger(1)))},
            output_tokens=1024,
            token_count=lambda _messages: candidate,
            producer_identity={"schema": "synthetic-producer"},
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
    ledger = _ledger(25)
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


def test_direct_hybrid_mode_freezes_source_without_mining_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    miner = Mock(side_effect=AssertionError("source-driven chapter never mines lessons"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", miner)
    result = chapter.evaluate_hybrid_chapter(runner, selected, section)
    assert result["status"] == "AWAITING_EXPLICIT_CPU_HYBRID_CHAPTER_ARGUMENT"
    assert result["mining_provenance"] == "DIRECT_SOURCE_NO_MINING"
    assert result["parent_original_reverified_on_host"] is False
    miner.assert_not_called()
    base = runner.root / "book-candidates" / "chapter-evaluations"
    stored = list(base.glob("*/source-ledger.json"))
    assert len(stored) == 1
    ledger = json.loads(stored[0].read_bytes())
    assert len(hybrid.validate_ledger(ledger)) == (
        result["included_units"] + result["skipped_units"]
    )
    assert all(item["source_id"].startswith(("unit-", "gap-")) for item in ledger["entries"])


def test_model_free_hybrid_argument_audits_reconciliation_review_gate_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    miner = Mock(side_effect=AssertionError("never mine in direct chapter mode"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", miner)
    call_counts = {"argument": 0, "audit": 0, "reconcile": 0}

    def completion(kind: str):
        def run(request: dict[str, object]) -> str:
            call_counts[kind] += 1
            return _injected_hybrid_response(kind, request)

        return run

    plan = chapter.ChapterPlan(brief_cpu=True)
    args = {
        "plan": plan,
        "argument_completion": completion("argument"),
        "audit_completion": completion("audit"),
        "reconcile_completion": completion("reconcile"),
        "token_counter": lambda _messages: 1000,
    }
    pending = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    assert pending["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert pending["argument_attempts"] == 1
    assert pending["brief_attempts"] == 1
    assert pending["audit_attempts"] >= 1
    miner.assert_not_called()
    assert pending["automation_claim"] == "INJECTED_TEST"
    assert not pending["publish_eligible"]
    root = runner.root / "book-candidates"
    ledger_path = next((root / "chapter-evaluations").glob("*/source-ledger.json"))
    ledger = json.loads(ledger_path.read_bytes())
    brief_path = next((root / "hybrid_reconcile").glob("*/validated.json"))
    brief = json.loads(brief_path.read_bytes())["payload"]
    assert len(brief["coverage"]) == len(ledger["entries"])
    assert len(brief["beats"]) == 2
    packet = hybrid.human_review_packet(brief, ledger)
    assert packet["review_status"] == "PENDING_ACTUAL_HUMAN_ATTESTATION"

    frozen = dict(call_counts)
    replayed = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        expected_brief_identity=pending["brief_identity"],
        **args,
    )
    assert replayed["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert call_counts == frozen
    review = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "brief_artifact_identity": pending["brief_identity"],
        "reviewer": "Synthetic test human (not production approval)",
        "unit_decisions": [
            {"id": item["id"], "review": "accept", "reason": "Compared source and gap"}
            for item in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": item["beat_id"],
                "review": "support_attested",
                "reason": "Synthetic test meaning and qualifier check",
            }
            for item in brief["beats"]
        ],
    }
    review_path = tmp_path / "synthetic-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    after_review = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=review_path,
        expected_brief_identity=pending["brief_identity"],
        **args,
    )
    assert after_review["status"] == "AWAITING_EXPLICIT_CPU_CHAPTER_ADAPTATION"
    assert after_review["human_attested_beat_count"] == 2
    assert call_counts == frozen
    miner.assert_not_called()


@pytest.mark.parametrize("script_only,configured", [(False, False), (True, False), (True, True)])
def test_full_hybrid_model_free_creative_path_keeps_spans_and_review_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script_only: bool, configured: bool
) -> None:
    from clipper.books import pipeline_plan

    runner, selected, section = _selected_hybrid_source(tmp_path)
    fake_miner = Mock(side_effect=AssertionError("no mandatory per-unit mining"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", fake_miner)
    plan = chapter.ChapterPlan(
        brief_cpu=True,
        adaptation_cpu=True,
        script_cpu=True,
        direction_cpu=not script_only,
        kokoro_cpu=not script_only,
    )
    calls: list[str] = []

    def producer(kind: str):
        def run(req: dict[str, object]) -> str:
            calls.append(kind)
            if kind in {"argument", "audit", "reconcile"}:
                return _injected_hybrid_response(kind, req)
            return _hybrid_creative_output(kind, req)

        return run

    args = {
        "plan": plan,
        "argument_completion": producer("argument"),
        "audit_completion": producer("audit"),
        "reconcile_completion": producer("reconcile"),
        "adaptation_completion": producer("adaptation"),
        "script_completion": producer("chapter_script"),
        "direction_completion": None if script_only else producer("chapter_direction"),
        "narration_producer": None if script_only else _hybrid_wav,
        "script_only": script_only and not configured,
        "token_counter": lambda _messages: 1200,
    }
    if configured:
        args["pipeline_config"] = pipeline_plan.template(selected.stem, "brief")
    waiting = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    assert waiting["status"] == (
        "BRIEF_READY_UNREVIEWED" if configured else "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    )
    if configured:
        assert waiting["configured_stage_results"]["human_review"] == "REQUIRED_NEXT"
        assert waiting["configured_execution_receipt_identity"]
    base = runner.root / "book-candidates"
    ledger = json.loads(
        next((base / "chapter-evaluations").glob("*/source-ledger.json")).read_bytes()
    )
    brief = json.loads(next((base / "hybrid_reconcile").glob("*/validated.json")).read_bytes())[
        "payload"
    ]
    review = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "brief_artifact_identity": waiting["brief_identity"],
        "reviewer": "Synthetic test human",
        "unit_decisions": [
            {"id": entry["id"], "review": "accept", "reason": "Original source considered"}
            for entry in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": beat["beat_id"],
                "review": "support_attested",
                "reason": "Synthetic source-support check only",
            }
            for beat in brief["beats"]
        ],
    }
    review_file = tmp_path / "hybrid-human-test.json"
    review_file.write_text(json.dumps(review), encoding="utf-8")
    downstream = Mock(side_effect=AssertionError("script-only downstream work"))
    if script_only:
        monkeypatch.setattr(chapter, "_timings", downstream)
        monkeypatch.setattr(chapter, "_narrate", downstream)
    if configured:
        config = pipeline_plan.template(selected.stem, "script")
        for name in ("argument", "audits", "reconcile", "human_review"):
            config["stages"][name]["enabled"] = False
        config["stages"]["narration"]["semantic"]["voice"] = "af_bella"
        args["pipeline_config"] = config
        args["plan"] = chapter.ChapterPlan(adaptation_cpu=True, script_cpu=True, voice="af_bella")
    checkpoint = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=review_file,
        expected_brief_identity=waiting["brief_identity"],
        **args,
    )
    if configured:
        assert checkpoint["chapter_scope_identity"] == waiting["chapter_scope_identity"]
        assert checkpoint["brief_identity"] == waiting["brief_identity"]
        results = checkpoint["configured_stage_results"]
        assert results["argument"] == "REUSED"
        assert results["audits"] == "REUSED"
        assert results["reconcile"] == "REUSED"
        assert results["adaptation"] == "RAN"
        assert results["script"] == "RAN"
        assert results["narration"] == "DISABLED"
        assert results["direction"] == "DISABLED"
        for name in ("adaptation", "script"):
            config["stages"][name]["enabled"] = False
        args["plan"] = chapter.ChapterPlan()
        before = list(calls)
        cached = chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            review_file=review_file,
            expected_brief_identity=waiting["brief_identity"],
            **args,
        )
        assert cached["final_result_identity"] == checkpoint["final_result_identity"]
        assert cached["configured_stage_results"]["script"] == "REUSED"
        assert (
            cached["configured_execution_receipt_identity"]
            != (checkpoint["configured_execution_receipt_identity"])
        )
        receipt_key = cached["configured_execution_receipt_identity"]
        receipt_path = (
            base
            / "chapter-evaluations"
            / cached["chapter_scope_identity"]
            / f"configured-execution-{receipt_key}.json"
        )
        receipt = json.loads(receipt_path.read_bytes())
        assert chapter.content_fingerprint(receipt) == receipt_key
        assert receipt["stage_results"]["script"] == "REUSED"
        assert receipt["configuration_sha256"] == chapter.content_fingerprint(config)
        assert receipt["configured_lineage"] == pipeline_plan.semantic_lineage(
            config, pipeline_plan.registry()
        )
        assert receipt["source_policy"] == "STRICT_SELECTED_INPUT_ONLY_NO_ACQUISITION"
        assert receipt["artifacts"]["final_result_identity"] == cached["final_result_identity"]
        assert calls == before
    if script_only:
        assert checkpoint["status"] == "TEST_DOUBLE_CHAPTER_SCRIPT_READY_UNREVIEWED"
        assert checkpoint["script_review"] == "UNREVIEWED"
        assert checkpoint["semantic_intervention"] == "HUMAN_ATTESTED_NOT_INDEPENDENTLY_PROVEN"
        assert checkpoint["final_result_identity"]
        assert "narration_report_identity" not in checkpoint
        assert "direction_identity" not in checkpoint
        assert calls[-2:] == ["adaptation", "chapter_script"]
        downstream.assert_not_called()
    else:
        assert checkpoint["status"] == "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
        assert checkpoint["direction_attempts"] == 1
        assert checkpoint["narration_report_identity"]
    assert checkpoint["script_scene_count"] == 3
    script_path = next((base / "chapter_script").glob("*/validated.json"))
    script = json.loads(script_path.read_bytes())["payload"]
    assert all(scene["source_spans"] for scene in script["scenes"])
    assert all(scene["reviewed_qualifications"] for scene in script["scenes"])
    assert all(
        scene["semantic_review"] == "NO_AUTOMATIC_SEMANTIC_CERTIFICATION"
        for scene in script["scenes"]
    )
    assert len([name for name in calls if name == "argument"]) == 1
    assert len([name for name in calls if name == "reconcile"]) == 1
    fake_miner.assert_not_called()
    assert not checkpoint["publish_eligible"]


def test_hybrid_reconciliation_raw_tampering_refuses_fresh_replay(
    tmp_path: Path,
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    args = {
        "plan": chapter.ChapterPlan(brief_cpu=True),
        "argument_completion": lambda req: _injected_hybrid_response("argument", req),
        "audit_completion": lambda req: _injected_hybrid_response("audit", req),
        "reconcile_completion": lambda req: _injected_hybrid_response("reconcile", req),
        "token_counter": lambda _messages: 1000,
    }
    waiting = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    root = runner.root / "book-candidates" / "hybrid_reconcile"
    record_path = next(root.glob("*/raw-1.json"))
    raw_record = json.loads(record_path.read_bytes())
    raw_record["raw"] = raw_record["raw"].replace("revisable", "altered")
    record_path.write_text(json.dumps(raw_record), encoding="utf-8")
    with pytest.raises(chapter.ChapterStop, match="cannot be verified"):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            expected_brief_identity=waiting["brief_identity"],
            **args,
        )


def test_export_import_101_plus_unresolved_gap_preserves_original_scoped_source(
    tmp_path: Path,
) -> None:
    original = tmp_path / "synthetic-source"
    original.mkdir()
    runner, acquired, _transport = ingested(original, paragraphs=300, with_image=True)
    section_id = acquired.book.sections[1].chapter_id
    expected = whole_book.plan_for(acquired.book, section_ids=(section_id,))
    assert len(expected.units) == 101
    assert len(expected.skipped) == 1
    selected_path, selected_sha = selected_scope.export_chapter(
        runner,
        acquired.analysis_ref_path,
        chapter_number=1,
        chapter_section_ids=(section_id,),
    )
    assert selected_sha == hashlib.sha256(selected_path.read_bytes()).hexdigest()
    host_root = tmp_path / "synthetic-host"
    copied = host_root / "book-candidates" / "selected-scopes" / f"{selected_sha}.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(selected_path.read_bytes())
    verified = selected_scope.import_chapter(host_root, copied)
    assert verified.work == expected
    units, gaps, ordered = chapter._full_source_scope(verified.book, verified.work, None, [])
    assert (len(units), len(gaps), len(ordered)) == (101, 1, 102)
    ledger = hybrid.freeze(
        units,
        gaps,
        ordered,
        {"chapter": 1, "selected_bundle_sha256": selected_sha},
    )
    assert len(hybrid.validate_ledger(ledger)) == 102
    assert [item["source_id"] for item in ledger["entries"]] == ordered
    assert all(item["kind"] == "CANONICAL_WORK_UNIT" for item in ledger["entries"][:-1])
    assert (
        sum(item["kind"] == "UNRECOVERED_OR_EMPTY_SOURCE_MARKER" for item in ledger["entries"]) == 1
    )
    assert not list(host_root.rglob("*.epub"))
    assert not list(host_root.rglob("analysis-refs"))


def test_selected_chapter_rejects_overcap_export_and_rehashed_reading_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = tmp_path / "overcap-source"
    owner.mkdir()
    runner, acquired, _transport = ingested(owner, paragraphs=384, with_image=True)
    section_id = acquired.book.sections[1].chapter_id
    plan = whole_book.plan_for(acquired.book, section_ids=(section_id,))
    assert len(plan.units) + len(plan.skipped) > selected_scope.MAX_CHAPTER_ENTRIES
    with pytest.raises(selected_scope.SelectedSourceStop, match="original-unit budget"):
        selected_scope.export_chapter(
            runner,
            acquired.analysis_ref_path,
            chapter_number=1,
            chapter_section_ids=(section_id,),
        )

    host, valid_path, data = _bundle(tmp_path)
    baseline = selected_scope.import_chapter(host.root, valid_path)
    assert baseline.book.selected_reading_blocks
    base = data["reading_blocks"]
    assert 0 < len(base) < selected_scope.MAX_CHAPTER_ENTRIES
    count = selected_scope.MAX_CHAPTER_ENTRIES + 1
    forged = {**data, "reading_blocks": (base * count)[:count]}
    assert len(forged["reading_blocks"]) == 129
    encoded = selected_scope._json(forged)
    digest = hashlib.sha256(encoded).hexdigest()
    path = host.root / "book-candidates" / "selected-scopes" / f"{digest}.json"
    path.write_bytes(encoded)

    # The public exception must stay sanitized. Inspect the exception event in
    # the original importer frame to prove WHICH guard refused the validly
    # rehashed 129-block bundle, before any reading-block consumer constructs it.
    raised_inside: list[str] = []

    def guard_trace(frame: object, event: str, arg: object) -> object:
        if (
            event == "exception"
            and getattr(frame, "f_code", None) is selected_scope.import_chapter.__code__
            and isinstance(arg, tuple)
            and arg[0] is selected_scope.SelectedSourceStop
        ):
            raised_inside.append(str(arg[1]))
        return guard_trace

    constructor = Mock(side_effect=AssertionError("never construct over-cap original blocks"))
    previous = sys.gettrace()
    try:
        with monkeypatch.context() as patched:
            patched.setattr(selected_scope, "ReadingBlock", constructor)
            sys.settrace(guard_trace)
            with pytest.raises(
                selected_scope.SelectedSourceStop,
                match=r"^selected chapter evidence cannot be verified$",
            ):
                selected_scope.import_chapter(host.root, path)
    finally:
        sys.settrace(previous)
    assert raised_inside[0] == "selected original reading blocks absent/oversized"
    assert raised_inside[-1] == "selected chapter evidence cannot be verified"
    constructor.assert_not_called()


def test_45_units_plus_gap_shards_have_single_core_owner() -> None:
    ledger = _ledger(45)
    windows = hybrid.audit_windows(ledger)
    assert len(ledger["entries"]) == 46
    assert len(windows) == 4
    cores = [sid for window in windows for sid in window["core_ids"]]
    assert cores == [item["id"] for item in ledger["entries"]]
    assert len(cores) == len(set(cores))
    for window in windows:
        assert not set(window["core_ids"]) & {item[0] for item in window["halo_context_only"]}


def test_runtime_only_change_invalidates_actual_hybrid_stage_artifacts_and_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    original = whole_book.local_producer_material()
    runtime = {"probe": "runtime-A"}
    encoding = {
        "token_ids": [11] * 1200,
        "tokenizer_config_sha256": "c" * 64,
        "chat_template_sha256": "d" * 64,
        "tokenizer_json_sha256": hybrid.constrained.PINNED_TOKENIZER_SHA256,
    }
    monkeypatch.setattr(
        whole_book,
        "local_producer_material",
        lambda: {**original, "synthetic_runtime_probe": runtime["probe"]},
    )
    calls = {
        name: 0
        for name in (
            "argument",
            "audit",
            "reconcile",
            "adaptation",
            "chapter_script",
            "chapter_direction",
        )
    }

    def completion(name: str):
        def execute(req: dict[str, object]) -> str:
            calls[name] += 1
            if name in {"argument", "audit", "reconcile"}:
                return _injected_hybrid_response(name, req)
            return _hybrid_creative_output(name, req)

        return execute

    args = {
        "plan": chapter.ChapterPlan(
            brief_cpu=True,
            adaptation_cpu=True,
            script_cpu=True,
            direction_cpu=True,
            kokoro_cpu=True,
        ),
        "argument_completion": completion("argument"),
        "audit_completion": completion("audit"),
        "reconcile_completion": completion("reconcile"),
        "adaptation_completion": completion("adaptation"),
        "script_completion": completion("chapter_script"),
        "direction_completion": completion("chapter_direction"),
        "narration_producer": _hybrid_wav,
        "token_counter": lambda _messages: {**encoding, "token_ids": list(encoding["token_ids"])},
    }
    base = runner.root / "book-candidates"
    ledger: dict[str, object] | None = None

    def approve(pending: dict[str, object], label: str) -> Path:
        nonlocal ledger
        if ledger is None:
            ledger = json.loads(
                next((base / "chapter-evaluations").glob("*/source-ledger.json")).read_bytes()
            )
        candidates = [
            json.loads(path.read_bytes())
            for path in (base / "hybrid_reconcile").glob("*/validated.json")
        ]
        matches = [
            candidate
            for candidate in candidates
            if candidate["artifact_identity"] == pending["brief_identity"]
        ]
        assert len(matches) == 1
        brief = matches[0]["payload"]
        review = {
            "schema": hybrid.REVIEW_SCHEMA,
            "brief_sha256": brief["brief_sha256"],
            "brief_artifact_identity": pending["brief_identity"],
            "reviewer": "Synthetic human for isolated cache test",
            "unit_decisions": [
                {"id": unit["id"], "review": "accept", "reason": "Source checked by fixture"}
                for unit in ledger["entries"]
            ],
            "beat_decisions": [
                {
                    "beat_id": beat["beat_id"],
                    "review": "support_attested",
                    "reason": "Explicit synthetic approval of context",
                }
                for beat in brief["beats"]
            ],
        }
        review_path = tmp_path / f"review-{label}.json"
        review_path.write_text(json.dumps(review), encoding="utf-8")
        return review_path

    pending_a = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    approval_a = approve(pending_a, "runtime-a")
    result_a = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=approval_a,
        expected_brief_identity=pending_a["brief_identity"],
        **args,
    )
    assert result_a["status"] == "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
    frozen = dict(calls)
    repeated_a = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=approval_a,
        expected_brief_identity=pending_a["brief_identity"],
        **args,
    )
    assert repeated_a["final_result_identity"] == result_a["final_result_identity"]
    assert calls == frozen, "same runtime must recover exact cached raw stages without calls"

    runtime["probe"] = "runtime-B"
    with pytest.raises(chapter.ChapterStop, match="cache missing or changed"):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            review_file=approval_a,
            expected_brief_identity=pending_a["brief_identity"],
            **args,
        )
    assert calls == frozen, "old approved identity must fail before a new producer call"
    pending_b = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    assert pending_b["producer_identity_sha256"] != pending_a["producer_identity_sha256"]
    for key in ("argument_identity", "brief_identity"):
        assert pending_b[key] != pending_a[key]
    assert pending_b["audit_identities"] != pending_a["audit_identities"]
    assert calls["argument"] == frozen["argument"] + 1
    assert calls["reconcile"] == frozen["reconcile"] + 1
    assert calls["audit"] > frozen["audit"]

    with pytest.raises(chapter.ChapterReviewPending):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            review_file=approval_a,
            expected_brief_identity=pending_b["brief_identity"],
            **args,
        )
    approval_b = approve(pending_b, "runtime-b")
    result_b = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=approval_b,
        expected_brief_identity=pending_b["brief_identity"],
        **args,
    )
    for key in ("adaptation_identity", "script_identity", "direction_identity"):
        assert result_b[key] != result_a[key]
    assert result_b["final_result_identity"] != result_a["final_result_identity"]
    for name in ("adaptation", "chapter_script", "chapter_direction"):
        assert calls[name] == frozen[name] + 1
    source_ledger = next((base / "chapter-evaluations").glob("*/source-ledger.json"))
    source = json.loads(source_ledger.read_bytes())
    assert (
        chapter_host._hybrid_producer_sha256(
            source,
            pending_b["section_scope_sha256"],
            pending_b["selected_source_bundle_sha256"],
            args["plan"],
        )
        == pending_b["producer_identity_sha256"]
    )
    runtime["probe"] = "runtime-A"
    assert (
        chapter_host._hybrid_producer_sha256(
            source,
            pending_a["section_scope_sha256"],
            pending_a["selected_source_bundle_sha256"],
            args["plan"],
        )
        == pending_a["producer_identity_sha256"]
    )

    # Same effective encoding is a zero-producer-call replay. Neither the same
    # count with different IDs nor a changed effective template/config can
    # authorize the prior raw, approval, or post-review creative artifacts.
    before_encoding_change = dict(calls)
    same_encoding = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=approval_a,
        expected_brief_identity=pending_a["brief_identity"],
        **args,
    )
    assert same_encoding["final_result_identity"] == result_a["final_result_identity"]
    assert calls == before_encoding_change
    former = pending_a
    for mutated_field, mutated_value in (
        ("token_ids", [12] * 1200),
        ("chat_template_sha256", "e" * 64),
        ("tokenizer_config_sha256", "f" * 64),
    ):
        encoding[mutated_field] = mutated_value
        with pytest.raises(chapter.ChapterStop, match="cache missing or changed"):
            chapter.evaluate_hybrid_chapter(
                runner,
                selected,
                section,
                review_file=approval_a,
                expected_brief_identity=former["brief_identity"],
                **args,
            )
        assert calls == before_encoding_change
        pending_next = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
        assert pending_next["brief_identity"] != former["brief_identity"]
        assert pending_next["argument_identity"] != former["argument_identity"]
        assert pending_next["audit_identities"] != former["audit_identities"]
        former = pending_next
        before_encoding_change = dict(calls)


def test_hybrid_private_host_journal_recovery_is_runtime_bound_and_mining_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, manifest, inputs = _private_setup(tmp_path, monkeypatch)
    _locked(runner, manifest, inputs)
    original = whole_book.local_producer_material()
    runtime = {"probe": "A"}
    monkeypatch.setattr(
        whole_book,
        "local_producer_material",
        lambda: {**original, "synthetic_probe": runtime["probe"]},
    )
    miner = Mock(side_effect=AssertionError("direct source must not call the miner"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", miner)
    evaluate = chapter.evaluate_hybrid_chapter
    calls = {"argument": 0, "audit": 0, "reconcile": 0}

    def fake(name: str):
        def call(request: dict[str, object]) -> str:
            calls[name] += 1
            return _injected_hybrid_response(name, request)

        return call

    def injected(*args: object, **kwargs: object) -> dict[str, object]:
        kwargs.update(
            argument_completion=fake("argument"),
            audit_completion=fake("audit"),
            reconcile_completion=fake("reconcile"),
            adaptation_completion=lambda req: _hybrid_creative_output("adaptation", req),
            script_completion=lambda req: _hybrid_creative_output("chapter_script", req),
            direction_completion=lambda req: _hybrid_creative_output("chapter_direction", req),
            narration_producer=_hybrid_wav,
            token_counter=lambda _messages: 1000,
        )
        outcome = evaluate(*args, **kwargs)
        if kwargs.get("review_file") is not None:
            assert outcome["status"] == "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
            return {**outcome, "status": "CHAPTER_PREPRODUCTION_READY_UNREVIEWED"}
        return outcome

    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", injected)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    report = _call(runner, manifest, "hybrid-brief")
    assert report["status"] == "PRIVATE_TWO_CHAPTER_HYBRID_CHECKPOINT"
    assert set(report["chapters"]) == {"1", "7"}
    assert all(
        record["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
        for record in report["chapters"].values()
    )
    assert calls["argument"] == 2
    assert calls["reconcile"] == 2
    assert calls["audit"] >= 2
    miner.assert_not_called()

    for number in (1, 7):
        block = inputs[f"chapter_{number}"]
        inventory = chapter_host._hybrid_output_manifest(
            runner.root,
            number,
            block["bundle_sha256"],
            block["scope_sha256"],
        )
        assert inventory["source_mode"] == "DIRECT_SOURCE_HYBRID"
        assert inventory["publish_eligible"] is False
        assert inventory["verified_artifact_identities"]["producer_identity_sha256"]
        kinds = {record["kind"] for record in inventory["files"]}
        assert "source_ledger" in kinds
        assert "text" in kinds
        assert "review" in kinds
    frozen = dict(calls)
    awaiting_review = _call(runner, manifest, "hybrid-resume")
    assert all(
        outcome["review_status"] == "REVIEW_REQUIRED"
        for outcome in awaiting_review["chapters"].values()
    )
    assert calls == frozen, "resume without a semantic review must not call any producer"
    runtime["probe"] = "B"
    for number in (1, 7):
        block = inputs[f"chapter_{number}"]
        with pytest.raises(chapter_host.PrivateHostStop, match="producer/runtime"):
            chapter_host._hybrid_output_manifest(
                runner.root,
                number,
                block["bundle_sha256"],
                block["scope_sha256"],
            )
    blocked = _call(runner, manifest, "hybrid-resume")
    assert all(
        record["review_status"] == "RESUME_PREFLIGHT_BLOCKED"
        for record in blocked["chapters"].values()
    )
    assert calls == frozen
    miner.assert_not_called()

    # Explicit synthetic human review of both complete, independently frozen briefs.
    runtime["probe"] = "A"
    private = runner.root / "book-candidates"
    for number in (1, 7):
        packet_path = private / "private-host" / f"chapter-{number}-review-packet.json"
        packet = json.loads(packet_path.read_bytes())
        approval = packet["review_template"]
        approval["reviewer"] = "Synthetic human (test fixture only)"
        for unit in approval["unit_decisions"]:
            unit.update(review="accept", reason="Inspected original synthetic unit and gap")
        for beat in approval["beat_decisions"]:
            beat.update(
                review="support_attested",
                reason="Synthetic meaning and qualification review",
            )
        review_path = private / "private-host" / f"chapter-{number}-synthetic-approval.json"
        evaluation._store(private, review_path, approval)
        inputs[f"chapter_{number}"]["review_file"] = str(review_path)
    manifest.write_bytes(evaluation._json_bytes(inputs))
    resumed = _call(runner, manifest, "hybrid-resume")
    assert all(
        chapter_result["status"] == "CHAPTER_PREPRODUCTION_READY_UNREVIEWED"
        for chapter_result in resumed["chapters"].values()
    )
    assert calls == frozen, "unchanged producer must reuse argument/audit/reconciliation"
    for number in (1, 7):
        block = inputs[f"chapter_{number}"]
        inventory = chapter_host._hybrid_output_manifest(
            runner.root,
            number,
            block["bundle_sha256"],
            block["scope_sha256"],
        )
        assert inventory["source_mode"] == "DIRECT_SOURCE_HYBRID"
        kinds = {file["kind"] for file in inventory["files"]}
        assert {"human_review", "final_result", "wav", "source_ledger", "text"} <= kinds
    recovery = chapter_host.recovery_report(runner.root)
    assert {
        chapter_result["recovery_state"] for chapter_result in recovery["chapters"].values()
    } == {"READY"}
    archive = chapter_host.private_recovery_archive(runner.root)
    assert archive["file_count"] >= 20
    assert archive["files_retrieved"] is False
    assert archive["host_cleanup_verified"] is False

    # ACTUAL OWNER CONSUMER: feed both fully host-produced hybrid chapter
    # archives (including synthetic WAV) through the unchanged Python verifier
    # embedded in the owner watchdog, not a surrogate manifest-only validator.
    owner_script = (
        Path(__file__).resolve().parents[1] / "scripts/private_chapter_external_supervisor.sh"
    )
    owner_shell = owner_script.read_text(encoding="utf-8")
    owner_verifier = owner_shell.split("if critical archive_verification python3 -c '\n", 1)[
        1
    ].split('\n\' "$session/recovery-created.json"', 1)[0]
    archive_path = private / "private-host/chapter-recovery.tar"
    with tarfile.open(archive_path, "r:") as recovered_tar:
        original_members = {
            member.name: recovered_tar.extractfile(member).read()
            for member in recovered_tar.getmembers()
        }

    def owner_accepts(payloads: dict[str, bytes]) -> tuple[int, dict[str, object] | None]:
        test_archive = tmp_path / "actual-hybrid-owner-verifier.tar"
        with tarfile.open(test_archive, "w") as output_tar:
            for name, payload in payloads.items():
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o600
                output_tar.addfile(member, io.BytesIO(payload))
        receipt = {
            "schema": "clipper-private-chapter-recovery-created-v1",
            "archive_sha256": hashlib.sha256(test_archive.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                payloads["private-recovery-manifest.json"]
            ).hexdigest(),
        }
        receipt_path = tmp_path / "actual-hybrid-owner-receipt.json"
        report_path = tmp_path / "actual-hybrid-owner-result.json"
        status_path = tmp_path / "actual-hybrid-owner-safe.json"
        receipt_path.write_bytes(evaluation._json_bytes(receipt))
        report_path.unlink(missing_ok=True)
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                owner_verifier,
                str(receipt_path),
                str(test_archive),
                str(report_path),
                str(status_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return (
            process.returncode,
            json.loads(report_path.read_bytes()) if report_path.exists() else None,
        )

    status, verified_owner = owner_accepts(original_members)
    assert status == 0, "owner must accept ACTUAL two-chapter hybrid host archive"
    assert verified_owner is not None
    assert verified_owner["chapters"] == {
        "1": "READY_OUTPUTS_VERIFIED",
        "7": "READY_OUTPUTS_VERIFIED",
    }
    assert verified_owner["output_scope"] == "COMPLETE"
    assert verified_owner["recovered_file_count"] == archive["file_count"]
    original_listing = json.loads(original_members["private-recovery-manifest.json"])
    for chapter_number in (1, 7):
        inventory = original_listing["chapters"][chapter_number == 7]
        assert inventory["chapter"] == chapter_number
        assert inventory["source_mode"] == "DIRECT_SOURCE_HYBRID"
        assert {"review_record", "human_review", "final_result", "wav", "source_ledger"} <= {
            item["kind"] for item in inventory["files"]
        }
        reports = [item for item in inventory["files"] if item["kind"] == "narration_report"]
        assert len(reports) == 1
        narration = json.loads(original_members[reports[0]["relative_path"]])
        assert narration["records"]
        for record in narration["records"]:
            wav_path = "book-candidates/" + record["wav_relative_path"]
            assert hashlib.sha256(original_members[wav_path]).hexdigest() == record["sha256"]

    def fully_rebind_metadata(changed: dict[str, bytes]) -> dict[str, bytes]:
        """Rehash hostile TAR/manifest/status/receipt, leaving semantic links untrusted."""
        listing = json.loads(changed["private-recovery-manifest.json"])
        safe = json.loads(changed["safe-recovery-status.json"])
        for row in listing["files"]:
            content = changed[row["relative_path"]]
            row["sha256"] = hashlib.sha256(content).hexdigest()
            row["bytes"] = len(content)
        for scope in listing["chapters"]:
            for row in scope["files"]:
                content = changed[row["relative_path"]]
                row["sha256"] = hashlib.sha256(content).hexdigest()
                row["bytes"] = len(content)
            manuscript = {k: v for k, v in scope.items() if k != "inventory_status"}
            safe["chapters"][str(scope["chapter"])]["output_manifest_sha256"] = hashlib.sha256(
                evaluation._json_bytes(manuscript)
            ).hexdigest()
        changed["safe-recovery-status.json"] = evaluation._json_bytes(safe)
        listing["runtime_status_sha256"] = hashlib.sha256(
            changed["safe-recovery-status.json"]
        ).hexdigest()
        changed["private-recovery-manifest.json"] = evaluation._json_bytes(listing)
        return changed

    missing = dict(original_members)
    missing.pop("book-candidates/private-host/chapter-1-brief-record.json")
    assert owner_accepts(missing)[0] != 0
    for attack in (
        "wrong_mode",
        "review_record",
        "source_ledger",
        "reconcile_ledger",
        "argument_payload",
        "audit_payload",
        "reconcile_payload",
        "adaptation_payload",
        "adaptation_payload_sha256",
        "adaptation_artifact_identity",
        "adaptation_rebound_payload_sha256",
        "chapter_script_payload",
        "chapter_direction_payload",
        "final_result",
        "human_review",
        "wav",
    ):
        forged = dict(original_members)
        listing = json.loads(forged["private-recovery-manifest.json"])
        if attack == "wrong_mode":
            listing["chapters"][0]["source_mode"] = "LEGACY_MINED_CHAPTER"
            forged["private-recovery-manifest.json"] = evaluation._json_bytes(listing)
        else:
            stage_attack = {
                "reconcile_ledger": "hybrid_reconcile",
                "argument_payload": "hybrid_argument",
                "audit_payload": "hybrid_audit",
                "reconcile_payload": "hybrid_reconcile",
                "adaptation_payload": "adaptation",
                "adaptation_payload_sha256": "adaptation",
                "adaptation_artifact_identity": "adaptation",
                "adaptation_rebound_payload_sha256": "adaptation",
                "chapter_script_payload": "chapter_script",
                "chapter_direction_payload": "chapter_direction",
            }.get(attack)
            kind = "text" if stage_attack else attack
            item = next(
                row
                for row in listing["chapters"][0]["files"]
                if row["kind"] == kind
                and (
                    stage_attack is None
                    or (
                        row["relative_path"].startswith(f"book-candidates/{stage_attack}/")
                        and row["relative_path"].endswith("/validated.json")
                    )
                )
            )
            path = item["relative_path"]
            if attack == "wav":
                forged[path] += b"tampered-real-narration-reference"
            else:
                evidence = json.loads(forged[path])
                if attack == "human_review":
                    evidence["review"]["reviewer"] = "Unauthorized archive-side editor"
                elif attack == "source_ledger":
                    evidence["identity"]["selected_bundle_sha256"] = "0" * 64
                elif attack == "reconcile_ledger":
                    evidence["payload"]["ledger_sha256"] = "0" * 64
                elif stage_attack is not None:
                    if attack == "adaptation_payload_sha256":
                        evidence["payload_sha256"] = "0" * 64
                    elif attack == "adaptation_artifact_identity":
                        evidence["artifact_identity"] = "0" * 64
                    else:
                        # Forge actual host-produced, originally validated
                        # chapter text while preserving its terminal identity.
                        evidence["payload"]["unauthorized_editorial_change"] = attack
                        if attack == "adaptation_rebound_payload_sha256":
                            # Even rehashing the inner payload is insufficient:
                            # exact _text_artifact_identity must also match
                            # the committed downstream final result.
                            evidence["payload_sha256"] = hashlib.sha256(
                                evaluation._json_bytes(evidence["payload"])
                            ).hexdigest()
                else:
                    evidence["producer_identity_sha256"] = "0" * 64
                forged[path] = evaluation._json_bytes(evidence)
        fully_rebind_metadata(forged)
        exit_code, rejected = owner_accepts(forged)
        # The verifier intentionally preserves the other independently
        # recoverable chapter: per-chapter INCOMPLETE with a PARTIAL report is
        # fail-closed, even when the outer TAR is checksum-valid (exit 0).
        assert exit_code == 0 and rejected is not None, attack
        assert rejected["chapters"] == {
            "1": "INCOMPLETE",
            # Synthetic WAV assets can be deduplicated between chapters.
            # Tampering a shared RIFF asset must invalidate BOTH consumers.
            "7": "INCOMPLETE" if attack == "wav" else "READY_OUTPUTS_VERIFIED",
        }, attack
        assert rejected["output_scope"] == "PARTIAL", attack
        assert rejected["retrieval_status"] == "VERIFIED_PRIVATE_ARCHIVE_PARTIAL"

    # Corrupted plan records must not become recoverable post-review phases,
    # even if a caller presents the historical valid stage identifiers.
    journal = private / "private-host" / "chapter-1-brief-record.json"
    journal_bytes = journal.read_bytes()
    for field in ("plan", "plan_fingerprint"):
        mutated = json.loads(journal_bytes)
        if field == "plan":
            mutated["plan"]["max_attempts_per_stage"] = True
        else:
            mutated["plan_fingerprint"] = "0" * 64
        try:
            journal.write_bytes(evaluation._json_bytes(mutated))
            with pytest.raises(chapter_host.PrivateHostStop, match="CPU plan"):
                chapter_host._hybrid_output_manifest(
                    runner.root,
                    1,
                    inputs["chapter_1"]["bundle_sha256"],
                    inputs["chapter_1"]["scope_sha256"],
                )
        finally:
            journal.write_bytes(journal_bytes)

    # Every original byte is restored after the intentionally corrupted private artifact.
    # Recovery must reject exact-source, producer, reviewer, raw, and WAV tampering.
    first = inputs["chapter_1"]

    def checked_inventory() -> dict[str, object]:
        return chapter_host._hybrid_output_manifest(
            runner.root, 1, first["bundle_sha256"], first["scope_sha256"]
        )

    def artifact(kind: str, stage: str | None = None) -> Path:
        matches = [
            file
            for file in checked_inventory()["files"]
            if file["kind"] == kind
            and (stage is None or f"/{stage}/" in file["relative_path"])
            and (stage is None or file["relative_path"].endswith("/raw-1.json"))
        ]
        if kind == "wav":
            assert matches, "at least one recorded synthetic WAV is required"
        else:
            assert len(matches) == 1
        return runner.root / matches[0]["relative_path"]

    for kind, stage, field in (
        ("source_ledger", None, "ledger_sha256"),
        ("review_record", None, "producer_identity_sha256"),
        ("review", None, "producer_identity_sha256"),
        ("final_result", None, "producer_identity_sha256"),
        ("human_review", None, "review_sha256"),
        ("narration_report", None, "provenance"),
        ("wav", None, None),
        ("text", "hybrid_argument", "raw"),
        ("text", "hybrid_audit", "raw"),
        ("text", "hybrid_reconcile", "raw"),
        ("text", "adaptation", "raw"),
        ("text", "chapter_script", "raw"),
        ("text", "chapter_direction", "raw"),
    ):
        path = artifact(kind, stage)
        unchanged = path.read_bytes()
        try:
            if kind == "wav":
                path.write_bytes(unchanged + b"synthetic-tamper")
            else:
                forged = json.loads(unchanged)
                forged[field] = (
                    "untrusted-test-text" if field in {"raw", "provenance"} else "0" * 64
                )
                path.write_bytes(evaluation._json_bytes(forged))
            with pytest.raises((chapter_host.PrivateHostStop, ValueError, OSError)):
                chapter_host._hybrid_output_manifest(
                    runner.root,
                    1,
                    first["bundle_sha256"],
                    first["scope_sha256"],
                )
        finally:
            path.write_bytes(unchanged)
    assert checked_inventory()["source_mode"] == "DIRECT_SOURCE_HYBRID"
    miner.assert_not_called()


def test_hybrid_producer_binding_reacts_to_tokenizer_template_and_generation_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "ledger_sha256": "a" * 64,
        "chapter_scope_sha256": "b" * 64,
        "selected_bundle_sha256": "c" * 64,
        "plan_fingerprint": "d" * 64,
        "model_producer": {"runtime": "synthetic"},
    }

    def bound() -> str:
        return hybrid.digest(hybrid.producer_identity(**identity))

    original = bound()
    with monkeypatch.context() as patch:
        patch.setattr(hybrid.constrained, "PINNED_TOKENIZER_SHA256", "0" * 64)
        assert bound() != original
    with monkeypatch.context() as patch:
        patch.setattr(hybrid, "CHAT_TEMPLATE_CONTRACT", "synthetic-new-template")
        assert bound() != original
    with monkeypatch.context() as patch:
        patch.setattr(
            hybrid.scripting,
            "QWEN_GENERATION_SETTINGS",
            {**hybrid.scripting.QWEN_GENERATION_SETTINGS, "temperature": 0.123},
        )
        assert bound() != original
    with monkeypatch.context() as patch:
        patch.setattr(hybrid.scripting, "SCRIPT_REVISION", "synthetic-new-revision")
        assert bound() != original


@pytest.mark.parametrize(
    "case",
    (
        "empty",
        "overcap",
        "duplicate_id",
        "bad_text",
        "wrong_hash",
        "unknown_kind",
        "gap_claims_text",
        "missing_order",
        "oversize_ledger",
    ),
)
def test_hybrid_source_ledger_rejects_forged_metadata_and_unbounded_input(
    case: str,
) -> None:
    units, gaps, original_order = _original(3)
    if case == "empty":
        units, gaps, original_order = [], [], []
    elif case == "overcap":
        units, gaps, original_order = _original(129)
    elif case == "duplicate_id":
        units[1]["source_id"] = units[0]["source_id"]
    elif case == "bad_text":
        units[0]["text"] = None
    elif case == "wrong_hash":
        units[0]["text"] = "changed without digest"
    elif case == "unknown_kind":
        units[0]["kind"] = "UNSUPPORTED"
    elif case == "gap_claims_text":
        gaps[0]["text"] = "invented original"
    elif case == "missing_order":
        original_order.pop()
    elif case == "oversize_ledger":
        units[0]["text"] = "x" * hybrid.MAX_SOURCE_BYTES
        units[0]["text_sha256"] = hashlib.sha256(units[0]["text"].encode()).hexdigest()
    with pytest.raises(hybrid.HybridStop):
        hybrid.freeze(units, gaps, original_order, {"chapter": 1})


@pytest.mark.parametrize(
    "case",
    ("bad_schema", "empty_units", "wrong_recomputed_order", "bad_identity"),
)
def test_hybrid_replay_fails_even_with_canonical_attacker_rehash(case: str) -> None:
    forged = copy.deepcopy(_ledger(3))
    if case == "bad_schema":
        forged["schema"] = "synthetic-other-schema"
    elif case == "empty_units":
        forged["entries"] = []
    elif case == "wrong_recomputed_order":
        forged["entries"][0]["order"] = 2
    elif case == "bad_identity":
        forged["identity"] = None
    forged["ledger_sha256"] = hybrid.digest(
        {key: value for key, value in forged.items() if key != "ledger_sha256"}
    )
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_ledger(forged)


@pytest.mark.parametrize(
    "case",
    ("no_hybrid_key", "invalid_hybrid_key", "mismatched_stage_contract"),
)
def test_stage_rejects_unbound_hybrid_attempt_before_any_generator(
    tmp_path: Path,
    case: str,
) -> None:
    callback = Mock(side_effect=AssertionError("no generator may be called"))
    args: dict[str, object] = {
        "prepared_request": {"max_new_tokens": 4096},
        "contract_version": "clipper-hybrid-argument-v1",
        "hybrid_request_identity": "a" * 64,
        "safe_context_tokens": hybrid.SAFE_CONTEXT_TOKENS,
    }
    if case == "no_hybrid_key":
        args.pop("hybrid_request_identity")
    elif case == "invalid_hybrid_key":
        args["hybrid_request_identity"] = "not-a-sha"
    else:
        args.pop("contract_version")
    with pytest.raises(chapter.ChapterStop, match="hybrid"):
        chapter._stage(
            tmp_path / "book-candidates",
            "hybrid_argument",
            {},
            chapter.ChapterPlan(),
            lambda raw: raw,
            callback,
            **args,
        )
    callback.assert_not_called()


@pytest.mark.parametrize("case", ("reject_unit", "missing_beat", "reject_beat"))
def test_human_semantic_review_rejects_omission_or_unattested_meaning(
    case: str,
) -> None:
    ledger = _ledger(25)
    brief = hybrid.reconcile(
        _reconciliation(ledger), ledger, _argument(ledger), _audit_outputs(ledger)
    )
    review = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "reviewer": "Synthetic semantic reviewer",
        "unit_decisions": [
            {"id": unit["id"], "review": "accept", "reason": "Read complete original"}
            for unit in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": beat["beat_id"],
                "review": "support_attested",
                "reason": "Compared meaning, contradiction and qualification",
            }
            for beat in brief["beats"]
        ],
    }
    if case == "reject_unit":
        review["unit_decisions"][0]["review"] = "reject"
    elif case == "missing_beat":
        review["beat_decisions"].pop()
    else:
        review["beat_decisions"][0]["review"] = "reject"
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_human_review(review, brief, ledger)


@pytest.mark.parametrize(
    "case",
    (
        "missing_argument",
        "missing_audit",
        "amend_same_decision",
        "amend_no_evidence",
        "unknown_resolution",
        "no_beats",
        "wrong_beat_id",
        "bad_beat_qualification",
        "empty_beat_evidence",
        "single_beat",
        "missing_source_link",
    ),
)
def test_reconciliation_refuses_nonexistent_decisions_and_broken_grounding(
    case: str,
) -> None:
    ledger = _ledger(25)
    audits = _audit_outputs(ledger)
    argument: object = _argument(ledger)
    raw = _reconciliation(ledger)
    if case == "missing_argument":
        argument = None
    elif case == "missing_audit":
        audits.pop()
    elif case.startswith("amend_"):
        raw["amendments"] = [
            {
                "id": "s0001",
                "from": "keep",
                "to": "keep" if case == "amend_same_decision" else "omit",
                "reason": "Human must assess a grounded editorial change",
                "evidence": []
                if case == "amend_no_evidence"
                else [{"id": "s0001", "start": 0, "end": 10, "role": "ASSERTION"}],
            }
        ]
    elif case == "unknown_resolution":
        raw["resolutions"] = [
            {
                "id": "a001-d01",
                "resolution": "qualified",
                "reason": "No matching local disagreement exists",
                "evidence": [{"id": "s0001", "start": 0, "end": 10, "role": "ASSERTION"}],
            }
        ]
    elif case == "no_beats":
        raw["beats"] = []
    elif case == "wrong_beat_id":
        raw["beats"][0]["beat_id"] = "invented"
    elif case == "bad_beat_qualification":
        raw["beats"][0]["qualification"] = None
    elif case == "empty_beat_evidence":
        raw["beats"][0]["spans"] = []
    elif case == "single_beat":
        raw["beats"] = [raw["beats"][0]]
    else:
        raw["beats"][0]["spans"].pop(0)
    with pytest.raises(hybrid.HybridStop):
        hybrid.reconcile(raw, ledger, argument, audits)


@pytest.mark.parametrize(
    "case",
    (
        "malformed_core",
        "wrong_audit_identity",
        "outside_core",
        "wrong_disagreement_id",
        "unsupported_disagreement",
        "empty_argument_links",
    ),
)
def test_audits_refuse_unlinked_qualifications_and_invalid_core(
    case: str,
) -> None:
    ledger = _ledger(25)
    window = copy.deepcopy(hybrid.audit_windows(ledger)[0])
    audit = copy.deepcopy(_audit_outputs(ledger)[0])
    if case == "malformed_core":
        window["core_ids"] = None
    elif case == "wrong_audit_identity":
        audit["audit_id"] = "a999"
    elif case == "outside_core":
        audit["decisions"][0]["evidence"][0]["id"] = "s0013"
    elif case.startswith("wrong_disagreement") or case == "unsupported_disagreement":
        audit["disagreements"] = [
            {
                "id": "wrong" if case == "wrong_disagreement_id" else "a001-d01",
                "issue": "A contrary case must qualify the main claim",
                "proposed_change": "Change the interpretation before final synthesis",
                "evidence": []
                if case == "unsupported_disagreement"
                else [{"id": "s0001", "start": 0, "end": 10, "role": "COUNTEREXAMPLE"}],
            }
        ]
    elif case == "empty_argument_links":
        argument = _argument(ledger)
        argument["source_links"] = []
        with pytest.raises(hybrid.HybridStop, match="no source-linked"):
            hybrid.validate_argument(argument, ledger)
        return
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_audit(audit, ledger, window)


def test_invalid_creative_replies_still_fail_despite_source_span_metadata() -> None:
    ledger = _ledger(25)
    brief = hybrid.reconcile(
        _reconciliation(ledger), ledger, _argument(ledger), _audit_outputs(ledger)
    )
    plan = chapter.ChapterPlan()
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_adaptation({}, brief, plan)
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_adaptation({}, {"beats": []}, plan)
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_script({}, {"beats": []}, brief, plan)


@pytest.mark.parametrize(
    "case",
    (
        "wrong_stage",
        "low_output",
        "high_output",
        "missing_producer",
        "zero_tokens",
        "noninteger_tokens",
        "over_context",
        "oversize_request",
    ),
)
def test_hybrid_real_request_preflight_refuses_bad_context_and_contract(
    case: str,
) -> None:
    stage = "argument"
    output_tokens = 4096
    producer = {"runtime": "synthetic"}
    payload: dict[str, object] = {"original_source": [["s0001", "text", "source"]]}
    count: object = 1200
    if case == "wrong_stage":
        stage = "unknown"
    elif case == "low_output":
        output_tokens = 900
    elif case == "high_output":
        output_tokens = 16385
    elif case == "missing_producer":
        producer = {}
    elif case == "zero_tokens":
        count = 0
    elif case == "noninteger_tokens":
        count = True
    elif case == "over_context":
        count = 32000
    elif case == "oversize_request":
        payload = {"exact_original_source": "x" * hybrid.MAX_REQUEST_BYTES}
    with pytest.raises(hybrid.HybridStop):
        hybrid.request(
            stage,
            payload,
            output_tokens=output_tokens,
            token_count=lambda _messages: count,
            producer_identity=producer,
        )


def test_hybrid_window_finite_output_call_cap_is_not_silently_reduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger(45)
    monkeypatch.setattr(hybrid, "MAX_CORE_UNITS", 1)
    with pytest.raises(hybrid.HybridStop, match="audit count"):
        hybrid.audit_windows(ledger)


@pytest.mark.parametrize(
    "case",
    (
        "bad_question",
        "bad_links",
        "unknown_source",
        "negative_offset",
        "boolean_offset",
        "end_after_source",
        "invalid_role",
        "invalid_quote_range",
    ),
)
def test_argument_original_spans_reject_forged_unicode_and_source_boundaries(
    case: str,
) -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    if case == "bad_question":
        argument["questions"] = [None]
    elif case == "bad_links":
        argument["source_links"] = "not-a-list"
    else:
        link = argument["source_links"][0]
        if case == "unknown_source":
            link["id"] = "s9999"
        elif case == "negative_offset":
            link["start"] = -1
        elif case == "boolean_offset":
            link["start"] = True
        elif case == "end_after_source":
            link["end"] = 99999
        elif case == "invalid_role":
            link["role"] = "INVENTED_AUTHOR_ACCOUNT"
        elif case == "invalid_quote_range":
            link["start"] = link["end"]
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_argument(argument, ledger)


@pytest.mark.parametrize(
    "case",
    (
        "wrong_core",
        "unknown_core_id",
        "missing_decision",
        "duplicated_decision",
        "bad_decision",
        "nontext_reason",
        "oversize_reason",
        "evidence_in_halo",
        "bad_disagreement_change",
        "bad_disagreement_evidence",
        "gap_kept",
    ),
)
def test_hybrid_local_audit_requires_complete_core_grounding(
    case: str,
) -> None:
    ledger = _ledger(25)
    windows = hybrid.audit_windows(ledger)
    window = windows[-1] if case == "gap_kept" else windows[0]
    raw = copy.deepcopy(_audit_outputs(ledger)[-1 if case == "gap_kept" else 0])
    if case == "wrong_core":
        raw["core_ids"] = list(reversed(raw["core_ids"]))
    elif case == "unknown_core_id":
        raw["decisions"][0]["id"] = "s9999"
    elif case == "missing_decision":
        raw["decisions"].pop()
    elif case == "duplicated_decision":
        raw["decisions"][1]["id"] = raw["decisions"][0]["id"]
    elif case == "bad_decision":
        raw["decisions"][0]["decision"] = "silently_omit"
    elif case == "nontext_reason":
        raw["decisions"][0]["reason"] = None
    elif case == "oversize_reason":
        raw["decisions"][0]["reason"] = "x" * 801
    elif case == "evidence_in_halo":
        raw["decisions"][0]["evidence"][0]["id"] = "s0013"
    elif case == "gap_kept":
        raw["decisions"][-1]["decision"] = "keep"
    else:
        raw["disagreements"] = [
            {
                "id": "a001-d01",
                "issue": "Reconsider contrary text",
                "proposed_change": None
                if case == "bad_disagreement_change"
                else "Qualify the argument",
                "evidence": []
                if case == "bad_disagreement_evidence"
                else [{"id": "s0001", "start": 0, "end": 10, "role": "ASSERTION"}],
            }
        ]
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_audit(raw, ledger, window)


@pytest.mark.parametrize(
    "case",
    (
        "duplicate_amendment",
        "gap_amendment",
        "unknown_amendment",
        "bad_amendment_reason",
        "duplicate_resolution",
        "invalid_resolution",
        "resolution_without_evidence",
        "beat_without_source",
        "beat_bad_essential",
        "beat_bad_causality",
    ),
)
def test_hybrid_reconciliation_rejects_unjustified_global_rewrites(
    case: str,
) -> None:
    ledger = _ledger(25)
    audits = _audit_outputs(ledger)
    response = _reconciliation(ledger)
    amendment = {
        "id": "s0001",
        "from": "keep",
        "to": "omit",
        "reason": "Explicit revision from a source-supported counterexample",
        "evidence": [{"id": "s0002", "start": 0, "end": 10, "role": "COUNTEREXAMPLE"}],
    }
    resolution = {
        "id": "a001-d01",
        "resolution": "qualified",
        "reason": "Bounded argument after contradictory source review",
        "evidence": [{"id": "s0002", "start": 0, "end": 10, "role": "COUNTEREXAMPLE"}],
    }
    if case == "duplicate_amendment":
        response["amendments"] = [amendment, copy.deepcopy(amendment)]
    elif case == "gap_amendment":
        response["amendments"] = [{**amendment, "id": "s0026", "from": "omit", "to": "keep"}]
    elif case == "unknown_amendment":
        response["amendments"] = [{**amendment, "id": "s9999"}]
    elif case == "bad_amendment_reason":
        response["amendments"] = [{**amendment, "reason": None}]
    elif case == "duplicate_resolution":
        audits[0]["disagreements"] = [
            {
                "id": resolution["id"],
                "issue": "Main argument needs qualification",
                "proposed_change": "Revise scope",
                "evidence": resolution["evidence"],
            }
        ]
        response["resolutions"] = [resolution, copy.deepcopy(resolution)]
    elif case == "invalid_resolution":
        audits[0]["disagreements"] = [
            {
                "id": resolution["id"],
                "issue": "Main argument needs qualification",
                "proposed_change": "Revise scope",
                "evidence": resolution["evidence"],
            }
        ]
        response["resolutions"] = [{**resolution, "resolution": "unresolved"}]
    elif case == "resolution_without_evidence":
        audits[0]["disagreements"] = [
            {
                "id": resolution["id"],
                "issue": "Main argument needs qualification",
                "proposed_change": "Revise scope",
                "evidence": resolution["evidence"],
            }
        ]
        response["resolutions"] = [{**resolution, "evidence": []}]
    elif case == "beat_without_source":
        response["beats"][0]["spans"][0]["id"] = "s9999"
    elif case == "beat_bad_essential":
        response["beats"][0]["essential"] = 1
    else:
        response["beats"][0]["causal_link"] = ""
    with pytest.raises(hybrid.HybridStop):
        hybrid.reconcile(response, ledger, _argument(ledger), audits)


def test_review_packets_reject_rehashed_but_semantically_mismatched_brief() -> None:
    ledger = _ledger(25)
    brief = hybrid.reconcile(
        _reconciliation(ledger), ledger, _argument(ledger), _audit_outputs(ledger)
    )
    brief["argument"] = "Test-only altered interpretation."
    with pytest.raises(hybrid.HybridStop, match="immutable source"):
        hybrid.human_review_packet(brief, ledger)


@pytest.mark.parametrize(
    "case", ("duplicate_unit", "missing_unit", "duplicate_beat", "missing_beat", "no_reason")
)
def test_human_attestation_refuses_partial_editorial_signoff(case: str) -> None:
    ledger = _ledger(25)
    brief = hybrid.reconcile(
        _reconciliation(ledger), ledger, _argument(ledger), _audit_outputs(ledger)
    )
    review = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "reviewer": "Synthetic editor",
        "unit_decisions": [
            {"id": unit["id"], "review": "accept", "reason": "Original unit checked"}
            for unit in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": beat["beat_id"],
                "review": "support_attested",
                "reason": "Qualification and logical meaning checked",
            }
            for beat in brief["beats"]
        ],
    }
    if case == "duplicate_unit":
        review["unit_decisions"][0]["id"] = review["unit_decisions"][1]["id"]
    elif case == "missing_unit":
        review["unit_decisions"].pop()
    elif case == "duplicate_beat":
        review["beat_decisions"][0]["beat_id"] = review["beat_decisions"][1]["beat_id"]
    elif case == "missing_beat":
        review["beat_decisions"].pop()
    elif case == "no_reason":
        review["beat_decisions"][0]["reason"] = ""
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_human_review(review, brief, ledger)


def test_interrupted_hybrid_host_has_no_review_and_cannot_duplicate_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, manifest, settings = _private_setup(tmp_path, monkeypatch)
    _locked(runner, manifest, settings)
    called = Mock(side_effect=AssertionError("interrupted reconciliation cannot repeat"))
    miner = Mock(side_effect=AssertionError("direct source cannot mine"))
    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", called)
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", miner)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    for number in (1, 7):
        block = settings[f"chapter_{number}"]
        chapter_host._phase(
            runner, number, block["bundle_sha256"], block["scope_sha256"], "NOT_STARTED"
        )
        chapter_host._phase(
            runner,
            number,
            block["bundle_sha256"],
            block["scope_sha256"],
            "HYBRID_BRIEF_IN_FLIGHT",
            {"claim_owner": "a" * 32, "chapter_mode": "DIRECT_SOURCE_HYBRID"},
        )
    result = _call(runner, manifest, "hybrid-brief")
    assert {chapter_result["status"] for chapter_result in result["chapters"].values()} == {
        "INTERRUPTED_DURING_HYBRID_RECONCILIATION"
    }
    for number in (1, 7):
        block = settings[f"chapter_{number}"]
        inventory = chapter_host._hybrid_output_manifest(
            runner.root, number, block["bundle_sha256"], block["scope_sha256"]
        )
        assert inventory["inflight_inventory"] == "UNATTESTED_FAIL_CLOSED"
        assert inventory["files"] == []
        assert inventory["verified_artifact_identities"] == {}
    state = chapter_host.recovery_report(runner.root)
    assert {item["recovery_state"] for item in state["chapters"].values()} == {"IN_FLIGHT"}
    called.assert_not_called()
    miner.assert_not_called()


@pytest.mark.parametrize(
    "case",
    (
        "invalid_plan",
        "source_overcap",
        "source_freeze_failure",
        "tampered_ledger",
        "missing_test_tokenizer",
        "context_overflow",
    ),
)
def test_original_hybrid_source_and_context_fail_before_any_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    producer = Mock(side_effect=AssertionError("model-free preflight must reject"))
    if case == "invalid_plan":
        with pytest.raises(chapter.ChapterStop, match="validated ChapterPlan"):
            chapter.evaluate_hybrid_chapter(runner, selected, section, plan=object())
    elif case == "source_overcap":
        monkeypatch.setattr(hybrid, "MAX_ENTRIES", 1)
        with pytest.raises(chapter.ChapterStop, match="finite original source cap"):
            chapter.evaluate_hybrid_chapter(runner, selected, section)
    elif case == "source_freeze_failure":
        monkeypatch.setattr(
            hybrid,
            "freeze",
            Mock(side_effect=hybrid.HybridStop("synthetic source schema rejected")),
        )
        with pytest.raises(chapter.ChapterStop, match="preflight refused"):
            chapter.evaluate_hybrid_chapter(runner, selected, section)
    elif case == "tampered_ledger":
        chapter.evaluate_hybrid_chapter(runner, selected, section)
        ledger_file = next(
            (runner.root / "book-candidates" / "chapter-evaluations").glob("*/source-ledger.json")
        )
        ledger_file.write_bytes(ledger_file.read_bytes() + b" ")
        with pytest.raises(chapter.ChapterStop, match="source ledger changed"):
            chapter.evaluate_hybrid_chapter(runner, selected, section)
    elif case == "missing_test_tokenizer":
        with pytest.raises(chapter.ChapterStop, match="tokenizer counter"):
            chapter.evaluate_hybrid_chapter(
                runner,
                selected,
                section,
                plan=chapter.ChapterPlan(brief_cpu=True),
                argument_completion=producer,
            )
    else:
        with pytest.raises(chapter.ChapterStop, match="token request preflight refused"):
            chapter.evaluate_hybrid_chapter(
                runner,
                selected,
                section,
                plan=chapter.ChapterPlan(brief_cpu=True),
                argument_completion=producer,
                token_counter=lambda _messages: 32768,
            )
    producer.assert_not_called()


def test_private_hybrid_review_packet_rejects_stale_source_and_brief(
    tmp_path: Path,
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    args = {
        "plan": chapter.ChapterPlan(brief_cpu=True),
        "argument_completion": lambda req: _injected_hybrid_response("argument", req),
        "audit_completion": lambda req: _injected_hybrid_response("audit", req),
        "reconcile_completion": lambda req: _injected_hybrid_response("reconcile", req),
        "token_counter": lambda _messages: 1200,
    }
    valid = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    assert valid["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    chapter_host._hybrid_review_packet(runner, 1, valid)
    packet = runner.root / "book-candidates" / "private-host" / "chapter-1-review-packet.json"
    assert packet.is_file()
    for field, invalid, expected in (
        ("chapter_scope_identity", "not-a-digest", "scope identity"),
        ("source_ledger_sha256", "0" * 64, "source ledger changed"),
        ("brief_identity", "0" * 64, "brief absent or ambiguous"),
    ):
        forged = {**valid, field: invalid}
        with pytest.raises(chapter_host.PrivateHostStop, match=expected):
            chapter_host._hybrid_review_packet(runner, 1, forged)
        assert packet.is_file()
    validated = next(
        (runner.root / "book-candidates" / "hybrid_reconcile").glob("*/validated.json")
    )
    saved = validated.read_bytes()
    envelope = json.loads(saved)
    envelope["payload"] = None
    try:
        validated.write_bytes(evaluation._json_bytes(envelope))
        with pytest.raises(chapter_host.PrivateHostStop, match="payload missing"):
            chapter_host._hybrid_review_packet(runner, 1, valid)
    finally:
        validated.write_bytes(saved)

    envelope = json.loads(saved)
    envelope["payload"]["ledger_sha256"] = "0" * 64
    try:
        validated.write_bytes(evaluation._json_bytes(envelope))
        with pytest.raises(chapter_host.PrivateHostStop, match="detached from source ledger"):
            chapter_host._hybrid_review_packet(runner, 1, valid)
    finally:
        validated.write_bytes(saved)

    ledger_path = (
        runner.root
        / "book-candidates"
        / "chapter-evaluations"
        / valid["chapter_scope_identity"]
        / "source-ledger.json"
    )
    original_ledger = ledger_path.read_bytes()
    malformed_ledger = evaluation._json_bytes([])
    try:
        ledger_path.write_bytes(malformed_ledger)
        bound_claim = {
            **valid,
            "source_ledger_sha256": hashlib.sha256(malformed_ledger).hexdigest(),
        }
        with pytest.raises(chapter_host.PrivateHostStop, match="source ledger malformed"):
            chapter_host._hybrid_review_packet(runner, 1, bound_claim)
    finally:
        ledger_path.write_bytes(original_ledger)

    stage_root = runner.root / "book-candidates" / "hybrid_reconcile"
    overflow = [
        stage_root / f"synthetic-untrusted-extra-reconcile-{index:03d}" for index in range(128)
    ]
    try:
        for directory in overflow:
            directory.mkdir()
            (directory / "validated.json").write_bytes(b"{}\n")
        with pytest.raises(chapter_host.PrivateHostStop, match="inventory exceeds finite bound"):
            chapter_host._hybrid_review_packet(runner, 1, valid)
    finally:
        for directory in overflow:
            (directory / "validated.json").unlink(missing_ok=True)
            directory.rmdir()
    chapter_host._hybrid_review_packet(runner, 1, valid)


@pytest.mark.parametrize(
    ("adaptation_cpu", "script_cpu", "direction_cpu", "kokoro_cpu", "status"),
    (
        (True, False, False, False, "AWAITING_EXPLICIT_CPU_CHAPTER_SCRIPT"),
        (True, True, False, False, "AWAITING_EXPLICIT_CPU_CHAPTER_DIRECTION"),
        (True, True, False, True, "AWAITING_EXPLICIT_CPU_CHAPTER_DIRECTION"),
        (True, True, True, False, "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"),
    ),
)
def test_source_preserving_hybrid_stage_stops_at_explicit_cpu_permissions(
    tmp_path: Path,
    adaptation_cpu: bool,
    script_cpu: bool,
    direction_cpu: bool,
    kokoro_cpu: bool,
    status: str,
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    limits = chapter.ChapterPlan(
        brief_cpu=True,
        adaptation_cpu=adaptation_cpu,
        script_cpu=script_cpu,
        direction_cpu=direction_cpu,
        kokoro_cpu=kokoro_cpu,
    )
    args = {
        "plan": limits,
        "argument_completion": lambda req: _injected_hybrid_response("argument", req),
        "audit_completion": lambda req: _injected_hybrid_response("audit", req),
        "reconcile_completion": lambda req: _injected_hybrid_response("reconcile", req),
        "adaptation_completion": lambda req: _hybrid_creative_output("adaptation", req),
        "script_completion": lambda req: _hybrid_creative_output("chapter_script", req),
        "direction_completion": lambda req: _hybrid_creative_output("chapter_direction", req),
        "narration_producer": _hybrid_wav,
        "token_counter": lambda _messages: 1200,
    }
    pending = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    root = runner.root / "book-candidates"
    ledger = json.loads(
        next((root / "chapter-evaluations").glob("*/source-ledger.json")).read_bytes()
    )
    stage_file = next((root / "hybrid_reconcile").glob("*/validated.json"))
    brief = json.loads(stage_file.read_bytes())["payload"]
    reviewed = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "brief_artifact_identity": pending["brief_identity"],
        "reviewer": "Synthetic controlled phase reviewer",
        "unit_decisions": [
            {"id": unit["id"], "review": "accept", "reason": "Original reviewed in fixture"}
            for unit in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": beat["beat_id"],
                "review": "support_attested",
                "reason": "Original meaning and qualifications checked",
            }
            for beat in brief["beats"]
        ],
    }
    review_file = tmp_path / "synthetic-phase-review.json"
    review_file.write_bytes(evaluation._json_bytes(reviewed))
    resumed = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=review_file,
        expected_brief_identity=pending["brief_identity"],
        **args,
    )
    assert resumed["status"] == status
    assert resumed["publish_eligible"] is False
    if not kokoro_cpu:
        assert "narration_report_identity" not in resumed or (
            resumed["narration_report_identity"] is None
        )
    else:
        assert resumed["narration_report_identity"]


def test_hybrid_resume_callback_requires_exact_brief_claim(
    tmp_path: Path,
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    args = {
        "plan": chapter.ChapterPlan(brief_cpu=True),
        "argument_completion": lambda req: _injected_hybrid_response("argument", req),
        "audit_completion": lambda req: _injected_hybrid_response("audit", req),
        "reconcile_completion": lambda req: _injected_hybrid_response("reconcile", req),
        "token_counter": lambda _messages: 1200,
    }
    pending = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    root = runner.root / "book-candidates"
    ledger = json.loads(
        next((root / "chapter-evaluations").glob("*/source-ledger.json")).read_bytes()
    )
    brief = json.loads(next((root / "hybrid_reconcile").glob("*/validated.json")).read_bytes())[
        "payload"
    ]
    reviewed = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "brief_artifact_identity": pending["brief_identity"],
        "reviewer": "Synthetic reviewer",
        "unit_decisions": [
            {"id": unit["id"], "review": "accept", "reason": "Read source"}
            for unit in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": beat["beat_id"],
                "review": "support_attested",
                "reason": "Read original meaning",
            }
            for beat in brief["beats"]
        ],
    }
    file = tmp_path / "synthetic-reviewed.json"
    file.write_bytes(evaluation._json_bytes(reviewed))
    callback = Mock(side_effect=AssertionError("no unbound phase claim"))
    with pytest.raises(chapter.ChapterStop, match="exact journaled brief"):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            review_file=file,
            on_resume_verified=callback,
            **args,
        )
    callback.assert_not_called()


def test_hybrid_host_rejects_partial_brief_and_never_retries_claimed_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, manifest, inputs = _private_setup(tmp_path, monkeypatch)
    _locked(runner, manifest, inputs)
    fake = Mock(return_value={"status": "UNREVIEWED_INCOMPLETE_SYNTHETIC_BRIEF"})
    miner = Mock(side_effect=AssertionError("direct source path never mines"))
    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", fake)
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", miner)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)

    stopped = _call(runner, manifest, "hybrid-brief")
    assert set(stopped["chapters"]) == {"1", "7"}
    assert all(
        result["status"] == "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE"
        for result in stopped["chapters"].values()
    )
    assert fake.call_count == 2
    assert not list((runner.root / "book-candidates" / "private-host").glob("*brief-record.json"))
    for number in (1, 7):
        block = inputs[f"chapter_{number}"]
        phases = chapter_host._phase_records(
            runner.root, number, block["bundle_sha256"], block["scope_sha256"]
        )
        assert [item["phase"] for item in phases] == [
            "NOT_STARTED",
            "HYBRID_BRIEF_IN_FLIGHT",
            "STOPPED",
        ]
    replay = _call(runner, manifest, "hybrid-brief")
    assert replay["chapters"] == stopped["chapters"]
    assert fake.call_count == 2, "stopped irreversible attempts must never replay"
    miner.assert_not_called()


@pytest.mark.parametrize("case", ("disabled_plan", "invalid_mode", "scope_drift"))
def test_hybrid_private_host_finite_scope_preflight_precedes_any_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    runner, manifest, settings = _private_setup(tmp_path, monkeypatch)
    _locked(runner, manifest, settings)
    prepared = chapter_host._prepare(runner, settings)
    plan = chapter.ChapterPlan.from_manifest(settings["plan"])
    mode = "hybrid-brief"
    if case == "disabled_plan":
        plan = chapter.ChapterPlan()
    elif case == "invalid_mode":
        mode = "invalid-synthetic-hybrid-mode"
    else:
        prepared[0][1]["scope_sha256"] = "0" * 64
    no_producer = Mock(side_effect=AssertionError("preflight refuses before CPU"))
    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", no_producer)
    with pytest.raises(chapter_host.PrivateHostStop, match=r"plan|scope"):
        chapter_host._execute_hybrid(
            runner, prepared, plan, mode, runner.root, "unused-before-guard"
        )
    no_producer.assert_not_called()
    assert not list((runner.root / "book-candidates" / "private-host").glob("*phase-events"))


@pytest.mark.parametrize(
    "failure",
    ("mismatched_output_budget", "real_context_overflow", "missing_effective_identity"),
)
def test_hybrid_reserved_stage_rejects_invalid_output_before_any_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    callback = Mock(side_effect=AssertionError("no model call before bound preflight"))
    plan = chapter.ChapterPlan()
    request = {"max_new_tokens": plan.max_new_tokens}
    if failure == "mismatched_output_budget":
        request["max_new_tokens"] += 1
    else:
        if failure == "real_context_overflow":
            request["verified_input_encoding"] = {
                "input_token_ids_sha256": "c" * 64,
                "tokenizer_config_sha256": "d" * 64,
                "chat_template_sha256": "e" * 64,
                "tokenizer_json_sha256": "f" * 64,
            }
        monkeypatch.setattr(scripting, "_ORIGINAL_LOCAL_CHAPTER_COMPLETION", callback)
        monkeypatch.setattr(scripting, "_local_chapter_completion", callback)
        monkeypatch.setattr(
            scripting, "_chapter_context_tokens", lambda _request: hybrid.SAFE_CONTEXT_TOKENS
        )
    with pytest.raises(
        chapter.ChapterStop,
        match=r"reservation differs from plan|reserved output exceed safe context|requires a complete rendered-token identity",
    ):
        chapter._stage(
            tmp_path / "book-candidates",
            "hybrid_argument",
            {"synthetic": "exact original context preflight"},
            plan,
            lambda raw: raw,
            callback,
            prepared_request=request,
            contract_version="clipper-hybrid-argument-v1",
            safe_context_tokens=hybrid.SAFE_CONTEXT_TOKENS,
            hybrid_request_identity="a" * 64,
        )
    callback.assert_not_called()
    assert not list(tmp_path.rglob("started-*.json"))


@pytest.mark.parametrize(
    "case", ["wrong_source", "unsupported", "disabled_prefix", "disabled_late"]
)
def test_configured_hybrid_preflight_blocks_before_producer(tmp_path: Path, case: str) -> None:
    from clipper.books import pipeline_plan

    runner, selected, section = _selected_hybrid_source(tmp_path)
    config = pipeline_plan.template(selected.stem, "brief")
    if case == "wrong_source":
        config["source_sha256"] = "0" * 64
    elif case == "unsupported":
        config["targets"] = ["video"]
    elif case == "disabled_prefix":
        config["stages"]["argument"]["enabled"] = False
    else:
        config["stages"]["reconcile"]["enabled"] = False
    forbidden = Mock(side_effect=AssertionError("provider called before preflight"))
    with pytest.raises(chapter.ChapterStop):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            plan=chapter.ChapterPlan(brief_cpu=True),
            pipeline_config=config,
            argument_completion=forbidden,
            audit_completion=forbidden,
            reconcile_completion=forbidden,
            token_counter=lambda _: 1200,
        )
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "fault",
    [
        "source_disabled",
        "mixed_script_only",
        "downstream_permission",
        "text_descriptor",
        "source_descriptor",
        "human_disabled",
        "missing_permission",
        "missing_callback",
        "mixed_origin",
        "review_without_identity",
        "identity_without_review",
        "brief_review",
    ],
)
def test_configured_execution_guard_matrix_no_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from clipper.books import pipeline_plan

    runner, selected, section = _selected_hybrid_source(tmp_path)
    config = pipeline_plan.template(selected.stem, "script")
    forbidden = Mock(side_effect=AssertionError("forbidden provider"))
    callbacks = {
        "argument_completion": forbidden,
        "audit_completion": forbidden,
        "reconcile_completion": forbidden,
        "adaptation_completion": forbidden,
        "script_completion": forbidden,
    }
    permissions = chapter.ChapterPlan(brief_cpu=True, adaptation_cpu=True, script_cpu=True)
    extra: dict[str, object] = {}
    if fault == "source_disabled":
        config["stages"]["source"]["enabled"] = False
    elif fault == "mixed_script_only":
        extra["script_only"] = True
    elif fault == "downstream_permission":
        permissions = chapter.ChapterPlan(direction_cpu=True)
    elif fault in {"text_descriptor", "source_descriptor"}:
        name = "source" if fault == "source_descriptor" else "argument"
        config["stages"][name]["input"] = {
            "schema": pipeline_plan.DESCRIPTOR_SCHEMA,
            "identity": pipeline_plan.stage_input_identity(
                name,
                pipeline_plan.registry()[name],
                selected.stem,
                config["stages"][name]["semantic"],
                (),
            ),
            "output_sha256": "a" * 64,
            "path": "input.json",
        }
    elif fault == "human_disabled":
        config["stages"]["human_review"]["enabled"] = False
    elif fault == "missing_permission":
        permissions = chapter.ChapterPlan()
    elif fault == "missing_callback":
        callbacks["script_completion"] = None
    elif fault == "mixed_origin":
        callbacks["audit_completion"] = scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION
    elif fault == "review_without_identity":
        extra["review_file"] = tmp_path / "no-approval.json"
    elif fault == "identity_without_review":
        extra["expected_brief_identity"] = "a" * 64
    elif fault == "brief_review":
        config["targets"] = ["brief"]
        extra["review_file"] = tmp_path / "no-approval.json"
    with pytest.raises(chapter.ChapterStop):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            plan=permissions,
            pipeline_config=config,
            token_counter=lambda _: 1000,
            **callbacks,
            **extra,
        )
    forbidden.assert_not_called()
    assert not list(runner.root.rglob("started-*.json"))


def _real_argument_contract() -> dict[str, object]:
    """The placeholder values the producer prompt really sends (shared constants)."""
    return {
        "hypothesis": hybrid.HYPOTHESIS_PLACEHOLDER,
        "questions": [hybrid.QUESTION_PLACEHOLDER],
    }


def test_argument_rejects_request_template_echo_with_clear_error() -> None:
    ledger = _ledger(25)
    echoed = {
        "stage": "argument",
        "schema": hybrid.ARGUMENT_SCHEMA,
        "response_contract": _real_argument_contract(),
    }
    with pytest.raises(hybrid.HybridStop, match="echoed the request contract"):
        hybrid.validate_argument(echoed, ledger)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda a: a.update(hypothesis=hybrid.HYPOTHESIS_PLACEHOLDER),
        lambda a: a.update(hypothesis=hybrid.HYPOTHESIS_PLACEHOLDER + " extra words"),
        lambda a: a.update(questions=[hybrid.QUESTION_PLACEHOLDER]),
        lambda a: a.update(questions=["Real question?", "<anything in angle brackets>"]),
    ),
)
def test_argument_rejects_contract_placeholders_in_answer_keys(mutate) -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    mutate(argument)
    with pytest.raises(hybrid.HybridStop, match="placeholder"):
        hybrid.validate_argument(argument, ledger)


def test_argument_accepts_real_answer_and_prompt_uses_shared_placeholders() -> None:
    ledger = _ledger(25)
    assert hybrid.validate_argument(_argument(ledger), ledger)["hypothesis"]
    import inspect

    source = inspect.getsource(chapter)
    assert "chapter_hybrid.HYPOTHESIS_PLACEHOLDER" in source
    assert "chapter_hybrid.QUESTION_PLACEHOLDER" in source


def test_argument_prompt_contract_has_no_copyable_filled_example() -> None:
    import inspect

    source = inspect.getsource(chapter)
    assert '"source_links": [{"id": "s0001", "start": 0, "end": 10' not in source


@pytest.mark.parametrize(
    "prompt_tokens,expected", [(1200, 16_384), (20_000, 32_768 - 20_000 - 512)]
)
def test_configured_auto_output_budget_reaches_every_brief_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt_tokens: int, expected: int
) -> None:
    from clipper.books import pipeline_plan

    runner, selected, section = _selected_hybrid_source(tmp_path)
    monkeypatch.setattr(
        BookOrchestrator, "mine_candidate", Mock(side_effect=AssertionError("no mining"))
    )
    config = pipeline_plan.template(selected.stem, "brief")
    for name in ("argument", "audits", "reconcile"):
        config["stages"][name]["semantic"]["max_new_tokens"] = "auto"
    seen: list[tuple[str, int]] = []

    def producer(kind: str):
        def run(req: dict[str, object]) -> str:
            seen.append((kind, int(req["max_new_tokens"])))  # type: ignore[call-overload]
            return _injected_hybrid_response(kind, req)

        return run

    waiting = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        plan=chapter.ChapterPlan(brief_cpu=True),
        pipeline_config=config,
        argument_completion=producer("argument"),
        audit_completion=producer("audit"),
        reconcile_completion=producer("reconcile"),
        token_counter=lambda _messages: prompt_tokens,
    )
    assert waiting["status"] == "BRIEF_READY_UNREVIEWED"
    assert {kind for kind, _ in seen} == {"argument", "audit", "reconcile"}
    assert {budget for _, budget in seen} == {expected}


_ARGUMENT_CONTRACT = {
    "hypothesis": hybrid.HYPOTHESIS_PLACEHOLDER,
    "questions": [hybrid.QUESTION_PLACEHOLDER],
    "source_links": [{"id": "<id>", "start": "<n>", "end": "<n>", "role": "<role>"}],
}


def _argument_request(contract: object = _ARGUMENT_CONTRACT) -> dict[str, object]:
    payload = {
        "schema": hybrid.ARGUMENT_SCHEMA,
        "original_source": hybrid.projection(hybrid.validate_ledger(_ledger())),
        "response_contract": contract,
    }
    request, _key, _count = hybrid.request(
        "argument",
        payload,
        output_tokens=4096,
        token_count=lambda _messages: 7968,
        producer_identity={"schema": "runtime-v1", "runtime": "CPU", "producer": "a" * 64},
    )
    return request


def test_hybrid_prompt_reads_the_source_first_and_the_task_last() -> None:
    request = _argument_request()
    system, user = (message["content"] for message in request["messages"])  # type: ignore[index]
    assert "argument step" in system
    assert "Stage:" not in system
    # The shape is prose after the source, not a JSON key a model can fill in.
    assert user.index("INPUT (JSON") < user.index("TASK:") < user.index("hypothesis")
    assert "top-level keys are exactly: hypothesis, questions, source_links" in user
    assert "Do not wrap it in another object" in user
    assert user.rstrip().endswith("Now write the JSON object.")
    assert "COUNTEREXAMPLE" in user and "Do not write character offsets" in user
    parsed = hybrid.input_payload(request["messages"])  # type: ignore[arg-type]
    assert "response_contract" not in parsed
    assert parsed["schema"] == hybrid.ARGUMENT_SCHEMA
    assert parsed["original_source"] == hybrid.projection(hybrid.validate_ledger(_ledger()))


def test_hybrid_prompt_states_a_prose_contract_for_creative_stages() -> None:
    request = _argument_request("Existing chapter adaptation JSON: throughline, beats.")
    user = request["messages"][1]["content"]  # type: ignore[index]
    assert "Required content: Existing chapter adaptation JSON: throughline, beats." in user


def test_wrapped_answer_in_the_contract_slot_is_still_rejected_as_an_echo() -> None:
    ledger = _ledger()
    wrapped = {"stage": "argument", "response_contract": dict(_ARGUMENT_CONTRACT)}
    with pytest.raises(hybrid.HybridStop, match="echoed the request contract"):
        hybrid.validate_argument(wrapped, ledger)


def test_input_payload_rejects_a_malformed_request() -> None:
    with pytest.raises(ValueError):
        hybrid.input_payload([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}])


def _text_of(ledger: dict[str, object], sid: str) -> str:
    return next(e["text"] for e in hybrid.validate_ledger(ledger) if e["id"] == sid)


def test_argument_id_reference_resolves_whole_unit_in_code() -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    argument["source_links"] = [{"id": "s0003", "role": "QUALIFICATION"}]
    result = hybrid.validate_argument(argument, ledger)
    text = _text_of(ledger, "s0003")
    assert result["source_links"] == [
        {"id": "s0003", "start": 0, "end": len(text), "role": "QUALIFICATION"}
    ]


def test_argument_quote_reference_resolves_offsets_from_exact_unique_text() -> None:
    ledger = _ledger(25)
    text = _text_of(ledger, "s0003")
    quote = text[2:9]
    assert text.count(quote) == 1
    argument = _argument(ledger)
    argument["source_links"] = [{"id": "s0003", "role": "COUNTEREXAMPLE", "quote": quote}]
    result = hybrid.validate_argument(argument, ledger)
    assert result["source_links"] == [
        {"id": "s0003", "start": 2, "end": 9, "role": "COUNTEREXAMPLE"}
    ]


@pytest.mark.parametrize("quote", ("", "not in the unit text at all", None, 7))
def test_argument_quote_reference_rejects_missing_empty_or_nonstring(quote: object) -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    argument["source_links"] = [{"id": "s0003", "role": "ASSERTION", "quote": quote}]
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_argument(argument, ledger)


def test_argument_quote_reference_rejects_a_quote_that_occurs_twice() -> None:
    ledger = _ledger(25)
    text = _text_of(ledger, "s0003")
    repeated = next(char for char in text if text.count(char) > 1)
    argument = _argument(ledger)
    argument["source_links"] = [{"id": "s0003", "role": "ASSERTION", "quote": repeated}]
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_argument(argument, ledger)


@pytest.mark.parametrize(
    "link",
    (
        {"id": "s9999", "role": "ASSERTION"},
        {"id": "s0003", "role": "INVENTED"},
        {"id": "gap-unresolved", "role": "ASSERTION"},
        {"id": "s0003", "role": "ASSERTION", "start": 0},
        {"id": "s0003", "role": "ASSERTION", "extra": 1},
        {"id": "s0003"},
    ),
)
def test_argument_id_reference_rejects_unknown_gap_role_and_extra_fields(
    link: dict[str, object],
) -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    argument["source_links"] = [link]
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_argument(argument, ledger)


def test_legacy_offset_links_still_validate_and_a_repeated_constant_span_is_rejected() -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    assert hybrid.validate_argument(argument, ledger)["source_links"][0]["end"] == 13
    short = min(
        (e for e in hybrid.validate_ledger(ledger) if e["kind"] == "CANONICAL_WORK_UNIT"),
        key=lambda e: len(e["text"]),
    )
    argument["source_links"] = [
        {"id": short["id"], "start": 0, "end": len(short["text"]) + 50, "role": "ASSERTION"}
    ]
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_argument(argument, ledger)


def test_audit_and_reconcile_accept_id_references_and_store_canonical_offsets() -> None:
    ledger = _ledger(25)
    outputs = _audit_outputs(ledger)
    for output in outputs:
        for decision in output["decisions"]:
            decision["evidence"] = [
                {"id": e["id"], "role": e["role"]} for e in decision["evidence"]
            ]
    audits = [
        hybrid.validate_audit(output, ledger, window)
        for output, window in zip(outputs, hybrid.audit_windows(ledger), strict=True)
    ]
    kept = [d for a in audits for d in a["decisions"] if d["evidence"]]
    assert kept and all(
        set(span) == {"id", "start", "end", "role"} for d in kept for span in d["evidence"]
    )
    reconciliation = _reconciliation(ledger)
    for beat in reconciliation["beats"]:
        beat["spans"] = [{"id": s["id"], "role": s["role"]} for s in beat["spans"]]
    brief = hybrid.reconcile(reconciliation, ledger, _argument(ledger), audits)
    assert brief["beats"]


def test_hybrid_contracts_ask_for_ids_not_character_offsets() -> None:
    source = Path(chapter.__file__).read_text(encoding="utf-8")
    contract_region = source[source.index('"hybrid_argument"') :]
    assert "integer character offset" not in contract_region
    assert '"start": 0' not in contract_region


def _unit_entries(text: str) -> dict[str, dict[str, object]]:
    return {"s1": {"kind": "CANONICAL_WORK_UNIT", "text": text, "source_id": "u1"}}


def test_quote_uniqueness_counts_overlapping_matches_and_rejects_blank_quotes() -> None:
    entry = _unit_entries("aaaa")["s1"]
    with pytest.raises(hybrid.HybridStop):
        hybrid._resolve_reference({"id": "s1", "role": "ASSERTION", "quote": "aaa"}, entry)
    for blank in (" ", "\n\t", ""):
        with pytest.raises(hybrid.HybridStop):
            hybrid._resolve_reference({"id": "s1", "role": "ASSERTION", "quote": blank}, entry)
    unique = hybrid._resolve_reference({"id": "s1", "role": "ASSERTION", "quote": "aaaa"}, entry)
    assert (unique["start"], unique["end"]) == (0, 4)
    # Offsets are character positions, not UTF-8 byte positions.
    entry = _unit_entries("é🧭 ok")["s1"]
    unicode_ref = hybrid._resolve_reference({"id": "s1", "role": "ASSERTION", "quote": "ok"}, entry)
    assert (unicode_ref["start"], unicode_ref["end"]) == (3, 5)


def test_the_700_character_bound_applies_to_every_citation_form() -> None:
    # A unit whose long tail is unique, so a quote of it is unambiguous.
    entries = _unit_entries("head " + "".join(f"{n:04d}" for n in range(200)))
    ok = {"id": "s1", "start": 0, "end": 700, "role": "ASSERTION"}
    assert hybrid._canonical_span(ok, entries)["end"] == 700
    tail = entries["s1"]["text"][5:]
    assert len(tail) == 800
    forms = {
        "whole unit": {"id": "s1", "role": "ASSERTION"},
        "quote": {"id": "s1", "role": "ASSERTION", "quote": tail[:701]},
        "legacy offsets": {**ok, "end": 701},
    }
    for name, bad in forms.items():
        with pytest.raises(hybrid.HybridStop) as caught:
            hybrid._canonical_span(bad, entries)
        assert caught.value.diagnostics == [{"code": "SPAN_TOO_LONG"}], name
    quote = {"id": "s1", "role": "ASSERTION", "quote": tail[:700]}
    assert hybrid._canonical_span(quote, entries) == {
        "id": "s1",
        "start": 5,
        "end": 705,
        "role": "ASSERTION",
    }


@pytest.mark.parametrize(
    "link",
    (
        {"id": ["s0003"], "role": "ASSERTION"},
        {"id": {"a": 1}, "role": "ASSERTION"},
        {"id": "s0003", "role": ["ASSERTION"]},
        {"id": "s0003", "role": {"x": 1}},
        {"id": ["s0003"], "start": 0, "end": 3, "role": "ASSERTION"},
        {"id": "s0003", "start": 0, "end": 3, "role": ["ASSERTION"]},
    ),
)
def test_malformed_id_and_role_types_raise_hybrid_stop_not_type_error(
    link: dict[str, object],
) -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    argument["source_links"] = [link]
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_argument(argument, ledger)


def test_identical_repeated_citations_are_kept_once_in_first_seen_order() -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    argument["source_links"] = [
        {"id": "s0003", "role": "ASSERTION"},
        {"id": "s0002", "role": "QUALIFICATION"},
        {"id": "s0003", "role": "ASSERTION"},
        {"id": "s0003", "role": "COUNTEREXAMPLE"},
    ]
    links = hybrid.validate_argument(argument, ledger)["source_links"]
    assert [(link["id"], link["role"]) for link in links] == [
        ("s0003", "ASSERTION"),
        ("s0002", "QUALIFICATION"),
        ("s0003", "COUNTEREXAMPLE"),
    ]


def test_rejection_reports_source_free_per_citation_diagnostics() -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    argument["source_links"] = [
        {"id": "s0003", "role": "ASSERTION"},
        {"id": "ATTACKER-TEXT", "role": "ASSERTION"},
        {"id": "s0004", "role": "INVENTED"},
        {"id": "s0005", "role": "ASSERTION", "quote": "never in the unit zzz"},
        {"id": "s0004", "start": 0, "end": 9999, "role": "ASSERTION"},
    ]
    with pytest.raises(hybrid.HybridStop) as caught:
        hybrid.validate_argument(argument, ledger)
    assert caught.value.diagnostics == [
        {"where": "source_links", "index": 1, "id": None, "code": "UNKNOWN_ID"},
        {"where": "source_links", "index": 2, "id": "s0004", "code": "BAD_ROLE"},
        {"where": "source_links", "index": 3, "id": "s0005", "code": "QUOTE_ABSENT"},
        {"where": "source_links", "index": 4, "id": "s0004", "code": "BAD_OFFSETS"},
    ]
    # The message (and therefore the stored rejection digest) is the first failure's.
    assert str(caught.value) == "hybrid source span outside allowed original units"
    assert "ATTACKER-TEXT" not in json.dumps(caught.value.diagnostics)


def test_every_evidence_consumer_stores_canonical_offsets_from_id_references() -> None:
    ledger = _ledger()
    audits = _audit_outputs(ledger)
    ref = {"id": "s0004", "role": "COUNTEREXAMPLE"}
    audits[0]["disagreements"] = [
        {
            "id": "a001-d01",
            "issue": "Counterexample changes the opening claim.",
            "proposed_change": "Qualify the initial chapter argument.",
            "evidence": [dict(ref)],
        }
    ]
    window = hybrid.audit_windows(ledger)[0]
    checked = hybrid.validate_audit(audits[0], ledger, window)
    text = _text_of(ledger, "s0004")
    expected = {"id": "s0004", "start": 0, "end": len(text), "role": "COUNTEREXAMPLE"}
    assert checked["disagreements"][0]["evidence"] == [expected]
    response = _reconciliation(ledger)
    response["resolutions"] = [
        {
            "id": "a001-d01",
            "resolution": "qualified",
            "reason": "A narrower argument survives the contrary example.",
            "evidence": [dict(ref)],
        }
    ]
    response["amendments"] = [
        {
            "id": "s0002",
            "from": "keep",
            "to": "omit",
            "reason": "Superseded by an explicit counterexample.",
            "evidence": [dict(ref)],
        }
    ]
    response["beats"][0]["spans"] = [
        span for span in response["beats"][0]["spans"] if span["id"] != "s0002"
    ]
    hybrid.reconcile(response, ledger, _argument(ledger), audits)
    assert response["resolutions"][0]["evidence"] == [expected]
    assert response["amendments"][0]["evidence"] == [expected]


def test_repeated_legacy_offset_citations_keep_their_historical_normalized_form() -> None:
    ledger = _ledger(25)
    argument = _argument(ledger)
    link = {"id": "s0003", "start": 0, "end": 5, "role": "ASSERTION"}
    argument["source_links"] = [dict(link), dict(link)]
    assert hybrid.validate_argument(argument, ledger)["source_links"] == [link, link]

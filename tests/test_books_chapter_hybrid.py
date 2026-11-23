"""Model-free adversarial checks for hybrid chapter source/decision contracts."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_books_chapter_host import _call, _locked, _private_setup
from test_books_selected_scope import _bundle
from test_books_whole_book import ingested

from clipper.books import chapter, chapter_host, evaluation, selected_scope, whole_book
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


def _selected_hybrid_source(tmp_path: Path) -> tuple[BookOrchestrator, Path, Path]:
    runner, selected, _data = _bundle(tmp_path, with_image=True)
    verified = selected_scope.import_chapter(runner.root, selected)
    section = selected_scope._section_manifest(verified.ref, verified.book, verified.work)
    path = runner.root / "book-candidates" / "section-scopes" / "chapter.json"
    evaluation._store(runner.root / "book-candidates", path, section)
    return runner, selected, path


def _injected_hybrid_response(kind: str, request: dict[str, object]) -> str:
    context = json.loads(request["messages"][1]["content"])
    if kind == "argument":
        source = next(item for item in context["original_source"] if item[1] == "text")
        return json.dumps(
            {
                "hypothesis": "The entire chapter remains an editable hypothesis.",
                "questions": ["Does the local evidence narrow the claim?"],
                "source_links": [{"id": source[0], "start": 0, "end": 10, "role": "ASSERTION"}],
            }
        )
    if kind == "audit":
        window = context["window"]
        return json.dumps(
            {
                "audit_id": window["audit_id"],
                "core_ids": window["core_ids"],
                "decisions": [
                    {
                        "id": item[0],
                        "decision": "keep" if item[1] == "text" else "omit",
                        "reason": "Explicit model-free original source or gap decision.",
                        "evidence": [{"id": item[0], "start": 0, "end": 10, "role": "ASSERTION"}]
                        if item[1] == "text"
                        else [],
                    }
                    for item in window["core"]
                ],
                "disagreements": [],
            }
        )
    if kind == "reconcile":
        entries = [item for item in context["complete_original_source"] if item[1] == "text"]
        halfway = max(1, len(entries) // 2)
        groups = [entries[:halfway], entries[halfway:]]
        if not groups[1]:
            groups = [[entries[0]], [entries[0]]]
        return json.dumps(
            {
                "argument": "The whole source is preserved as one revisable causal argument.",
                "amendments": [],
                "resolutions": [],
                "beats": [
                    {
                        "beat_id": f"beat-{index:03d}",
                        "spans": [
                            {"id": entry[0], "start": 0, "end": 10, "role": "ASSERTION"}
                            for entry in group
                        ],
                        "meaning": "A source-grounded, still unreviewed argument.",
                        "qualification": "A limitation expressed as a faithful paraphrase.",
                        "causal_link": "Connect this source with the next original passage.",
                        "essential": True,
                    }
                    for index, group in enumerate(groups, 1)
                ],
            }
        )
    raise AssertionError("unexpected synthetic stage")


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


def _hybrid_creative_output(kind: str, request: dict[str, object]) -> str:
    data = json.loads(request["messages"][1]["content"])
    if kind == "adaptation":
        return json.dumps(
            {
                "throughline": "An invented person tests, learns and changes a physical procedure.",
                "narrative_form": "DRAMATIZED_ILLUSTRATION",
                "choice_reason": "Fiction is never mistaken for a source-authenticated author example.",
                "beats": [
                    {
                        "beat_id": beat["beat_id"],
                        "decision": "keep",
                        "reason": "Retain the source-grounded link to the whole argument.",
                        "qualification": "A faithful paraphrase that omits literal source terms.",
                        "illustration": "INVENTED_ILLUSTRATION",
                    }
                    for beat in data["brief"]["beats"]
                ],
                "characters": [
                    {
                        "id": "worker",
                        "name": "Invented worker",
                        "initial_state": "examining the unchanged instructions",
                    }
                ],
                "locations": [
                    {
                        "id": "workroom",
                        "name": "Imaginary workroom",
                        "initial_state": "a bench containing unmodified tools",
                    }
                ],
                "props": [
                    {
                        "id": "card",
                        "name": "Paper checklist",
                        "initial_state": "blank",
                    }
                ],
            }
        )
    if kind == "chapter_script":
        beat_ids = [beat["beat_id"] for beat in data["adaptation"]["beats"]]
        narration = (
            "A worker sees that the shared instruction assumes a result without a test. "
            "She records the original observation before attempting a small change. "
            "The next limited experiment produces a different observation, which narrows "
            "the original claim instead of proving that every circumstance is identical. "
            "The group can now revise its procedure while admitting that other conditions "
            "might require additional verification."
        )
        scenes = []
        for index in range(3):
            scenes.append(
                {
                    "scene_id": f"scene-{index + 1:02d}",
                    "previous_id": f"scene-{index:02d}" if index else None,
                    "next_id": f"scene-{index + 2:02d}" if index < 2 else None,
                    "purpose": "Show one limited example and connect it to the next claim.",
                    "action": "The worker marks a revised step on the checklist.",
                    "environment_id": "workroom",
                    "narration": narration,
                    "dialogue": [],
                    "beat_ids": [beat_ids[index % len(beat_ids)]],
                    "illustration_label": "INVENTED_ILLUSTRATION",
                    "transition": "The prior observation motivates the next test.",
                    "consequence": "The next step changes without asserting universal proof.",
                }
            )
        return json.dumps({"title": "The revised checklist", "scenes": scenes})
    if kind == "chapter_direction":
        shots = []
        before = "blank"
        for index, timing in enumerate(data["timings"]):
            after = f"marked step {index + 1}"
            shots.append(
                {
                    "shot_id": f"shot-{index + 1:03d}",
                    "scene_id": timing["scene_id"],
                    "purpose": "Show a measured revision of the invented checklist.",
                    "action": "The character marks a practical change.",
                    "environment_id": "workroom",
                    "camera": "Medium close shot on the same card and hand.",
                    "transition": "Continue following the checklist into the next test.",
                    "entity_ids": ["card"],
                    "state_before": {"card": before},
                    "state_after": {"card": after},
                    "start_seconds": timing["start_seconds"],
                    "end_seconds": timing["end_seconds"],
                }
            )
            before = after
        return json.dumps({"shots": shots})
    raise AssertionError("unrecognized creative stage")


def _hybrid_wav(_text: str, voice: str) -> bytes:
    assert voice == "af_heart"
    file = io.BytesIO()
    with wave.open(file, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 24000)
    return file.getvalue()


def test_full_hybrid_model_free_creative_path_keeps_spans_and_review_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    fake_miner = Mock(side_effect=AssertionError("no mandatory per-unit mining"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", fake_miner)
    plan = chapter.ChapterPlan(
        brief_cpu=True,
        adaptation_cpu=True,
        script_cpu=True,
        direction_cpu=True,
        kokoro_cpu=True,
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
        "direction_completion": producer("chapter_direction"),
        "narration_producer": _hybrid_wav,
        "token_counter": lambda _messages: 1200,
    }
    waiting = chapter.evaluate_hybrid_chapter(runner, selected, section, **args)
    assert waiting["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
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
    checkpoint = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=review_file,
        expected_brief_identity=waiting["brief_identity"],
        **args,
    )
    assert checkpoint["status"] == "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
    assert checkpoint["script_scene_count"] == 3
    assert checkpoint["direction_attempts"] == 1
    assert checkpoint["narration_report_identity"]
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
    tmp_path: Path,
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

    host, _valid_path, data = _bundle(tmp_path)
    base = data["reading_blocks"]
    assert 0 < len(base) < selected_scope.MAX_CHAPTER_ENTRIES
    forged = {
        **data,
        "reading_blocks": (base * (selected_scope.MAX_CHAPTER_ENTRIES + 1))[
            : selected_scope.MAX_CHAPTER_ENTRIES + 1
        ],
    }
    encoded = selected_scope._json(forged)
    digest = hashlib.sha256(encoded).hexdigest()
    path = host.root / "book-candidates" / "selected-scopes" / f"{digest}.json"
    path.write_bytes(encoded)
    with pytest.raises(selected_scope.SelectedSourceStop, match="evidence cannot be verified"):
        selected_scope.import_chapter(host.root, path)


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
        "token_counter": lambda _messages: 1200,
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
        output_tokens = 8193
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
    "case", ("invalid_plan", "source_overcap", "source_freeze_failure", "tampered_ledger",
             "missing_test_tokenizer", "context_overflow")
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
            hybrid, "freeze",
            Mock(side_effect=hybrid.HybridStop("synthetic source schema rejected"))
        )
        with pytest.raises(chapter.ChapterStop, match="preflight refused"):
            chapter.evaluate_hybrid_chapter(runner, selected, section)
    elif case == "tampered_ledger":
        chapter.evaluate_hybrid_chapter(runner, selected, section)
        ledger_file = next(
            (runner.root / "book-candidates" / "chapter-evaluations").glob(
                "*/source-ledger.json"
            )
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
    packet = (
        runner.root / "book-candidates" / "private-host" / "chapter-1-review-packet.json"
    )
    assert packet.is_file()
    for field, invalid, expected in (
        ("chapter_scope_identity", "not-a-digest", "scope identity"),
        ("source_ledger_sha256", "0" * 64, "source ledger changed"),
        ("brief_identity", "0" * 64, "brief absent or ambiguous"),
    ):
        forged = {**valid, field: invalid}
        with pytest.raises(chapter_host.PrivateHostStop):
            chapter_host._hybrid_review_packet(runner, 1, forged)
        assert packet.is_file()
    validated = next(
        (runner.root / "book-candidates" / "hybrid_reconcile").glob(
            "*/validated.json"
        )
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
    brief = json.loads(
        next((root / "hybrid_reconcile").glob("*/validated.json")).read_bytes()
    )["payload"]
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

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
from test_books_selected_scope import _bundle

from clipper.books import chapter, evaluation, selected_scope
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
                "source_links": [
                    {"id": source[0], "start": 0, "end": 10, "role": "ASSERTION"}
                ],
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
                        "evidence": [
                            {"id": item[0], "start": 0, "end": 10, "role": "ASSERTION"}
                        ]
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
    ledger = json.loads(next((base / "chapter-evaluations").glob("*/source-ledger.json")).read_bytes())
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
    raw_record["raw"] = raw_record["raw"].replace("editable", "uncertain")
    record_path.write_text(json.dumps(raw_record), encoding="utf-8")
    with pytest.raises(chapter.ChapterStop, match="cannot be verified"):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            expected_brief_identity=waiting["brief_identity"],
            **args,
        )

"""Private previsual evaluation regression tests; synthetic source and fake producers only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_books_whole_book import ingested

from clipper.books import cli, constrained, evaluation, narration, scripting, selection, whole_book
from clipper.books.orchestrator import BookOrchestrator
from clipper.stage_contracts import content_fingerprint


def _mined(tmp_path: Path) -> tuple[BookOrchestrator, Path, Path, list[dict]]:
    runner, acquired, _transport = ingested(tmp_path, paragraphs=3, with_image=False)
    work = whole_book.plan_for(acquired.book)
    assert 0 < len(work.units) <= 8
    model_free_calls: list[dict] = []

    def fake_miner(request: dict[str, object]) -> str:
        model_free_calls.append(request)
        span = request["constraints"]["spans"][0]  # type: ignore[index]
        return json.dumps(
            {
                "lessons": [
                    {
                        "id": span["id"],
                        "topic": "Rehearsal and observation",
                        "claim": "An invented workshop recorded a specific observation before revising a step.",
                        "hook_angle": "What changed after a small rehearsal?",
                        "visual_action": "A worker revises the named tool on a checklist.",
                        "evidence": span["text"],
                        "evidence_field": request["constraints"]["evidence_field"],  # type: ignore[index]
                    }
                ]
            }
        )

    result = runner.evaluate_candidate(
        acquired.analysis_ref_path,
        options=evaluation.EvaluationOptions(mine_cpu=True, max_new_units=len(work.units)),
        mine_completion=fake_miner,
    )
    assert result["status"] == "AWAITING_HUMAN_SOURCE_SUPPORT_REVIEW"
    assert result["candidate_count"] == len(work.units)
    assert len(model_free_calls) == len(work.units)
    paths = list((runner.root / "book-candidates/analysis/progress").rglob("*.json"))
    assert len(paths) == 1
    candidate_paths = list(
        (runner.root / "book-candidates/evaluations").rglob("mined-candidates.json")
    )
    assert len(candidate_paths) == 1
    data = json.loads(candidate_paths[0].read_bytes())
    assert data["review_status"] == "MINED_UNREVIEWED"
    return runner, acquired.analysis_ref_path, paths[0], data["candidates"]


def _review(
    tmp_path: Path,
    progress: Path,
    candidates: list[dict],
    *,
    selected: str | None = "first",
    corrupt: bool = False,
) -> Path:
    raw_progress = progress.read_bytes()
    decisions = [
        {
            "candidate_id": item["id"],
            "decision": "accept_direct" if selected == "first" and i == 0 else "reject",
            "reason": "Human test review attests this exact narrow synthetic example."
            if selected == "first" and i == 0
            else "This illustrative candidate was not selected for this evaluation.",
            "qualifier_quote": "",
        }
        for i, item in enumerate(candidates)
    ]
    manifest = {
        "schema": evaluation.REVIEW_SCHEMA,
        "progress_sha256": hashlib.sha256(raw_progress).hexdigest() if not corrupt else "a" * 64,
        "reviewer": "synthetic-test-reviewer",
        "selected_candidate_id": candidates[0]["id"] if selected == "first" else None,
        "decisions": decisions,
    }
    path = tmp_path / "human-review-private.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _episode() -> dict[str, object]:
    sentences = [
        "A workshop team rehearses one instruction and notices that the required tool is missing from the checklist today.",
        "The people record what they saw, without deciding too early that every mistake has the same explanation.",
        "They name the tool, repeat the rehearsal, and compare the second attempt with the original observation.",
        "A useful next action is to change one instruction and test it again, not promise that mistakes disappear forever.",
    ]
    return {
        "title": "Test the next instruction",
        "scenes": [
            {"headline": f"Scene {i + 1}", "narration": sentence}
            for i, sentence in enumerate(sentences)
        ],
    }


def _shots(req: dict[str, object]) -> str:
    context = json.loads(req["messages"][1]["content"])  # type: ignore[index]
    return json.dumps(
        {
            "shots": [
                {
                    "scene_id": scene["scene_id"],
                    "purpose": f"Show the mechanism in scene {i + 1}",
                    "subject": "A workshop worker and a shared instruction card",
                    "action": "The worker marks the missing tool and retests the instruction",
                    "environment": "A small fabrication workshop with shared tools",
                    "mechanism": "One wording change removes an ambiguous physical action",
                    "continuity": "The same worker and marked instruction remain in view",
                    "evidence_id": context["source_context"]["candidate_id"],
                }
                for i, scene in enumerate(context["scenes"])
            ]
        }
    )


def _wav(narration: str, voice: str) -> bytes:
    assert voice == "af_heart"
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * (24_000 * 8))
    return output.getvalue()


def _private_report(root: Path, name: str) -> dict:
    paths = list((root / "book-candidates/evaluations").rglob(name))
    assert len(paths) == 1
    return json.loads(paths[0].read_bytes())


def test_mined_handoff_review_checkpoint_and_no_catalog_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    monkeypatch.setattr(selection, "choose_lesson", Mock(side_effect=AssertionError("catalog")))
    monkeypatch.setattr(runner, "plan", Mock(side_effect=AssertionError("catalog plan")))
    monkeypatch.setattr(runner, "run", Mock(side_effect=AssertionError("legacy run")))
    result = runner.evaluate_candidate(ref, progress_path=progress)
    assert result["status"] == "AWAITING_HUMAN_SOURCE_SUPPORT_REVIEW"
    assert result["publish_eligible"] is False
    assert result["source_authorization"] == "unverified"
    for candidate in candidates:
        assert candidate["evidence_anchor"]["review_status"] == "source_unverified"
        assert candidate["_evaluation_context"]["source_path"]
        assert candidate["_evaluation_context"]["evidence_quote"]
        assert candidate["_evaluation_context"]["authorization_status"] == "unverified"
    review = _review(tmp_path, progress, candidates)
    blocked = runner.evaluate_candidate(ref, progress_path=progress, review_file=review)
    assert blocked["status"] == "AWAITING_EXPLICIT_CPU_SCRIPT_PERMISSION"
    assert blocked["accepted_lessons"] == 1
    selection_record = _private_report(runner.root, "selection-*.json")
    assert selection_record["human_assisted"] is True
    assert selection_record["publish_eligible"] is False
    assert selection_record["source_context"]["source_path"]
    assert selection_record["source_context"]["source_context"]
    assert selection_record["source_context"]["source_review_status"] == "source_unverified"


def test_abstention_and_review_rejections_are_private_and_no_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    bad_producer = Mock(side_effect=AssertionError("should abstain without inference"))
    review = _review(tmp_path, progress, candidates, selected=None)
    result = runner.evaluate_candidate(
        ref,
        progress_path=progress,
        review_file=review,
        options=evaluation.EvaluationOptions(script_cpu=True, direction_cpu=True, kokoro_cpu=True),
        script_completion=bad_producer,
        direction_completion=bad_producer,
        narration_producer=bad_producer,
    )
    assert result["status"] == "ABSTAINED_ZERO_SOURCE_SUPPORTED_LESSONS"
    assert result["accepted_lessons"] == 0
    bad_producer.assert_not_called()
    monkeypatch.setattr(narration, "synthesize_scene", bad_producer)
    review_data = json.loads(review.read_text())
    review_data["decisions"][0]["decision"] = "accept_direct"
    review_data["selected_candidate_id"] = None
    review.write_text(json.dumps(review_data))
    with pytest.raises(evaluation.EvaluationStop, match="exactly one"):
        runner.evaluate_candidate(ref, progress_path=progress, review_file=review)
    bad_producer.assert_not_called()


def test_actual_script_injected_generated_direction_and_no_visual_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates)
    blocked_visual = Mock(side_effect=AssertionError("visual stage must never run"))
    monkeypatch.setattr(runner, "require_unported", blocked_visual)
    monkeypatch.setattr(selection, "choose_lesson", blocked_visual)
    monkeypatch.setattr(narration, "synthesize_scene", blocked_visual)
    script_raw = json.dumps(_episode())
    scripts = Mock(return_value=script_raw)
    director = Mock(return_value="{}")

    def retry_direction(req: dict[str, object]) -> str:
        if director.call_count == 1:
            return "{}"
        return _shots(req)

    director.side_effect = retry_direction
    result = runner.evaluate_candidate(
        ref,
        progress_path=progress,
        review_file=review,
        options=evaluation.EvaluationOptions(script_cpu=True, direction_cpu=True),
        script_completion=scripts,
        direction_completion=director,
    )
    assert result["status"] == "PREVISUAL_TEXT_READY_UNREVIEWED"
    assert result["visual_generation_calls"] == 0
    assert result["narration_included"] is False
    assert result["timing_basis"] == "estimated_no_audio"
    assert result["script_scene_count"] == result["text_shot_count"] == 4
    assert result["direction_attempts"] == 2
    assert scripts.call_count == 1
    assert director.call_count == 2
    assert "evaluation_source_context" in scripts.call_args.args[0]["messages"][1]["content"]
    assert json.loads(scripts.call_args.args[0]["messages"][1]["content"])["approved_facts"] == []
    blocked_visual.assert_not_called()
    script_report = _private_report(runner.root, "script-report-*.json")
    assert script_report["script_semantic_review"] == "PENDING"
    assert script_report["raw_attempt_count"] == 1
    assert (
        _private_report(runner.root, "narration-report-*.json")["voice_revision_resolved_commit"]
        is None
    )
    shots = _private_report(runner.root, "validated-text-shots.json")
    assert all(shot["scene_id"] == f"scene-{i:02d}" for i, shot in enumerate(shots["shots"]))
    assert all(shot["timing"]["basis"] == "estimated_no_audio" for shot in shots["shots"])
    assert all(shot["evidence_id"] == candidates[0]["id"] for shot in shots["shots"])
    assert len(list((runner.root / "book-candidates/evaluations").rglob("raw-*.json"))) == 3


def test_opt_in_kokoro_real_wav_metadata_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates)
    audio = Mock(side_effect=_wav)
    monkeypatch.setattr(runner, "require_unported", Mock(side_effect=AssertionError("visual")))
    script = Mock(return_value=json.dumps(_episode()))
    shots = Mock(side_effect=_shots)
    kwargs = {
        "progress_path": progress,
        "review_file": review,
        "options": evaluation.EvaluationOptions(
            script_cpu=True, direction_cpu=True, kokoro_cpu=True
        ),
        "script_completion": script,
        "direction_completion": shots,
        "narration_producer": audio,
    }
    result = runner.evaluate_candidate(ref, **kwargs)
    assert result["status"] == "PREVISUAL_TEXT_READY_UNREVIEWED"
    assert result["narration_included"] is True
    assert result["timing_basis"] == "measured_wav"
    assert result["measured_duration_seconds"] == 32.0
    assert audio.call_count == 4
    report = _private_report(runner.root, "narration-report-*.json")
    assert len(report["narration"]) == 4
    assert all(track["duration_seconds"] == 8 for track in report["narration"])
    assert all(track["source"] == "actual_cpu_kokoro_wav" for track in report["narration"])
    assert report["voice_model"] == scripting.VOICE_MODEL
    assert report["voice_revision_label"] == "v1.0"
    assert report["voice_identity_verified"] is False
    directions = _private_report(runner.root, "validated-text-shots.json")
    assert directions["shots"][0]["timing"]["start_seconds"] == 0.0
    assert directions["shots"][1]["timing"]["start_seconds"] == 8.0
    assert directions["shots"][-1]["timing"]["start_seconds"] == 24.0
    # This is a replay proof, not authority to run again: existing immutable
    # text direction must NOT allow the model producer to be reinvoked.
    no_more = Mock(side_effect=AssertionError("no repeat inference"))
    replay = runner.evaluate_candidate(
        ref,
        progress_path=progress,
        review_file=review,
        options=kwargs["options"],
        script_completion=no_more,
        direction_completion=no_more,
        narration_producer=no_more,
    )
    assert replay["script_attempts"] == 0
    assert replay["direction_cache_reused"] is True
    no_more.assert_not_called()
    script_before = _private_report(runner.root, "script-report-*.json")
    estimate = runner.evaluate_candidate(
        ref,
        progress_path=progress,
        review_file=review,
        options=evaluation.EvaluationOptions(script_cpu=True, direction_cpu=True, kokoro_cpu=False),
        script_completion=no_more,
        direction_completion=shots,
        narration_producer=no_more,
    )
    assert estimate["narration_included"] is False
    assert estimate["timing_basis"] == "estimated_no_audio"
    assert _private_report(runner.root, "script-report-*.json") == script_before
    assert (
        len(list((runner.root / "book-candidates/evaluations").rglob("narration-report-*.json")))
        == 2
    )
    assert audio.call_count == 4
    no_more.assert_not_called()


def test_bounded_script_retries_record_every_raw_and_no_source_repair(tmp_path: Path) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates)
    broken = Mock(side_effect=['{"title":"bad","scenes":[]}', json.dumps(_episode())])
    result = runner.evaluate_candidate(
        ref,
        progress_path=progress,
        review_file=review,
        options=evaluation.EvaluationOptions(script_cpu=True),
        script_completion=broken,
    )
    assert result["status"] == "AWAITING_EXPLICIT_CPU_TEXT_DIRECTION_PERMISSION"
    assert result["script_attempts"] == 2
    assert broken.call_count == 2
    paths = list((runner.root / "book-candidates/evaluations").rglob("raw-*.json"))
    assert len(paths) == 2
    audit = _private_report(runner.root, "script-report-*.json")
    assert [a["accepted"] for a in audit["attempts"]] == [False, True]
    for i, path in enumerate(sorted(paths), 1):
        record = json.loads(path.read_bytes())
        assert record["attempt"] == i
        assert record["sha256"] == hashlib.sha256(record["raw"].encode()).hexdigest()


def test_review_identity_qualifiers_private_errors_and_tampering(tmp_path: Path) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates, corrupt=True)
    with pytest.raises(evaluation.EvaluationStop, match="mining identity") as exc:
        runner.evaluate_candidate(ref, progress_path=progress, review_file=review)
    assert "Invented paragraph" not in str(exc.value)
    review = _review(tmp_path, progress, candidates)
    body = json.loads(review.read_text())
    body["decisions"][0]["qualifier_quote"] = "PRIVATE_SECRET_NOT_IN_SOURCE"
    review.write_text(json.dumps(body))
    with pytest.raises(evaluation.EvaluationStop, match="qualifier"):
        runner.evaluate_candidate(ref, progress_path=progress, review_file=review)
    # No existing candidate may be silently promoted by tampering with its export.
    export = runner.root / "book-candidates/analysis/exports" / progress.parent.name / progress.name
    export.write_text('{"books":[]}', encoding="utf-8")
    with pytest.raises(evaluation.EvaluationStop, match="mined export"):
        runner.evaluate_candidate(ref, progress_path=progress)


@pytest.mark.parametrize(
    "mutation",
    [
        {"max_new_units": 0},
        {"max_new_units": True},
        {"mine_cpu": "true"},
        {"voice": "not-a-voice"},
        {"pillar": "other"},
    ],
)
def test_cpu_manifest_must_be_explicit_and_bounded(mutation: dict) -> None:
    params = {
        "schema": evaluation.MANIFEST_SCHEMA,
        "mine_cpu": False,
        "script_cpu": False,
        "direction_cpu": False,
        "kokoro_cpu": False,
        "max_new_units": 1,
        "pillar": "business",
        "voice": "af_heart",
    }
    params.update(mutation)
    with pytest.raises(evaluation.EvaluationStop):
        evaluation.EvaluationOptions.from_manifest(params)


def test_no_implicit_inference_from_default_options(tmp_path: Path) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    script = Mock(side_effect=AssertionError("implicit script inference"))
    result = runner.evaluate_candidate(
        ref,
        progress_path=progress,
        review_file=_review(tmp_path, progress, candidates),
        script_completion=script,
        direction_completion=script,
        narration_producer=script,
    )
    assert result["status"] == "AWAITING_EXPLICIT_CPU_SCRIPT_PERMISSION"
    script.assert_not_called()


def test_reject_invalid_direction_twice_and_keep_raw(tmp_path: Path) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    failure = Mock(return_value=json.dumps({"shots": []}))
    with pytest.raises(evaluation.EvaluationStop, match="two recorded attempts"):
        runner.evaluate_candidate(
            ref,
            progress_path=progress,
            review_file=_review(tmp_path, progress, candidates),
            options=evaluation.EvaluationOptions(script_cpu=True, direction_cpu=True),
            script_completion=Mock(return_value=json.dumps(_episode())),
            direction_completion=failure,
        )
    assert failure.call_count == 2
    raw = list((runner.root / "book-candidates/evaluations").rglob("raw-*.json"))
    assert len(raw) == 3  # 1 script and 2 direction attempts
    assert (
        len(list((runner.root / "book-candidates/evaluations").rglob("validated-text-shots.json")))
        == 0
    )


def test_prompt_version_keeps_legacy_cache_stable_and_strengthens_evidence() -> None:
    request = scripting._script_request(
        scripting.NicheConfig(
            niche="Private",
            audience="Adults",
            brand="CLIPPER",
            voice="af_heart",
            facts=(),
            topic="An invented workshop",
        )
    )
    assert "evaluation_source_context" not in str(request)
    assert whole_book.LOCAL_PRODUCER_CONTRACT.endswith("v8")
    assert whole_book.LOCAL_GENERATION["max_new_tokens"] == 960
    assert len(scripting.QWEN_GENERATION_SETTINGS) == 3
    assert (
        "exact selected evidence entry ALONE"
        in whole_book._request_for_identity.__code__.co_consts.__str__()
    )


@pytest.mark.parametrize("kind", ["exception", "nontext"])
def test_script_failed_provider_never_reinvoked_on_identical_resume(
    tmp_path: Path, kind: str
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates)
    provider = Mock(
        side_effect=RuntimeError("PRIVATE_SOURCE_DO_NOT_ECHO") if kind == "exception" else None,
        return_value={"PRIVATE_SOURCE_DO_NOT_ECHO": True} if kind == "nontext" else None,
    )
    kwargs = {
        "progress_path": progress,
        "review_file": review,
        "options": evaluation.EvaluationOptions(script_cpu=True),
        "script_completion": provider,
    }
    with pytest.raises(evaluation.EvaluationStop):
        runner.evaluate_candidate(ref, **kwargs)
    with pytest.raises(evaluation.EvaluationStop, match="previous script attempt"):
        runner.evaluate_candidate(ref, **kwargs)
    assert provider.call_count == 1
    started = list((runner.root / "book-candidates/evaluations").rglob("started-1.json"))
    assert len(started) == 1
    assert json.loads(started[0].read_bytes())["status"] == "STARTED_UNRESOLVED"


@pytest.mark.parametrize("kind", ["exception", "nontext"])
def test_direction_failed_provider_never_reinvoked_on_identical_resume(
    tmp_path: Path, kind: str
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates)
    script = Mock(return_value=json.dumps(_episode()))
    provider = Mock(
        side_effect=RuntimeError("PRIVATE_DIRECTION_DO_NOT_ECHO") if kind == "exception" else None,
        return_value={"PRIVATE_DIRECTION_DO_NOT_ECHO": True} if kind == "nontext" else None,
    )
    kwargs = {
        "progress_path": progress,
        "review_file": review,
        "options": evaluation.EvaluationOptions(script_cpu=True, direction_cpu=True),
        "script_completion": script,
        "direction_completion": provider,
    }
    with pytest.raises(evaluation.EvaluationStop):
        runner.evaluate_candidate(ref, **kwargs)
    with pytest.raises(evaluation.EvaluationStop, match="previous direction attempt"):
        runner.evaluate_candidate(ref, **kwargs)
    assert script.call_count == 1
    assert provider.call_count == 1
    direction_starts = list(
        (runner.root / "book-candidates/evaluations").rglob("direction-*/started-1.json")
    )
    assert len(direction_starts) == 1


def test_immutable_exact_five_unit_section_scope_and_incomplete_fail_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=12, with_image=False)
    parent = whole_book.plan_for(acquired.book)
    grouped: dict[str, int] = {}
    for unit in parent.units:
        grouped[unit.section_id] = grouped.get(unit.section_id, 0) + 1
    chosen = next((sid for sid, n in grouped.items() if n == 5), None)
    assert chosen is not None, grouped
    scoped = whole_book.plan_for(acquired.book, section_ids=(chosen,))
    assert len(scoped.units) == 5
    assert len(parent.units) > 5
    assert scoped.fingerprint != parent.fingerprint
    assert [u.ordinal for u in scoped.units] == [
        u.ordinal for u in parent.units if u.section_id == chosen
    ]
    manifest = {
        "schema": evaluation.SECTION_MANIFEST_SCHEMA,
        "book_fingerprint": acquired.book.fingerprint,
        "source_fingerprint": acquired.book.source_fingerprint,
        "ingest_stage_key": acquired.stage_key,
        "parent_plan_fingerprint": parent.fingerprint,
        "section_ids": [chosen],
        "parent_total_units": len(parent.units),
        "included_units": 5,
        "excluded_units": len(parent.units) - 5,
    }
    path = tmp_path / "private-canonical-section.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    calls = Mock()

    def producer(request: dict[str, object]) -> str:
        calls(request)
        span = request["constraints"]["spans"][0]  # type: ignore[index]
        return json.dumps(
            {
                "lessons": [
                    {
                        "id": span["id"],
                        "topic": "Synthetic section evidence",
                        "claim": "The exact synthetic source states a limited observation.",
                        "hook_angle": "A single concrete observation",
                        "visual_action": "A worker marks a tool on a card",
                        "evidence": span["text"],
                        "evidence_field": request["constraints"]["evidence_field"],  # type: ignore[index]
                    }
                ]
            }
        )

    with pytest.raises(evaluation.EvaluationStop, match="incomplete"):
        runner.evaluate_candidate(
            acquired.analysis_ref_path,
            section_manifest=path,
            options=evaluation.EvaluationOptions(mine_cpu=True, max_new_units=4),
            mine_completion=producer,
        )
    assert calls.call_count == 4
    outcome = runner.evaluate_candidate(
        acquired.analysis_ref_path,
        section_manifest=path,
        options=evaluation.EvaluationOptions(mine_cpu=True, max_new_units=5),
        mine_completion=producer,
    )
    assert outcome["status"] == "AWAITING_HUMAN_SOURCE_SUPPORT_REVIEW"
    assert outcome["section_scope"]["included_units"] == 5
    assert outcome["section_scope"]["excluded_units"] == len(parent.units) - 5
    assert outcome["parent_full_book_complete"] is False
    assert calls.call_count == 5
    ledger_paths = list((runner.root / "book-candidates/analysis/progress").rglob("*.json"))
    assert len(ledger_paths) == 2
    complete = next(
        json.loads(p.read_bytes())
        for p in ledger_paths
        if json.loads(p.read_bytes())["status"] == "complete"
    )
    assert complete["section_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert complete["counts"]["unresolved"] == 0
    assert complete["total_work_units"] == 5
    assert complete["parent_book_fingerprint"] == acquired.book.fingerprint
    assert complete["section_scope"]["parent_full_book_complete"] is False
    parser = argparse.ArgumentParser()
    root = parser.add_subparsers(dest="command", required=True)
    cli.configure_books_parser(root)
    complete_path = next(
        p for p in ledger_paths if json.loads(p.read_bytes())["status"] == "complete"
    )
    args = parser.parse_args(
        [
            "books",
            "evaluate",
            "--artifact-root",
            str(runner.root),
            "--candidate-ref",
            str(acquired.analysis_ref_path),
            "--progress",
            str(complete_path),
            "--section-manifest",
            str(path),
        ]
    )
    assert cli.dispatch_books(args) == 2
    public_stdout, public_stderr = capsys.readouterr()
    assert chosen not in public_stdout + public_stderr
    public_summary = json.loads(public_stdout)
    assert public_summary["scope_included_units"] == 5
    assert public_summary["scope_excluded_units"] == len(parent.units) - 5
    assert "section_scope" not in public_summary
    manifest["included_units"] = 4
    path.write_text(json.dumps(manifest))
    with pytest.raises(evaluation.EvaluationStop, match="bounds"):
        runner.evaluate_candidate(
            acquired.analysis_ref_path,
            section_manifest=path,
            progress_path=next(
                p for p in ledger_paths if json.loads(p.read_bytes())["status"] == "complete"
            ),
        )


def test_cli_private_model_keys_and_provider_output_do_not_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates)
    private_marker = "PRIV_SYNT_SENTINEL_8CF1AA12"
    private_token = private_marker

    def malicious(_request: dict[str, object]) -> str:
        print(private_token)
        print(private_token, file=sys.stderr)
        return json.dumps({private_token: "private synthetic value", "scenes": []})

    monkeypatch.setattr(scripting, "_local_script_completion", malicious)
    manifest = {
        "schema": evaluation.MANIFEST_SCHEMA,
        "mine_cpu": False,
        "script_cpu": True,
        "direction_cpu": False,
        "kokoro_cpu": False,
        "max_new_units": 1,
        "pillar": "self_improvement",
        "voice": "af_heart",
    }
    manifest_path = tmp_path / "private-cpu-opt-in.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    parser = argparse.ArgumentParser()
    root = parser.add_subparsers(dest="command", required=True)
    cli.configure_books_parser(root)
    args = parser.parse_args(
        [
            "books",
            "evaluate",
            "--artifact-root",
            str(runner.root),
            "--candidate-ref",
            str(ref),
            "--progress",
            str(progress),
            "--review-file",
            str(review),
            "--run-manifest",
            str(manifest_path),
        ]
    )
    assert cli.dispatch_books(args) == 2
    stdout, stderr = capsys.readouterr()
    assert private_token not in stdout + stderr
    assert "PRIV_SYNT_SENTINEL" not in stdout + stderr
    assert "PRIVATE_EVALUATION_FAILED_CLOSED" in stderr
    paths = list((runner.root / "book-candidates/evaluations").rglob("raw-*.json"))
    assert len(paths) == 3
    assert private_token in paths[0].read_text(encoding="utf-8")
    console_logs = list((runner.root / "book-candidates/evaluations").rglob("log-*.json"))
    assert console_logs
    assert any(private_token in item.read_text(encoding="utf-8") for item in console_logs)


def test_recorded_source_pairs_are_independently_validated_without_rewrites(tmp_path: Path) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    work = whole_book.plan_for(acquired.book)
    responses: dict[str, object] = {}
    for unit in work.units:
        request = whole_book.request_for(acquired.book, unit)
        span = constrained.span_pool(unit.text, unit.evidence_field)["spans"][0]
        responses[content_fingerprint(request)] = {
            "lessons": [
                {
                    "id": span["id"],
                    "topic": "An invented observed workshop detail",
                    "claim": "The invented source describes a specific workshop observation.",
                    "hook_angle": "What was observed during the synthetic workshop?",
                    "visual_action": "A worker points to a labeled instruction",
                    "evidence": span["text"],
                    "evidence_field": unit.evidence_field,
                }
            ]
        }
    raw = json.dumps(responses, sort_keys=True).encode()
    result = runner.evaluate_candidate(
        acquired.analysis_ref_path,
        recorded_responses=responses,
        response_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert result["status"] == "AWAITING_HUMAN_SOURCE_SUPPORT_REVIEW"
    assert result["candidate_count"] == len(work.units)
    exports = list((runner.root / "book-candidates/evaluations").rglob("mined-candidates.json"))
    assert len(exports) == 1
    originals = json.loads(exports[0].read_bytes())["candidates"]
    assert all(
        item["evidence_anchor"].get("selected_occurrence") is None
        and item["_evaluation_context"]["evidence_quote"] == item["evidence_anchor"]["quote"]
        for item in originals
    )
    corrupted = json.loads(raw)
    first = next(iter(corrupted))
    corrupted[first]["lessons"][0]["id"] = "unpaired-but-syntactically-valid"
    changed = json.dumps(corrupted, sort_keys=True).encode()
    with pytest.raises(evaluation.EvaluationStop, match="source-exact"):
        runner.evaluate_candidate(
            acquired.analysis_ref_path,
            recorded_responses=corrupted,
            response_file_sha256=hashlib.sha256(changed).hexdigest(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "invalid-json",
        "missing-field",
        "wrong-source",
        "wrong-count",
        "not-list",
        "unknown-id",
        "duplicate-id",
        "reverse-order",
        "oversize",
        "symlink",
    ],
)
def test_private_section_manifest_rejects_unpinned_or_inexact_boundaries(
    tmp_path: Path, mutation: str
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=12, with_image=False)
    parent = whole_book.plan_for(acquired.book)
    section_id = parent.units[0].section_id
    scoped = whole_book.plan_for(acquired.book, section_ids=(section_id,))
    doc: dict[str, object] = {
        "schema": evaluation.SECTION_MANIFEST_SCHEMA,
        "book_fingerprint": acquired.book.fingerprint,
        "source_fingerprint": acquired.book.source_fingerprint,
        "ingest_stage_key": acquired.stage_key,
        "parent_plan_fingerprint": parent.fingerprint,
        "section_ids": [section_id],
        "parent_total_units": len(parent.units),
        "included_units": len(scoped.units),
        "excluded_units": scoped.excluded_units,
    }
    file = tmp_path / "bounded-section.json"
    if mutation == "invalid-json":
        file.write_text("{")
    elif mutation == "oversize":
        file.write_text(" " * 16_385)
    elif mutation == "symlink":
        target = tmp_path / "real.json"
        target.write_text(json.dumps(doc))
        file.symlink_to(target)
    else:
        if mutation == "missing-field":
            del doc["parent_plan_fingerprint"]
        elif mutation == "wrong-source":
            doc["source_fingerprint"] = "b" * 64
        elif mutation == "wrong-count":
            doc["included_units"] = 2000
        elif mutation == "not-list":
            doc["section_ids"] = section_id
        elif mutation == "unknown-id":
            doc["section_ids"] = ["missing-parent-section"]
        elif mutation == "duplicate-id":
            doc["section_ids"] = [section_id, section_id]
        elif mutation == "reverse-order":
            another = next(u.section_id for u in parent.units if u.section_id != section_id)
            doc["section_ids"] = [another, section_id]
        file.write_text(json.dumps(doc))
    with pytest.raises(evaluation.EvaluationStop):
        evaluation._section_scope(runner, acquired.analysis_ref_path, file)


@pytest.mark.parametrize(
    "mutation",
    [
        "bad-reviewer",
        "missing-decisions",
        "duplicate-decision",
        "wrong-candidate",
        "wrong-decision",
        "selected-rejected",
        "invalid-record",
        "unmatched-qualifier",
    ],
)
def test_private_human_review_rejects_unverified_semantic_attestation(
    tmp_path: Path, mutation: str
) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    path = _review(tmp_path, progress, candidates)
    doc = json.loads(path.read_text())
    if mutation == "bad-reviewer":
        doc["reviewer"] = 77
    elif mutation == "missing-decisions":
        doc["decisions"] = []
    elif mutation == "duplicate-decision":
        doc["decisions"].append(dict(doc["decisions"][0]))
    elif mutation == "wrong-candidate":
        doc["decisions"][0]["candidate_id"] = "nonexistent-candidate"
    elif mutation == "wrong-decision":
        doc["decisions"][0]["decision"] = "approve_for_publication"
    elif mutation == "selected-rejected":
        doc["decisions"][0]["decision"] = "reject"
    elif mutation == "invalid-record":
        del doc["decisions"][0]["reason"]
    elif mutation == "unmatched-qualifier":
        doc["decisions"][0]["qualifier_quote"] = "PRIVATE_UNMATCHED_CONTEXT"
    path.write_text(json.dumps(doc))
    with pytest.raises(evaluation.EvaluationStop):
        runner.evaluate_candidate(ref, progress_path=progress, review_file=path)


@pytest.mark.parametrize(
    "mutation",
    [
        "too-large",
        "invalid-json",
        "wrong-schema",
        "no-scenes",
        "wrong-scene-id",
        "wrong-evidence-id",
        "missing-field",
        "empty-action",
        "extra-field",
    ],
)
def test_generated_text_direction_fail_closed_schema(mutation: str) -> None:
    episode = scripting.Episode.from_json(_episode())
    evidence_id = "synthetic-evidence-id"
    timings = [
        {"start_seconds": i * 8.0, "duration_seconds": 8.0, "basis": "measured_wav"}
        for i in range(len(episode.scenes))
    ]
    shots = [
        {
            "scene_id": f"scene-{i:02d}",
            "purpose": "Explain a small observed change",
            "subject": "A workshop worker",
            "action": "Marks the named tool on a checklist",
            "environment": "A small local workshop",
            "mechanism": "Identifies the missing tool to reduce ambiguity",
            "continuity": "Same worker and same paper card throughout",
            "evidence_id": evidence_id,
        }
        for i in range(len(episode.scenes))
    ]
    doc: dict[str, object] = {"shots": shots}
    if mutation == "too-large":
        raw = "x" * (evaluation.MAX_DIRECTION_BYTES + 1)
    elif mutation == "invalid-json":
        raw = "{"
    else:
        if mutation == "wrong-schema":
            doc["unapproved"] = True
        elif mutation == "no-scenes":
            doc["shots"] = []
        elif mutation == "wrong-scene-id":
            shots[0]["scene_id"] = "scene-99"
        elif mutation == "wrong-evidence-id":
            shots[0]["evidence_id"] = "unrelated-evidence"
        elif mutation == "missing-field":
            del shots[0]["purpose"]
        elif mutation == "empty-action":
            shots[0]["action"] = " "
        elif mutation == "extra-field":
            shots[0]["invented_visual"] = "Already rendered 4K video"
        raw = json.dumps(doc)
    with pytest.raises(evaluation.EvaluationStop):
        evaluation._shots(raw, episode, evidence_id, timings)


def test_kokoro_failed_provider_leaves_durable_started_marker(tmp_path: Path) -> None:
    runner, ref, progress, candidates = _mined(tmp_path)
    review = _review(tmp_path, progress, candidates)
    audio = Mock(side_effect=RuntimeError("PRIVATE_AUDIO_SENTINEL"))
    script = Mock(return_value=json.dumps(_episode()))
    kwargs = {
        "progress_path": progress,
        "review_file": review,
        "options": evaluation.EvaluationOptions(script_cpu=True, kokoro_cpu=True),
        "script_completion": script,
        "narration_producer": audio,
    }
    with pytest.raises(evaluation.EvaluationStop, match="Kokoro"):
        runner.evaluate_candidate(ref, **kwargs)
    with pytest.raises(evaluation.EvaluationStop, match="previous Kokoro attempt"):
        runner.evaluate_candidate(ref, **kwargs)
    audio.assert_called_once()
    script.assert_called_once()
    markers = list((runner.root / "book-candidates/evaluations").rglob("audio-*/attempts/*.json"))
    assert len(markers) == 1
    assert json.loads(markers[0].read_bytes())["status"] == "STARTED_UNRESOLVED"

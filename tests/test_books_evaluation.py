"""Private previsual evaluation regression tests; synthetic source and fake producers only."""

from __future__ import annotations

import hashlib
import io
import json
import wave
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_books_whole_book import ingested

from clipper.books import evaluation, narration, scripting, selection, whole_book
from clipper.books.orchestrator import BookOrchestrator


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
    assert script_report["voice_revision_resolved_commit"] is None
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
    report = _private_report(runner.root, "script-report-*.json")
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
    runner.evaluate_candidate(
        ref,
        progress_path=progress,
        review_file=review,
        options=kwargs["options"],
        script_completion=no_more,
        direction_completion=no_more,
        narration_producer=no_more,
    )
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

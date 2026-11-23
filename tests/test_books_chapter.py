"""Synthetic chapter preproduction only: no model, private source or video calls."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest
from test_books_whole_book import ingested

from clipper.books import chapter, cli, evaluation, narration, scripting, selection, whole_book
from clipper.books.orchestrator import BookOrchestrator
from clipper.stage_contracts import content_fingerprint


def _source(
    tmp_path: Path, *, zero_candidate_unit: bool = False, with_image: bool = False
) -> tuple[BookOrchestrator, Path, Path, Path, list[dict]]:
    runner, acquired, _ = ingested(tmp_path, paragraphs=12, with_image=with_image)
    parent = whole_book.plan_for(acquired.book)
    grouped: dict[str, int] = {}
    for unit in parent.units:
        grouped[unit.section_id] = grouped.get(unit.section_id, 0) + 1
    selected = next(
        (
            sid
            for sid, count in grouped.items()
            if count == 5
            and (not with_image or any(item["section_id"] == sid for item in parent.skipped))
        ),
        None,
    )
    assert selected is not None
    scoped = whole_book.plan_for(acquired.book, section_ids=(selected,))
    manifest = {
        "schema": evaluation.SECTION_MANIFEST_SCHEMA,
        "book_fingerprint": acquired.book.fingerprint,
        "source_fingerprint": acquired.book.source_fingerprint,
        "ingest_stage_key": acquired.stage_key,
        "parent_plan_fingerprint": parent.fingerprint,
        "section_ids": [selected],
        "parent_total_units": len(parent.units),
        "included_units": 5,
        "excluded_units": len(parent.units) - 5,
    }
    section_path = tmp_path / "private-test-section.json"
    section_path.write_text(json.dumps(manifest), encoding="utf-8")

    zero_target = scoped.units[1] if zero_candidate_unit else None

    def fake_miner(req: dict[str, object]) -> str:
        source = json.loads(req["messages"][1]["content"])  # type: ignore[index]
        if zero_target is not None and (source["anchor"], source["word_offset"]) == (
            zero_target.anchor,
            zero_target.word_offset,
        ):
            return json.dumps({"lessons": []})
        span = req["constraints"]["spans"][0]  # type: ignore[index]
        return json.dumps(
            {
                "lessons": [
                    {
                        "id": span["id"],
                        "evidence": span["text"],
                        "evidence_field": req["constraints"]["evidence_field"],  # type: ignore[index]
                        "topic": "Synthetic learning-by-doing example",
                        "claim": "One synthetic rehearsal recorded an observed step.",
                        "hook_angle": "A specific limited observation",
                        "visual_action": "A participant marks a missing tool.",
                    }
                ]
            }
        )

    output = runner.evaluate_candidate(
        acquired.analysis_ref_path,
        section_manifest=section_path,
        options=evaluation.EvaluationOptions(mine_cpu=True, max_new_units=5),
        mine_completion=fake_miner,
    )
    assert output["scope_included_units"] == 5
    progress = next((runner.root / "book-candidates/analysis/progress").rglob("*.json"))
    _ledger, candidates, _book = evaluation._mined_export(
        runner,
        acquired.analysis_ref_path,
        progress,
        section_plan=scoped,
        section_manifest_sha256=hashlib.sha256(section_path.read_bytes()).hexdigest(),
    )
    return runner, acquired.analysis_ref_path, progress, section_path, candidates


def _input(request: dict[str, object]) -> dict:
    return json.loads(request["messages"][1]["content"])["input"]  # type: ignore[index]


def _brief(request: dict[str, object]) -> str:
    source = _input(request)["source"]
    units = source["units"]
    gaps = source["gaps"]
    assert source["scope"]["included_units"] == 5
    assert len(source["reading_order"]) == len(units) + len(gaps)
    return json.dumps(
        {
            "argument": "An invented workshop learns through repeated observation and action.",
            "beats": [
                {
                    "beat_id": f"beat-{i:02d}",
                    "unit_id": item["source_id"],
                    "span_start": 0,
                    "span_end": min(170, len(item["text"])),
                    "quote": item["text"][:170],
                    "candidate_id": item["candidate_ids"][0] if item["candidate_ids"] else None,
                    "meaning": "An original bounded synthetic example or contrast from this source unit.",
                    "qualification": "",
                    "essential": True,
                    "source_role": "AUTHOR_EXAMPLE",
                }
                for i, item in enumerate(units, 1)
            ],
            "coverage": [
                {
                    "source_id": item["source_id"],
                    "decision": "keep" if item["kind"] == "CANONICAL_WORK_UNIT" else "omit",
                    "reason": "Preserve this original unit in reading order or flag its source gap.",
                }
                for item in [*units, *gaps]
            ],
        }
    )


def _adapt(request: dict[str, object]) -> str:
    context = _input(request)
    assert context["source"]["authorization_status"] == "unverified"
    assert context["human_review"]["reviewer"] == "synthetic-human-reviewer"
    return json.dumps(
        {
            "throughline": "A worker notices ambiguity, retests it, and changes the shared procedure.",
            "narrative_form": "DRAMATIZED_ILLUSTRATION",
            "choice_reason": "Invented people enact a mechanism without claiming real author events.",
            "beats": [
                {
                    "beat_id": item["beat_id"],
                    "decision": "keep",
                    "reason": "The event makes the chapter argument more concrete.",
                    "qualification": "",
                    "illustration": "INVENTED_ILLUSTRATION",
                }
                for item in context["brief"]["beats"]
            ],
            "characters": [
                {
                    "id": "worker",
                    "name": "Invented workshop worker",
                    "initial_state": "observing a blank procedure",
                }
            ],
            "locations": [
                {
                    "id": "workroom",
                    "name": "Invented workroom",
                    "initial_state": "unchanged workbench",
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


def _script(request: dict[str, object]) -> str:
    context = _input(request)
    beats = [item["beat_id"] for item in context["adaptation"]["beats"]]
    narration = (
        "An invented workshop worker notices a small ambiguity in a shared instruction. "
        "Rather than claim that every team has the same problem, the worker tests one "
        "change and records exactly what happens next. The process remains limited "
        "to this example, which motivates a more careful next action for the group."
    )
    scenes = []
    for i in range(3):
        scenes.append(
            {
                "scene_id": f"scene-{i + 1:02d}",
                "previous_id": f"scene-{i:02d}" if i else None,
                "next_id": f"scene-{i + 2:02d}" if i < 2 else None,
                "purpose": "Explain one physically demonstrated step of a limited argument",
                "action": "The worker tests and then revises the named procedure.",
                "environment_id": "workroom",
                "narration": narration,
                "dialogue": [
                    {
                        "speaker_id": "worker",
                        "text": "I will check the next step rather than assume a cause.",
                        "fiction_label": "INVENTED_DIALOGUE",
                    }
                ]
                if i == 1
                else [],
                "beat_ids": beats[0:2] if i == 0 else beats[2:4] if i == 1 else beats[4:],
                "illustration_label": "INVENTED_ILLUSTRATION",
                "transition": "The prior observation motivates the next controlled action.",
                "consequence": "The card is visibly changed, with no unearned general claim.",
            }
        )
    return json.dumps({"title": "An invented chapter rehearsal", "scenes": scenes})


def _direction(request: dict[str, object]) -> str:
    content = _input(request)
    before = "blank"
    shots = []
    for i, timing in enumerate(content["timings"]):
        after = f"marked step {i + 1}"
        shots.append(
            {
                "shot_id": f"shot-{i + 1:03d}",
                "scene_id": timing["scene_id"],
                "purpose": "Show how observing one limitation changes a physical instruction",
                "action": "A worker marks the shared card and retries the procedure.",
                "environment_id": "workroom",
                "camera": "Medium close tracking shot on the workbench and worker's hand",
                "transition": "Follow the modified card into the next causal action",
                "entity_ids": ["card"],
                "state_before": {"card": before},
                "state_after": {"card": after},
                "start_seconds": timing["start_seconds"],
                "end_seconds": timing["end_seconds"],
            }
        )
        before = after
    return json.dumps({"shots": shots})


def _wav(_text: str, voice: str) -> bytes:
    assert voice == "af_heart"
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 24_000 * 8)
    return output.getvalue()


def _plan(**updates: object) -> chapter.ChapterPlan:
    payload: dict[str, object] = {
        "schema": chapter.PLAN_SCHEMA,
        "brief_cpu": True,
        "adaptation_cpu": True,
        "script_cpu": True,
        "direction_cpu": True,
        "kokoro_cpu": False,
        "max_scenes": 8,
        "max_words": 1000,
        "max_beats": 20,
        "max_shots_per_scene": 3,
        "max_new_tokens": 4096,
        "max_attempts_per_stage": 2,
        "max_runtime_seconds": 900,
        "voice": "af_heart",
    }
    payload.update(updates)
    return chapter.ChapterPlan.from_manifest(payload)


def _review(tmp_path: Path, brief_id: str, brief: dict, *, reject_all: bool = False) -> Path:
    file = tmp_path / "private-human-chapter-review.json"
    file.write_text(
        json.dumps(
            {
                "schema": chapter.REVIEW_SCHEMA,
                "brief_sha256": brief_id,
                "reviewer": "synthetic-human-reviewer",
                "decisions": [
                    {
                        "beat_id": item["beat_id"],
                        "decision": "reject" if reject_all else "support_attested",
                        "reason": "Synthetic reviewer explicitly checks a narrowly supported example.",
                        "qualifier_quote": "",
                    }
                    for item in brief["beats"]
                ],
                "unit_decisions": [
                    {
                        "source_id": item["source_id"],
                        "decision": item["decision"],
                        "review": "accept_coverage",
                        "reason": "Reviewer assessed original unit or gap and its declared treatment.",
                        "qualifier_quote": "",
                    }
                    for item in brief["coverage"]
                ],
            }
        ),
        encoding="utf-8",
    )
    return file


def _first_brief(
    runner: BookOrchestrator, ref: Path, progress: Path, section: Path, plan: chapter.ChapterPlan
) -> tuple[dict, dict]:
    result = chapter.evaluate_chapter(
        runner,
        ref,
        progress,
        section,
        plan=plan,
        brief_completion=_brief,
    )
    assert result["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    paths = list((runner.root / "book-candidates/brief").rglob("validated.json"))
    assert len(paths) == 1
    return result, json.loads(paths[0].read_bytes())["payload"]


def _full_kwargs(review: Path, *, kokoro: bool = False) -> dict:
    return {
        "review_file": review,
        "plan": _plan(kokoro_cpu=kokoro),
        "brief_completion": _brief,
        "adaptation_completion": _adapt,
        "script_completion": _script,
        "direction_completion": _direction,
        "narration_producer": _wav if kokoro else None,
    }


def test_chapter_default_plan_is_model_free_and_source_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, section, source = _source(tmp_path)
    no_model = Mock(side_effect=AssertionError("unexpected model"))
    monkeypatch.setattr(scripting, "_local_script_completion", no_model)
    monkeypatch.setattr(narration, "_kokoro_wav", no_model)
    monkeypatch.setattr(selection, "choose_lesson", no_model)
    monkeypatch.setattr(runner, "run", no_model)
    result = chapter.evaluate_chapter(runner, ref, progress, section)
    assert result["status"] == "AWAITING_EXPLICIT_CPU_CHAPTER_BRIEF"
    assert result["source_review"] == "MINED_UNREVIEWED"
    assert result["parent_full_book_complete"] is False
    assert result["included_units"] == 5
    assert result["candidate_count"] == len(source)
    assert result["visual_calls"] == 0
    assert result["actual_audio"] is False
    assert result["automation_claim"] == "NOT_YET_GENERATED"
    no_model.assert_not_called()


def test_complete_injected_chapter_is_connected_and_exceeds_short_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, section, evidence = _source(tmp_path)
    initial, brief = _first_brief(
        runner,
        ref,
        progress,
        section,
        _plan(adaptation_cpu=False, script_cpu=False, direction_cpu=False),
    )
    review = _review(tmp_path, initial["brief_identity"], brief)
    forbidden = Mock(side_effect=AssertionError("catalog/render/short compactor called"))
    monkeypatch.setattr(selection, "choose_lesson", forbidden)
    monkeypatch.setattr(runner, "run", forbidden)
    monkeypatch.setattr(runner, "require_unported", forbidden)
    monkeypatch.setattr(scripting, "_compact_book_sentences", forbidden)
    monkeypatch.setattr(narration, "_kokoro_wav", forbidden)
    result = chapter.evaluate_chapter(runner, ref, progress, section, **_full_kwargs(review))
    assert result["status"] == "TEST_DOUBLE_CHAPTER_TEXT_READY_UNREVIEWED"
    assert result["actual_audio"] is False
    assert result["timing_basis"] == "NO_AUDIO_ESTIMATE"
    assert result["visual_calls"] == 0
    assert result["script_scene_count"] == 3
    assert result["script_words"] > 125
    assert result["shot_count"] == 3
    assert result["human_attested_beat_count"] == len(evidence)
    assert result["automation_claim"] == "INJECTED_TEST"
    source_record = next(
        (runner.root / "book-candidates/chapter-evaluations").rglob("source-ledger.json")
    )
    original = json.loads(source_record.read_bytes())
    assert original["source_review"] == "MINED_UNREVIEWED"
    assert original["scope"]["included_units"] == 5
    assert original["parent_full_book_complete"] is False
    assert original["evidence"][0]["source_path"]
    assert original["evidence"][0]["anchor"]
    assert original["evidence"][0]["candidate_id"] == evidence[0]["id"]
    assert original["evidence"][0]["review_status"] == "MINED_UNREVIEWED"
    script_saved = next((runner.root / "book-candidates/chapter_script").rglob("validated.json"))
    scenes = json.loads(script_saved.read_bytes())["payload"]["scenes"]
    assert scenes[0]["previous_id"] is None
    assert scenes[-1]["next_id"] is None
    assert scenes[1]["dialogue"][0]["fiction_label"] == "INVENTED_DIALOGUE"
    assert sum(len(x["narration"].split()) for x in scenes) > 125
    forbidden.assert_not_called()


def test_injected_real_wav_geometry_and_immutable_audio_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    first, brief = _first_brief(runner, ref, progress, section, _plan())
    review = _review(tmp_path, first["brief_identity"], brief)
    audio = Mock(side_effect=_wav)
    supplied = _full_kwargs(review, kokoro=True)
    supplied["narration_producer"] = audio
    first_run = chapter.evaluate_chapter(runner, ref, progress, section, **supplied)
    assert first_run["actual_audio"] is False
    assert first_run["audio_provenance"] == "INJECTED_TEST_WAV"
    assert first_run["timing_basis"] == "MEASURED_WAV"
    assert first_run["duration_seconds"] == 32
    assert audio.call_count == 2  # Four utterances share two text/voice cache keys.
    reports = list((runner.root / "book-candidates").rglob("narration-report-*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_bytes())
    assert len(report["records"]) == 4
    assert [
        (x["scene_id"], x["utterance"], x["kind"], x["speaker_id"]) for x in report["records"]
    ] == [
        ("scene-01", 0, "narration", "narrator"),
        ("scene-02", 0, "narration", "narrator"),
        ("scene-02", 1, "invented_dialogue", "worker"),
        ("scene-03", 0, "narration", "narrator"),
    ]
    assert len({x["tts_request_sha256"] for x in report["records"]}) == 2
    assert all(x["duration_seconds"] == 8.0 for x in report["records"])
    assert all(x["sample_rate"] == 24000 for x in report["records"])
    assert report["resolved_model_commit"] is None
    assert report["resolved_commit_verified"] is False
    never = Mock(side_effect=AssertionError("must not regenerate"))
    repeat = chapter.evaluate_chapter(
        runner,
        ref,
        progress,
        section,
        review_file=review,
        plan=_plan(kokoro_cpu=True),
        brief_completion=never,
        adaptation_completion=never,
        script_completion=never,
        direction_completion=never,
        narration_producer=never,
    )
    # All stage caches are content-addressed and replayed without inference.
    assert repeat["status"] == "TEST_DOUBLE_CHAPTER_TEXT_READY_UNREVIEWED"
    assert repeat["brief_attempts"] == 0
    assert repeat["direction_attempts"] == 0
    never.assert_not_called()
    estimated = chapter.evaluate_chapter(runner, ref, progress, section, **_full_kwargs(review))
    assert estimated["timing_basis"] == "NO_AUDIO_ESTIMATE"
    assert estimated["direction_identity"] != first_run["direction_identity"]
    assert estimated["script_identity"] == first_run["script_identity"]
    assert audio.call_count == 2


def test_human_review_abstains_before_adaptation_and_no_provider_called(tmp_path: Path) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    initial, brief = _first_brief(runner, ref, progress, section, _plan())
    review = _review(tmp_path, initial["brief_identity"], brief, reject_all=True)
    never = Mock(side_effect=AssertionError("review abstention must stop"))
    result = chapter.evaluate_chapter(
        runner,
        ref,
        progress,
        section,
        review_file=review,
        plan=_plan(),
        brief_completion=_brief,
        adaptation_completion=never,
        script_completion=never,
        direction_completion=never,
        narration_producer=never,
    )
    assert result["status"] == "ABSTAINED_INSUFFICIENT_ATTESTED_CHAPTER_BEATS"
    assert result["human_attested_beat_count"] == 0
    never.assert_not_called()


@pytest.mark.parametrize("stage", chapter.TEXT_STAGES)
def test_durable_provider_failure_does_not_reinvoke_identical_stage(
    tmp_path: Path, stage: str
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    initial, brief = _first_brief(runner, ref, progress, section, _plan())
    review = _review(tmp_path, initial["brief_identity"], brief)
    kwargs = _full_kwargs(review)
    injections = {
        "brief": "brief_completion",
        "adaptation": "adaptation_completion",
        "chapter_script": "script_completion",
        "chapter_direction": "direction_completion",
    }
    if stage == "brief":
        # A different explicit test provenance+request budget creates new authority.
        altered = _plan(max_attempts_per_stage=1)
        kwargs["plan"] = altered
    failure = Mock(side_effect=RuntimeError("PRIVATE_UNTRUSTED_MODEL_TEXT"))
    kwargs[injections[stage]] = failure
    with pytest.raises(chapter.ChapterStop):
        chapter.evaluate_chapter(runner, ref, progress, section, **kwargs)
    with pytest.raises(chapter.ChapterStop, match="prior unresolved"):
        chapter.evaluate_chapter(runner, ref, progress, section, **kwargs)
    failure.assert_called_once()
    started = list((runner.root / "book-candidates" / stage).rglob("started-1.json"))
    assert started


@pytest.mark.parametrize("failure_mode", ["invalid-json", "nontext", "oversize"])
def test_failed_brief_attempts_are_recorded_and_not_repeated(
    tmp_path: Path, failure_mode: str
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    out: object = (
        "{"
        if failure_mode == "invalid-json"
        else (
            {"bad": "nontext"} if failure_mode == "nontext" else "x" * (chapter.MAX_RAW_BYTES + 1)
        )
    )
    generated = Mock(return_value=out)
    with pytest.raises(chapter.ChapterStop):
        chapter.evaluate_chapter(
            runner,
            ref,
            progress,
            section,
            plan=_plan(brief_cpu=True),
            brief_completion=generated,
        )
    assert generated.call_count == (2 if failure_mode == "invalid-json" else 1)
    with pytest.raises(chapter.ChapterStop, match="prior unresolved"):
        chapter.evaluate_chapter(
            runner,
            ref,
            progress,
            section,
            plan=_plan(brief_cpu=True),
            brief_completion=generated,
        )
    assert generated.call_count == (2 if failure_mode == "invalid-json" else 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-candidate",
        "missing-coverage",
        "duplicate-beat",
        "wrong-essential",
        "wrong-beat-id",
        "empty-argument",
        "single-beat",
        "changed-quote",
        "invalid-qualification",
        "dropped-retained-beat",
        "unknown-original-unit",
    ],
)
def test_brief_rejects_source_and_coverage_forgery(mutation: str) -> None:
    units = [
        {
            "source_id": f"unit-{i}",
            "text": (
                f"Invented example unit {i} describes an observable contrast "
                "between two synthetic checklist actions."
            ),
            "candidate_ids": [f"candidate-{i}"],
        }
        for i in range(3)
    ]
    fake = {
        "argument": "A narrow example",
        "beats": [
            {
                "beat_id": f"beat-{i + 1:02d}",
                "unit_id": unit["source_id"],
                "span_start": 0,
                "span_end": len(unit["text"]),
                "quote": unit["text"],
                "candidate_id": unit["candidate_ids"][0],
                "meaning": "A narrow original example",
                "qualification": "",
                "essential": True,
                "source_role": "AUTHOR_EXAMPLE",
            }
            for i, unit in enumerate(units)
        ],
        "coverage": [
            {
                "source_id": unit["source_id"],
                "decision": "keep",
                "reason": "Retain the original contrast explicitly.",
            }
            for unit in units
        ],
    }
    if mutation == "unknown-candidate":
        fake["beats"][0]["candidate_id"] = "not-pinned"
    elif mutation == "missing-coverage":
        fake["coverage"].pop()
    elif mutation == "duplicate-beat":
        fake["beats"][1]["unit_id"] = fake["beats"][0]["unit_id"]
        fake["beats"][1]["quote"] = fake["beats"][0]["quote"]
        fake["beats"][1]["span_end"] = fake["beats"][0]["span_end"]
        fake["beats"][1]["candidate_id"] = fake["beats"][0]["candidate_id"]
    elif mutation == "wrong-essential":
        fake["beats"][0]["essential"] = "yes"
    elif mutation == "wrong-beat-id":
        fake["beats"][1]["beat_id"] = "beat-89"
    elif mutation == "empty-argument":
        fake["argument"] = " "
    elif mutation == "single-beat":
        fake["beats"] = fake["beats"][:1]
    elif mutation == "changed-quote":
        fake["beats"][0]["quote"] = "Invented unsupported assertion"
    elif mutation == "invalid-qualification":
        fake["beats"][0]["qualification"] = ["not-text"]
    elif mutation == "dropped-retained-beat":
        fake["beats"].pop()
    elif mutation == "unknown-original-unit":
        fake["coverage"][0]["source_id"] = "noncanonical"
    with pytest.raises(chapter.ChapterStop):
        chapter._brief_validate(fake, units, [], _plan())


@pytest.mark.parametrize(
    "mutation",
    [
        "unreviewed-beat",
        "bad-qualifier",
        "deleted-essential",
        "missing-beat",
        "unknown-entity",
        "duplicate-entity",
        "bad-narrative-form",
    ],
)
def test_adaptation_fiction_qualifier_and_essential_coverage_enforced(mutation: str) -> None:
    brief = {
        "beats": [
            {
                "beat_id": "beat-01",
                "candidate_id": "one",
                "essential": True,
                "qualification": "only in this one case",
            },
            {
                "beat_id": "beat-02",
                "candidate_id": "two",
                "essential": True,
                "qualification": "",
            },
        ]
    }
    decisions = {
        key: {"decision": "support_attested", "qualifier_quote": ""}
        for key in ("beat-01", "beat-02")
    }
    adap = {
        "throughline": "A specific limited example motivates an action.",
        "narrative_form": "DRAMATIZED_ILLUSTRATION",
        "choice_reason": "Invented characters demonstrate a mechanism.",
        "beats": [
            {
                "beat_id": item["beat_id"],
                "decision": "keep",
                "reason": "This detail is necessary for the causal argument.",
                "qualification": item["qualification"],
                "illustration": "INVENTED_ILLUSTRATION",
            }
            for item in brief["beats"]
        ],
        "characters": [
            {
                "id": "worker",
                "name": "Worker",
                "initial_state": "observing",
            }
        ],
        "locations": [
            {
                "id": "workroom",
                "name": "Workshop",
                "initial_state": "lit",
            }
        ],
        "props": [
            {
                "id": "card",
                "name": "Card",
                "initial_state": "blank",
            }
        ],
    }
    if mutation == "unreviewed-beat":
        decisions["beat-01"]["decision"] = "reject"
    elif mutation == "bad-qualifier":
        adap["beats"][0]["qualification"] = ""
    elif mutation == "deleted-essential":
        adap["beats"][0]["decision"] = "omit"
    elif mutation == "missing-beat":
        adap["beats"].pop()
    elif mutation == "unknown-entity":
        adap["props"][0]["id"] = "wrong&invalid"
    elif mutation == "duplicate-entity":
        adap["props"][0]["id"] = "worker"
    elif mutation == "bad-narrative-form":
        adap["narrative_form"] = "AUTOMATIC_FILM_RENDER"
    with pytest.raises(chapter.ChapterStop):
        chapter._adapt_validate(adap, brief, decisions, _plan())


@pytest.mark.parametrize(
    "mutation",
    [
        "short-words",
        "wrong-link",
        "unknown-beat",
        "missing-beat",
        "unlabeled-dialogue",
        "unknown-speaker",
        "wrong-location",
        "lost-qualifier",
    ],
)
def test_chapter_script_graph_fiction_and_qualification_fail_closed(mutation: str) -> None:
    dummy = {
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "input": {
                            "adaptation": {
                                "beats": [{"beat_id": "beat-01"}, {"beat_id": "beat-02"}]
                            }
                        }
                    }
                ),
            },
        ]
    }
    text = json.loads(_script(dummy))
    adapt = json.loads(
        _adapt(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "input": {
                                    "source": {"authorization_status": "unverified"},
                                    "human_review": {"reviewer": "synthetic-human-reviewer"},
                                    "brief": {
                                        "beats": [{"beat_id": "beat-01"}, {"beat_id": "beat-02"}]
                                    },
                                }
                            }
                        ),
                    },
                ]
            }
        )
    )
    decisions = {
        "beat-01": {"qualifier_quote": ""},
        "beat-02": {"qualifier_quote": ""},
    }
    # Restore generated coverage: two beats only.
    text["scenes"][1]["beat_ids"] = ["beat-02"]
    text["scenes"][2]["beat_ids"] = ["beat-02"]
    text["scenes"][0]["beat_ids"] = ["beat-01"]
    if mutation == "short-words":
        for item in text["scenes"]:
            item["narration"] = "Brief example"
    elif mutation == "wrong-link":
        text["scenes"][1]["previous_id"] = "scene-99"
    elif mutation == "unknown-beat":
        text["scenes"][0]["beat_ids"] = ["beat-99"]
    elif mutation == "missing-beat":
        text["scenes"][1]["beat_ids"] = ["beat-01"]
        text["scenes"][2]["beat_ids"] = ["beat-01"]
    elif mutation == "unlabeled-dialogue":
        text["scenes"][1]["illustration_label"] = "SOURCE_ACCOUNT"
    elif mutation == "unknown-speaker":
        text["scenes"][1]["dialogue"][0]["speaker_id"] = "unlisted"
    elif mutation == "wrong-location":
        text["scenes"][0]["environment_id"] = "unknown"
    elif mutation == "lost-qualifier":
        decisions["beat-01"]["qualifier_quote"] = "exact source qualification"
    with pytest.raises(chapter.ChapterStop):
        chapter._script_validate(text, adapt, decisions, _plan())


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-entity",
        "bad-before",
        "missing-after",
        "wrong-location",
        "gap",
        "overlap",
        "orphan-scene",
        "missing-scene",
        "no-state-change",
        "reordered-array",
    ],
)
def test_continuity_and_timing_are_enforced_without_visual_generation(mutation: str) -> None:
    script = json.loads(
        _script(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "input": {
                                    "adaptation": {
                                        "beats": [{"beat_id": f"beat-{i:02d}"} for i in range(1, 6)]
                                    }
                                }
                            }
                        ),
                    },
                ]
            }
        )
    )
    adaptation = json.loads(
        _adapt(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "input": {
                                    "source": {"authorization_status": "unverified"},
                                    "human_review": {"reviewer": "synthetic-human-reviewer"},
                                    "brief": {
                                        "beats": [{"beat_id": f"beat-{i:02d}"} for i in range(1, 6)]
                                    },
                                }
                            }
                        ),
                    },
                ]
            }
        )
    )
    timings = chapter._timings(script, _plan(), None)
    shots = json.loads(
        _direction(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": json.dumps({"input": {"timings": timings}})},
                ]
            }
        )
    )
    if mutation == "unknown-entity":
        shots["shots"][0]["entity_ids"] = ["unregistered"]
    elif mutation == "bad-before":
        shots["shots"][1]["state_before"]["card"] = "blank"
    elif mutation == "missing-after":
        shots["shots"][1]["state_after"] = {}
    elif mutation == "wrong-location":
        shots["shots"][0]["environment_id"] = "not-a-room"
    elif mutation == "gap":
        shots["shots"][1]["start_seconds"] += 1.0
    elif mutation == "overlap":
        shots["shots"][1]["start_seconds"] -= 1.0
    elif mutation == "orphan-scene":
        shots["shots"][1]["scene_id"] = "scene-99"
    elif mutation == "missing-scene":
        shots["shots"].pop()
    elif mutation == "no-state-change":
        shots["shots"][0]["state_after"] = {"card": "blank"}
        shots["shots"][1]["state_before"] = {"card": "blank"}
    elif mutation == "reordered-array":
        shots["shots"][0], shots["shots"][1] = shots["shots"][1], shots["shots"][0]
    with pytest.raises(chapter.ChapterStop):
        chapter._direction_validate(shots, script, adaptation, timings, _plan())


def test_cli_chapter_mode_does_not_leak_private_model_keys_or_source_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, ref, progress, section, evidence = _source(tmp_path)
    marker = "PRIVATE_CHAPTER_SENTINEL_9BCD"

    def hostile(_request: dict[str, object]) -> str:
        print(marker)
        print(marker, file=sys.stderr)
        return json.dumps({marker: "a book excerpt should not leak"})

    monkeypatch.setattr(scripting, "_local_script_completion", hostile)
    plan = _plan(adaptation_cpu=False, script_cpu=False, direction_cpu=False)
    from dataclasses import asdict

    manifest = {"schema": chapter.PLAN_SCHEMA, **asdict(plan)}
    manifest_path = tmp_path / "private-chapter-run-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    parser = argparse.ArgumentParser()
    cli.configure_books_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "books",
            "chapter-evaluate",
            "--artifact-root",
            str(runner.root),
            "--candidate-ref",
            str(ref),
            "--section-manifest",
            str(section),
            "--progress",
            str(progress),
            "--run-manifest",
            str(manifest_path),
        ]
    )
    assert cli.dispatch_books(args) == 2
    stdout, stderr = capsys.readouterr()
    assert marker not in stdout + stderr
    assert evidence[0]["id"] not in stdout + stderr
    assert "PRIVATE_CHAPTER_BOUNDED_HOST_UNVERIFIED" in stderr
    raw = list((runner.root / "book-candidates/brief").rglob("raw-*.json"))
    assert raw == []  # The real host is unverified, so no producer starts or raw appears.


def test_zero_candidate_source_example_remains_in_brief_and_human_unit_review(
    tmp_path: Path,
) -> None:
    runner, ref, progress, section, candidates = _source(tmp_path, zero_candidate_unit=True)
    assert len(candidates) == 4  # One original work unit mined no lesson.
    plan = _plan(adaptation_cpu=False, script_cpu=False, direction_cpu=False)
    first, brief = _first_brief(runner, ref, progress, section, plan)
    source_path = next(
        (runner.root / "book-candidates/chapter-evaluations").rglob("source-ledger.json")
    )
    source = json.loads(source_path.read_bytes())
    assert len(source["units"]) == 5
    assert source["zero_candidate_units"] == 1
    assert len(source["reading_order"]) == 5
    zero = next(unit for unit in source["units"] if unit["mined_candidate_count"] == 0)
    assert zero["candidate_ids"] == []
    assert "synthetic learning example" in zero["text"]
    assert zero["unit_fingerprint"] in zero["source_id"]
    assert zero["text_sha256"] == hashlib.sha256(zero["text"].encode()).hexdigest()
    assert zero["original_parent_ordinal"] >= 0
    beat = next(item for item in brief["beats"] if item["unit_id"] == zero["source_id"])
    assert beat["candidate_id"] is None
    assert beat["quote"] == zero["text"][beat["span_start"] : beat["span_end"]]
    assert {entry["source_id"] for entry in brief["coverage"]} == {
        entry["source_id"] for entry in source["units"]
    }
    review = _review(tmp_path, first["brief_identity"], brief)
    data = json.loads(review.read_text())
    incomplete = dict(data)
    incomplete["unit_decisions"] = data["unit_decisions"][:-1]
    review.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(chapter.ChapterStop, match="did not assess"):
        chapter.evaluate_chapter(
            runner,
            ref,
            progress,
            section,
            review_file=review,
            plan=plan,
            brief_completion=_brief,
        )
    data["unit_decisions"][1]["review"] = "reject_coverage"
    review.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(chapter.ChapterStop, match="rejected chapter unit coverage"):
        chapter.evaluate_chapter(
            runner,
            ref,
            progress,
            section,
            review_file=review,
            plan=plan,
            brief_completion=_brief,
        )
    data["unit_decisions"][1]["review"] = "accept_coverage"
    review.write_text(json.dumps(data), encoding="utf-8")
    accepted = chapter.evaluate_chapter(
        runner,
        ref,
        progress,
        section,
        review_file=review,
        plan=plan,
        brief_completion=_brief,
    )
    assert accepted["status"] == "AWAITING_EXPLICIT_CPU_CHAPTER_ADAPTATION"
    assert accepted["human_attested_beat_count"] == 5


def test_entire_chapter_request_and_output_preflight_abstains_before_provider(
    tmp_path: Path,
) -> None:
    provider = Mock(side_effect=AssertionError("must not reach weights"))
    with pytest.raises(chapter.ChapterStop, match="context"):
        chapter._stage(
            tmp_path / "private-test",
            "brief",
            {
                "schema": chapter.BRIEF_SCHEMA,
                "source": {"units": [{"source_id": "unit-original", "text": "synthetic" * 6_000}]},
            },
            _plan(max_new_tokens=8192),
            lambda _: {"should": "never validate"},
            provider,
        )
    provider.assert_not_called()
    assert not list(tmp_path.rglob("started-*.json"))


def test_skipped_image_marker_kept_in_original_order_and_human_review(tmp_path: Path) -> None:
    runner, ref, progress, section, _ = _source(tmp_path, with_image=True)
    plan = _plan(adaptation_cpu=False, script_cpu=False, direction_cpu=False)
    initial, brief = _first_brief(runner, ref, progress, section, plan)
    source_file = next(
        (runner.root / "book-candidates/chapter-evaluations").rglob("source-ledger.json")
    )
    source = json.loads(source_file.read_bytes())
    assert len(source["units"]) == 5
    assert len(source["gaps"]) >= 1
    assert len(source["reading_order"]) == len(source["units"]) + len(source["gaps"])
    gap_ids = {gap["source_id"] for gap in source["gaps"]}
    assert gap_ids.issubset(set(source["reading_order"]))
    assert all(gap["kind"] == "UNRECOVERED_OR_EMPTY_SOURCE_MARKER" for gap in source["gaps"])
    assert any(gap["reason"] == "unrecovered_image" for gap in source["gaps"])
    assert all(gap["text"] == "" for gap in source["gaps"])
    covered = {row["source_id"]: row for row in brief["coverage"]}
    assert all(covered[sid]["decision"] == "omit" for sid in gap_ids)
    review = _review(tmp_path, initial["brief_identity"], brief)
    accepted = chapter.evaluate_chapter(
        runner, ref, progress, section, review_file=review, plan=plan, brief_completion=_brief
    )
    assert accepted["status"] == "AWAITING_EXPLICIT_CPU_CHAPTER_ADAPTATION"
    bad = json.loads(json.dumps(brief))
    next(row for row in bad["coverage"] if row["source_id"] in gap_ids)["decision"] = "keep"
    with pytest.raises(chapter.ChapterStop, match="unrecovered chapter source"):
        chapter._brief_validate(bad, source["units"], source["gaps"], plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-unit",
        "duplicate-unit",
        "wrong-unit-decision",
        "unit-qualifier",
        "reject-unit",
        "bad-brief",
        "missing-beat",
        "duplicate-beat",
        "unknown-beat",
        "beat-qualifier",
    ],
)
def test_human_review_checks_every_original_unit_and_exact_span(
    tmp_path: Path, mutation: str
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    initial, brief = _first_brief(runner, ref, progress, section, _plan())
    review = _review(tmp_path, initial["brief_identity"], brief)
    source_file = next(
        (runner.root / "book-candidates/chapter-evaluations").rglob("source-ledger.json")
    )
    source = json.loads(source_file.read_bytes())
    doc = json.loads(review.read_bytes())
    if mutation == "missing-unit":
        doc["unit_decisions"].pop()
    elif mutation == "duplicate-unit":
        doc["unit_decisions"][1]["source_id"] = doc["unit_decisions"][0]["source_id"]
    elif mutation == "wrong-unit-decision":
        doc["unit_decisions"][0]["decision"] = "omit"
    elif mutation == "unit-qualifier":
        doc["unit_decisions"][0]["qualifier_quote"] = "NOT_PRESENT_IN_SOURCE"
    elif mutation == "reject-unit":
        doc["unit_decisions"][0]["review"] = "reject_coverage"
    elif mutation == "bad-brief":
        doc["brief_sha256"] = "a" * 64
    elif mutation == "missing-beat":
        doc["decisions"].pop()
    elif mutation == "duplicate-beat":
        doc["decisions"][1]["beat_id"] = doc["decisions"][0]["beat_id"]
    elif mutation == "unknown-beat":
        doc["decisions"][0]["beat_id"] = "beat-99"
    elif mutation == "beat-qualifier":
        doc["decisions"][0]["qualifier_quote"] = "NOT_PRESENT_IN_SOURCE"
    with pytest.raises(chapter.ChapterStop):
        chapter._review_validate(
            doc, brief, initial["brief_identity"], source["units"], source["gaps"]
        )


def test_chapter_source_budget_fails_without_truncation_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    monkeypatch.setattr(chapter, "MAX_SOURCE_BYTES", 128)
    producer = Mock(side_effect=AssertionError("no model permitted"))
    with pytest.raises(chapter.ChapterStop, match="source preflight budget"):
        chapter.evaluate_chapter(
            runner,
            ref,
            progress,
            section,
            plan=_plan(),
            brief_completion=producer,
        )
    producer.assert_not_called()


@pytest.mark.parametrize("hold", ["script", "direction"])
def test_chapter_explicit_stage_hold_never_silently_calls_later_provider(
    tmp_path: Path, hold: str
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    initial, brief = _first_brief(runner, ref, progress, section, _plan())
    review = _review(tmp_path, initial["brief_identity"], brief)
    never = Mock(side_effect=AssertionError("held producer invoked"))
    kwargs = {
        "review_file": review,
        "plan": _plan(script_cpu=hold != "script", direction_cpu=False),
        "brief_completion": _brief,
        "adaptation_completion": _adapt,
        "script_completion": never if hold == "script" else _script,
        "direction_completion": never,
    }
    result = chapter.evaluate_chapter(runner, ref, progress, section, **kwargs)
    assert result["status"] == (
        "AWAITING_EXPLICIT_CPU_CHAPTER_SCRIPT"
        if hold == "script"
        else "AWAITING_EXPLICIT_CPU_CHAPTER_DIRECTION"
    )
    never.assert_not_called()


@pytest.mark.parametrize("mutation", ["schema", "nonbool", "over-limit", "voice"])
def test_chapter_manifest_permissions_and_budgets_reject_bad_inputs(mutation: str) -> None:
    from dataclasses import asdict

    settings: dict[str, object] = {"schema": chapter.PLAN_SCHEMA, **asdict(_plan())}
    if mutation == "schema":
        settings["schema"] = "unexpected-bypass"
    elif mutation == "nonbool":
        settings["brief_cpu"] = "true"
    elif mutation == "over-limit":
        settings["max_beats"] = 3000
    elif mutation == "voice":
        settings["voice"] = "unknown-speaker"
    with pytest.raises(chapter.ChapterStop):
        chapter.ChapterPlan.from_manifest(settings)


def test_concurrent_chapter_stage_exclusive_owner_and_legacy_cache_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = Barrier(2)
    original_claim = chapter._claim

    def synchronized_claim(*args: object) -> str:
        barrier.wait(timeout=5)
        return original_claim(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(chapter, "_claim", synchronized_claim)
    provider = Mock(return_value='{"ok": true}')
    args = (tmp_path / "root", "brief", {"scope": "synthetic"}, _plan(max_attempts_per_stage=1))

    def run() -> tuple[dict, str, str, int]:
        return chapter._stage(*args, lambda v: v, provider)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run) for _ in range(2)]
        results = []
        errors = []
        for future in futures:
            try:
                results.append(future.result())
            except chapter.ChapterStop as exc:
                errors.append(str(exc))
    assert len(results) == 1
    assert len(errors) == 1 and "reservation" in errors[0]
    provider.assert_called_once()
    stage_dir = next((tmp_path / "root" / "brief").iterdir())
    complete = json.loads((stage_dir / "validated.json").read_bytes())
    assert complete["owner"]
    assert json.loads((stage_dir / "started-1.json").read_bytes())["owner"] == complete["owner"]
    monkeypatch.setattr(chapter, "_claim", original_claim)
    assert run()[3] == 0
    (stage_dir / "started-1.json").unlink()
    with pytest.raises(chapter.ChapterStop, match="artifact cannot be verified"):
        run()
    provider.assert_called_once()


def test_concurrent_chapter_narration_single_winner_and_interrupted_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = Barrier(2)
    original_claim = chapter._claim

    def synchronized_claim(*args: object) -> str:
        barrier.wait(timeout=5)
        return original_claim(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(chapter, "_claim", synchronized_claim)
    producer = Mock(side_effect=_wav)
    script = {"scenes": [{"scene_id": "scene-01", "narration": "synthetic", "dialogue": []}]}

    def run() -> tuple[list[dict], str, str]:
        return chapter._narrate(tmp_path / "root", script, _plan(), producer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run) for _ in range(2)]
        results = []
        failures = []
        for future in futures:
            try:
                results.append(future.result())
            except chapter.ChapterStop as exc:
                failures.append(str(exc))
    assert len(results) == 1
    assert len(failures) == 1 and "reservation" in failures[0]
    producer.assert_called_once()
    monkeypatch.setattr(chapter, "_claim", original_claim)
    assert run() == results[0]
    producer.assert_called_once()
    root = tmp_path / "root"
    marker = next((root / "chapter-audio-injected_test_wav" / "attempts").glob("completed-*.json"))
    marker.unlink()
    with pytest.raises(chapter.ChapterStop, match="unresolved"):
        run()
    producer.assert_called_once()


def test_qualified_invented_beats_cannot_be_relabeled_or_silently_denarrated() -> None:
    request = {
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "input": {
                            "adaptation": {
                                "beats": [{"beat_id": f"beat-{i:02d}"} for i in range(1, 6)]
                            }
                        }
                    }
                ),
            },
        ]
    }
    script = json.loads(_script(request))
    adaptation = json.loads(
        _adapt(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "input": {
                                    "source": {"authorization_status": "unverified"},
                                    "human_review": {"reviewer": "synthetic-human-reviewer"},
                                    "brief": {
                                        "beats": [{"beat_id": f"beat-{i:02d}"} for i in range(1, 6)]
                                    },
                                }
                            }
                        ),
                    },
                ]
            }
        )
    )
    reviews = {f"beat-{i:02d}": {"qualifier_quote": ""} for i in range(1, 6)}
    for scene in script["scenes"]:
        scene["illustration_label"] = "SOURCE_ACCOUNT"
        scene["dialogue"] = []
    with pytest.raises(chapter.ChapterStop, match="relabel invented"):
        chapter._script_validate(script, adaptation, reviews, _plan())
    for scene in script["scenes"]:
        scene["illustration_label"] = "INVENTED_ILLUSTRATION"
    adaptation["beats"][0]["qualification"] = "only within this synthetic trial"
    with pytest.raises(chapter.ChapterStop, match="narration lost"):
        chapter._script_validate(script, adaptation, reviews, _plan())
    script["scenes"][0]["narration"] += " only within this synthetic trial"
    brief = {
        "beats": [
            {
                "beat_id": f"beat-{i:02d}",
                "unit_id": f"unit-{i}",
                "qualification": "source specifically limits this trial" if i == 1 else "",
            }
            for i in range(1, 6)
        ]
    }
    review = {
        "unit_decisions": [
            {
                "source_id": f"unit-{i}",
                "qualifier_quote": "qualified unit reviewer" if i == 1 else "",
            }
            for i in range(1, 6)
        ]
    }
    with pytest.raises(chapter.ChapterStop, match="narration lost"):
        chapter._script_validate(
            script, adaptation, reviews, _plan(), brief=brief, human_review=review
        )
    script["scenes"][0]["narration"] += " source specifically limits this trial"
    script["scenes"][0]["narration"] += " qualified unit reviewer"
    assert (
        chapter._script_validate(
            script, adaptation, reviews, _plan(), brief=brief, human_review=review
        )
        == script
    )


@pytest.mark.parametrize("delta", [0.01, -0.01, 0.0000002, float("nan"), float("inf")])
def test_chapter_shot_timing_is_gap_free_and_finite(delta: float) -> None:
    dummy = {
        "messages": [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "input": {
                            "adaptation": {
                                "beats": [{"beat_id": f"beat-{i:02d}"} for i in range(1, 6)]
                            }
                        }
                    }
                ),
            },
        ]
    }
    script = json.loads(_script(dummy))
    adaptation = json.loads(
        _adapt(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "input": {
                                    "source": {"authorization_status": "unverified"},
                                    "human_review": {"reviewer": "synthetic-human-reviewer"},
                                    "brief": {
                                        "beats": [{"beat_id": f"beat-{i:02d}"} for i in range(1, 6)]
                                    },
                                }
                            }
                        ),
                    },
                ]
            }
        )
    )
    timings = chapter._timings(script, _plan(), None)
    directions = json.loads(
        _direction(
            {
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": json.dumps({"input": {"timings": timings}})},
                ]
            }
        )
    )
    directions["shots"][1]["start_seconds"] += delta
    with pytest.raises(chapter.ChapterStop, match="timing gap"):
        chapter._direction_validate(directions, script, adaptation, timings, _plan())


def test_final_result_identity_binds_full_provenance_and_same_length_wav_bytes(
    tmp_path: Path,
) -> None:
    identities: list[tuple[str, str, dict]] = []
    for number in range(2):
        runner, ref, progress, section, _ = _source(tmp_path / f"run-{number}")
        first, brief = _first_brief(runner, ref, progress, section, _plan())
        review = _review(tmp_path / f"run-{number}", first["brief_identity"], brief)

        def alternate_wav(text: str, voice: str, variant: int = number) -> bytes:
            pcm = bytearray(_wav(text, voice))
            pcm[44] = variant
            return bytes(pcm)

        kwargs = _full_kwargs(review, kokoro=True)
        kwargs["narration_producer"] = alternate_wav
        result = chapter.evaluate_chapter(runner, ref, progress, section, **kwargs)
        saved = next(
            (runner.root / "book-candidates/chapter-evaluations").rglob("chapter-result-*.json")
        )
        payload = json.loads(saved.read_bytes())
        assert saved.stem == "chapter-result-" + result["final_result_identity"]
        assert payload["narration_report_key"] == result["narration_report_identity"]
        assert payload["audio_records"][0]["start_sample"] == 0
        assert payload["audio_records"][-1]["end_sample"] == 24000 * 32
        identities.append((result["direction_identity"], result["final_result_identity"], payload))
    assert identities[0][2]["timings"] == identities[1][2]["timings"]
    assert identities[0][1] != identities[1][1]  # Different WAV bytes: result cannot alias.
    assert (
        identities[0][2]["audio_records"][0]["sha256"]
        != (identities[1][2]["audio_records"][0]["sha256"])
    )
    copy = dict(identities[0][2])
    copy["script_key"] = "different-producer-script-same-scenes"
    assert content_fingerprint(copy) != identities[0][1]


def test_omitted_qualified_unit_needs_human_disposition(tmp_path: Path) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    first, brief = _first_brief(runner, ref, progress, section, _plan())
    review_path = _review(tmp_path, first["brief_identity"], brief)
    source_file = next(
        (runner.root / "book-candidates/chapter-evaluations").rglob("source-ledger.json")
    )
    source = json.loads(source_file.read_bytes())
    review = json.loads(review_path.read_bytes())
    brief["coverage"][0]["decision"] = "omit"
    # Directly validate explicit reviewer treatment of an omitted, qualified unit.
    review["unit_decisions"][0]["decision"] = "omit"
    review["unit_decisions"][0]["qualifier_quote"] = source["units"][0]["text"][:40]
    with pytest.raises(chapter.ChapterStop, match="explicit reviewer disposition"):
        chapter._review_validate(review, brief, first["brief_identity"], source["units"], [])
    review["unit_decisions"][0]["reason"] += review["unit_decisions"][0]["qualifier_quote"]
    assert (
        chapter._review_validate(review, brief, first["brief_identity"], source["units"], [])[0]
        == review
    )


def test_cached_chapter_stage_owner_tamper_is_not_silently_replayed(tmp_path: Path) -> None:
    root = tmp_path / "private"
    plan = _plan(max_attempts_per_stage=1)
    producer = Mock(return_value=json.dumps({"story": "synthetic"}))
    args = (root, "brief", {"source": "synthetic"}, plan)
    first = chapter._stage(*args, lambda item: item, producer)
    assert first[3] == 1
    started = next((root / "brief").rglob("started-1.json"))
    marker = json.loads(started.read_bytes())
    marker["owner"] = "synthetic-other-owner"
    started.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(chapter.ChapterStop, match="artifact cannot be verified"):
        chapter._stage(*args, lambda item: item, producer)
    producer.assert_called_once()


def test_chapter_wav_cache_rejects_changed_bytes_and_bad_completion(tmp_path: Path) -> None:
    root = tmp_path / "private"
    plan = _plan()
    producer = Mock(side_effect=_wav)
    script = {"scenes": [{"scene_id": "scene-01", "narration": "synthetic", "dialogue": []}]}
    records, provenance, _key = chapter._narrate(root, script, plan, producer)
    assert provenance == "INJECTED_TEST_WAV"
    assert records[0]["start_sample"] == 0
    wav_path = root / records[0]["wav_relative_path"]
    original = wav_path.read_bytes()
    changed = bytearray(original)
    changed[44] ^= 1
    wav_path.write_bytes(changed)
    with pytest.raises(chapter.ChapterStop, match="cache changed"):
        chapter._narrate(root, script, plan, producer)
    wav_path.write_bytes(original)
    completion = next(root.rglob("completed-*.json"))
    saved = json.loads(completion.read_bytes())
    completion.write_text('{"broken":', encoding="utf-8")
    with pytest.raises(chapter.ChapterStop, match="completion record invalid"):
        chapter._narrate(root, script, plan, producer)
    completion.write_text(json.dumps({**saved, "status": "UNRESOLVED"}), encoding="utf-8")
    with pytest.raises(chapter.ChapterStop, match="conflicting publication"):
        chapter._narrate(root, script, plan, producer)
    producer.assert_called_once()


def test_chapter_audio_failure_and_unverified_host_are_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "private"
    plan = _plan(max_attempts_per_stage=1)
    script = {"scenes": [{"scene_id": "scene-01", "narration": "synthetic", "dialogue": []}]}
    with pytest.raises(chapter.ChapterStop, match="verified bounded private host"):
        chapter._narrate(root, script, plan, None)
    with pytest.raises(chapter.ChapterStop, match="verified bounded private host"):
        chapter._stage(root, "brief", {"source": "synthetic"}, plan, lambda item: item, None)
    failed = Mock(side_effect=RuntimeError("synthetic producer failure"))
    with pytest.raises(chapter.ChapterStop, match="failed"):
        chapter._narrate(root, script, plan, failed)
    with pytest.raises(chapter.ChapterStop, match="unresolved"):
        chapter._narrate(root, script, plan, failed)
    failed.assert_called_once()


def test_chapter_timing_rejects_noncontiguous_pcm_sample_offsets() -> None:
    plan = _plan()
    script = {"scenes": [{"scene_id": "scene-01", "narration": "synthetic", "dialogue": []}]}
    frame_record = {
        "scene_id": "scene-01",
        "samples": 8000,
        "sample_rate": 24000,
        "start_sample": 1,
        "end_sample": 8001,
    }
    with pytest.raises(chapter.ChapterStop, match="offsets are discontinuous"):
        chapter._timings(script, plan, [frame_record])


@pytest.mark.parametrize("stage", ["brief", "chapter_direction"])
@pytest.mark.parametrize(
    "tamper", ["payload", "raw", "missing_raw", "raw_digest", "artifact_digest"]
)
def test_text_artifact_replay_verifies_exact_raw_and_validated_output(
    tmp_path: Path, stage: str, tamper: str
) -> None:
    root = tmp_path / "private"
    plan = _plan()
    callback = Mock(return_value='{"value":"original synthetic response"}')
    args = (root, stage, {"fixture": "synthetic"}, plan)
    first = chapter._stage(*args, lambda x: x, callback)
    assert first[3] == 1
    directory = next((root / stage).iterdir())
    complete = directory / "validated.json"
    saved = json.loads(complete.read_bytes())
    raw_path = directory / "raw-1.json"
    raw = json.loads(raw_path.read_bytes())
    if tamper == "payload":
        saved["payload"]["value"] = "changed synthetic response"
    elif tamper == "raw":
        raw["raw"] = '{"value":"changed synthetic response"}'
        raw_path.write_bytes(evaluation._json_bytes(raw))
    elif tamper == "missing_raw":
        raw_path.unlink()
    elif tamper == "raw_digest":
        raw["sha256"] = "f" * 64
        raw_path.write_bytes(evaluation._json_bytes(raw))
    else:
        saved["artifact_identity"] = "f" * 64
    if tamper in {"payload", "artifact_digest"}:
        complete.write_bytes(evaluation._json_bytes(saved))
    with pytest.raises(chapter.ChapterStop, match="artifact cannot be verified"):
        chapter._stage(*args, lambda x: x, callback)
    callback.assert_called_once()


def test_text_artifact_identity_separates_changed_output_not_equivalent_content(
    tmp_path: Path,
) -> None:
    ids: list[str] = []
    requests: list[str] = []
    for index, content in enumerate(["same", "same", "changed"]):
        root = tmp_path / f"run-{index}"
        output = '{"value":"' + content + '"}'
        _, key, _, attempts = chapter._stage(
            root,
            "brief",
            {"fixture": "same-source-request"},
            _plan(),
            lambda x: x,
            lambda _request, answer=output: answer,
        )
        assert attempts == 1
        ids.append(key)
        requests.append(next((root / "brief").iterdir()).name)
    assert requests[0] == requests[1] == requests[2]
    assert ids[0] == ids[1]
    assert ids[0] != ids[2]


@pytest.mark.parametrize("tamper", ["delete", "replace", "partial"])
def test_text_reservation_loss_before_publish_never_commits_success(
    tmp_path: Path, tamper: str
) -> None:
    root = tmp_path / "private"
    calls = Mock(return_value='{"value":"valid synthetic output"}')

    def hostile(request: dict[str, object]) -> str:
        marker = next((root / "brief").rglob("started-1.json"))
        if tamper == "delete":
            marker.unlink()
        elif tamper == "partial":
            marker.write_bytes(b"{")
        else:
            record = json.loads(marker.read_bytes())
            record["owner"] = "0" * 32
            marker.write_text(json.dumps(record), encoding="utf-8")
        return calls(request)

    args = (root, "brief", {"fixture": "claim-loss"}, _plan(max_attempts_per_stage=1))
    with pytest.raises(chapter.ChapterStop):
        chapter._stage(*args, lambda x: x, hostile)
    assert list((root / "brief").rglob("validated.json")) == []
    # Even a deleted marker leaves a raw orphan; no new provider call is permitted.
    replay = Mock(side_effect=AssertionError("must not call provider twice"))
    with pytest.raises(chapter.ChapterStop, match="prior unresolved"):
        chapter._stage(*args, lambda x: x, replay)
    replay.assert_not_called()
    calls.assert_called_once()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_attempts_per_stage", 3),
        ("max_attempts_per_stage", 0),
        ("max_attempts_per_stage", True),
        ("max_runtime_seconds", float("nan")),
        ("max_runtime_seconds", float("inf")),
        ("max_runtime_seconds", 901),
        ("max_new_tokens", 100000),
        ("max_beats", -1),
        ("brief_cpu", 1),
        ("kokoro_cpu", "true"),
        ("voice", []),
    ],
)
def test_direct_chapter_plan_rejects_invalid_type_or_limit(name: str, value: object) -> None:
    with pytest.raises(chapter.ChapterStop):
        chapter.ChapterPlan(**{name: value})


def test_internal_chapter_stage_revalidates_mutated_direct_plan(tmp_path: Path) -> None:
    plan = _plan()
    object.__setattr__(plan, "max_attempts_per_stage", 3)
    callback = Mock(return_value="invalid JSON")
    with pytest.raises(chapter.ChapterStop, match="finite limits"):
        chapter._stage(
            tmp_path / "private",
            "brief",
            {"fixture": "synthetic"},
            plan,
            lambda x: x,
            callback,
        )
    callback.assert_not_called()
    assert not list(tmp_path.rglob("started-*.json"))


def test_existing_human_review_rejects_changed_verified_brief_artifact(tmp_path: Path) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    first, brief = _first_brief(runner, ref, progress, section, _plan())
    review_path = _review(tmp_path, first["brief_identity"], brief)
    saved_path = next((runner.root / "book-candidates/brief").rglob("validated.json"))
    directory = saved_path.parent
    saved = json.loads(saved_path.read_bytes())
    raw_path = directory / "raw-1.json"
    original_raw = json.loads(raw_path.read_bytes())
    changed_brief = dict(brief, argument=brief["argument"] + " A distinct synthetic qualifier.")
    changed_raw = json.dumps(changed_brief)
    raw_digest = hashlib.sha256(changed_raw.encode()).hexdigest()
    raw_path.write_bytes(
        evaluation._json_bytes(
            {
                **original_raw,
                "raw": changed_raw,
                "sha256": raw_digest,
            }
        )
    )
    payload_digest = hashlib.sha256(evaluation._json_bytes(changed_brief)).hexdigest()
    saved.update(
        {
            "payload": changed_brief,
            "raw_sha256": raw_digest,
            "payload_sha256": payload_digest,
            "artifact_identity": chapter._text_artifact_identity(
                "brief",
                saved["request_key"],
                saved["provenance"],
                raw_digest,
                payload_digest,
            ),
        }
    )
    saved_path.write_bytes(evaluation._json_bytes(saved))
    callback = Mock(side_effect=AssertionError("must reuse verified altered cache"))
    with pytest.raises(chapter.ChapterStop, match="review does not match generated brief"):
        chapter.evaluate_chapter(
            runner,
            ref,
            progress,
            section,
            review_file=review_path,
            plan=_plan(),
            brief_completion=callback,
        )
    callback.assert_not_called()
    assert first["brief_identity"] != saved["artifact_identity"]


def test_reviewed_qualified_retained_unit_cannot_lose_last_adapted_beat(
    tmp_path: Path,
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path, with_image=True)

    def nonessential(request: dict[str, object]) -> str:
        response = json.loads(_brief(request))
        response["beats"][0]["essential"] = False
        return json.dumps(response)

    first = chapter.evaluate_chapter(
        runner,
        ref,
        progress,
        section,
        plan=_plan(),
        brief_completion=nonessential,
    )
    brief_path = next((runner.root / "book-candidates/brief").rglob("validated.json"))
    brief = json.loads(brief_path.read_bytes())["payload"]
    review_path = _review(tmp_path, first["brief_identity"], brief)
    source_path = next(
        (runner.root / "book-candidates/chapter-evaluations").rglob("source-ledger.json")
    )
    source = json.loads(source_path.read_bytes())
    first_unit = brief["beats"][0]["unit_id"]
    unit = next(x for x in source["units"] if x["source_id"] == first_unit)
    review_data = json.loads(review_path.read_bytes())
    review_row = next(x for x in review_data["unit_decisions"] if x["source_id"] == first_unit)
    review_row["qualifier_quote"] = unit["text"][:40]
    review_path.write_text(json.dumps(review_data), encoding="utf-8")

    def omitted_last_beat(request: dict[str, object]) -> str:
        adaptation = json.loads(_adapt(request))
        adaptation["beats"][0]["decision"] = "omit"
        return json.dumps(adaptation)

    script = Mock(side_effect=AssertionError("omitted source must block script"))
    with pytest.raises(chapter.ChapterStop, match="exhausted"):
        chapter.evaluate_chapter(
            runner,
            ref,
            progress,
            section,
            review_file=review_path,
            plan=_plan(),
            brief_completion=nonessential,
            adaptation_completion=omitted_last_beat,
            script_completion=script,
        )
    script.assert_not_called()
    # The generated raw remains private and records the disallowed omission.
    failed = list((runner.root / "book-candidates/adaptation").rglob("failed-*.json"))
    assert failed


def test_cli_unexpected_chapter_failure_stays_generic_and_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    marker = "PRIVATE_UNEXPECTED_HOST_ERROR_7FCB"

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(marker)

    monkeypatch.setattr(chapter, "evaluate_chapter", unexpected)
    parser = argparse.ArgumentParser()
    cli.configure_books_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "books",
            "chapter-evaluate",
            "--artifact-root",
            str(runner.root),
            "--candidate-ref",
            str(ref),
            "--section-manifest",
            str(section),
            "--progress",
            str(progress),
        ]
    )
    assert cli.dispatch_books(args) == 2
    stdout, stderr = capsys.readouterr()
    assert marker not in stdout + stderr
    assert "PRIVATE_EVALUATION_FAILED_CLOSED" in stderr
    assert "PRIVATE_CHAPTER_BOUNDED_HOST_UNVERIFIED" not in stderr


def test_text_artifact_envelope_rejects_trailing_unbound_bytes(tmp_path: Path) -> None:
    root = tmp_path / "private"
    producer = Mock(return_value='{"value":"synthetic"}')
    args = (root, "brief", {"fixture": "synthetic"}, _plan())
    chapter._stage(*args, lambda x: x, producer)
    raw_path = next((root / "brief").rglob("raw-1.json"))
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(chapter.ChapterStop, match="artifact cannot be verified"):
        chapter._stage(*args, lambda x: x, producer)
    producer.assert_called_once()


def test_changed_verified_direction_text_rekeys_final_result_without_new_producer(
    tmp_path: Path,
) -> None:
    runner, ref, progress, section, _ = _source(tmp_path)
    initial, brief = _first_brief(runner, ref, progress, section, _plan())
    review = _review(tmp_path, initial["brief_identity"], brief)
    first = chapter.evaluate_chapter(runner, ref, progress, section, **_full_kwargs(review))
    root = runner.root / "book-candidates"
    direction_file = next((root / "chapter_direction").rglob("validated.json"))
    raw_file = direction_file.parent / "raw-1.json"
    validated = json.loads(direction_file.read_bytes())
    raw_record = json.loads(raw_file.read_bytes())
    changed = json.loads(json.dumps(validated["payload"]))
    changed["shots"][0]["purpose"] += " Distinct synthetic evidence presentation."
    new_raw = json.dumps(changed)
    raw_record["raw"] = new_raw
    raw_record["sha256"] = hashlib.sha256(new_raw.encode()).hexdigest()
    raw_file.write_bytes(evaluation._json_bytes(raw_record))
    validated["payload"] = changed
    validated["raw_sha256"] = raw_record["sha256"]
    validated["payload_sha256"] = hashlib.sha256(evaluation._json_bytes(changed)).hexdigest()
    validated["artifact_identity"] = chapter._text_artifact_identity(
        "chapter_direction",
        validated["request_key"],
        validated["provenance"],
        validated["raw_sha256"],
        validated["payload_sha256"],
    )
    direction_file.write_bytes(evaluation._json_bytes(validated))
    no_provider = Mock(side_effect=AssertionError("text replay must not generate"))
    replay = chapter.evaluate_chapter(
        runner,
        ref,
        progress,
        section,
        review_file=review,
        plan=_plan(),
        brief_completion=no_provider,
        adaptation_completion=no_provider,
        script_completion=no_provider,
        direction_completion=no_provider,
    )
    assert replay["direction_attempts"] == 0
    assert replay["direction_identity"] != first["direction_identity"]
    assert replay["final_result_identity"] != first["final_result_identity"]
    assert len(list((root / "chapter-evaluations").rglob("chapter-result-*.json"))) == 2
    no_provider.assert_not_called()


@pytest.mark.parametrize("claim_change", ["delete", "replace", "partial"])
@pytest.mark.parametrize("rejection", ["malformed", "schema", "source"])
def test_rejected_chapter_text_cannot_retry_with_lost_claim(
    tmp_path: Path, claim_change: str, rejection: str
) -> None:
    root = tmp_path / "private"
    first_response = "{" if rejection == "malformed" else json.dumps(
        {"source": "unattested"} if rejection == "source" else {"wrong": "schema"}
    )
    responses: list[str] = []

    def producer(_request: dict[str, object]) -> str:
        responses.append(first_response)
        if len(responses) != 1:
            raise AssertionError("changed first claim must forbid second producer call")
        started = next((root / "brief").rglob("started-1.json"))
        if claim_change == "delete":
            started.unlink()
        elif claim_change == "partial":
            started.write_bytes(b"{")
        else:
            record = json.loads(started.read_bytes())
            record["owner"] = "0" * 32
            started.write_bytes(evaluation._json_bytes(record))
        return first_response

    def validator(value: object) -> dict:
        if not isinstance(value, dict) or (
            rejection == "schema" and "source" not in value
        ) or (rejection == "source" and value.get("source") == "unattested"):
            raise chapter.ChapterStop("synthetic schema/source rejection")
        return value

    args = (root, "brief", {"fixture": "claim-invalid-response"}, _plan())
    with pytest.raises(chapter.ChapterStop):
        chapter._stage(*args, validator, producer)
    assert len(responses) == 1
    raw_files = list((root / "brief").rglob("raw-1.json"))
    assert len(raw_files) == 1
    recorded = json.loads(raw_files[0].read_bytes())
    assert recorded["raw"] == first_response
    assert recorded["attempt"] == 1
    assert recorded["sha256"] == hashlib.sha256(first_response.encode()).hexdigest()
    assert not list((root / "brief").rglob("started-2.json"))
    assert not list((root / "brief").rglob("validated.json"))
    with pytest.raises(chapter.ChapterStop, match="prior unresolved"):
        chapter._stage(*args, validator, producer)
    assert len(responses) == 1


def test_healthy_rejected_chapter_first_attempt_retries_with_distinct_claims(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    answers = Mock(side_effect=["{", '{"value":"accepted synthetic"}'])
    args = (root, "brief", {"fixture": "healthy-two-attempts"}, _plan())
    first, identity, _, attempts = chapter._stage(*args, lambda x: x, answers)
    assert first == {"value": "accepted synthetic"}
    assert attempts == 2
    directory = next((root / "brief").iterdir())
    claims = [
        json.loads((directory / f"started-{index}.json").read_bytes())
        for index in (1, 2)
    ]
    raws = [
        json.loads((directory / f"raw-{index}.json").read_bytes())
        for index in (1, 2)
    ]
    assert [type(claim["attempt"]) for claim in claims] == [int, int]
    assert [raw["attempt"] for raw in raws] == [1, 2]
    assert [claim["attempt"] for claim in claims] == [1, 2]
    assert claims[0]["owner"] != claims[1]["owner"]
    assert [raw["owner"] for raw in raws] == [claim["owner"] for claim in claims]
    assert json.loads((directory / "failed-1.json").read_bytes())["status"] == (
        "SCHEMA_OR_SOURCE_REJECTED"
    )
    replay, replay_id, _, replay_attempts = chapter._stage(*args, lambda x: x, answers)
    assert (replay, replay_id, replay_attempts) == (first, identity, 0)
    assert answers.call_count == 2


@pytest.mark.parametrize("envelope", ["claim", "raw", "validated"])
@pytest.mark.parametrize("attempt", [True, False, 0, -1, 2, 3, 1.0, "1"])
def test_canonical_chapter_text_attempt_envelope_requires_exact_bounded_int(
    tmp_path: Path, envelope: str, attempt: object
) -> None:
    root = tmp_path / "private"
    producer = Mock(return_value='{"value":"synthetic"}')
    args = (
        root,
        "brief",
        {"fixture": "attempt-envelope"},
        _plan(max_attempts_per_stage=1),
    )
    chapter._stage(*args, lambda x: x, producer)
    directory = next((root / "brief").iterdir())
    target = (
        directory / "started-1.json"
        if envelope == "claim"
        else directory / "raw-1.json"
        if envelope == "raw"
        else directory / "validated.json"
    )
    record = json.loads(target.read_bytes())
    record["attempt"] = attempt
    target.write_bytes(evaluation._json_bytes(record))
    with pytest.raises(chapter.ChapterStop):
        chapter._stage(*args, lambda x: x, producer)
    producer.assert_called_once()


def test_consistently_rewritten_second_attempt_rejected_under_one_attempt_plan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    producer = Mock(return_value='{"value":"synthetic"}')
    args = (
        root,
        "brief",
        {"fixture": "attempt-ceiling"},
        _plan(max_attempts_per_stage=1),
    )
    chapter._stage(*args, lambda x: x, producer)
    directory = next((root / "brief").iterdir())
    for prefix in ("started", "raw"):
        original = directory / f"{prefix}-1.json"
        record = json.loads(original.read_bytes())
        record["attempt"] = 2
        (directory / f"{prefix}-2.json").write_bytes(evaluation._json_bytes(record))
        original.unlink()
    validated = directory / "validated.json"
    record = json.loads(validated.read_bytes())
    record["attempt"] = 2
    validated.write_bytes(evaluation._json_bytes(record))
    with pytest.raises(chapter.ChapterStop, match="artifact cannot be verified"):
        chapter._stage(*args, lambda x: x, producer)
    producer.assert_called_once()

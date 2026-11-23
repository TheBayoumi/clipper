"""Model-free two-chapter Codespace integration and full-source runtime gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_books_whole_book import ingested

from clipper.books import chapter, chapter_host, evaluation, scripting
from clipper.books.orchestrator import BookOrchestrator


def _private_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[BookOrchestrator, Path, dict]:
    runner, acquired, _ = ingested(tmp_path, paragraphs=12, with_image=True)
    monkeypatch.setenv("CODESPACE_NAME", "synthetic-private-host")
    section_ids = [item.chapter_id for item in acquired.book.sections][:2]
    assert len(section_ids) == 2
    plan = asdict(
        chapter.ChapterPlan(
            brief_cpu=True,
            adaptation_cpu=True,
            script_cpu=True,
            direction_cpu=True,
            kokoro_cpu=True,
        )
    )
    config = {
        "schema": chapter_host.BATCH_SCHEMA,
        "candidate_ref": str(acquired.analysis_ref_path),
        "chapter_1": {
            "section_ids": [section_ids[0]],
            "scope_sha256": None,
            "review_file": None,
        },
        "chapter_7": {
            "section_ids": [section_ids[1]],
            "scope_sha256": None,
            "review_file": None,
        },
        "plan": {"schema": chapter.PLAN_SCHEMA, **plan},
    }
    path = runner.root / "two-private-chapters.json"
    path.write_bytes(evaluation._json_bytes(config))
    return runner, path, config


def _call(
    runner: BookOrchestrator, path: Path, mode: str, *, age_seconds: int = 60
) -> dict[str, object]:
    return chapter_host.execute(
        mode=mode,
        artifact_root=runner.root,
        manifest_path=path,
        created_at=(datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat(),
    )


def _locked(runner: BookOrchestrator, path: Path, config: dict) -> dict[str, object]:
    inspection = _call(runner, path, "inspect")
    assert inspection["status"] == "SCOPE_INSPECTION_REQUIRES_HUMAN_SELECTION_LOCK"
    assert inspection["publish_eligible"] is False
    for number in (1, 7):
        config[f"chapter_{number}"]["scope_sha256"] = inspection["chapters"][str(number)][
            "scope_sha256"
        ]
    path.write_bytes(evaluation._json_bytes(config))
    return inspection


def test_scope_inspection_only_two_sections_and_exposes_source_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    result = _locked(runner, path, config)
    assert set(result["chapters"]) == {"1", "7"}
    for info in result["chapters"].values():
        assert info["work_units"] >= 1
        assert info["words"] >= 1
        assert "skipped_markers" not in info
    report_file = runner.root / "book-candidates/private-host/scope-inspection.json"
    report = json.loads(report_file.read_bytes())
    assert report["semantic_scope_approval"] == "NOT_ATTESTED"
    assert report["footnote_caption_figure_completeness"] == (
        "UNVERIFIED_REQUIRES_SOURCE_INSPECTION"
    )
    assert report["parent_full_book_complete"] is False
    assert report["unselected_sections"]
    assert "source_counts" in report["chapters"]["1"]
    assert len(list((runner.root / "book-candidates/section-scopes").glob("*.json"))) == 2


@pytest.mark.parametrize(
    "bad",
    ["missing", "overlap", "bad_digest", "invalid_plan", "unknown_key"],
)
def test_codespace_manifest_rejects_unlocked_or_malformed_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    if bad == "missing":
        del config["chapter_7"]
    elif bad == "overlap":
        config["chapter_7"]["section_ids"] = config["chapter_1"]["section_ids"]
    elif bad == "bad_digest":
        config["chapter_1"]["scope_sha256"] = "not-a-sha"
    elif bad == "invalid_plan":
        config["plan"]["max_attempts_per_stage"] = True
    else:
        config["unapproved"] = "third chapter"
    path.write_bytes(evaluation._json_bytes(config))
    with pytest.raises((chapter_host.PrivateHostStop, chapter.ChapterStop)):
        _call(runner, path, "inspect")


def test_codespace_scope_lock_mismatch_blocks_mining_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    config["chapter_1"]["scope_sha256"] = "f" * 64
    path.write_bytes(evaluation._json_bytes(config))
    called = Mock(side_effect=AssertionError("no model before scope lock"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", called)
    with pytest.raises(chapter_host.PrivateHostStop, match="locked"):
        _call(runner, path, "brief")
    called.assert_not_called()


def test_codespace_created_time_and_private_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, _ = _private_setup(tmp_path, monkeypatch)
    now = datetime.now(UTC)
    assert chapter_host._created_at((now - timedelta(minutes=30)).isoformat(), now) < now
    for time in (
        "not-a-time",
        "2026-01-01T12:00:00",
        (now + timedelta(minutes=1)).isoformat(),
        (now - timedelta(seconds=6900)).isoformat(),
        (now - timedelta(seconds=7201)).isoformat(),
    ):
        with pytest.raises(chapter_host.PrivateHostStop):
            chapter_host._created_at(time, now)
    with pytest.raises(chapter_host.PrivateHostStop):
        chapter_host._private_path(str(tmp_path / "outside.json"), runner.root)
    monkeypatch.delenv("CODESPACE_NAME")
    with pytest.raises(chapter_host.PrivateHostStop, match="Codespace"):
        _call(runner, path, "inspect")


def test_two_chapters_brief_stops_for_independent_review_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    requests: list[tuple[str, ...]] = []
    callbacks: list[tuple[object, object]] = []
    progress_root = runner.root / "book-candidates/analysis/progress/synthetic"
    progress_root.mkdir(parents=True)

    def fake_mine(self: BookOrchestrator, ref_path: Path, **kwargs: object) -> dict:
        assert kwargs["local_cpu"] is True
        assert kwargs["max_new_units"] >= 1
        section_ids = kwargs["section_ids"]
        assert isinstance(section_ids, tuple)
        requests.append(section_ids)
        progress = progress_root / f"{len(requests)}.json"
        progress.write_bytes(evaluation._json_bytes({"synthetic": True}))
        return {
            "status": "complete",
            "progress_path": str(progress),
            "total_work_units": kwargs["max_new_units"],
        }

    def fake_evaluate(
        _runner: BookOrchestrator,
        _ref: Path,
        _progress: Path,
        section: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        callbacks.append((kwargs["brief_completion"], kwargs["narration_producer"]))
        assert kwargs["brief_completion"] is scripting._local_chapter_completion
        assert kwargs["script_completion"] is scripting._local_chapter_completion
        assert kwargs["direction_completion"] is scripting._local_chapter_completion
        assert kwargs["narration_producer"] is chapter_host.narration._kokoro_wav
        assert section.is_file()
        if kwargs["review_file"] is None:
            scope = json.loads(section.read_bytes())
            # The phase is deliberately non-semantic and yields no approval.
            return {
                "status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
                "brief_identity": hashlib.sha256(evaluation._json_bytes(scope)).hexdigest(),
                "section_scope_sha256": hashlib.sha256(section.read_bytes()).hexdigest(),
            }
        assert isinstance(kwargs["review_file"], Path)
        assert kwargs["review_file"].is_file()
        return {
            "status": "CHAPTER_PREPRODUCTION_READY_UNREVIEWED",
            "brief_identity": "a" * 64,
            "final_result_identity": "b" * 64,
            "audio_provenance": "ACTUAL_CPU_KOKORO",
        }

    packets: list[int] = []
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", fake_mine)
    monkeypatch.setattr(chapter, "evaluate_chapter", fake_evaluate)
    monkeypatch.setattr(chapter_host, "_review_packet", lambda _r, n, _s: packets.append(n))
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    phase = _call(runner, path, "brief")
    assert phase["status"] == "PRIVATE_TWO_CHAPTER_CHECKPOINT"
    assert [phase["chapters"][str(n)]["status"] for n in (1, 7)] == [
        "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
        "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
    ]
    assert len(set(requests)) == 2
    assert packets == [1, 7]
    assert len(callbacks) == 2
    for number in (1, 7):
        record = json.loads(
            (
                runner.root / f"book-candidates/private-host/chapter-{number}-brief-record.json"
            ).read_bytes()
        )
        assert record["phase"] == "AWAITING_HUMAN_REVIEW"
        review_path = runner.root / f"human-review-{number}.json"
        review_path.write_text("UNATTESTED SYNTHETIC FILE", encoding="utf-8")
        config[f"chapter_{number}"]["review_file"] = str(review_path)
    path.write_bytes(evaluation._json_bytes(config))
    resumed = _call(runner, path, "resume")
    assert all(
        item["status"] == "CHAPTER_PREPRODUCTION_READY_UNREVIEWED"
        for item in resumed["chapters"].values()
    )
    assert len(requests) == 2
    assert len(callbacks) == 4
    # The actual chapter.evaluate_chapter validator (not mocked in production)
    # alone decides whether the human review file is valid.


def test_missing_review_or_incomplete_mining_never_auto_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    called = Mock(return_value={"status": "incomplete", "total_work_units": 2})
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", called)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    assert all(
        item["status"] == "SCOPED_MINING_INCOMPLETE"
        for item in _call(runner, path, "brief")["chapters"].values()
    )
    assert called.call_count == 2
    assert not list((runner.root / "book-candidates/private-host").glob("*brief-record.json"))
    evaluation_called = Mock(side_effect=AssertionError("no model without brief and review"))
    monkeypatch.setattr(chapter, "evaluate_chapter", evaluation_called)
    assert all(
        item["status"] == "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE"
        for item in _call(runner, path, "resume")["chapters"].values()
    )
    evaluation_called.assert_not_called()


def test_pinned_real_chapter_preflight_occurs_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = Mock(return_value='{"value":"private synthetic"}')
    verify = Mock(side_effect=scripting.NicheVideoError("token context full"))
    monkeypatch.setattr(scripting, "_local_chapter_completion", callback)
    monkeypatch.setattr(scripting, "_chapter_context_tokens", verify)
    root = tmp_path / "private"
    with pytest.raises(chapter.ChapterStop, match="token context"):
        chapter._stage(
            root,
            "brief",
            {"fixture": "real-adapter-bound"},
            chapter.ChapterPlan(),
            lambda item: item,
            callback,
        )
    callback.assert_not_called()
    verify.assert_called_once()
    assert not list(root.rglob("started-*.json"))


def test_real_model_identity_is_separate_from_synthetic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = '{"value":"synthetic exact raw"}'
    real = Mock(return_value=reply)
    monkeypatch.setattr(scripting, "_local_chapter_completion", real)
    monkeypatch.setattr(scripting, "_chapter_context_tokens", lambda _: 12)
    root = tmp_path / "private"
    args = (root, "brief", {"fixture": "provenance"}, chapter.ChapterPlan())
    _, real_key, real_origin, real_attempt = chapter._stage(*args, lambda x: x, real)
    assert (real_origin, real_attempt) == ("ACTUAL_CPU_QWEN", 1)
    synthetic = Mock(return_value=reply)
    _, synthetic_key, origin, attempts = chapter._stage(*args, lambda x: x, synthetic)
    assert (origin, attempts) == ("INJECTED_TEST_PRODUCER", 1)
    assert real_key != synthetic_key or len(list((root / "brief").iterdir())) == 2
    real.assert_called_once()
    synthetic.assert_called_once()


def test_chapter_token_preflight_model_free_with_mocked_public_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = list(range(40))
    monkeypatch.setattr(scripting, "_public_book_context", lambda: (tokenizer, 40960))
    request: dict[str, object] = {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 8192,
        "messages": [{"role": "user", "content": "purely synthetic"}],
    }
    assert scripting._chapter_context_tokens(request) == 40
    tokenizer.apply_chat_template.return_value = list(range(40000))
    with pytest.raises(scripting.NicheVideoError, match="token context"):
        scripting._chapter_context_tokens(request)
    request["seed"] = True
    with pytest.raises(scripting.NicheVideoError, match="pinned model"):
        scripting._chapter_context_tokens(request)


def test_chapter_incomplete_real_generation_retains_exact_raw(
    tmp_path: Path,
) -> None:
    raw = '{"value":"valid JSON but missing EOS"}'
    producer = Mock(side_effect=scripting.IncompleteConstrainedResponse(raw))
    root = tmp_path / "private"
    args = (root, "brief", {"fixture": "incomplete"}, chapter.ChapterPlan())
    with pytest.raises(chapter.ChapterStop, match="without EOS"):
        chapter._stage(*args, lambda x: x, producer)
    stored = json.loads(next(root.rglob("raw-1.json")).read_bytes())
    assert stored["raw"] == raw
    assert stored["sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert json.loads(next(root.rglob("failed-1.json")).read_bytes())["status"] == ("MISSING_EOS")
    with pytest.raises(chapter.ChapterStop, match="prior unresolved"):
        chapter._stage(*args, lambda x: x, producer)
    producer.assert_called_once()


def test_real_chapter_allows_full_large_request_only_after_exact_context_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # >25 KiB of synthetic source: conservative offline preflight rejects it;
    # verified CPU mode may accept only with pinned-tokenizer exact counting.
    payload = {
        "schema": chapter.BRIEF_SCHEMA,
        "source": {"units": [{"source_id": "unit-synthetic", "text": "invented " * 4200}]},
    }
    plan = chapter.ChapterPlan(max_new_tokens=8192)
    with pytest.raises(chapter.ChapterStop, match="context"):
        chapter._stage(
            tmp_path / "conservative",
            "brief",
            payload,
            plan,
            lambda value: value,
            Mock(side_effect=AssertionError("must not call")),
        )
    observed: list[dict[str, object]] = []

    def fake_context(request: dict[str, object]) -> int:
        observed.append(request)
        assert "invented " * 4200 in request["messages"][1]["content"]
        return 12_345

    monkeypatch.setattr(scripting, "_chapter_context_tokens", fake_context)
    real = Mock(return_value='{"value":"synthetic"}')
    monkeypatch.setattr(scripting, "_local_chapter_completion", real)
    result = chapter._stage(
        tmp_path / "exact",
        "brief",
        payload,
        plan,
        lambda value: value,
        real,
    )
    assert result[0] == {"value": "synthetic"}
    assert result[2] == "ACTUAL_CPU_QWEN"
    assert len(observed) == 1
    real.assert_called_once()


def test_actual_kokoro_provenance_requires_exact_real_adapter_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_books_chapter import _wav

    real = Mock(side_effect=_wav)
    monkeypatch.setattr(chapter.narration, "_kokoro_wav", real)
    script = {
        "scenes": [{"scene_id": "scene-01", "narration": "Invented narration.", "dialogue": []}]
    }
    records, origin, digest = chapter._narrate(
        tmp_path / "private", script, chapter.ChapterPlan(), real
    )
    assert origin == "ACTUAL_CPU_KOKORO"
    assert records[0]["provenance"] == origin
    assert len(digest) == 64
    real.assert_called_once()

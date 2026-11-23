"""Model-free two-chapter Codespace integration and full-source runtime gates."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_books_whole_book import ingested

from clipper.books import (
    chapter,
    chapter_host,
    chapter_lifecycle,
    evaluation,
    scripting,
    selected_scope,
)
from clipper.books.orchestrator import BookOrchestrator


def _private_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[BookOrchestrator, Path, dict]:
    # Export from retained source, transfer ONLY two selected bundles to host.
    # The host never receives an ordinary ingest ref, extracted BookText, or EPUB.
    owner = tmp_path / "owner-retained-source"
    owner.mkdir()
    exporter, acquired, _ = ingested(owner, paragraphs=12, with_image=True)
    monkeypatch.setenv("CODESPACE_NAME", "synthetic-private-host")
    monkeypatch.setattr(
        chapter_lifecycle,
        "verify_lifecycle",
        lambda _root, *, caller_created_at, codespace_name: chapter_lifecycle.VerifiedLifecycle(
            42,
            codespace_name,
            datetime.fromisoformat(caller_created_at).astimezone(UTC),
            "a" * 40,
            datetime.fromisoformat(caller_created_at).astimezone(UTC) + timedelta(seconds=7200),
        ),
    )
    sections = [item.chapter_id for item in acquired.book.sections]
    host = tmp_path / "host-only"
    bundle_dir = host / "book-candidates" / "selected-scopes"
    bundle_dir.mkdir(parents=True)
    scopes: dict[str, dict] = {}
    for number, section in ((1, sections[0]), (7, sections[1])):
        exported, sha = selected_scope.export_chapter(
            exporter,
            acquired.analysis_ref_path,
            chapter_number=number,
            chapter_section_ids=(section,),
        )
        target = bundle_dir / f"{sha}.json"
        target.write_bytes(exported.read_bytes())
        scopes[f"chapter_{number}"] = {
            "bundle_path": str(target),
            "bundle_sha256": sha,
            "scope_sha256": None,
            "review_file": None,
        }
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
        **scopes,
        "plan": {"schema": chapter.PLAN_SCHEMA, **plan},
    }
    path = host / "two-private-chapters.json"
    path.write_bytes(evaluation._json_bytes(config))
    runner = BookOrchestrator(host)
    assert not list(host.rglob("*.epub"))
    assert not list(host.rglob("analysis-refs"))
    assert not list(host.rglob("extracted"))
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
    report_files = list(
        (runner.root / "book-candidates/private-host").glob("scope-inspection-*.json")
    )
    assert len(report_files) == 1
    report = json.loads(report_files[0].read_bytes())
    assert report["semantic_approval"] == "NOT_ATTESTED"
    assert report["source_linkage_review"] == "REQUIRED"
    assert report["parent_full_book_complete"] is False
    assert report["parent_original_bytes_reverified_on_host"] is False
    assert report["chapters"]["1"]["selected_reading_blocks"]
    assert report["chapters"]["1"]["bundle_sha256"] == config["chapter_1"]["bundle_sha256"]
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
        config["chapter_7"]["bundle_path"] = config["chapter_1"]["bundle_path"]
    elif bad == "bad_digest":
        config["chapter_1"]["scope_sha256"] = "not-a-sha"
    elif bad == "invalid_plan":
        config["plan"]["max_attempts_per_stage"] = True
    else:
        config["unapproved"] = "third chapter"
    path.write_bytes(evaluation._json_bytes(config))
    with pytest.raises((chapter_host.PrivateHostStop, chapter.ChapterStop)):
        _call(runner, path, "inspect")


@pytest.mark.parametrize("number", [1, 7])
def test_codespace_scope_lock_mismatch_blocks_mining_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, number: int
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    config[f"chapter_{number}"]["scope_sha256"] = "f" * 64
    path.write_bytes(evaluation._json_bytes(config))
    called = Mock(side_effect=AssertionError("no model before scope lock"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", called)
    with pytest.raises(chapter_host.PrivateHostStop, match="mismatch"):
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
        assert kwargs["selected_scope_path"] == _ref
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
        assert kwargs["expected_brief_identity"] is not None
        assert callable(kwargs["on_resume_verified"])
        kwargs["on_resume_verified"]()
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
        item["status"] == "SCOPED_MINING_INCOMPLETE"
        for item in _call(runner, path, "resume")["chapters"].values()
    )
    evaluation_called.assert_not_called()


def test_pinned_real_chapter_preflight_occurs_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = Mock(return_value='{"value":"private synthetic"}')
    verify = Mock(side_effect=scripting.NicheVideoError("token context full"))
    monkeypatch.setattr(scripting, "_local_chapter_completion", callback)
    monkeypatch.setattr(scripting, "_ORIGINAL_LOCAL_CHAPTER_COMPLETION", callback)
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
    real = Mock(
        return_value=scripting.PinnedQwenRawReceipt(
            raw=reply,
            raw_sha256=hashlib.sha256(reply.encode()).hexdigest(),
            generated_token_count=17,
            terminal_token_id=151645,
            eos_decision="EOS_VERIFIED",
        )
    )
    monkeypatch.setattr(scripting, "_local_chapter_completion", real)
    monkeypatch.setattr(scripting, "_ORIGINAL_LOCAL_CHAPTER_COMPLETION", real)
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
    raw = '{"value":"synthetic"}'
    real = Mock(
        return_value=scripting.PinnedQwenRawReceipt(
            raw=raw,
            raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            generated_token_count=17,
            terminal_token_id=151645,
            eos_decision="EOS_VERIFIED",
        )
    )
    # This fixture deliberately stands in for the original trusted provider.
    # Generic injected callbacks must never receive real-model provenance.
    monkeypatch.setattr(scripting, "_local_chapter_completion", real)
    monkeypatch.setattr(scripting, "_ORIGINAL_LOCAL_CHAPTER_COMPLETION", real)
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


def test_private_scope_inspection_repeat_is_content_addressed_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, _ = _private_setup(tmp_path, monkeypatch)
    first = _call(runner, path, "inspect")
    second = _call(runner, path, "inspect")
    assert first == second
    report_dir = runner.root / "book-candidates/private-host"
    reports = list(report_dir.glob("scope-inspection-*.json"))
    assert len(reports) == 1
    assert hashlib.sha256(reports[0].read_bytes()).hexdigest() == first["inspection_sha256"]


def test_private_review_packet_contains_full_source_and_invalid_human_template(
    tmp_path: Path,
) -> None:
    from test_books_chapter import _first_brief, _plan, _source

    runner, ref, progress, section, _candidates = _source(tmp_path, with_image=True)
    summary, brief = _first_brief(runner, ref, progress, section, _plan())
    chapter_host._review_packet(runner, 1, summary)
    packet_path = runner.root / "book-candidates/private-host/chapter-1-review-packet.json"
    packet = json.loads(packet_path.read_bytes())
    assert packet["brief_identity"] == summary["brief_identity"]
    assert packet["brief"] == brief
    assert len(packet["source_ledger"]["units"]) == summary["included_units"]
    assert packet["source_ledger"]["reading_order"]
    assert packet["source_ledger"]["gaps"]
    template = packet["review_template"]
    assert template["brief_sha256"] == packet["brief_identity"]
    assert template["reviewer"] == ""
    assert {row["decision"] for row in template["decisions"]} == {"REVIEW_REQUIRED"}
    assert {row["review"] for row in template["unit_decisions"]} == {"REVIEW_REQUIRED"}
    assert packet["publish_eligible"] is False
    with pytest.raises(chapter_host.PrivateHostStop, match="source ledger"):
        chapter_host._review_packet(runner, 7, {**summary, "section_scope_sha256": "f" * 64})
    with pytest.raises(chapter_host.PrivateHostStop, match="verified brief"):
        chapter_host._review_packet(runner, 7, {**summary, "brief_identity": "f" * 64})


def test_pinned_chapter_cpu_adapter_generate_eos_and_memory_error_paths_are_model_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    from clipper.books import narration

    seeds: list[int] = []

    class FakeTorch:
        @staticmethod
        def manual_seed(value: int) -> None:
            seeds.append(value)

        @staticmethod
        def inference_mode() -> object:
            return nullcontext()

    class FakeTokenizer:
        eos_token_id = 99

        def __init__(self) -> None:
            self.length = 2

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> dict:
            return {"input_ids": SimpleNamespace(shape=(1, self.length))}

        def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            assert ids in ([99], [55])
            return '{"value":"synthetic CPU generation with no weights"}'

    tokenizer = FakeTokenizer()
    generated = Mock(return_value=[[10, 20, 99]])
    model = SimpleNamespace(
        device="cpu",
        generate=generated,
        generation_config=SimpleNamespace(eos_token_id=99),
    )
    original_import = scripting.importlib.import_module
    monkeypatch.setattr(
        scripting.importlib,
        "import_module",
        lambda name: FakeTorch() if name == "torch" else original_import(name),
    )
    monkeypatch.setattr(scripting, "_chapter_context_tokens", lambda _: 2)
    monkeypatch.setattr(scripting, "_script_pipeline", lambda: (model, tokenizer))
    narration._kokoro_pipeline.cache_clear()
    request: dict[str, object] = {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 1024,
        "messages": [{"role": "user", "content": "synthetic"}],
    }
    result = scripting._local_chapter_completion(request)
    expected_raw = '{"value":"synthetic CPU generation with no weights"}'
    assert isinstance(result, scripting.PinnedQwenRawReceipt)
    assert result.raw == expected_raw
    assert result.raw_sha256 == hashlib.sha256(expected_raw.encode("utf-8")).hexdigest()
    assert result.generated_token_count == 1
    assert result.terminal_token_id == 99
    assert result.eos_decision == "EOS_VERIFIED"
    assert seeds == [42]
    assert generated.call_count == 1
    assert generated.call_args.kwargs["max_new_tokens"] == 1024
    generated.return_value = [[10, 20, 55]]
    with pytest.raises(scripting.IncompleteConstrainedResponse) as incomplete:
        scripting._local_chapter_completion(request)
    assert incomplete.value.raw == expected_raw
    model.device = "cuda"
    with pytest.raises(scripting.NicheVideoError, match="CPU"):
        scripting._local_chapter_completion(request)
    model.device = "cpu"
    tokenizer.length = 40960
    with pytest.raises(scripting.NicheVideoError, match="context"):
        scripting._local_chapter_completion(request)
    assert generated.call_count == 2


def test_private_host_cli_status_codes_do_not_expose_private_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "brief",
        "--artifact-root",
        str(tmp_path / "private"),
        "--manifest",
        str(tmp_path / "private/manifest.json"),
        "--codespace-created-at",
        "2026-11-23T06:30:00+00:00",
    ]

    def output(status: str) -> dict[str, object]:
        return {
            "status": "PRIVATE_TWO_CHAPTER_CHECKPOINT",
            "chapters": {"1": {"status": status}, "7": {"status": status}},
            "publish_eligible": False,
        }

    monkeypatch.setattr(
        chapter_host, "execute", lambda **_: output("AWAITING_HUMAN_CHAPTER_BEAT_REVIEW")
    )
    assert chapter_host.main(args) == 0
    stdout, stderr = capsys.readouterr()
    assert "PRIVATE_TWO_CHAPTER_CHECKPOINT" in stdout
    assert not stderr
    monkeypatch.setattr(chapter_host, "execute", lambda **_: output("SCOPED_MINING_INCOMPLETE"))
    assert chapter_host.main(args) == 2
    capsys.readouterr()

    def failure(**_kwargs: object) -> None:
        raise ValueError("SYNTHETIC_PRIVATE_CONTENT_MUST_NOT_LEAK")

    monkeypatch.setattr(chapter_host, "execute", failure)
    assert chapter_host.main(args) == 2
    stdout, stderr = capsys.readouterr()
    assert not stdout
    assert "PRIVATE_CHAPTER_HOST_FAILED_CLOSED" in stderr
    assert "SYNTHETIC_PRIVATE_CONTENT" not in stderr


def test_host_deadline_paths_and_strict_manifest_permissions_do_not_start_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy

    runner, path, config = _private_setup(tmp_path, monkeypatch)
    with pytest.raises(chapter_host.PrivateHostStop, match="budget exhausted"):
        chapter_host._check_deadline(datetime.now(UTC) - timedelta(hours=2))
    with pytest.raises(chapter_host.PrivateHostStop, match="absolute"):
        chapter_host._private_path("relative-private-file.json", runner.root)
    with pytest.raises(chapter_host.PrivateHostStop, match="does not exist"):
        chapter_host._private_path(str(runner.root / "missing.json"), runner.root, must_exist=True)
    for change in ("candidate_ref", "section_ids", "review_file", "bad_schema"):
        corrupt = copy.deepcopy(config)
        if change == "candidate_ref":
            corrupt["candidate_ref"] = 23
        elif change == "section_ids":
            corrupt["chapter_1"]["section_ids"] = []
        elif change == "review_file":
            corrupt["chapter_7"]["review_file"] = 123
        else:
            corrupt["schema"] = "unrecognized"
        with pytest.raises(chapter_host.PrivateHostStop):
            chapter_host._config(corrupt)
    with pytest.raises(chapter_host.PrivateHostStop, match="unsupported"):
        _call(runner, path, "unknown_phase")
    config["plan"]["brief_cpu"] = False
    path.write_bytes(evaluation._json_bytes(config))
    with pytest.raises(chapter_host.PrivateHostStop, match="permissions"):
        _call(runner, path, "brief")
    # The repository checkout is never an authorized private artifact root.
    with pytest.raises(chapter_host.PrivateHostStop, match="outside"):
        chapter_host.execute(
            mode="inspect",
            artifact_root=Path(chapter_host.__file__).resolve().parents[3],
            manifest_path=path,
            created_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        )


def test_direct_host_api_rejects_missing_external_supervisor_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, _ = _private_setup(tmp_path, monkeypatch)
    called = Mock(side_effect=AssertionError("no provider without signed supervisor"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", called)
    monkeypatch.setattr(
        chapter_lifecycle,
        "verify_lifecycle",
        Mock(side_effect=chapter_lifecycle.LifecycleStop("external proof absent")),
    )
    with pytest.raises(chapter_lifecycle.LifecycleStop, match="external proof"):
        _call(runner, path, "brief")
    called.assert_not_called()


def test_interrupted_chapter_7_retains_chapter_1_verified_phase_and_no_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    seen: list[str] = []

    def mining(_self: BookOrchestrator, bundle: Path, **_options: object) -> dict:
        seen.append(bundle.stem)
        progress = runner.root / "progress-synthetic.json"
        progress.write_bytes(evaluation._json_bytes({"synthetic": True}))
        return {
            "status": "complete",
            "progress_path": str(progress),
            "total_work_units": 1,
        }

    def chapter_brief(
        _runner: BookOrchestrator,
        _bundle: Path,
        _progress: Path,
        _section: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {
            "status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
            "brief_identity": "b" * 64,
            "section_scope_sha256": "c" * 64,
        }

    monkeypatch.setattr(BookOrchestrator, "mine_candidate", mining)
    monkeypatch.setattr(chapter, "evaluate_chapter", chapter_brief)
    monkeypatch.setattr(chapter_host, "_review_packet", lambda *_: None)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    original_check = chapter_host._check_deadline
    checks = 0

    def deadline(created: datetime) -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise chapter_host.PrivateHostStop("one-host shared execution budget exhausted")
        original_check(created)

    monkeypatch.setattr(chapter_host, "_check_deadline", deadline)
    summary = _call(runner, path, "brief")
    assert summary["chapters"]["1"]["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert summary["chapters"]["7"]["status"] == (
        "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE"
    )
    assert len(seen) == 1
    persisted = list(
        (runner.root / "book-candidates/private-host/chapter-1-phase-events").glob("*.json")
    )
    assert persisted
    assert json.loads(sorted(persisted)[-1].read_bytes())["phase"] == "AWAITING_HUMAN_REVIEW"
    replay = _call(runner, path, "brief")
    assert replay["chapters"]["1"]["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert replay["chapters"]["7"]["status"] == (
        "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE"
    )
    assert len(seen) == 1


def test_in_flight_chapter_phase_reentry_stays_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    bundle = config["chapter_1"]["bundle_sha256"]
    scope = config["chapter_1"]["scope_sha256"]
    chapter_host._phase(runner, 1, bundle, scope, "NOT_STARTED")
    chapter_host._phase(runner, 1, bundle, scope, "MINING_IN_FLIGHT")
    # The other selected chapter is independently NOT_STARTED. In order
    # to assert zero calls, mark its own unrelated request interrupted too.
    other = config["chapter_7"]
    chapter_host._phase(runner, 7, other["bundle_sha256"], other["scope_sha256"], "NOT_STARTED")
    chapter_host._phase(
        runner, 7, other["bundle_sha256"], other["scope_sha256"], "MINING_IN_FLIGHT"
    )
    provider = Mock(side_effect=AssertionError("interrupted attempts were consumed"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", provider)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    result = _call(runner, path, "brief")
    assert result["chapters"]["1"]["status"] == "INTERRUPTED_DURING_MINING"
    assert result["chapters"]["7"]["status"] == "INTERRUPTED_DURING_MINING"
    provider.assert_not_called()


@pytest.mark.parametrize("damage", ["empty", "folder_symlink", "entry_symlink", "canonical_tamper"])
def test_private_phase_journal_missing_or_rewritten_records_never_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    digest = config["chapter_1"]["bundle_sha256"]
    scope = config["chapter_1"]["scope_sha256"]
    chapter_host._phase(runner, 1, digest, scope, "NOT_STARTED")
    folder = runner.root / "book-candidates/private-host/chapter-1-phase-events"
    event = next(folder.glob("*.json"))
    if damage == "empty":
        event.unlink()
    elif damage == "folder_symlink":
        original = folder.with_name("phase-events-retained")
        folder.rename(original)
        folder.symlink_to(original, target_is_directory=True)
    elif damage == "entry_symlink":
        copied = folder / "retained-record"
        event.rename(copied)
        event.symlink_to(copied)
    else:
        record = json.loads(event.read_bytes())
        record["publish_eligible"] = True
        event.write_bytes(evaluation._json_bytes(record))
    with pytest.raises(chapter_host.PrivateHostStop, match="phase history"):
        chapter_host._phase_records(runner.root, 1, digest, scope)
    no_model = Mock(side_effect=AssertionError("missing phase journal cannot restart mining"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", no_model)
    with pytest.raises(chapter_host.PrivateHostStop, match="phase history"):
        _call(runner, path, "brief")
    no_model.assert_not_called()


def test_private_host_rejects_unapproved_mode_and_incomplete_cpu_plan_before_mining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    producer = Mock(side_effect=AssertionError("never infer for invalid runtime settings"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", producer)
    with pytest.raises(chapter_host.PrivateHostStop, match="unsupported"):
        _call(runner, path, "unsupported-private-mode")
    config["plan"]["brief_cpu"] = False
    path.write_bytes(evaluation._json_bytes(config))
    with pytest.raises(chapter_host.PrivateHostStop, match="explicit full CPU"):
        _call(runner, path, "brief")
    producer.assert_not_called()


def test_private_host_checks_original_section_manifest_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    runner, _manifest, config = _private_setup(tmp_path, monkeypatch)
    bundle = Path(config["chapter_1"]["bundle_path"])
    verified = selected_scope.import_chapter(runner.root, bundle)
    changed = replace(verified, section_manifest_sha256="f" * 64)
    with pytest.raises(chapter_host.PrivateHostStop, match="manifest changed"):
        chapter_host._section_manifest(runner, changed)


def test_private_host_cli_never_leaks_missing_private_source_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "inspect",
        "--artifact-root",
        str(tmp_path),
        "--manifest",
        str(tmp_path / "synthetic-private-missing.json"),
        "--codespace-created-at",
        datetime.now(UTC).isoformat(),
    ]
    assert chapter_host.main(args) == 2
    printed = capsys.readouterr().err
    assert "PRIVATE_CHAPTER_HOST_FAILED_CLOSED" in printed
    assert "synthetic-private-missing.json" not in printed


def test_private_host_sequential_model_cache_release_is_model_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    qwen_clear = Mock()
    kokoro_clear = Mock()
    monkeypatch.setattr(
        chapter_host.scripting,
        "_script_pipeline",
        SimpleNamespace(cache_clear=qwen_clear),
    )
    monkeypatch.setattr(
        chapter_host.narration,
        "_kokoro_pipeline",
        SimpleNamespace(cache_clear=kokoro_clear),
    )
    chapter_host._release_models()
    qwen_clear.assert_called_once_with()
    kokoro_clear.assert_called_once_with()


def test_private_host_rejects_rest_creation_clock_disagreement_before_any_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, _ = _private_setup(tmp_path, monkeypatch)
    original_verify = chapter_lifecycle.verify_lifecycle

    def mismatched(
        root: Path, *, caller_created_at: str, codespace_name: str
    ) -> chapter_lifecycle.VerifiedLifecycle:
        valid = original_verify(
            root, caller_created_at=caller_created_at, codespace_name=codespace_name
        )
        return chapter_lifecycle.VerifiedLifecycle(
            valid.codespace_id,
            valid.codespace_name,
            valid.created_at - timedelta(minutes=1),
            valid.approved_sha,
            valid.expires_at,
        )

    monkeypatch.setattr(chapter_lifecycle, "verify_lifecycle", mismatched)
    producer = Mock(side_effect=AssertionError("REST time mismatch reached model"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", producer)
    with pytest.raises(chapter_host.PrivateHostStop, match="creation time mismatch"):
        _call(runner, path, "brief")
    producer.assert_not_called()


def test_committed_private_journal_head_detects_missing_last_valid_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    digest = config["chapter_1"]["bundle_sha256"]
    scope = config["chapter_1"]["scope_sha256"]
    chapter_host._phase(runner, 1, digest, scope, "NOT_STARTED")
    chapter_host._phase(runner, 1, digest, scope, "MINING_IN_FLIGHT")
    folder = runner.root / "book-candidates/private-host/chapter-1-phase-events"
    last = sorted(folder.glob("*.json"))[-1]
    last.unlink()
    with pytest.raises(chapter_host.PrivateHostStop, match="committed head"):
        chapter_host._phase_records(runner.root, 1, digest, scope)
    blocked = Mock(side_effect=AssertionError("deleted tail must not reopen mining"))
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", blocked)
    with pytest.raises(chapter_host.PrivateHostStop):
        _call(runner, path, "brief")
    blocked.assert_not_called()


def test_concurrent_private_phase_claims_allow_only_one_mining_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    digest = config["chapter_1"]["bundle_sha256"]
    scope = config["chapter_1"]["scope_sha256"]
    chapter_host._phase(runner, 1, digest, scope, "NOT_STARTED")

    def claim(_: int) -> object:
        try:
            return chapter_host._phase(runner, 1, digest, scope, "MINING_IN_FLIGHT")
        except chapter_host.PrivateHostStop:
            return "REJECTED_DUPLICATE"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))
    assert results.count("REJECTED_DUPLICATE") == 1
    assert sum(isinstance(item, dict) for item in results) == 1
    assert [
        item["phase"] for item in chapter_host._phase_records(runner.root, 1, digest, scope)
    ] == ["NOT_STARTED", "MINING_IN_FLIGHT"]


def test_model_free_recovery_report_has_precise_phase_and_only_safe_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    baseline = chapter_host.recovery_report(runner.root)
    assert baseline["chapters"]["1"]["recovery_state"] == "NEVER_STARTED"
    digest = config["chapter_1"]["bundle_sha256"]
    scope = config["chapter_1"]["scope_sha256"]
    chapter_host._phase(runner, 1, digest, scope, "NOT_STARTED")
    chapter_host._phase(runner, 1, digest, scope, "MINING_IN_FLIGHT")
    report = chapter_host.recovery_report(runner.root)
    assert report["chapters"]["1"]["last_subphase"] == "MINING_IN_FLIGHT"
    assert report["chapters"]["1"]["recovery_state"] == "IN_FLIGHT"
    chapter_host._phase(runner, 1, digest, scope, "BRIEF_IN_FLIGHT")
    chapter_host._phase(
        runner, 1, digest, scope, "AWAITING_HUMAN_REVIEW",
        {"brief_identity": "a" * 64, "progress_path": "/private/never-print-this.json"},
    )
    report = chapter_host.recovery_report(runner.root)
    assert report["chapters"]["1"]["recovery_state"] == "AWAITING_REVIEW"
    assert report["chapters"]["1"]["last_verified_artifact_identities"] == {
        "brief_identity": "a" * 64
    }
    assert "never-print-this" not in json.dumps(report)
    chapter_host._phase(
        runner, 1, digest, scope, "RESUME_IN_FLIGHT",
        {"brief_identity": "a" * 64},
    )
    chapter_host._phase(
        runner, 1, digest, scope, "PREPRODUCTION_READY",
        {"final_result_identity": "b" * 64, "audio_provenance": "ACTUAL_CPU_KOKORO"},
    )
    report = chapter_host.recovery_report(runner.root)
    assert report["chapters"]["1"]["recovery_state"] == "READY"
    assert report["chapters"]["1"]["last_verified_artifact_identities"] == {
        "final_result_identity": "b" * 64,
        "brief_identity": "a" * 64,
    }


def test_resume_missing_review_or_changed_plan_keeps_same_review_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    counter = Mock()
    progress = runner.root / "synthetic-progress.json"
    progress.write_bytes(evaluation._json_bytes({"synthetic": True}))

    def mining(_self: BookOrchestrator, _bundle: Path, **_kwargs: object) -> dict:
        return {"status": "complete", "progress_path": str(progress), "total_work_units": 1}

    def evaluate(
        _runner: BookOrchestrator, _ref: Path, _progress: Path, _scope: Path,
        **kwargs: object
    ) -> dict[str, object]:
        counter()
        if kwargs["review_file"] is None:
            return {
                "status": "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW",
                "brief_identity": "b" * 64,
                "section_scope_sha256": "c" * 64,
            }
        raise chapter.ChapterReviewPending("synthetic review intentionally invalid")

    monkeypatch.setattr(BookOrchestrator, "mine_candidate", mining)
    monkeypatch.setattr(chapter, "evaluate_chapter", evaluate)
    monkeypatch.setattr(chapter_host, "_review_packet", lambda *_: None)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    first = _call(runner, path, "brief")
    assert first["chapters"]["1"]["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert counter.call_count == 2
    missing = _call(runner, path, "resume")
    assert all(c["review_status"] == "REVIEW_REQUIRED" for c in missing["chapters"].values())
    assert counter.call_count == 2
    review = runner.root / "synthetic-invalid-human-review.json"
    review.write_text("not valid human review", encoding="utf-8")
    for n in (1, 7):
        config[f"chapter_{n}"]["review_file"] = str(review)
    path.write_bytes(evaluation._json_bytes(config))
    invalid = _call(runner, path, "resume")
    assert all(c["review_status"] == "REVIEW_REQUIRED" for c in invalid["chapters"].values())
    assert counter.call_count == 4
    for n in (1, 7):
        records = chapter_host._phase_records(
            runner.root, n,
            config[f"chapter_{n}"]["bundle_sha256"],
            config[f"chapter_{n}"]["scope_sha256"],
        )
        assert records[-1]["phase"] == "AWAITING_HUMAN_REVIEW"
    config["plan"]["max_words"] += 1
    path.write_bytes(evaluation._json_bytes(config))
    blocked = _call(runner, path, "resume")
    assert all(
        c["review_status"] == "RESUME_PREFLIGHT_BLOCKED"
        for c in blocked["chapters"].values()
    )
    assert counter.call_count == 4

"""Model-free two-chapter Codespace integration and full-source runtime gates."""

from __future__ import annotations

import hashlib
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
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
        stages = (
            ("ADAPTATION_IN_FLIGHT", None),
            ("ADAPTATION_VERIFIED", "a" * 64),
            ("SCRIPT_IN_FLIGHT", None),
            ("SCRIPT_VERIFIED", "b" * 64),
            ("NARRATION_IN_FLIGHT", None),
            ("NARRATION_VERIFIED", "c" * 64),
            ("DIRECTION_IN_FLIGHT", None),
            ("DIRECTION_VERIFIED", "d" * 64),
        )
        for stage, identity in stages:
            kwargs["on_stage_progress"](stage, identity)
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
    monkeypatch.setattr(chapter_host, "_save_final_manifest", lambda *_args: "e" * 64)
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
        runner,
        1,
        digest,
        scope,
        "AWAITING_HUMAN_REVIEW",
        {"brief_identity": "a" * 64, "progress_path": "/private/never-print-this.json"},
    )
    report = chapter_host.recovery_report(runner.root)
    assert report["chapters"]["1"]["recovery_state"] == "AWAITING_REVIEW"
    assert report["chapters"]["1"]["last_verified_artifact_identities"] == {
        "brief_identity": "a" * 64
    }
    assert "never-print-this" not in json.dumps(report)
    chapter_host._phase(
        runner,
        1,
        digest,
        scope,
        "RESUME_IN_FLIGHT",
        {"brief_identity": "a" * 64},
    )
    chapter_host._phase(
        runner,
        1,
        digest,
        scope,
        "PREPRODUCTION_READY",
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
        _runner: BookOrchestrator, _ref: Path, _progress: Path, _scope: Path, **kwargs: object
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
    # An explicitly named but unavailable private review is still a checkpoint,
    # not a consumed attempt or permission to invoke a model callback.
    for n in (1, 7):
        config[f"chapter_{n}"]["review_file"] = str(runner.root / "not-yet-supplied.json")
    path.write_bytes(evaluation._json_bytes(config))
    unavailable = _call(runner, path, "resume")
    assert all(c["review_status"] == "REVIEW_REQUIRED" for c in unavailable["chapters"].values())
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
            runner.root,
            n,
            config[f"chapter_{n}"]["bundle_sha256"],
            config[f"chapter_{n}"]["scope_sha256"],
        )
        assert records[-1]["phase"] == "AWAITING_HUMAN_REVIEW"
    config["plan"]["max_words"] += 1
    path.write_bytes(evaluation._json_bytes(config))
    blocked = _call(runner, path, "resume")
    assert all(
        c["review_status"] == "RESUME_PREFLIGHT_BLOCKED" for c in blocked["chapters"].values()
    )
    assert counter.call_count == 4


def test_recovery_status_cli_is_sanitized_and_rejects_invalid_journal_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    assert chapter_host.main(["status", "--artifact-root", str(runner.root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "PRIVATE_CHAPTER_RECOVERY_CHECKPOINT_ONLY"
    assert status["chapters"]["1"]["recovery_state"] == "NEVER_STARTED"
    assert "source_ledger" not in json.dumps(status)
    assert chapter_host.main(["brief", "--artifact-root", str(runner.root)]) == 2
    assert "PRIVATE_CHAPTER_HOST_FAILED_CLOSED" in capsys.readouterr().err
    with pytest.raises(chapter_host.PrivateHostStop, match="absolute artifact root"):
        chapter_host.recovery_report(Path("relative-only"))

    _locked(runner, path, config)
    digest = config["chapter_1"]["bundle_sha256"]
    scope = config["chapter_1"]["scope_sha256"]
    chapter_host._phase(runner, 1, digest, scope, "NOT_STARTED")
    head_path = chapter_host._phase_head(runner.root, 1)
    original = head_path.read_bytes()
    changed = json.loads(original)
    changed["bundle_sha256"] = "non-digest"
    head_path.write_bytes(evaluation._json_bytes(changed))
    with pytest.raises(chapter_host.PrivateHostStop, match="cannot be trusted"):
        chapter_host.recovery_report(runner.root)
    head_path.write_bytes(original)

    folder = runner.root / "book-candidates/private-host/chapter-1-phase-events"
    for event in folder.iterdir():
        event.unlink()
    folder.rmdir()
    with pytest.raises(chapter_host.PrivateHostStop, match="missing from committed head"):
        chapter_host._phase_records(runner.root, 1, digest, scope)


def test_full_execute_losing_concurrent_mining_claim_cannot_stop_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Competing complete host calls, not merely the lower-level append helper."""
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    first = config["chapter_1"]
    second = config["chapter_7"]
    chapter_host._phase(runner, 1, first["bundle_sha256"], first["scope_sha256"], "NOT_STARTED")
    chapter_host._phase(runner, 7, second["bundle_sha256"], second["scope_sha256"], "NOT_STARTED")
    chapter_host._phase(
        runner, 7, second["bundle_sha256"], second["scope_sha256"], "MINING_IN_FLIGHT"
    )
    both_preflighted = Barrier(2)
    original = chapter_host._phase
    models = Mock(return_value={"status": "incomplete", "total_work_units": 1})

    def competing_phase(
        owner: BookOrchestrator,
        chapter_number: int,
        digest: str,
        scope: str,
        phase: str,
        artifacts: dict[str, object] | None = None,
    ) -> dict:
        if chapter_number == 1 and phase == "MINING_IN_FLIGHT":
            both_preflighted.wait(timeout=15)
        return original(owner, chapter_number, digest, scope, phase, artifacts)

    monkeypatch.setattr(chapter_host, "_phase", competing_phase)
    monkeypatch.setattr(BookOrchestrator, "mine_candidate", models)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_call, runner, path, "brief") for _ in range(2)]
        both = [future.result(timeout=25) for future in futures]
    models.assert_called_once()
    assert not any(
        outcome["chapters"]["1"]["status"]
        == "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE"
        for outcome in both
    )
    records = chapter_host._phase_records(
        runner.root, 1, first["bundle_sha256"], first["scope_sha256"]
    )
    assert [record["phase"] for record in records] == [
        "NOT_STARTED",
        "MINING_IN_FLIGHT",
        "SCOPED_MINING_INCOMPLETE",
    ]


@pytest.mark.parametrize(
    "stop_after",
    [
        "ADAPTATION_IN_FLIGHT",
        "SCRIPT_IN_FLIGHT",
        "NARRATION_IN_FLIGHT",
        "DIRECTION_IN_FLIGHT",
        "DIRECTION_VERIFIED",
    ],
)
def test_resumed_host_interruption_reports_precise_last_subphase_and_verified_prior_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_after: str
) -> None:
    runner, path, config = _private_setup(tmp_path, monkeypatch)
    _locked(runner, path, config)
    chosen = config["chapter_1"]
    other = config["chapter_7"]
    bundle, scope = chosen["bundle_sha256"], chosen["scope_sha256"]
    owner = "a" * 32
    _ = chapter_host._phase(runner, 1, bundle, scope, "NOT_STARTED")
    chapter_host._phase(runner, 1, bundle, scope, "MINING_IN_FLIGHT", {"claim_owner": owner})
    chapter_host._phase(runner, 1, bundle, scope, "BRIEF_IN_FLIGHT", {"claim_owner": owner})
    progress = runner.root / "synthetic-progress.json"
    progress.write_bytes(evaluation._json_bytes({"progress": "synthetic"}))
    brief = "b" * 64
    chapter_host._phase(
        runner,
        1,
        bundle,
        scope,
        "AWAITING_HUMAN_REVIEW",
        {"claim_owner": owner, "brief_identity": brief},
    )
    saved = runner.root / "book-candidates/private-host/chapter-1-brief-record.json"
    evaluation._store(
        runner.root / "book-candidates",
        saved,
        {
            "chapter": 1,
            "phase": "AWAITING_HUMAN_REVIEW",
            "scope_sha256": scope,
            "bundle_sha256": bundle,
            "progress_path": str(progress),
            "progress_sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
            "brief_identity": brief,
            "plan_fingerprint": chapter.ChapterPlan.from_manifest(config["plan"]).fingerprint,
        },
    )
    review = runner.root / "synthetic-reviewed.json"
    review.write_bytes(evaluation._json_bytes({"synthetic": "review placeholder"}))
    config["chapter_1"]["review_file"] = str(review)
    path.write_bytes(evaluation._json_bytes(config))
    chapter_host._phase(runner, 7, other["bundle_sha256"], other["scope_sha256"], "NOT_STARTED")
    chapter_host._phase(
        runner, 7, other["bundle_sha256"], other["scope_sha256"], "MINING_IN_FLIGHT"
    )
    steps = (
        ("ADAPTATION_IN_FLIGHT", None),
        ("ADAPTATION_VERIFIED", "c" * 64),
        ("SCRIPT_IN_FLIGHT", None),
        ("SCRIPT_VERIFIED", "d" * 64),
        ("NARRATION_IN_FLIGHT", None),
        ("NARRATION_VERIFIED", "e" * 64),
        ("DIRECTION_IN_FLIGHT", None),
        ("DIRECTION_VERIFIED", "f" * 64),
    )

    def fake_evaluation(*_args: object, **kwargs: object) -> dict:
        kwargs["on_resume_verified"]()
        for step, identity in steps:
            kwargs["on_stage_progress"](step, identity)
            if step == stop_after:
                raise RuntimeError("synthetic process interruption")
        raise AssertionError("fixture must stop inside a subphase")

    monkeypatch.setattr(chapter, "evaluate_chapter", fake_evaluation)
    monkeypatch.setattr(chapter_host, "_release_models", lambda: None)
    result = _call(runner, path, "resume")
    assert result["chapters"]["1"]["status"] == (
        "PRIVATE_CHAPTER_PHASE_STOPPED_REVIEW_PRIVATE_EVIDENCE"
    )
    status = chapter_host.recovery_report(runner.root)
    entry = status["chapters"]["1"]
    assert entry["runtime_journal_status"] == "STOPPED"
    assert entry["last_subphase"] == stop_after
    assert entry["retrieval_status"] == "NOT_ATTEMPTED"
    verified = entry["last_verified_artifact_identities"]
    assert verified["brief_identity"] == brief
    if stop_after in {
        "SCRIPT_IN_FLIGHT",
        "NARRATION_IN_FLIGHT",
        "DIRECTION_IN_FLIGHT",
        "DIRECTION_VERIFIED",
    }:
        assert verified["adaptation_identity"] == "c" * 64
    if stop_after in {"NARRATION_IN_FLIGHT", "DIRECTION_IN_FLIGHT", "DIRECTION_VERIFIED"}:
        assert verified["script_identity"] == "d" * 64
    if stop_after in {"DIRECTION_IN_FLIGHT", "DIRECTION_VERIFIED"}:
        assert verified["narration_report_identity"] == "e" * 64
    if stop_after == "DIRECTION_VERIFIED":
        assert verified["direction_identity"] == "f" * 64


@pytest.mark.parametrize(
    "damage",
    [
        "wav_changed",
        "wav_missing",
        "report_missing",
        "source_changed",
        "source_missing",
        "human_review_missing",
        "final_result_changed",
        "final_result_missing",
        "stage_validated_missing",
        "saved_manifest_changed",
        "saved_manifest_missing",
        "recovery_archive_symlink",
        "invalid_final_identity",
        "forged_selected_bundle",
        "forged_source_boundary",
        "forged_parent_attestation",
        "forged_publish_eligible",
        "forged_brief",
        "forged_adaptation",
        "forged_script",
        "forged_direction",
        "forged_narration",
        "forged_human_review",
        "invalid_brief_checkpoint",
        "valid_brief_checkpoint",
        "too_many_text_files",
        "changed_between_inventory_and_archive",
        "missing_winning_claim_before_manifest",
        "missing_prior_rejection_before_manifest",
        "valid_prior_rejection_before_manifest",
        "shared_wav_both_chapters",
        "second_chapter_invalid_inventory",
    ],
)
def test_scoped_private_recovery_archive_contains_and_hashes_wavs_reports_text(
    tmp_path: Path,
    damage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic outputs only; all archive members are independently content-bound."""
    import io
    import tarfile
    import wave

    from clipper.stage_contracts import content_fingerprint

    root = tmp_path / "synthetic-private-host"
    root.mkdir()
    runner = BookOrchestrator(root)
    bundle, scope = "1" * 64, "2" * 64
    first_owner, resumed_owner = "3" * 32, "4" * 32
    base = root / "book-candidates"

    def stage_file(stage: str) -> str:
        key = hashlib.sha256(stage.encode("ascii")).hexdigest()
        payload = {
            "brief": {"argument": "synthetic", "beats": [], "coverage": []},
            "adaptation": {
                "throughline": "synthetic",
                "narrative_form": "CONCEPTUAL_SEQUENCE",
                "choice_reason": "synthetic",
                "beats": [],
                "characters": [],
                "locations": [],
                "props": [],
            },
            "chapter_script": {"title": "synthetic", "scenes": []},
            "chapter_direction": {"shots": []},
        }[stage]
        raw = json.dumps(payload, sort_keys=True)
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        payload_sha = hashlib.sha256(evaluation._json_bytes(payload)).hexdigest()
        directory = base / stage / key
        retry = stage == "adaptation" and damage in {
            "valid_prior_rejection_before_manifest",
            "missing_prior_rejection_before_manifest",
        }
        prior_digest = None
        winning_attempt, winning_owner = 1, first_owner
        if retry:
            previous_owner = "8" * 32
            failed_raw = json.dumps({"not": "valid-adaptation"}, sort_keys=True)
            failed_raw_sha = hashlib.sha256(failed_raw.encode("utf-8")).hexdigest()
            evaluation._store(
                base,
                directory / "started-1.json",
                {
                    "request_key": key,
                    "attempt": 1,
                    "owner": previous_owner,
                    "status": "STARTED_UNRESOLVED",
                },
            )
            evaluation._store(
                base,
                directory / "raw-1.json",
                {
                    "attempt": 1,
                    "owner": previous_owner,
                    "raw": failed_raw,
                    "sha256": failed_raw_sha,
                },
            )
            evaluation._store(
                base,
                directory / "failed-1.json",
                {
                    "attempt": 1,
                    "owner": previous_owner,
                    "status": "SCHEMA_OR_SOURCE_REJECTED",
                    "validation_evidence": chapter._rejection_evidence(
                        directory,
                        key,
                        1,
                        previous_owner,
                        failed_raw_sha,
                        chapter.ChapterStop("chapter adaptation schema"),
                    ),
                },
            )
            prior_digest = hashlib.sha256((directory / "failed-1.json").read_bytes()).hexdigest()
            winning_attempt, winning_owner = 2, first_owner
        identity = chapter._text_artifact_identity(
            stage,
            key,
            "INJECTED_TEST_PRODUCER",
            raw_sha,
            payload_sha,
            prior_rejection_sha256=prior_digest,
        )
        evaluation._store(
            base,
            directory / f"raw-{winning_attempt}.json",
            {
                "attempt": winning_attempt,
                "owner": winning_owner,
                "raw": raw,
                "sha256": raw_sha,
            },
        )
        evaluation._store(
            base,
            directory / f"started-{winning_attempt}.json",
            {
                "request_key": key,
                "attempt": winning_attempt,
                "owner": winning_owner,
                "status": "STARTED_UNRESOLVED",
            },
        )
        evaluation._store(
            base,
            directory / "validated.json",
            {
                "request_key": key,
                "provenance": "INJECTED_TEST_PRODUCER",
                "contract": stage,
                "validator_contract": chapter.TEXT_ARTIFACT_CONTRACT,
                "payload": payload,
                "owner": winning_owner,
                "attempt": winning_attempt,
                "raw_sha256": raw_sha,
                "payload_sha256": payload_sha,
                "token_receipt": None,
                "artifact_identity": identity,
                **({"prior_rejection_sha256": prior_digest} if prior_digest is not None else {}),
            },
        )
        return identity

    identities = {
        stage: stage_file(stage)
        for stage in ("brief", "adaptation", "chapter_script", "chapter_direction")
    }
    wav_stream = io.BytesIO()
    with wave.open(wav_stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 400)
    wav_bytes = wav_stream.getvalue()
    wav_path = base / "chapter-audio-injected_test_wav/synthetic.wav"
    wav_path.parent.mkdir(parents=True)
    wav_path.write_bytes(wav_bytes)
    report = {
        "schema": "clipper-chapter-narration-v2",
        "provenance": "INJECTED_TEST_WAV",
        "records": [
            {
                "wav_relative_path": "chapter-audio-injected_test_wav/synthetic.wav",
                "sha256": hashlib.sha256(wav_bytes).hexdigest(),
            }
        ],
    }
    report_identity = content_fingerprint(report)
    evaluation._store(base, base / f"narration-report-{report_identity}.json", report)

    chapter_host._phase(runner, 1, bundle, scope, "NOT_STARTED")
    chapter_host._phase(runner, 1, bundle, scope, "MINING_IN_FLIGHT", {"claim_owner": first_owner})
    chapter_host._phase(runner, 1, bundle, scope, "BRIEF_IN_FLIGHT", {"claim_owner": first_owner})
    chapter_host._phase(
        runner,
        1,
        bundle,
        scope,
        "AWAITING_HUMAN_REVIEW",
        {"claim_owner": first_owner, "brief_identity": identities["brief"]},
    )
    chapter_host._phase(
        runner, 1, bundle, scope, "RESUME_IN_FLIGHT", {"claim_owner": resumed_owner}
    )
    for start, verified, identity in (
        ("ADAPTATION_IN_FLIGHT", "ADAPTATION_VERIFIED", identities["adaptation"]),
        ("SCRIPT_IN_FLIGHT", "SCRIPT_VERIFIED", identities["chapter_script"]),
        ("NARRATION_IN_FLIGHT", "NARRATION_VERIFIED", report_identity),
        ("DIRECTION_IN_FLIGHT", "DIRECTION_VERIFIED", identities["chapter_direction"]),
    ):
        chapter_host._phase(runner, 1, bundle, scope, start, {"claim_owner": resumed_owner})
        stage_key = {
            "ADAPTATION_VERIFIED": "adaptation_identity",
            "SCRIPT_VERIFIED": "script_identity",
            "NARRATION_VERIFIED": "narration_report_identity",
            "DIRECTION_VERIFIED": "direction_identity",
        }[verified]
        chapter_host._phase(
            runner,
            1,
            bundle,
            scope,
            verified,
            {"claim_owner": resumed_owner, stage_key: identity},
        )

    base_dir = base / "chapter-evaluations" / ("5" * 64)
    source_bytes = evaluation._json_bytes({"synthetic": "selected scope source, private"})
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    (base_dir).mkdir(parents=True)
    (base_dir / "source-ledger.json").write_bytes(source_bytes)
    review_id = "6" * 64
    evaluation._store(
        base,
        base_dir / f"human-review-{review_id}.json",
        {"synthetic": "private human review artifact"},
    )
    result = {
        "selected_source_bundle_sha256": bundle,
        "source_boundary": "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT",
        "parent_original_reverified_on_host": False,
        "publish_eligible": False,
        "brief_key": identities["brief"],
        "adaptation_key": identities["adaptation"],
        "script_key": identities["chapter_script"],
        "direction_key": identities["chapter_direction"],
        "narration_report_key": report_identity,
        "source_ledger_sha256": source_sha,
        "human_review_sha256": review_id,
    }
    final_key = content_fingerprint(result)
    evaluation._store(base, base_dir / f"chapter-result-{final_key}.json", result)
    stage_directory = base / "adaptation" / hashlib.sha256(b"adaptation").hexdigest()
    if damage == "missing_winning_claim_before_manifest":
        (stage_directory / "started-1.json").unlink()
        with pytest.raises(chapter_host.PrivateHostStop, match="replay history"):
            chapter_host._save_final_manifest(root, 1, bundle, scope, final_key)
        return
    if damage == "missing_prior_rejection_before_manifest":
        # The winning identity still commits the true prior failure, now missing.
        (stage_directory / "failed-1.json").unlink()
        with pytest.raises(chapter_host.PrivateHostStop, match="replay history"):
            chapter_host._save_final_manifest(root, 1, bundle, scope, final_key)
        return
    manifest_sha = chapter_host._save_final_manifest(root, 1, bundle, scope, final_key)
    chapter_host._phase(
        runner,
        1,
        bundle,
        scope,
        "PREPRODUCTION_READY",
        {
            "claim_owner": resumed_owner,
            "final_result_identity": final_key,
            "output_manifest_sha256": manifest_sha,
        },
    )
    if damage in {"shared_wav_both_chapters", "second_chapter_invalid_inventory"}:
        second_bundle, second_scope = "7" * 64, "8" * 64
        second_owner = "a" * 32
        chapter_host._phase(runner, 7, second_bundle, second_scope, "NOT_STARTED")
        for phase, artifacts in (
            ("MINING_IN_FLIGHT", {}),
            ("BRIEF_IN_FLIGHT", {}),
            ("AWAITING_HUMAN_REVIEW", {"brief_identity": identities["brief"]}),
            ("RESUME_IN_FLIGHT", {"brief_identity": identities["brief"]}),
            ("ADAPTATION_IN_FLIGHT", {}),
            ("ADAPTATION_VERIFIED", {"adaptation_identity": identities["adaptation"]}),
            ("SCRIPT_IN_FLIGHT", {}),
            ("SCRIPT_VERIFIED", {"script_identity": identities["chapter_script"]}),
            ("NARRATION_IN_FLIGHT", {}),
            (
                "NARRATION_VERIFIED",
                {
                    "narration_report_identity": (
                        report_identity if damage == "shared_wav_both_chapters" else "9" * 64
                    )
                },
            ),
        ):
            chapter_host._phase(
                runner,
                7,
                second_bundle,
                second_scope,
                phase,
                {"claim_owner": second_owner, **artifacts},
            )
        recovered = chapter_host.private_recovery_archive(root)
        with tarfile.open(base / "private-host/chapter-recovery.tar", "r:") as shared:
            recovered_manifest = json.load(shared.extractfile("private-recovery-manifest.json"))
            audio_members = [
                name
                for name in shared.getnames()
                if name == "book-candidates/chapter-audio-injected_test_wav/synthetic.wav"
            ]
            assert len(audio_members) == 1
            wav_entry = next(row for row in recovered_manifest["files"] if row["kind"] == "wav")
            if damage == "shared_wav_both_chapters":
                assert wav_entry["chapters"] == [1, 7]
                assert [row["inventory_status"] for row in recovered_manifest["chapters"]] == [
                    "VERIFIED",
                    "VERIFIED",
                ]
            else:
                assert wav_entry["chapters"] == [1]
                assert [row["inventory_status"] for row in recovered_manifest["chapters"]] == [
                    "VERIFIED",
                    "INCOMPLETE",
                ]
                assert recovered_manifest["chapters"][1]["files"] == []
                assert recovered["file_count"] > 0
        return
    status = chapter_host.recovery_report(root)
    assert status["chapters"]["1"]["recovery_state"] == "READY"
    assert status["chapters"]["1"]["last_subphase"] == "PREPRODUCTION_READY"
    assert status["chapters"]["1"]["retrieval_status"] == "NOT_ATTEMPTED"
    assert status["chapters"]["1"]["output_manifest_sha256"] == manifest_sha
    archive_report = chapter_host.private_recovery_archive(root)
    assert archive_report["files_retrieved"] is False
    assert archive_report["host_cleanup_verified"] is False
    with tarfile.open(base / "private-host/chapter-recovery.tar", "r:") as archive:
        names = archive.getnames()
        assert "book-candidates/chapter-audio-injected_test_wav/synthetic.wav" in names
        assert f"book-candidates/narration-report-{report_identity}.json" in names
        assert f"book-candidates/chapter-evaluations/{'5' * 64}/source-ledger.json" in names
        for stage in identities:
            assert any(name.endswith("/validated.json") and f"/{stage}/" in name for name in names)
        saved_manifest = json.load(archive.extractfile("private-recovery-manifest.json"))
        assert saved_manifest["schema"] == "clipper-chapter-private-recovery-archive-manifest-v1"
        assert any(entry["kind"] == "wav" for entry in saved_manifest["files"])
    if damage == "valid_prior_rejection_before_manifest":
        assert archive_report["file_count"] > 0
        return
    final_path = base_dir / f"chapter-result-{final_key}.json"
    report_path = base / f"narration-report-{report_identity}.json"
    source_path = base_dir / "source-ledger.json"
    human_path = base_dir / f"human-review-{review_id}.json"
    saved_manifest = base / "private-host/chapter-1-output-manifest.json"
    archived_path = base / "private-host/chapter-recovery.tar"
    if damage == "invalid_final_identity":
        with pytest.raises(chapter_host.PrivateHostStop, match="final result identity invalid"):
            chapter_host._chapter_output_manifest(
                root, 1, bundle, scope, final_result_identity="not-a-digest"
            )
        return
    if damage == "valid_brief_checkpoint":
        review_checkpoint = base / "private-host/chapter-1-brief-record.json"
        evaluation._store(base, review_checkpoint, {"brief_identity": identities["brief"]})
        scoped = chapter_host._chapter_output_manifest(root, 1, bundle, scope)
        assert any(
            entry["relative_path"] == "book-candidates/private-host/chapter-1-brief-record.json"
            for entry in scoped["files"]
        )
        return
    if damage == "too_many_text_files":
        key = hashlib.sha256("brief".encode("ascii")).hexdigest()
        stage_dir = base / "brief" / key
        for index in range(22):
            (stage_dir / f"synthetic-foreign-{index:03}.json").write_bytes(b"{}")
        scoped = chapter_host.private_recovery_archive(root)
        assert scoped["file_count"] == 0
        with tarfile.open(base / "private-host/chapter-recovery.tar", "r:") as skipped:
            state = json.load(skipped.extractfile("private-recovery-manifest.json"))
            assert state["chapters"][0]["inventory_status"] == "INCOMPLETE"
            assert state["chapters"][0]["files"] == []
        return
    if damage == "changed_between_inventory_and_archive":
        original_inventory = chapter_host._chapter_output_manifest

        def concurrent_change(*args: object, **kwargs: object) -> dict:
            verified_manifest = original_inventory(*args, **kwargs)
            wav_path.write_bytes(b"synthetic file changed after inventory")
            return verified_manifest

        monkeypatch.setattr(chapter_host, "_chapter_output_manifest", concurrent_change)
        with pytest.raises(chapter_host.PrivateHostStop, match="changed during archive"):
            chapter_host.private_recovery_archive(root)
        return
    if damage == "invalid_brief_checkpoint":
        evaluation._store(
            base,
            base / "private-host/chapter-1-brief-record.json",
            {"brief_identity": "0" * 64, "status": "synthetic invalid"},
        )
        with pytest.raises(chapter_host.PrivateHostStop, match="review record changed"):
            chapter_host._chapter_output_manifest(root, 1, bundle, scope)
        return
    if damage.startswith("forged_"):
        # Rehash the complete forged JSON and give it a correctly named path.
        # A digest-valid altered final result must still fail the chapter's
        # attested source boundary and exact reviewed-stage identity checks.
        field, bad_value = {
            "forged_selected_bundle": ("selected_source_bundle_sha256", "9" * 64),
            "forged_source_boundary": ("source_boundary", "UNVERIFIED_FULL_BOOK"),
            "forged_parent_attestation": ("parent_original_reverified_on_host", True),
            "forged_publish_eligible": ("publish_eligible", True),
            "forged_brief": ("brief_key", "9" * 64),
            "forged_adaptation": ("adaptation_key", "9" * 64),
            "forged_script": ("script_key", "9" * 64),
            "forged_direction": ("direction_key", "9" * 64),
            "forged_narration": ("narration_report_key", "9" * 64),
            "forged_human_review": ("human_review_sha256", "invalid-review-id"),
        }[damage]
        tampered_result = {**result, field: bad_value}
        altered_id = content_fingerprint(tampered_result)
        evaluation._store(base, base_dir / f"chapter-result-{altered_id}.json", tampered_result)
        expected_error = (
            "human review identity missing"
            if damage == "forged_human_review"
            else "final result differs from reviewed stages"
        )
        with pytest.raises(chapter_host.PrivateHostStop, match=expected_error):
            chapter_host._chapter_output_manifest(
                root, 1, bundle, scope, final_result_identity=altered_id
            )
        return
    if damage == "wav_changed":
        wav_path.write_bytes(b"tampered synthetic audio")
    elif damage == "wav_missing":
        wav_path.unlink()
    elif damage == "report_missing":
        report_path.unlink()
    elif damage == "source_changed":
        source_path.write_bytes(evaluation._json_bytes({"synthetic": "modified source"}))
    elif damage == "source_missing":
        source_path.unlink()
    elif damage == "human_review_missing":
        human_path.unlink()
    elif damage == "final_result_changed":
        final_path.write_bytes(evaluation._json_bytes({**result, "publish_eligible": True}))
    elif damage == "final_result_missing":
        final_path.unlink()
    elif damage == "stage_validated_missing":
        key = hashlib.sha256("adaptation".encode("ascii")).hexdigest()
        (base / "adaptation" / key / "validated.json").unlink()
    elif damage == "saved_manifest_changed":
        saved_manifest.write_bytes(saved_manifest.read_bytes() + b" ")
    elif damage == "saved_manifest_missing":
        saved_manifest.unlink()
    else:
        assert damage == "recovery_archive_symlink"
        archived_path.unlink()
        forbidden = tmp_path / "not-private-archive"
        forbidden.write_bytes(b"outside synthetic marker")
        archived_path.symlink_to(forbidden)
    if damage == "recovery_archive_symlink":
        with pytest.raises(chapter_host.PrivateHostStop, match="redirected"):
            chapter_host.private_recovery_archive(root)
    else:
        receipt = chapter_host.private_recovery_archive(root)
        assert receipt["file_count"] == 0
        with tarfile.open(base / "private-host/chapter-recovery.tar", "r:") as isolated:
            manifest = json.load(isolated.extractfile("private-recovery-manifest.json"))
            assert manifest["chapters"][0]["inventory_status"] == "INCOMPLETE"
            assert manifest["chapters"][0]["files"] == []
            assert manifest["files"] == []


@pytest.mark.parametrize(
    "tamper",
    [
        "none",
        "missing_winning_claim",
        "missing_first_rejection",
        "changed_original_rejection_receipt",
        "changed_first_raw",
    ],
)
def test_same_shape_semantically_invalid_source_retries_and_recovers_with_original_receipt(
    tmp_path: Path, tamper: str
) -> None:
    """No model: the FIRST complete JSON reply fails its SOURCE identity, not its shape."""
    root = tmp_path / "synthetic-host"
    base = root / "book-candidates"
    first = {"argument": "synthetic", "beats": [], "coverage": ["wrong-source"]}
    second = {"argument": "synthetic", "beats": [], "coverage": ["unit-verified"]}
    producer = Mock(side_effect=[json.dumps(first), json.dumps(second)])
    plan = chapter.ChapterPlan()

    def real_source_bound_validator(raw: object) -> dict:
        obj = chapter._shape(
            raw, {"argument", "beats", "coverage"}, "brief full source schema invalid"
        )
        if obj["coverage"] != ["unit-verified"]:
            raise chapter.ChapterStop("chapter original unit identity mismatch")
        return obj

    arguments = (
        base,
        "brief",
        {"fixture": "synthetic original unit unit-verified"},
        plan,
        real_source_bound_validator,
    )
    validated, identity, provenance, attempts = chapter._stage(*arguments, producer)
    assert validated == second
    assert provenance == "INJECTED_TEST_PRODUCER"
    assert attempts == 2
    replay, replay_identity, _, replay_attempts = chapter._stage(
        *arguments, Mock(side_effect=AssertionError("no new model calls"))
    )
    assert (replay, replay_identity, replay_attempts) == (second, identity, 0)
    # Replaying the fully validated second attempt never asks the producer again.
    assert producer.call_count == 2

    stage_dir = next((base / "brief").iterdir())
    first_failure = json.loads((stage_dir / "failed-1.json").read_bytes())
    receipt = first_failure["validation_evidence"]
    assert receipt["contract"] == chapter.REJECTION_EVIDENCE_CONTRACT
    assert receipt["request_key"] == stage_dir.name
    assert receipt["raw_sha256"] == json.loads((stage_dir / "raw-1.json").read_bytes())["sha256"]

    runner = BookOrchestrator(root)
    bundle, scope, owner = "1" * 64, "2" * 64, "3" * 32
    for phase, evidence in (
        ("NOT_STARTED", {}),
        ("MINING_IN_FLIGHT", {}),
        ("BRIEF_IN_FLIGHT", {}),
        ("AWAITING_HUMAN_REVIEW", {"brief_identity": identity}),
    ):
        chapter_host._phase(
            runner,
            1,
            bundle,
            scope,
            phase,
            {"claim_owner": owner, **evidence} if phase != "NOT_STARTED" else {},
        )
    if tamper == "missing_winning_claim":
        (stage_dir / "started-2.json").unlink()
    elif tamper == "missing_first_rejection":
        (stage_dir / "failed-1.json").unlink()
    elif tamper == "changed_original_rejection_receipt":
        first_failure["validation_evidence"]["failure_sha256"] = "f" * 64
        (stage_dir / "failed-1.json").write_bytes(evaluation._json_bytes(first_failure))
    elif tamper == "changed_first_raw":
        path = stage_dir / "raw-1.json"
        changed = json.loads(path.read_bytes())
        changed["raw"] = json.dumps(second)
        path.write_bytes(evaluation._json_bytes(changed))

    if tamper == "none":
        entries = chapter_host._stage_output_entries(root, 1, "brief", identity)
        assert len(entries) == 6
        manifest = chapter_host._chapter_output_manifest(root, 1, bundle, scope)
        assert manifest["files"] == entries or len(manifest["files"]) == len(entries)
        archived = chapter_host.private_recovery_archive(root)
        assert archived["file_count"] == 6
        with tarfile.open(base / "private-host/chapter-recovery.tar", "r:") as output:
            rows = json.load(output.extractfile("private-recovery-manifest.json"))
            assert rows["chapters"][0]["inventory_status"] == "VERIFIED"
            assert rows["chapters"][1]["inventory_status"] == "NOT_STARTED"
    else:
        with pytest.raises(chapter_host.PrivateHostStop):
            chapter_host._stage_output_entries(root, 1, "brief", identity)
        with pytest.raises(chapter_host.PrivateHostStop):
            chapter_host._chapter_output_manifest(root, 1, bundle, scope)
        chapter_host.private_recovery_archive(root)
        with tarfile.open(base / "private-host/chapter-recovery.tar", "r:") as output:
            rows = json.load(output.extractfile("private-recovery-manifest.json"))
            assert rows["chapters"][0]["inventory_status"] == "INCOMPLETE"
            assert rows["chapters"][0]["files"] == []
            assert rows["chapters"][1]["inventory_status"] == "NOT_STARTED"
    assert producer.call_count == 2


@pytest.mark.parametrize(
    "fault",
    [
        "missing_directory",
        "directory_symlink",
        "child_symlink",
        "missing_identity",
        "missing_raw_owner",
        "wrong_raw_digest",
        "missing_validated_fields",
        "too_many_stage_directories",
    ],
)
def test_scoped_recovery_text_stage_rejects_unverifiable_artifacts(
    tmp_path: Path, fault: str
) -> None:
    """Only canonical stage evidence may be copied, never guessed text."""
    root = tmp_path / "private"
    base = root / "book-candidates"
    stage = "adaptation"
    stage_root = base / stage
    key = "1" * 64
    raw = '{"synthetic":"text"}'
    owner = "2" * 32
    raw_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    payload = json.loads(raw)
    payload_sha = hashlib.sha256(evaluation._json_bytes(payload)).hexdigest()
    identity = chapter._text_artifact_identity(
        stage, key, "INJECTED_TEST_PRODUCER", raw_digest, payload_sha
    )
    if fault == "missing_directory":
        with pytest.raises(chapter_host.PrivateHostStop, match="stage directory missing"):
            chapter_host._stage_output_entries(root, 1, stage, identity)
        return
    stage_root.mkdir(parents=True)
    if fault == "too_many_stage_directories":
        for index in range(129):
            (stage_root / f"synthetic-{index:03}").mkdir()
        with pytest.raises(chapter_host.PrivateHostStop, match="inventory exceeds"):
            chapter_host._stage_output_entries(root, 1, stage, identity)
        return
    if fault == "directory_symlink":
        retained = stage_root.with_name("retained-synthetic-text")
        stage_root.rename(retained)
        stage_root.symlink_to(retained, target_is_directory=True)
        with pytest.raises(chapter_host.PrivateHostStop, match="stage directory missing"):
            chapter_host._stage_output_entries(root, 1, stage, identity)
        return
    if fault == "child_symlink":
        elsewhere = tmp_path / "not-in-private-stage"
        elsewhere.mkdir()
        (stage_root / key).symlink_to(elsewhere, target_is_directory=True)
        with pytest.raises(chapter_host.PrivateHostStop, match="stage directory redirected"):
            chapter_host._stage_output_entries(root, 1, stage, identity)
        return
    directory = stage_root / key
    directory.mkdir()
    if fault == "missing_identity":
        with pytest.raises(chapter_host.PrivateHostStop, match="absent or ambiguous"):
            chapter_host._stage_output_entries(root, 1, stage, identity)
        return
    record = {
        "request_key": key,
        "provenance": "INJECTED_TEST_PRODUCER",
        "contract": stage,
        "validator_contract": chapter.TEXT_ARTIFACT_CONTRACT,
        "payload": payload,
        "owner": owner,
        "attempt": 1,
        "raw_sha256": raw_digest,
        "payload_sha256": payload_sha,
        "token_receipt": None,
        "artifact_identity": identity,
    }
    if fault == "missing_validated_fields":
        del record["payload"]
    evaluation._store(base, directory / "validated.json", record)
    saved = {
        "attempt": 1,
        "owner": owner if fault != "missing_raw_owner" else "3" * 32,
        "raw": raw if fault != "wrong_raw_digest" else "mutated synthetic text",
        "sha256": raw_digest,
    }
    evaluation._store(base, directory / "raw-1.json", saved)
    with pytest.raises(chapter_host.PrivateHostStop, match="reviewed chapter"):
        chapter_host._stage_output_entries(root, 1, stage, identity)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_report",
        "bad_report_schema",
        "unapproved_wav_path",
        "malformed_wav_digest",
        "absent_wav",
        "changed_wav",
        "invalid_report_json",
        "oversized_report",
        "symlink_report",
        "non_object_wav_entry",
    ],
)
def test_scoped_recovery_narration_refuses_unverified_or_cross_scope_audio(
    tmp_path: Path, fault: str
) -> None:
    from clipper.stage_contracts import content_fingerprint

    root = tmp_path / "private"
    base = root / "book-candidates"
    audio = base / "chapter-audio-injected_test_wav/fixture.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"synthetic-wave-data")
    report = {
        "schema": "clipper-chapter-narration-v2",
        "provenance": "INJECTED_TEST_WAV",
        "records": [
            {
                "wav_relative_path": (
                    "../different-chapter/private.wav"
                    if fault == "unapproved_wav_path"
                    else "chapter-audio-injected_test_wav/fixture.wav"
                ),
                "sha256": (
                    "not-a-digest"
                    if fault == "malformed_wav_digest"
                    else hashlib.sha256(b"synthetic-wave-data").hexdigest()
                ),
            }
        ],
    }
    if fault == "bad_report_schema":
        report["schema"] = "different-synthetic-schema"
    if fault == "non_object_wav_entry":
        report["records"] = [None]
    key = content_fingerprint(report)
    if fault != "missing_report":
        evaluation._store(base, base / f"narration-report-{key}.json", report)
    if fault == "absent_wav":
        audio.unlink()
    if fault == "changed_wav":
        audio.write_bytes(b"synthetic-wave-data-hash-mismatch")
    report_path = base / f"narration-report-{key}.json"
    if fault == "invalid_report_json":
        report_path.write_bytes(b"untrusted synthetic report text, not JSON")
    if fault == "oversized_report":
        report_path.write_bytes(b"x" * 300_001)
    if fault == "symlink_report":
        report_path.unlink()
        report_path.symlink_to(audio)
    with pytest.raises(chapter_host.PrivateHostStop):
        chapter_host._narration_output_entries(root, 1, key)


@pytest.mark.parametrize("mode", ["status", "recover"])
@pytest.mark.parametrize("damage", ["missing", "symlink", "invalid_json", "oversized"])
def test_private_recovery_public_cli_normalizes_corrupt_journal_without_leaking_prose(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    damage: str,
) -> None:
    root = tmp_path / "synthetic-private-runtime"
    folder = root / "book-candidates/private-host/chapter-1-phase-events"
    folder.mkdir(parents=True)
    head = root / "book-candidates/private-host/chapter-1-phase-head.json"
    if damage == "symlink":
        secret = tmp_path / "synthetic-private-raw-do-not-print"
        secret.write_text("PRIVATE_SYNTHETIC_SHOULD_NOT_LEAK", encoding="utf-8")
        head.symlink_to(secret)
    elif damage == "invalid_json":
        head.write_bytes(b"PRIVATE_SYNTHETIC_SHOULD_NOT_LEAK malformed JSON")
    elif damage == "oversized":
        head.write_bytes(b"X" * 16_385)
    # Missing journal head is meaningful only with an existing phase folder.
    with pytest.raises(chapter_host.PrivateHostStop):
        chapter_host.recovery_report(root)
    if mode == "recover":
        checkpoint = chapter_host.private_recovery_archive(root)
        assert checkpoint["file_count"] == 0
        with tarfile.open(
            root / "book-candidates/private-host/chapter-recovery.tar", "r:"
        ) as archive:
            record = json.load(archive.extractfile("private-recovery-manifest.json"))
            assert record["chapters"][0]["inventory_status"] == "INCOMPLETE"
            assert record["chapters"][0]["files"] == []
        assert chapter_host.main([mode, "--artifact-root", str(root)]) == 0
        visible = capsys.readouterr()
        assert json.loads(visible.out)["files_retrieved"] is False
    else:
        assert chapter_host.main([mode, "--artifact-root", str(root)]) == 2
        visible = capsys.readouterr()
        assert visible.out == ""
        assert "PRIVATE_CHAPTER_HOST_FAILED_CLOSED" in visible.err
    assert "Traceback" not in visible.err
    assert "PRIVATE_SYNTHETIC_SHOULD_NOT_LEAK" not in visible.out + visible.err


@pytest.mark.parametrize("damage", ["outside", "missing", "symlink", "oversized"])
def test_private_recovery_member_read_is_bounded_and_never_trusts_retargeted_paths(
    tmp_path: Path, damage: str
) -> None:
    root = tmp_path / "synthetic-private-root"
    folder = root / "book-candidates/brief"
    folder.mkdir(parents=True)
    member = folder / "validated.json"
    if damage == "outside":
        member = tmp_path / "outside-book-candidates.json"
        member.write_bytes(b"synthetic outside artifact")
    elif damage == "symlink":
        elsewhere = tmp_path / "foreign-artifact.json"
        elsewhere.write_bytes(b"synthetic foreign artifact")
        member.symlink_to(elsewhere)
    elif damage == "oversized":
        member.write_bytes(b"x" * 9)
    with pytest.raises(chapter_host.PrivateHostStop):
        chapter_host._private_output_entry(root, member, 1, "text", 8)


def test_private_recovery_rejects_journal_that_rebinds_a_reviewed_brief(
    tmp_path: Path,
) -> None:
    """A later journal event cannot silently replace an already reviewed brief."""
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    bundle, scope = "a" * 64, "b" * 64
    chapter_host._phase(runner, 1, bundle, scope, "NOT_STARTED")
    chapter_host._phase(runner, 1, bundle, scope, "MINING_IN_FLIGHT")
    chapter_host._phase(runner, 1, bundle, scope, "BRIEF_IN_FLIGHT")
    chapter_host._phase(
        runner,
        1,
        bundle,
        scope,
        "AWAITING_HUMAN_REVIEW",
        {"brief_identity": "1" * 64},
    )
    chapter_host._phase(
        runner,
        1,
        bundle,
        scope,
        "RESUME_IN_FLIGHT",
        {"brief_identity": "2" * 64},
    )
    with pytest.raises(chapter_host.PrivateHostStop, match="identity changed within journal"):
        chapter_host._chapter_output_manifest(root, 1, bundle, scope)

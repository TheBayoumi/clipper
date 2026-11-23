"""Offline exact candidate-ref handoff and whole-book mining, with fabricated text only."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from clipper import cli
from clipper.books import acquisition, constrained, mining, scripting, whole_book
from clipper.books import orchestrator as orchestrator_module
from clipper.books.contracts import (
    BookText,
    ImageEvidence,
    ImageOccurrence,
    SourceBlock,
    SourceUnresolved,
)
from clipper.books.orchestrator import BookOrchestrator, BookRunError
from clipper.dag import DagStore
from clipper.stage_contracts import content_fingerprint

BASE = Path(__file__).resolve().parents[1] / "niches/book-source-candidates-v1.json"


def synthetic_candidate(
    tmp_path: Path, *, paragraphs: int = 129, with_image: bool = True
) -> tuple[acquisition.ReviewedCandidate, object]:
    """All prose and image bytes are original synthetic fixtures, never book extracts."""
    candidate = next(
        item for item in acquisition.load_registry(BASE) if item.book_id == "atomic-habits"
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<container><rootfile full-path="OPS/content.opf"/></container>',
        )
        manifest = "".join(
            f'<item id="c{i}" href="chapter{i}.xhtml" media-type="application/xhtml+xml"/>'
            for i in range(3)
        )
        z.writestr(
            "OPS/content.opf",
            '<package xmlns:dc="http://purl.org/dc/elements/1.1/" unique-identifier="uid">'
            f"<metadata><dc:title>{candidate.title}</dc:title>"
            f"<dc:creator>{candidate.author}</dc:creator>"
            f'<dc:identifier id="uid">urn:isbn:{candidate.isbn}</dc:identifier>'
            "</metadata><manifest>"
            + manifest
            + ('<item id="pic" href="figure.png" media-type="image/png"/>' if with_image else "")
            + "</manifest><spine>"
            + "".join(f'<itemref idref="c{i}"/>' for i in range(3))
            + "</spine></package>",
        )
        for chapter in range(3):
            parts = [f"<h2>Fabricated heading {chapter}</h2>"]
            for index in range(chapter, paragraphs, 3):
                parts.append(
                    f"<p>Invented paragraph {index} describes a reproducible "
                    "synthetic learning example, without borrowing text.</p>"
                )
            if with_image and chapter == 1:
                parts.append('<img src="figure.png" alt="diagram is not extracted words"/>')
            z.writestr(
                f"OPS/chapter{chapter}.xhtml",
                "<html><body>" + "".join(parts) + "</body></html>",
            )
        if with_image:
            z.writestr("OPS/figure.png", b"\x89PNG\r\n\x1a\nSYNTHETIC-ONLY")
    raw = stream.getvalue()
    selected = replace(
        candidate,
        expected_bytes=len(raw),
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )

    class RecordedTransport:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str, *, limit: int) -> acquisition.HTTPResult:
            self.calls += 1
            if "/metadata/" in url:
                metadata = {
                    "metadata": {"identifier": selected.record_url.rsplit("/", 1)[-1]},
                    "files": [
                        {
                            "name": selected.filename,
                            "size": str(selected.expected_bytes),
                            "source": "original",
                        }
                    ],
                }
                return acquisition.HTTPResult(
                    url, url, 200, "application/json", json.dumps(metadata).encode()
                )
            assert url.startswith("https://archive.org/download/")
            assert limit >= len(raw)
            return acquisition.HTTPResult(
                url,
                "https://ia801234.us.archive.org/synthetic/whole-epub",
                200,
                "application/epub+zip",
                raw,
            )

    return selected, RecordedTransport()


def ingested(tmp_path: Path, *, paragraphs: int = 129, with_image: bool = True):
    candidate, transport = synthetic_candidate(
        tmp_path, paragraphs=paragraphs, with_image=with_image
    )
    runner = BookOrchestrator(tmp_path / "artifacts")
    ingest_result = runner.ingest_candidate(
        candidate.title, registry=(candidate,), transport=transport
    )
    assert not isinstance(ingest_result, tuple)
    assert transport.calls == 2
    return runner, ingest_result, transport


def response_for(unit: whole_book.WorkUnit, *, count: int = 1) -> dict[str, object]:
    spans = constrained.span_pool(unit.text, unit.evidence_field)["spans"]
    return {
        "lessons": [
            {
                "id": spans[index]["id"] if index < len(spans) else f"idea-{index}",
                "topic": "Fabricated idea",
                "claim": "Unreviewed synthetic claim",
                "hook_angle": "An invented demonstration",
                "visual_action": "A cartoon actor places a token",
                "evidence_field": unit.evidence_field,
                "evidence": spans[index]["text"] if index < len(spans) else unit.text[:35],
            }
            for index in range(count)
        ]
    }


def test_canonical_prompt_states_existing_strict_validator_contract(
    tmp_path: Path,
) -> None:
    _runner, acquired, _transport = ingested(tmp_path, paragraphs=3, with_image=False)
    unit = whole_book.plan_for(acquired.book).units[0]
    request = whole_book.request_for(acquired.book, unit)
    assert request == whole_book._request_for_identity(
        acquired.book, unit, acquired.book.fingerprint
    )
    prompt = request["messages"][0]["content"]
    for requirement in (
        "one valid JSON object",
        'key "lessons"',
        "no prose, Markdown, or code fences",
        "0 to 3 candidate objects",
        "no supported candidate exists",
        "never as useful-lesson success",
        "nonempty string of at most 400 characters",
        "^[a-z0-9]+(?:-[a-z0-9]+)*$",
        '"source_evidence_choices"',
        "one exact paired entry",
        "Do not change either",
        "must not repeat in one reply",
        "specific claim",
        "selected entry exactly",
        '"evidence_field" must exactly repeat',
        "unreviewed",
        "not approved or publishable",
        "not authenticated original book prose",
    ):
        assert requirement in prompt
    for field in (
        "id",
        "topic",
        "claim",
        "hook_angle",
        "visual_action",
        "evidence",
        "evidence_field",
    ):
        assert f'"{field}"' in prompt
    supplied = json.loads(request["messages"][1]["content"])
    assert supplied["text"] == unit.text
    assert supplied["evidence_field"] == unit.evidence_field
    assert supplied["anchor"] == unit.anchor
    choices = supplied["source_evidence_choices"]
    assert len(choices) == len(request["constraints"]["spans"])
    for selected, span in zip(choices, request["constraints"]["spans"], strict=True):
        assert selected["id"] == span["id"]
        assert selected["evidence"] == unit.text[span["start"] : span["end"]]
        assert selected["start"] == span["start"]
        assert selected["end"] == span["end"]
    assert request["max_new_tokens"] == 960


def test_whole_book_validator_keeps_exact_bounded_unreviewed_evidence(
    tmp_path: Path,
) -> None:
    _runner, acquired, _transport = ingested(tmp_path, paragraphs=3, with_image=False)
    unit = replace(whole_book.plan_for(acquired.book).units[0], text="Z" * 401)
    valid = dict(response_for(unit)["lessons"][0], evidence="Z" * 400)
    assert whole_book.validate_reply({"lessons": [valid]}, unit) == [valid]
    candidate = whole_book.candidate_for(acquired.book, unit, valid)
    assert candidate["review_status"] == "MINED_UNREVIEWED"
    assert candidate["publish_eligible"] is False
    assert candidate["evidence_anchor"]["quote"] == "Z" * 400
    for field in valid:
        invalid = dict(valid, **{field: "Z" * 401})
        with pytest.raises(mining.BookMiningError, match="bounded nonempty"):
            whole_book.validate_reply({"lessons": [invalid]}, unit)
    for unsafe in ("source:section", "Invalid-Uppercase", "two--hyphens", "under_score"):
        with pytest.raises(mining.BookMiningError, match="invalid or duplicate"):
            whole_book.validate_reply({"lessons": [dict(valid, id=unsafe)]}, unit)
    with pytest.raises(mining.BookMiningError, match="invalid or duplicate"):
        whole_book.validate_reply({"lessons": [valid, valid]}, unit)


@pytest.mark.parametrize("kind", ["markdown-fence", "colon-id", "overlong-claim"])
def test_local_raw_invalid_response_is_not_repaired_or_accepted(tmp_path: Path, kind: str) -> None:
    runner, acquired, _transport = ingested(tmp_path, paragraphs=3, with_image=False)
    unit = whole_book.plan_for(acquired.book).units[0]
    lesson = dict(response_for(unit)["lessons"][0])
    if kind == "colon-id":
        lesson["id"] = "source:anchor:0"
    elif kind == "overlong-claim":
        lesson["claim"] = "Z" * 401
    raw = json.dumps({"lessons": [lesson]})
    if kind == "markdown-fence":
        raw = chr(96) * 3 + f"json\n{raw}\n" + chr(96) * 3
    result = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=1,
        local_completion=lambda _request: raw,
    )
    assert result["status"] == "incomplete"
    assert result["counts"]["failed"] == 1
    progress = json.loads(Path(result["progress_path"]).read_text())
    assert progress["progress"][0]["failure_category"] == "invalid_reply"
    raws = list((tmp_path / "artifacts/book-candidates/analysis/completions").glob("**/*.txt"))
    assert len(raws) == 1
    assert raws[0].read_bytes() == raw.encode("utf-8")


def test_incomplete_constrained_generation_preserves_raw_before_rejection(
    tmp_path: Path,
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    raw = '{"lessons":[{"id":"a","topic":"unfinished"'

    def incomplete(_request: dict[str, object]) -> str:
        raise scripting.IncompleteConstrainedResponse(raw)

    result = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=1,
        local_completion=incomplete,
    )
    assert result["status"] == "incomplete"
    assert result["counts"]["failed"] == 1
    progress = json.loads(Path(result["progress_path"]).read_bytes())
    assert progress["progress"][0]["failure_category"] == "incomplete_constrained_generation"
    raws = list((tmp_path / "artifacts/book-candidates/analysis/completions").glob("**/*.txt"))
    assert len(raws) == 1
    assert raws[0].read_bytes() == raw.encode("utf-8")


def test_local_raw_unselected_substring_fails_without_repair(
    tmp_path: Path,
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    unit = whole_book.plan_for(acquired.book).units[0]
    request = whole_book.request_for(acquired.book, unit)
    selected = request["constraints"]["spans"][0]
    assert len(selected["text"]) > 10
    unselected = selected["text"][:10]
    assert unselected in unit.text
    lesson = dict(response_for(unit)["lessons"][0], evidence=unselected)
    raw = json.dumps({"lessons": [lesson]}, ensure_ascii=False)
    result = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=1,
        local_completion=lambda _request: raw,
    )
    assert result["status"] == "incomplete"
    progress = json.loads(Path(result["progress_path"]).read_bytes())
    row = progress["progress"][0]
    assert row["failure_category"] == "invalid_reply"
    assert row["evidence_coverage"]["policy"] == constrained.SPAN_POLICY
    assert row["evidence_coverage"]["selectable_spans"][0]["sha256"] == selected["sha256"]
    assert "text" not in row["evidence_coverage"]["selectable_spans"][0]
    original = list((tmp_path / "artifacts/book-candidates/analysis/completions").glob("**/*.txt"))
    assert len(original) == 1
    assert original[0].read_bytes() == raw.encode("utf-8")


def test_local_fresh_and_replay_recheck_canonical_selected_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=1, with_image=False)
    units = whole_book.plan_for(acquired.book).units
    responses = responses_for(acquired.book)
    requests = [whole_book.request_for(acquired.book, unit) for unit in units]
    keys = [content_fingerprint(request) for request in requests]
    real_check = constrained.validate_local_choice
    checks: list[str] = []

    def checked(lessons: list[dict[str, str]], actual: dict[str, object]) -> None:
        checks.append(content_fingerprint(actual))
        real_check(lessons, actual)

    monkeypatch.setattr(constrained, "validate_local_choice", checked)
    first = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(units),
        local_completion=lambda request: json.dumps(responses[content_fingerprint(request)]),
    )
    assert first["counts"]["failed"] == 0
    assert first["status"] == "complete"
    second = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(units),
        local_completion=lambda _request: pytest.fail("cache must not generate"),
    )
    assert second == first
    assert checks == [key for key in keys for _ in range(2)] + keys
    report = json.loads(Path(first["progress_path"]).read_bytes())
    summary = report["evidence_coverage_summary"]
    assert summary["policy"] == constrained.SPAN_POLICY
    assert summary["covered_fragment_chars"] > 0
    assert summary["unit_start_context_unknown"] >= 1
    export = json.loads(Path(first["export_path"]).read_bytes())
    assert export["evidence_coverage_summary"] == summary
    assert export["publish_eligible"] is False


def test_declared_occurrence_pair_provenance_is_raw_derived_not_recorded(
    tmp_path: Path,
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=1, with_image=False)
    work = whole_book.plan_for(acquired.book)
    request = whole_book.request_for(acquired.book, work.units[0])
    response = response_for(work.units[0])
    raw = json.dumps(response, ensure_ascii=False)
    result = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(work.units),
        local_completion=lambda request: json.dumps(
            response_for(
                next(
                    unit
                    for unit in work.units
                    if unit.text == json.loads(request["messages"][1]["content"])["text"]
                )
            )
        ),
    )
    assert result["status"] == "complete"
    catalog = json.loads(Path(result["export_path"]).read_bytes())
    mined = catalog["books"][0]["lessons"]
    assert mined
    expected = constrained.selected_span(response["lessons"][0], request)
    chosen = mined[0]["evidence_anchor"]["selected_occurrence"]
    assert chosen == {
        "unit_start": expected["start"],
        "unit_end": expected["end"],
        "quote_sha256": expected["sha256"],
        "boundary": expected["boundary"],
        "id": expected["id"],
    }
    assert mined[0]["review_status"] == "MINED_UNREVIEWED"
    assert mined[0]["publish_eligible"] is False
    assert (
        "selected_occurrence"
        not in whole_book.candidate_for(acquired.book, work.units[0], response["lessons"][0])[
            "evidence_anchor"
        ]
    )
    # Distinct raw IDs disambiguate exact repeated-text occurrences.
    repeated = replace(work.units[0], text="Repeat. Repeat.")
    pool = constrained.span_pool(repeated.text, repeated.evidence_field)
    one, two = pool["spans"]
    assert one["text"] == two["text"] and one["id"] != two["id"]
    for span in (one, two):
        choice = dict(response["lessons"][0], id=span["id"], evidence=span["text"])
        validated = constrained.selected_span(
            choice,
            {
                "messages": [
                    {"role": "system", "content": "synthetic"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "text": repeated.text,
                                "evidence_field": repeated.evidence_field,
                            }
                        ),
                    },
                ],
                "constraints": pool,
            },
        )
        item = whole_book.candidate_for(acquired.book, repeated, choice, local_span=validated)
        occ = item["evidence_anchor"]["selected_occurrence"]
        assert occ["unit_start"] == span["start"]
        assert occ["unit_end"] == span["end"]
    assert raw.encode("utf-8") == json.dumps(response, ensure_ascii=False).encode("utf-8")


def test_fake_existing_generation_path_accepts_multi_eos_and_rejects_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    class FakeInput:
        shape = (1, 1)

    class FakeTokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"input_ids": FakeInput()}

        def decode(self, _tokens: object, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            if isinstance(_tokens, list) and _tokens and _tokens[0] == 222:
                return '{"lessons":[{"id":"unfinished"'
            return '{"lessons":[]}'

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=[0, 42])

        def __init__(self, terminal: int, *, partial: bool = False) -> None:
            self.terminal = terminal
            self.partial = partial

        def generate(self, **_kwargs: object) -> list[list[int]]:
            # Pure fake token output, not a loaded model or real inference.
            return [[999, 222 if self.partial else 111, self.terminal]]

    actual_import = scripting.importlib.import_module
    fake_torch = SimpleNamespace(
        manual_seed=lambda _seed: None,
        inference_mode=lambda: nullcontext(),
    )

    def imported(name: str) -> object:
        return fake_torch if name == "torch" else actual_import(name)

    with monkeypatch.context() as patch:
        patch.setattr(scripting.importlib, "import_module", imported)
        patch.setattr(constrained, "prefix_for", lambda *_args: lambda _batch, _ids: [0])
        patch.setattr(scripting, "_public_cpu_smoke_phase", lambda _phase: None)
        request = {"seed": 42, "max_new_tokens": 960, "messages": []}
        tok = FakeTokenizer()
        for last in (0, 42):
            patch.setattr(scripting, "_script_pipeline", lambda last=last: (FakeModel(last), tok))
            assert scripting._run_pinned_qwen(request, normalize_script=False) == ('{"lessons":[]}')
        # The injected configured EOS checks post-generation recognition only.
        # It does not claim the production LMFE callback permits this EOS.
        patch.setattr(scripting, "_script_pipeline", lambda: (FakeModel(42, partial=True), tok))
        partial = scripting._run_pinned_qwen(request, normalize_script=False)
        assert partial == '{"lessons":[{"id":"unfinished"'
        with pytest.raises(ValueError):
            whole_book.load_strict_reply(partial)
        patch.setattr(scripting, "_script_pipeline", lambda: (FakeModel(777), tok))
        with pytest.raises(scripting.IncompleteConstrainedResponse) as exc:
            scripting._run_pinned_qwen(request, normalize_script=False)
        assert exc.value.raw == '{"lessons":[]}'


def test_canonical_prompt_change_invalidates_cache_but_keeps_old_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, _transport = ingested(tmp_path, paragraphs=3, with_image=False)
    unit = whole_book.plan_for(acquired.book).units[0]
    current_request = whole_book.request_for(acquired.book, unit)
    legacy_prompt = (
        'Return JSON only: {"lessons":[{"id":str,"topic":str,"claim":str,'
        '"hook_angle":str,"visual_action":str,"evidence":str,'
        '"evidence_field":str}]}. Return zero to three unreviewed candidates.'
        " Quote an exact substring only from the supplied text field. "
        "Repeat its evidence_field verbatim. Image/OCR/research wording is "
        "not authenticated original book prose. Never assert approval."
    )
    legacy_request = {
        **current_request,
        "messages": [
            {**current_request["messages"][0], "content": legacy_prompt},
            current_request["messages"][1],
        ],
    }
    old_key = content_fingerprint(legacy_request)
    new_key = content_fingerprint(current_request)
    assert old_key != new_key
    assert whole_book.recorded_answer({old_key: {"lessons": []}}, current_request) is None
    assert whole_book.recorded_answer({new_key: {"lessons": []}}, current_request) == {
        "lessons": []
    }
    canonical_request = whole_book._request_for_identity

    def old_request(
        book: BookText, current_unit: whole_book.WorkUnit, fingerprint: str
    ) -> dict[str, object]:
        request = canonical_request(book, current_unit, fingerprint)
        request["messages"][0]["content"] = legacy_prompt
        return request

    produced: list[str] = []

    def synthetic_completion(request: dict[str, object]) -> str:
        produced.append(content_fingerprint(request))
        supplied = json.loads(request["messages"][1]["content"])
        return json.dumps(
            {
                "lessons": [
                    {
                        "id": request["constraints"]["spans"][0]["id"],
                        "topic": "Fabricated repetition",
                        "claim": "An invented example repeats a fabricated action",
                        "hook_angle": "Observe a repeated synthetic example",
                        "visual_action": "An actor places a token twice",
                        "evidence": request["constraints"]["spans"][0]["text"],
                        "evidence_field": supplied["evidence_field"],
                    }
                ]
            }
        )

    with monkeypatch.context() as patch:
        patch.setattr(whole_book, "_request_for_identity", old_request)
        earlier = runner.mine_candidate(
            acquired.analysis_ref_path,
            local_cpu=True,
            max_new_units=1,
            local_completion=synthetic_completion,
        )
    raw_root = tmp_path / "artifacts/book-candidates/analysis/completions"
    historic_raw = {p: p.read_bytes() for p in raw_root.glob("**/*.txt")}
    assert len(historic_raw) == 1
    current = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=1,
        local_completion=synthetic_completion,
    )
    assert produced == [old_key, new_key]
    old_progress = json.loads(Path(earlier["progress_path"]).read_text())
    new_progress = json.loads(Path(current["progress_path"]).read_text())
    assert (
        old_progress["progress"][0]["generation_stage_key"]
        != new_progress["progress"][0]["generation_stage_key"]
    )
    assert earlier["progress_path"] != current["progress_path"]
    assert len(list(raw_root.glob("**/*.txt"))) == 2
    assert all(p.is_file() and p.read_bytes() == data for p, data in historic_raw.items())


def responses_for(book: BookText, *, count: int = 1) -> dict[str, object]:
    plan = whole_book.plan_for(book)
    return {
        content_fingerprint(whole_book.request_for(book, unit)): response_for(unit, count=count)
        for unit in plan.units
    }


def _write_responses(path: Path, replies: dict[str, object]) -> str:
    data = (json.dumps(replies, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_title_only_ingest_handoff_more_than_128_chunks_64_candidates_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, transport = ingested(tmp_path)
    assert acquired.book.recovery_status == "incomplete_recovery"
    plan = whole_book.plan_for(acquired.book)
    assert len(plan.units) > 128
    assert plan.skipped
    assert acquired.analysis_ref_path.is_file()
    assert not (tmp_path / "artifacts/book-text/atomic-habits.json").exists()
    replies = responses_for(acquired.book)
    source = tmp_path / "responses.json"
    _write_responses(source, replies)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: pytest.fail("network call"))
    assert (
        cli.main(
            [
                "books",
                "mine",
                "--candidate-ref",
                str(acquired.analysis_ref_path),
                "--book-id",
                "atomic-habits",
                "--responses",
                str(source),
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert transport.calls == 2
    assert summary["total_work_units"] == len(plan.units)
    assert summary["candidate_count"] > 64
    assert summary["counts"]["unresolved"] == summary["counts"]["failed"] == 0
    assert summary["counts"]["skipped"] == len(plan.skipped)
    assert summary["recovery_status"] == "incomplete_recovery"
    assert summary["review_status"] == "MINED_UNREVIEWED"
    assert summary["publish_eligible"] is False
    assert summary["status"] == "complete"
    progress_path = Path(summary["progress_path"])
    export_path = Path(summary["export_path"])
    progress = json.loads(progress_path.read_text())
    export = json.loads(export_path.read_text())
    assert len(progress["progress"]) == len(plan.units)
    assert sorted(p["unit"] for p in progress["progress"]) == list(range(len(plan.units)))
    assert max(p["window"] for p in progress["progress"]) >= 4
    assert len(export["books"][0]["lessons"]) == len(plan.units)
    assert len({x["id"] for x in export["books"][0]["lessons"]}) == len(plan.units)
    assert all(x["review_status"] == "MINED_UNREVIEWED" for x in export["books"][0]["lessons"])
    assert all(x["publish_eligible"] is False for x in export["books"][0]["lessons"])
    assert not (tmp_path / "artifacts/mined/atomic-habits.json").exists()
    for path in (acquired.analysis_ref_path, progress_path, export_path):
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    stage_root = tmp_path / "artifacts/book-dag/books-mining-unit-v1"
    assert len(list(stage_root.glob("*/stage.json"))) == len(plan.units)
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in stage_root.glob("*/output.json"))
    original_export = export_path.read_bytes()
    original_progress = progress_path.read_bytes()
    assert (
        runner.mine_candidate(
            acquired.analysis_ref_path,
            responses=replies,
            response_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        == summary
    )
    assert export_path.read_bytes() == original_export
    assert progress_path.read_bytes() == original_progress


def test_private_analysis_hierarchy_with_permissive_umask(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o755)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    previous = os.umask(0)
    try:
        artifact_root = tmp_path / "artifacts"
        runner, acquired, _ = ingested(tmp_path, paragraphs=9, with_image=False)
        replies = responses_for(acquired.book)
        report = runner.mine_candidate(
            acquired.analysis_ref_path,
            responses=replies,
            response_file_sha256=content_fingerprint(replies),
        )
        assert report["status"] == "complete"
        candidate_root = artifact_root / "book-candidates"
        paths = [
            acquired.analysis_ref_path,
            Path(report["progress_path"]),
            Path(report["export_path"]),
        ]
        for file in paths:
            assert file.stat().st_mode & 0o777 == 0o600
            current = file.parent
            while current != candidate_root.parent:
                assert current.stat().st_mode & 0o777 == 0o700, current
                current = current.parent
        dag = artifact_root / "book-dag" / "books-mining-unit-v1"
        assert list(dag.glob("*/stage.json"))
        for file in dag.glob("*/stage.json"):
            assert file.stat().st_mode & 0o777 == 0o600
            assert file.parent.stat().st_mode & 0o777 == 0o700
        for file in dag.glob("*/output.json"):
            assert file.stat().st_mode & 0o777 == 0o600
        assert shared.stat().st_mode & 0o777 == 0o755
        assert tmp_path.stat().st_mode & 0o777 == 0o755
    finally:
        os.umask(previous)


def test_reused_analysis_ancestors_tightened_without_changing_artifact_bytes(
    tmp_path: Path,
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=9, with_image=False)
    replies = responses_for(acquired.book)
    digest = content_fingerprint(replies)
    initial = runner.mine_candidate(
        acquired.analysis_ref_path, responses=replies, response_file_sha256=digest
    )
    assert initial["status"] == "complete"
    root = tmp_path / "artifacts"
    candidate_root = root / "book-candidates"
    originals = {
        acquired.analysis_ref_path: acquired.analysis_ref_path.read_bytes(),
        Path(initial["progress_path"]): Path(initial["progress_path"]).read_bytes(),
        Path(initial["export_path"]): Path(initial["export_path"]).read_bytes(),
    }
    directories = [
        candidate_root,
        candidate_root / "analysis-refs",
        acquired.analysis_ref_path.parent,
        candidate_root / "analysis",
        candidate_root / "analysis" / "progress",
        Path(initial["progress_path"]).parent,
        candidate_root / "analysis" / "exports",
        Path(initial["export_path"]).parent,
    ]
    root.chmod(0o755)
    for directory in directories:
        directory.chmod(0o777)
    previous = os.umask(0)
    try:
        repeated = runner.ingest_candidate(
            acquired.candidate.title, registry=(acquired.candidate,), offline=True
        )
        assert not isinstance(repeated, SourceUnresolved)
        assert repeated.analysis_ref_path == acquired.analysis_ref_path
        assert (
            runner.mine_candidate(
                acquired.analysis_ref_path, responses=replies, response_file_sha256=digest
            )
            == initial
        )
    finally:
        os.umask(previous)
    for directory in directories:
        assert directory.stat().st_mode & 0o777 == 0o700, directory
    for file, original in originals.items():
        assert file.read_bytes() == original
        assert file.stat().st_mode & 0o777 == 0o600
    assert root.stat().st_mode & 0o777 == 0o755


def test_partial_missing_invalid_and_extended_response_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, _transport = ingested(tmp_path, paragraphs=96, with_image=False)
    book = acquired.book
    units = whole_book.plan_for(book).units
    assert len(units) > 64
    complete = responses_for(book)
    keys = [content_fingerprint(whole_book.request_for(book, unit)) for unit in units]
    partial = {key: complete[key] for key in keys[:38]}
    file = tmp_path / "recorded.json"
    digest = _write_responses(file, partial)
    first = runner.mine_candidate(
        acquired.analysis_ref_path, responses=partial, response_file_sha256=digest
    )
    assert first["status"] == "incomplete"
    assert first["counts"]["unresolved"] == len(units) - 38
    assert first["export_path"] is None
    assert Path(first["progress_path"]).is_file()
    recorded_before = sorted(
        p.parent.name
        for p in (tmp_path / "artifacts/book-dag/books-mining-unit-v1").glob("*/stage.json")
    )
    assert len(recorded_before) == 38
    partial[keys[40]] = {"lessons": [{"id": "bad", "evidence": "not anchored"}]}
    digest = _write_responses(file, partial)
    broken = runner.mine_candidate(
        acquired.analysis_ref_path, responses=partial, response_file_sha256=digest
    )
    assert broken["counts"]["failed"] == 1
    assert broken["status"] == "incomplete"
    assert broken["export_path"] is None
    stage_records = (tmp_path / "artifacts/book-dag/books-mining-unit-v1").glob("*/stage.json")
    assert len(list(stage_records)) == 39
    corrected = dict(complete)
    corrected[keys[41]] = {"lessons": []}
    digest = _write_responses(file, corrected)
    original_execute = DagStore.execute
    seen: list[bool] = []

    def wrapped(self, identity, operation):
        value, hit = original_execute(self, identity, operation)
        if identity.stage_name == "books.mining.unit.v1":
            seen.append(hit)
        return value, hit

    monkeypatch.setattr(DagStore, "execute", wrapped)
    done = runner.mine_candidate(
        acquired.analysis_ref_path, responses=corrected, response_file_sha256=digest
    )
    assert done["status"] == "complete"
    assert done["counts"]["zero_candidate"] == 1
    assert done["candidate_count"] == len(units) - 1
    assert len(seen) == len(units)
    assert all(seen[:38])
    assert all(not x for x in seen[38:])
    assert len(recorded_before) == 38
    assert Path(done["export_path"]).is_file()


@pytest.mark.parametrize(
    "tamper",
    [
        "book-id",
        "stage-key",
        "source",
        "parser",
        "raw",
        "dossier",
        "extracted",
        "unsafe-path",
        "cross-book",
        "extra-field",
    ],
)
def test_candidate_reference_is_checked_before_any_mining_work(
    tmp_path: Path, tamper: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, result, _ = ingested(tmp_path, paragraphs=9)
    ref_path = result.analysis_ref_path
    reference = json.loads(ref_path.read_text())
    if tamper == "book-id":
        reference["book_id"] = "other"
    elif tamper == "stage-key":
        reference["ingest_stage_key"] = "f" * 64
    elif tamper == "source":
        reference["source_fingerprint"] = "f" * 64
    elif tamper == "parser":
        reference["parser_version"] = "unrelated-parser"
    elif tamper == "extra-field":
        reference["surprise"] = True
    elif tamper == "raw":
        result.acquired.raw_path.write_bytes(b"changed synthetic payload")
    elif tamper == "dossier":
        result.dossier_path.write_text('{"schema":"other"}')
    elif tamper == "extracted":
        result.extracted_path.write_text('{"changed":true}')
    elif tamper == "unsafe-path":
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps(reference))
        ref_path = outside
    if tamper in {"book-id", "stage-key", "source", "parser", "extra-field"}:
        ref_path.write_text(json.dumps(reference))
    monkeypatch.setattr(whole_book, "plan_for", lambda _book: pytest.fail("producer was invoked"))
    with pytest.raises(BookRunError, match="candidate analysis reference"):
        runner.mine_candidate(
            ref_path,
            responses={},
            response_file_sha256="a" * 64,
            book_id="psychology-of-money" if tamper == "cross-book" else None,
        )
    assert not (tmp_path / "artifacts/book-candidates/analysis").exists()


def test_zero_candidates_are_honest_not_editorial_approval(tmp_path: Path) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=9, with_image=False)
    responses = responses_for(acquired.book, count=0)
    digest = content_fingerprint(responses)
    result = runner.mine_candidate(
        acquired.analysis_ref_path, responses=responses, response_file_sha256=digest
    )
    assert result["status"] == "complete"
    assert result["candidate_count"] == 0
    assert result["counts"]["zero_candidate"] == result["total_work_units"]
    catalog = json.loads(Path(result["export_path"]).read_text())
    assert catalog["books"][0]["lessons"] == []
    assert catalog["publish_eligible"] is False
    assert catalog["review_status"] == "MINED_UNREVIEWED"


def test_evidence_field_and_origin_never_conflate_research_ocr_and_source(tmp_path: Path) -> None:
    _, acquired, _ = ingested(tmp_path, paragraphs=9)
    book = acquired.book
    original = book.blocks[-1]
    digest = "2" * 64
    reading = (
        SourceBlock("n-1", "chapter-2", "fig-n.png", "image", "", digest),
        SourceBlock("m-1", "chapter-2", "fig-m.png", "image", "", "3" * 64),
        SourceBlock("g-1", "chapter-2", "fig-g.png", "image", "", "4" * 64),
    )
    images = (
        ImageEvidence(
            digest,
            "image/png",
            5,
            (
                ImageOccurrence(
                    "n-1",
                    "fig-n.png",
                    "",
                    observation_id="synthetic-obs",
                    original_transcription="ORIGINAL synthetic unreviewed diagram",
                    normalized_transcription="NORMALIZED synthetic interpretation",
                    review_status="needs_review",
                    research_link_status="unverified",
                ),
            ),
            review_status="needs_review",
        ),
        ImageEvidence(
            "3" * 64,
            "image/png",
            5,
            (ImageOccurrence("m-1", "fig-m.png", ""),),
            machine_transcription="OCR synthetic ambiguous words",
            review_status="machine_unreviewed",
        ),
        ImageEvidence(
            "4" * 64,
            "image/png",
            5,
            (ImageOccurrence("g-1", "fig-g.png", ""),),
            review_status="needs_review",
        ),
    )
    synthetic = replace(book, blocks=(original, *reading), images=images)
    plan = whole_book.plan_for(synthetic)
    assert len(plan.skipped) == 1
    assert plan.skipped[0]["reason"] == "unrecovered_image"
    normalized = next(x for x in plan.units if x.origin == "normalized_research")
    machine = next(x for x in plan.units if x.origin == "machine_ocr")
    assert normalized.evidence_field == "normalized_research_text"
    assert normalized.review_status == "needs_review"
    assert normalized.research_link_status == "unverified"
    assert machine.evidence_field == "machine_ocr_text"
    original_quote = response_for(normalized)["lessons"][0]
    assert isinstance(original_quote, dict)
    mismatch = dict(original_quote, evidence="ORIGINAL synthetic unreviewed diagram")
    with pytest.raises(mining.BookMiningError, match="exact provenance field"):
        whole_book.validate_reply({"lessons": [mismatch]}, normalized)
    mismatch["evidence"] = normalized.text[:20]
    mismatch["evidence_field"] = "original_image_text"
    with pytest.raises(mining.BookMiningError, match="exact provenance field"):
        whole_book.validate_reply({"lessons": [mismatch]}, normalized)
    correct = response_for(normalized)["lessons"][0]
    assert isinstance(correct, dict)
    assert whole_book.validate_reply({"lessons": [correct]}, normalized)
    provisional = whole_book.candidate_for(synthetic, normalized, correct)
    assert provisional["evidence_anchor"]["field"] == "normalized_research_text"
    assert provisional["evidence_anchor"]["origin"] == "normalized_research"
    assert provisional["publish_eligible"] is False
    assert (
        whole_book.candidate_for(synthetic, machine, response_for(machine)["lessons"][0])[
            "evidence_anchor"
        ]["field"]
        == "machine_ocr_text"
    )


@pytest.mark.parametrize("target", ["ref", "dossier", "manifest", "stage"])
@pytest.mark.parametrize("tail", [b" ", b"{"])
def test_candidate_json_over_budget_rejects_complete_prefix_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    tail: bytes,
) -> None:
    runner, result, _ = ingested(tmp_path, paragraphs=9, with_image=False)
    paths = {
        "ref": result.analysis_ref_path,
        "dossier": result.dossier_path,
        "manifest": (
            tmp_path
            / "artifacts/book-candidates/manifests"
            / f"{result.candidate.fingerprint}.json"
        ),
        "stage": (tmp_path / "artifacts/book-dag/books-ingest" / result.stage_key / "stage.json"),
    }
    target_path = paths[target]
    max_bytes = 131_072 if target == "ref" else 2_000_000
    original_bytes = target_path.read_bytes()
    assert len(original_bytes) < max_bytes
    # The original JSON is complete and valid. The unread tail lies past the
    # old limit, so accepting the prefix would silently ignore malformed bytes.
    padding = b" " * (max_bytes - len(original_bytes) + 1)
    mutated = original_bytes + padding + tail
    assert len(mutated) > max_bytes
    target_path.write_bytes(mutated)
    with pytest.raises(BookRunError, match="bounded input"):
        orchestrator_module._bounded_candidate_json(target_path, max_bytes=max_bytes)
    monkeypatch.setattr(
        whole_book,
        "plan_for",
        lambda _book: pytest.fail("invalid candidate reached mining producer"),
    )
    with pytest.raises(BookRunError, match="candidate analysis reference"):
        runner.mine_candidate(
            result.analysis_ref_path,
            responses={},
            response_file_sha256="a" * 64,
        )
    assert not (tmp_path / "artifacts/book-candidates/analysis").exists()


@pytest.mark.parametrize("max_bytes", [1, 16, 131_072, 2_000_000])
def test_bounded_candidate_json_exact_eof_and_first_excess_byte(
    tmp_path: Path, max_bytes: int
) -> None:
    """Use real file bytes, not stat sizes or single-short-read assumptions."""
    file = tmp_path / "boundary.json"
    valid = b"1" + b" " * (max_bytes - 1)
    file.write_bytes(valid)
    assert orchestrator_module._bounded_candidate_json(file, max_bytes=max_bytes) == 1
    assert file.read_bytes() == valid
    file.write_bytes(valid + b" ")
    with pytest.raises(BookRunError, match="bounded input"):
        orchestrator_module._bounded_candidate_json(file, max_bytes=max_bytes)
    file.write_bytes(valid + b"invalid-tail")
    with pytest.raises(BookRunError, match="bounded input"):
        orchestrator_module._bounded_candidate_json(file, max_bytes=max_bytes)


def test_canonical_ref_remains_readable_at_exact_json_budget(tmp_path: Path) -> None:
    runner, result, _ = ingested(tmp_path, paragraphs=9, with_image=False)
    path = result.analysis_ref_path
    original = path.read_bytes()
    path.write_bytes(original + b" " * (131_072 - len(original)))
    assert len(path.read_bytes()) == 131_072
    reference, book = runner.load_candidate_for_analysis(path)
    assert reference.ingest_stage_key == result.stage_key
    assert book.fingerprint == result.book.fingerprint


def test_same_response_digest_failed_then_success_keeps_both_immutable_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, result, _ = ingested(tmp_path, paragraphs=18, with_image=False)
    plan = whole_book.plan_for(result.book)
    replies = responses_for(result.book)
    digest = content_fingerprint(replies)
    real_execute = DagStore.execute
    transient = ValueError("synthetic one-time mining DAG failure")
    attempts = 0

    def transient_once(self, identity, operation):
        nonlocal attempts
        if identity.stage_name == "books.mining.unit.v1":
            attempts += 1
            if attempts == 1:
                raise transient
        return real_execute(self, identity, operation)

    with monkeypatch.context() as patch:
        patch.setattr(DagStore, "execute", transient_once)
        incomplete = runner.mine_candidate(
            result.analysis_ref_path,
            responses=replies,
            response_file_sha256=digest,
        )
    assert incomplete["status"] == "incomplete"
    assert incomplete["counts"]["failed"] == 1
    assert incomplete["export_path"] is None
    assert attempts == len(plan.units)
    first_path = Path(incomplete["progress_path"])
    first_bytes = first_path.read_bytes()
    assert json.loads(first_bytes)["status"] == "incomplete"

    hits: list[bool] = []

    def track_hits(self, identity, operation):
        output, reused = real_execute(self, identity, operation)
        if identity.stage_name == "books.mining.unit.v1":
            hits.append(reused)
        return output, reused

    with monkeypatch.context() as patch:
        patch.setattr(DagStore, "execute", track_hits)
        complete = runner.mine_candidate(
            result.analysis_ref_path,
            responses=replies,
            response_file_sha256=digest,
        )
    assert complete["status"] == "complete"
    assert complete["counts"]["failed"] == 0
    assert complete["candidate_count"] == len(plan.units)
    assert len(hits) == len(plan.units)
    assert sum(hits) == len(plan.units) - 1
    completed_path = Path(complete["progress_path"])
    export_path = Path(complete["export_path"])
    completed_bytes = completed_path.read_bytes()
    original_export = export_path.read_bytes()
    assert first_path != completed_path
    assert first_path.read_bytes() == first_bytes
    assert json.loads(completed_bytes)["status"] == "complete"
    export = json.loads(original_export)
    assert export["progress_sha256"] == hashlib.sha256(completed_bytes).hexdigest()
    assert completed_path.name == export_path.name
    assert first_path.stat().st_mode & 0o777 == 0o600
    assert completed_path.stat().st_mode & 0o777 == 0o600
    assert export_path.stat().st_mode & 0o777 == 0o600

    hits.clear()
    with monkeypatch.context() as patch:
        patch.setattr(DagStore, "execute", track_hits)
        replay = runner.mine_candidate(
            result.analysis_ref_path,
            responses=replies,
            response_file_sha256=digest,
        )
    assert replay == complete
    assert len(hits) == len(plan.units) and all(hits)
    assert first_path.read_bytes() == first_bytes
    assert completed_path.read_bytes() == completed_bytes
    assert export_path.read_bytes() == original_export


@pytest.mark.parametrize("paragraphs", [9, 99])
def test_whole_book_fingerprint_traversals_are_constant_per_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paragraphs: int,
) -> None:
    runner, result, _ = ingested(tmp_path, paragraphs=paragraphs, with_image=False)
    replies = responses_for(result.book)
    counts = {"book": 0, "plan": 0}
    actual_book = BookText.fingerprint.fget
    actual_plan = whole_book.WorkPlan.fingerprint.fget
    assert actual_book is not None and actual_plan is not None

    def count_book(book: BookText) -> str:
        counts["book"] += 1
        return actual_book(book)

    def count_plan(work: whole_book.WorkPlan) -> str:
        counts["plan"] += 1
        return actual_plan(work)

    with monkeypatch.context() as patch:
        patch.setattr(BookText, "fingerprint", property(count_book))
        patch.setattr(whole_book.WorkPlan, "fingerprint", property(count_plan))
        result_status = runner.mine_candidate(
            result.analysis_ref_path,
            responses=replies,
            response_file_sha256=content_fingerprint(replies),
        )
    assert result_status["status"] == "complete"
    assert result_status["total_work_units"] >= paragraphs
    # Exact full-material traversal count is independent of the number of
    # units; unit-local SHA operations are intentionally not restricted.
    assert counts["book"] <= 3
    assert counts["plan"] == 1


def test_local_cpu_limited_resume_private_raw_and_exact_cache_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=9, with_image=False)
    units = whole_book.plan_for(acquired.book).units
    replies = responses_for(acquired.book)
    generated: list[str] = []
    root = tmp_path / "artifacts"

    def fake(request: dict[str, object]) -> str:
        key = content_fingerprint(request)
        generated.append(key)
        return json.dumps(replies[key], sort_keys=True)

    def forbidden(*_a: object, **_k: object) -> None:
        pytest.fail("no model initialization or network allowed for injected producer")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("clipper.books.scripting._script_pipeline", forbidden)
    old_umask = os.umask(0)
    try:
        first = runner.mine_candidate(
            acquired.analysis_ref_path,
            local_cpu=True,
            max_new_units=2,
            local_completion=fake,
        )
        assert first["status"] == "incomplete"
        assert first["counts"]["complete"] == 2
        assert first["counts"]["unresolved"] == len(units) - 2
        assert first["export_path"] is None
        first_bytes = Path(first["progress_path"]).read_bytes()
        assert first_bytes
        assert len(generated) == 2
        for _ in range(len(units)):
            current = runner.mine_candidate(
                acquired.analysis_ref_path,
                local_cpu=True,
                max_new_units=2,
                local_completion=fake,
            )
            if current["status"] == "complete":
                break
        assert current["status"] == "complete"
        assert len(generated) == len(units)
        assert current["candidate_count"] == len(units)
        assert first["progress_path"] != current["progress_path"]
        assert Path(first["progress_path"]).read_bytes() == first_bytes

        progress = json.loads(Path(current["progress_path"]).read_bytes())
        export = json.loads(Path(current["export_path"]).read_bytes())
        assert progress["completion_mode"] == "local_cpu"
        assert progress["response_file_sha256"] is None
        assert progress["counts"]["failed"] == 0
        assert (
            export["progress_sha256"]
            == hashlib.sha256(Path(current["progress_path"]).read_bytes()).hexdigest()
        )
        assert export["publish_eligible"] is False
        assert export["books"][0]["lessons"][0]["evidence_anchor"]["field"] == "native_text"
        assert all(
            row["completion_origin"] == "INJECTED_TEST_PRODUCER" for row in progress["progress"]
        )
        assert all(row["raw_completion_sha256"] for row in progress["progress"])
        completion_root = root / "book-candidates/analysis/completions"
        raw_files = list(completion_root.glob("**/*.txt"))
        assert len(raw_files) == len(units)
        for raw in raw_files:
            assert raw.stat().st_mode & 0o777 == 0o600
            parent = raw.parent
            while parent != root / "book-candidates":
                assert parent.stat().st_mode & 0o777 == 0o700, parent
                parent = parent.parent
        for report in (Path(first["progress_path"]), Path(current["progress_path"])):
            assert report.stat().st_mode & 0o777 == 0o600
        assert Path(current["export_path"]).stat().st_mode & 0o777 == 0o600
        generation_root = root / "book-dag/books-mining-local-completion-v1"
        stages = list(generation_root.glob("*/stage.json"))
        assert len(stages) == len(units)
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in stages)
        assert all(p.parent.stat().st_mode & 0o777 == 0o700 for p in stages)

        def must_not_generate(_request: dict[str, object]) -> str:
            pytest.fail("completed local generation must be reused")

        replay = runner.mine_candidate(
            acquired.analysis_ref_path,
            local_cpu=True,
            max_new_units=1,
            local_completion=must_not_generate,
        )
        assert replay == current
        assert (
            Path(replay["progress_path"]).read_bytes()
            == Path(current["progress_path"]).read_bytes()
        )
        assert len(generated) == len(units)
    finally:
        os.umask(old_umask)


def test_recorded_empty_does_not_alias_real_local_completion(tmp_path: Path) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    units = whole_book.plan_for(acquired.book).units
    recorded = responses_for(acquired.book, count=0)
    saved = runner.mine_candidate(
        acquired.analysis_ref_path,
        responses=recorded,
        response_file_sha256=content_fingerprint(recorded),
    )
    assert saved["status"] == "complete" and saved["candidate_count"] == 0
    expected = responses_for(acquired.book)
    seen: list[str] = []

    def producer(request: dict[str, object]) -> str:
        key = content_fingerprint(request)
        seen.append(key)
        return json.dumps(expected[key])

    result = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(units),
        local_completion=producer,
    )
    assert result["status"] == "complete"
    assert result["candidate_count"] == len(units)
    assert len(seen) == len(units)
    assert result["progress_path"] != saved["progress_path"]
    assert result["export_path"] != saved["export_path"]
    assert json.loads(Path(saved["export_path"]).read_text())["books"][0]["lessons"] == []
    assert all(
        row["completion_origin"] == "INJECTED_TEST_PRODUCER"
        for row in json.loads(Path(result["progress_path"]).read_text())["progress"]
    )


@pytest.mark.parametrize(
    "kind",
    ["invalid-json", "invalid-quote", "oversized", "runtime", "nontext", "invalid-utf8"],
)
def test_local_failure_retains_bounded_evidence_then_retries(tmp_path: Path, kind: str) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    units = whole_book.plan_for(acquired.book).units
    responses = responses_for(acquired.book)

    def failing(request: dict[str, object]) -> str:
        if kind == "runtime":
            raise RuntimeError("synthetic secret must not enter logs")
        if kind == "nontext":
            return None  # type: ignore[return-value]
        if kind == "oversized":
            return "a" * (whole_book.MAX_RECORDED_REPLY_BYTES + 1)
        if kind == "invalid-utf8":
            return "\ud800"
        if kind == "invalid-json":
            return "{"
        invalid = dict(responses[content_fingerprint(request)])
        lesson = dict(invalid["lessons"][0])
        lesson["evidence"] = "invented not in source"
        invalid["lessons"] = [lesson]
        return json.dumps(invalid)

    failed = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=1,
        local_completion=failing,
    )
    assert failed["status"] == "incomplete"
    assert failed["counts"]["failed"] == 1
    assert failed["counts"]["unresolved"] == len(units) - 1
    assert failed["export_path"] is None
    failure_bytes = Path(failed["progress_path"]).read_bytes()
    progress = json.loads(failure_bytes)
    assert progress["progress"][0]["failure_category"]
    failure_files = list(
        (tmp_path / "artifacts/book-candidates/analysis/failures").glob("**/*.json")
    )
    assert len(failure_files) == 1
    failure_event = json.loads(failure_files[0].read_bytes())
    assert failure_event["category"] == progress["progress"][0]["failure_category"]
    assert "synthetic secret" not in failure_files[0].read_text()
    assert failure_files[0].stat().st_mode & 0o777 == 0o600
    raw_files = list((tmp_path / "artifacts/book-candidates/analysis/completions").glob("**/*.txt"))
    assert len(raw_files) == (0 if kind in ("runtime", "nontext", "invalid-utf8") else 1)
    assert all(p.stat().st_size <= whole_book.MAX_RECORDED_REPLY_BYTES for p in raw_files)
    if kind == "oversized":
        assert raw_files[0].stat().st_size == whole_book.MAX_RECORDED_REPLY_BYTES

    def corrected(request: dict[str, object]) -> str:
        return json.dumps(responses[content_fingerprint(request)])

    done = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(units),
        local_completion=corrected,
    )
    assert done["status"] == "complete"
    assert done["counts"]["failed"] == 0
    assert done["candidate_count"] == len(units)
    assert Path(failed["progress_path"]).read_bytes() == failure_bytes
    generation_root = tmp_path / "artifacts/book-dag/books-mining-local-completion-v1"
    first_stage = generation_root / progress["progress"][0]["generation_stage_key"]
    assert json.loads((first_stage / "stage.json").read_text())["attempt_count"] == 2
    assert failure_files[0].is_file()


def test_local_producer_identity_change_invalidates_only_generated_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    responses = responses_for(acquired.book)
    calls: list[str] = []
    units = whole_book.plan_for(acquired.book).units

    def fake(request: dict[str, object]) -> str:
        key = content_fingerprint(request)
        calls.append(key)
        return json.dumps(responses[key])

    first = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(units),
        local_completion=fake,
    )
    assert first["status"] == "complete"
    before = Path(first["progress_path"]).read_bytes()
    monkeypatch.setattr(whole_book, "LOCAL_PRODUCER_CONTRACT", "synthetic-new-producer-v2")
    second = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(units),
        local_completion=fake,
    )
    assert second["status"] == "complete"
    assert len(calls) == 2 * len(units)
    assert first["progress_path"] != second["progress_path"]
    assert first["export_path"] != second["export_path"]
    assert Path(first["progress_path"]).read_bytes() == before

    altered = {**scripting.QWEN_GENERATION_SETTINGS, "temperature": 0.6}
    monkeypatch.setattr(scripting, "QWEN_GENERATION_SETTINGS", altered)
    changed_settings = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=len(units),
        local_completion=fake,
    )
    assert changed_settings["status"] == "complete"
    assert len(calls) == 3 * len(units)
    assert changed_settings["progress_path"] != second["progress_path"]
    assert Path(first["progress_path"]).read_bytes() == before


def test_cached_local_raw_tamper_fails_closed_without_regeneration(
    tmp_path: Path,
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    responses = responses_for(acquired.book)

    def fake(request: dict[str, object]) -> str:
        return json.dumps(responses[content_fingerprint(request)])

    runner.mine_candidate(
        acquired.analysis_ref_path, local_cpu=True, max_new_units=3, local_completion=fake
    )
    evidence = next((tmp_path / "artifacts/book-candidates/analysis/completions").glob("**/*.txt"))
    evidence.write_text("tampered fixture")
    with pytest.raises(BookRunError, match="raw evidence changed"):
        runner.mine_candidate(
            acquired.analysis_ref_path,
            local_cpu=True,
            max_new_units=3,
            local_completion=lambda _: pytest.fail("cache must not invoke model"),
        )
    assert evidence.read_text() == "tampered fixture"


def test_cli_recorded_and_opt_in_cpu_modes_are_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    source = tmp_path / "responses.json"
    replies = responses_for(acquired.book, count=0)
    _write_responses(source, replies)

    def forbidden(*_a: object, **_k: object) -> None:
        pytest.fail("recorded CLI may not load model weights or call network")

    monkeypatch.setattr("clipper.books.scripting._script_pipeline", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    common = [
        "books",
        "mine",
        "--candidate-ref",
        str(acquired.analysis_ref_path),
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    assert cli.main([*common, "--responses", str(source)]) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["candidate_count"] == 0

    def fake(request: dict[str, object]) -> str:
        assert request["model"] == "Qwen/Qwen3-1.7B"
        assert request["revision"] == "8d4744f9e13072f4920c326350fa81eedb74eae9"
        return json.dumps({"lessons": []})

    monkeypatch.setattr("clipper.books.scripting._local_book_completion", fake)
    assert cli.main([*common, "--local-cpu", "--max-new-units", "1"]) == 2
    local = json.loads(capsys.readouterr().out)
    assert local["status"] == "incomplete"
    assert local["counts"]["zero_candidate"] == 1
    assert local["counts"]["unresolved"] > 0
    assert cli.main([*common, "--responses", str(source), "--max-new-units", "1"]) == 1
    assert cli.main(["books", "mine", "--book-id", "legacy", "--local-cpu"]) == 1
    with pytest.raises(SystemExit):
        cli._parser().parse_args([*common, "--responses", str(source), "--local-cpu"])
    with pytest.raises(SystemExit):
        cli._parser().parse_args(common)


def test_local_mode_rejects_mixed_and_invalid_budgets(tmp_path: Path) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    ref_path = acquired.analysis_ref_path
    with pytest.raises(BookRunError, match="exclusive"):
        runner.mine_candidate(
            ref_path,
            local_cpu=True,
            responses={},
            response_file_sha256="a" * 64,
        )
    for invalid in (0, 20_001, True):
        with pytest.raises(BookRunError, match="max-new-units"):
            runner.mine_candidate(ref_path, local_cpu=True, max_new_units=invalid)
    with pytest.raises(BookRunError, match="local generation options"):
        runner.mine_candidate(
            ref_path, responses={}, response_file_sha256="a" * 64, max_new_units=1
        )


def test_local_cached_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    replies = responses_for(acquired.book)

    def fake(request: dict[str, object]) -> str:
        return json.dumps(replies[content_fingerprint(request)])

    count = len(whole_book.plan_for(acquired.book).units)
    completed = runner.mine_candidate(
        acquired.analysis_ref_path,
        local_cpu=True,
        max_new_units=count,
        local_completion=fake,
    )
    assert completed["status"] == "complete"
    base = tmp_path / "artifacts/book-candidates/analysis"
    originals = base / "completions"
    renamed = base / "completions-preserved"
    originals.rename(renamed)
    originals.symlink_to(renamed, target_is_directory=True)
    with pytest.raises(BookRunError, match="symlink"):
        runner.mine_candidate(
            acquired.analysis_ref_path,
            local_cpu=True,
            max_new_units=1,
            local_completion=lambda _: pytest.fail("generation cache must be used"),
        )
    assert list(renamed.glob("**/*.txt"))


def test_pinned_local_qwen_cpu_loader_uses_one_fake_model_without_weight_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    _runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    unit = whole_book.plan_for(acquired.book).units[0]
    request = whole_book.request_for(acquired.book, unit)
    snapshot = (
        tmp_path
        / "cache"
        / "hub"
        / "models--Qwen--Qwen3-1.7B"
        / "snapshots"
        / scripting.SCRIPT_REVISION
    )
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_bytes(b"synthetic model-cache placeholder")
    loaded: list[str] = []
    hub_calls: list[bool] = []
    seeds: list[int] = []
    generated: list[dict[str, object]] = []
    decoded = ' {"lessons": []} '

    class Inputs:
        shape = (1, 1)

    class Model:
        device = "cpu"

        def eval(self):
            return self

        def generate(self, **kwargs):
            generated.append(kwargs)
            return [[0, 7, 0]]

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["enable_thinking"] is False
            if kwargs.get("return_dict"):
                return {"input_ids": Inputs()}
            return [0]

        def decode(self, tokens, *, skip_special_tokens: bool):
            assert tokens == [7, 0] and skip_special_tokens
            return decoded

    model = Model()
    tokenizer = Tokenizer()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(location, *, local_files_only):
            # This rejects the old repository-ID call that causes the
            # Transformers 4.57.3 tokenizer metadata lookup while offline.
            assert Path(location) == snapshot
            assert local_files_only is True
            loaded.append("tokenizer")
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(location, *, local_files_only, dtype):
            assert Path(location) == snapshot
            assert local_files_only is True
            assert dtype == "auto"
            loaded.append("model")
            return model

    def snapshot_download(*, repo_id, revision, local_files_only):
        assert repo_id == scripting.SCRIPT_MODEL
        assert revision == scripting.SCRIPT_REVISION
        hub_calls.append(local_files_only)
        if not local_files_only:
            pytest.fail("offline pinned model cache must never query the Hub")
        return str(snapshot)

    fake_torch = SimpleNamespace(
        set_num_threads=lambda count: None,
        manual_seed=lambda seed: seeds.append(seed),
        inference_mode=lambda: nullcontext(),
    )
    fake_transformers = SimpleNamespace(AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModel)
    fake_hub = SimpleNamespace(snapshot_download=snapshot_download)
    real_import = scripting.importlib.import_module

    def injected(name: str):
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        if name == "huggingface_hub":
            return fake_hub
        return real_import(name)

    scripting._script_pipeline.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr("clipper.books.constrained._decoder", lambda: (None, None))
            patch.setattr(
                "clipper.books.constrained.prefix_for",
                lambda *_args: lambda _batch_id, _tokens: [0],
            )
            patch.setenv("HF_HUB_OFFLINE", "1")
            patch.setattr(scripting.importlib, "import_module", injected)
            patch.setattr(scripting, "_public_book_context", lambda: (tokenizer, 40_960))
            assert scripting._local_book_completion(request) == decoded
            assert scripting._local_book_completion(request) == decoded
            assert hub_calls == [True]
            assert loaded == ["tokenizer", "model"]
            assert seeds == [42, 42]
            assert len(generated) == 2
            assert all(call["temperature"] == 0.7 for call in generated)
            assert all(call["max_new_tokens"] == 960 for call in generated)
            assert all(call["prefix_allowed_tokens_fn"](0, []) == [0] for call in generated)
    finally:
        scripting._script_pipeline.cache_clear()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "offline cache"),
        ("missing-directory", "snapshot is missing"),
        ("incomplete", "incomplete"),
        ("tokenizer-empty", "incomplete"),
        ("wrong-revision", "identity mismatch"),
        ("shard-missing", "incomplete"),
        ("bad-index", "incomplete"),
        ("unsafe-shard", "incomplete"),
    ],
)
def test_pinned_qwen_offline_snapshot_fails_closed_without_metadata_or_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, message: str
) -> None:
    from types import SimpleNamespace

    model_cache = tmp_path / "cache" / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots"
    revision = "f" * 40 if kind == "wrong-revision" else scripting.SCRIPT_REVISION
    snapshot = model_cache / revision
    if kind not in ("missing", "missing-directory"):
        snapshot.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (snapshot / name).write_bytes(b"synthetic")
        if kind == "tokenizer-empty":
            (snapshot / "tokenizer.json").write_bytes(b"")
        if kind == "wrong-revision":
            (snapshot / "model.safetensors").write_bytes(b"synthetic")
        if kind in ("shard-missing", "bad-index", "unsafe-shard"):
            index = snapshot / "model.safetensors.index.json"
            if kind == "bad-index":
                index.write_text("{")
            else:
                name = (
                    "../wrong.safetensors"
                    if kind == "unsafe-shard"
                    else "model-00001-of-00002.safetensors"
                )
                index.write_text(json.dumps({"weight_map": {"layer": name}}))

    def snapshot_download(*, repo_id, revision, local_files_only):
        assert (repo_id, revision) == (scripting.SCRIPT_MODEL, scripting.SCRIPT_REVISION)
        assert local_files_only is True
        if kind == "missing":
            raise FileNotFoundError("synthetic missing pinned snapshot")
        return str(snapshot)

    hub = SimpleNamespace(snapshot_download=snapshot_download)
    original_import = scripting.importlib.import_module

    def injected(name: str):
        return hub if name == "huggingface_hub" else original_import(name)

    with monkeypatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setattr(scripting.importlib, "import_module", injected)
        with pytest.raises(scripting.PinnedQwenSnapshotError, match=message):
            scripting._pinned_qwen_snapshot()


def test_pinned_qwen_pre_cache_and_explicit_public_download_use_exact_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    snapshot = (
        tmp_path
        / "cache"
        / "hub"
        / "models--Qwen--Qwen3-1.7B"
        / "snapshots"
        / scripting.SCRIPT_REVISION
    )
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (snapshot / name).write_bytes(b"synthetic")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00002.safetensors"}})
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"synthetic shard")
    calls: list[bool] = []

    def snapshot_download(*, repo_id, revision, local_files_only):
        assert (repo_id, revision) == (scripting.SCRIPT_MODEL, scripting.SCRIPT_REVISION)
        calls.append(local_files_only)
        if local_files_only:
            raise FileNotFoundError("synthetic absent cache entry")
        return str(snapshot)

    hub = SimpleNamespace(snapshot_download=snapshot_download)
    original_import = scripting.importlib.import_module

    def injected(name: str):
        return hub if name == "huggingface_hub" else original_import(name)

    with monkeypatch.context() as patch:
        patch.delenv("HF_HUB_OFFLINE", raising=False)
        patch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        patch.setattr(scripting.importlib, "import_module", injected)
        assert scripting._pinned_qwen_snapshot() == snapshot
        assert calls == [True, False]


def test_pinned_qwen_download_failure_is_sanitized_and_no_revision_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls: list[bool] = []

    def snapshot_download(*, repo_id, revision, local_files_only):
        assert repo_id == scripting.SCRIPT_MODEL
        assert revision == scripting.SCRIPT_REVISION
        calls.append(local_files_only)
        if local_files_only:
            raise FileNotFoundError("synthetic offline snapshot not cached")
        raise OSError("synthetic network unavailable")

    hub = SimpleNamespace(snapshot_download=snapshot_download)
    original_import = scripting.importlib.import_module

    def injected(name: str):
        return hub if name == "huggingface_hub" else original_import(name)

    with monkeypatch.context() as patch:
        patch.delenv("HF_HUB_OFFLINE", raising=False)
        patch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        patch.setattr(scripting.importlib, "import_module", injected)
        with pytest.raises(scripting.PinnedQwenSnapshotError, match="download failed"):
            scripting._pinned_qwen_snapshot()
    assert calls == [True, False]


def test_pinned_qwen_missing_optional_hub_dependency_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = scripting.importlib.import_module

    def injected(name: str):
        if name == "huggingface_hub":
            raise ImportError("synthetic no hub installed")
        return original_import(name)

    with monkeypatch.context() as patch:
        patch.setattr(scripting.importlib, "import_module", injected)
        with pytest.raises(scripting.NicheVideoError, match="niche-models"):
            scripting._pinned_qwen_snapshot()


def test_local_cached_evidence_reprotects_reused_private_hierarchy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=4, with_image=False)
    count = len(whole_book.plan_for(acquired.book).units)
    recorded = responses_for(acquired.book)

    def fake(request: dict[str, object]) -> str:
        return json.dumps(recorded[content_fingerprint(request)], sort_keys=True)

    initial = runner.mine_candidate(
        acquired.analysis_ref_path, local_cpu=True, max_new_units=count, local_completion=fake
    )
    assert initial["status"] == "complete"
    artifact_root = tmp_path / "artifacts"
    private_root = artifact_root / "book-candidates"
    raw_files = list((private_root / "analysis/completions").glob("**/*.txt"))
    files = [*raw_files, Path(initial["progress_path"]), Path(initial["export_path"])]
    assert raw_files
    originals = {file: file.read_bytes() for file in files}
    shared = tmp_path / "unrelated-shared"
    shared.mkdir()
    tmp_path.chmod(0o755)
    artifact_root.chmod(0o755)
    shared.chmod(0o755)
    dirs: set[Path] = set()
    for file in files:
        file.chmod(0o666)
        current = file.parent
        while current != private_root.parent:
            dirs.add(current)
            current = current.parent
    for directory in dirs:
        directory.chmod(0o777)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("completed cache replay must not load a model or network")

    monkeypatch.setattr("clipper.books.scripting._script_pipeline", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    previous = os.umask(0)
    try:
        repeated = runner.mine_candidate(
            acquired.analysis_ref_path,
            local_cpu=True,
            max_new_units=1,
            local_completion=forbidden,
        )
    finally:
        os.umask(previous)
    assert repeated == initial
    for directory in dirs:
        assert directory.stat().st_mode & 0o777 == 0o700, directory
    for file, original in originals.items():
        assert file.read_bytes() == original
        assert file.stat().st_mode & 0o777 == 0o600, file
    assert tmp_path.stat().st_mode & 0o777 == 0o755
    assert artifact_root.stat().st_mode & 0o777 == 0o755
    assert shared.stat().st_mode & 0o777 == 0o755


def test_local_cached_raw_evidence_missing_raises_book_domain_error(tmp_path: Path) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    count = len(whole_book.plan_for(acquired.book).units)

    def fake(request: dict[str, object]) -> str:
        return json.dumps(responses_for(acquired.book)[content_fingerprint(request)])

    result = runner.mine_candidate(
        acquired.analysis_ref_path, local_cpu=True, max_new_units=count, local_completion=fake
    )
    assert result["status"] == "complete"
    raw = next((tmp_path / "artifacts/book-candidates/analysis/completions").glob("**/*.txt"))
    record = next(
        (tmp_path / "artifacts/book-dag/books-mining-local-completion-v1").glob("*/stage.json")
    )
    report_bytes = Path(result["progress_path"]).read_bytes()
    raw.unlink()

    def forbidden(_request: dict[str, object]) -> str:
        pytest.fail("missing PASS raw bytes must never trigger regeneration")

    with pytest.raises(BookRunError, match="cached local model raw evidence missing"):
        runner.mine_candidate(
            acquired.analysis_ref_path,
            local_cpu=True,
            max_new_units=1,
            local_completion=forbidden,
        )
    assert not raw.exists()
    assert json.loads(record.read_text())["status"] == "PASS"
    assert Path(result["progress_path"]).read_bytes() == report_bytes


def test_reused_failure_provenance_permissions_and_bytes_are_preserved(
    tmp_path: Path,
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)

    def invalid(_request: dict[str, object]) -> str:
        return "{"

    first = runner.mine_candidate(
        acquired.analysis_ref_path, local_cpu=True, max_new_units=1, local_completion=invalid
    )
    assert first["counts"]["failed"] == 1
    private_root = tmp_path / "artifacts/book-candidates"
    events = list((private_root / "analysis/failures").glob("**/*.json"))
    assert len(events) == 1
    old = events[0]
    original = old.read_bytes()
    old.chmod(0o666)
    directories: set[Path] = set()
    current = old.parent
    while current != private_root.parent:
        directories.add(current)
        current = current.parent
    for directory in directories:
        directory.chmod(0o777)

    second = runner.mine_candidate(
        acquired.analysis_ref_path, local_cpu=True, max_new_units=1, local_completion=invalid
    )
    assert second["counts"]["failed"] == 1
    assert first["progress_path"] == second["progress_path"]
    assert old.read_bytes() == original
    assert old.stat().st_mode & 0o777 == 0o600
    for directory in directories:
        assert directory.stat().st_mode & 0o777 == 0o700
    assert len(list((private_root / "analysis/failures").glob("**/*.json"))) == 2


def test_local_offline_missing_cache_is_failed_not_zero_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, acquired, _ = ingested(tmp_path, paragraphs=3, with_image=False)
    attempts: list[int] = []

    def unavailable(_request: dict[str, object]) -> str:
        attempts.append(1)
        raise scripting.PinnedQwenSnapshotError("pinned Qwen snapshot unavailable in offline cache")

    monkeypatch.setattr(scripting, "_local_book_completion", unavailable)
    outcome = runner.mine_candidate(acquired.analysis_ref_path, local_cpu=True, max_new_units=1)
    assert len(attempts) == 1
    assert outcome["status"] == "incomplete"
    assert outcome["counts"]["failed"] == 1
    assert outcome["counts"]["zero_candidate"] == 0
    assert outcome["export_path"] is None
    progress = json.loads(Path(outcome["progress_path"]).read_text())
    assert progress["progress"][0]["failure_category"] == "pinned_model_cache_unavailable"


def test_injected_source_mining_has_distinct_identity_from_cpu_material() -> None:
    from clipper.stage_contracts import content_fingerprint

    real = {
        **whole_book.local_producer_material(),
        "execution_provenance": "ACTUAL_CPU_QWEN",
    }
    injected = {
        **whole_book.local_producer_material(),
        "execution_provenance": "INJECTED_TEST_PRODUCER",
    }
    assert content_fingerprint(real) != content_fingerprint(injected)
    assert real["generation"] == injected["generation"]

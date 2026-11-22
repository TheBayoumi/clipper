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
from clipper.books import acquisition, mining, whole_book
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
    return {
        "lessons": [
            {
                "id": f"idea-{index}",
                "topic": "Fabricated idea",
                "claim": "Unreviewed synthetic claim",
                "hook_angle": "An invented demonstration",
                "visual_action": "A cartoon actor places a token",
                "evidence_field": unit.evidence_field,
                "evidence": unit.text[:35],
            }
            for index in range(count)
        ]
    }


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

"""Offline exact candidate-ref handoff and whole-book mining, with fabricated text only."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
import pytest

from clipper import cli
from clipper.books import acquisition, faithful, mining, whole_book
from clipper.books.contracts import BookText, ImageEvidence, ImageOccurrence, SourceBlock
from clipper.books.orchestrator import BookOrchestrator, BookRunError
from clipper.dag import DagStore
from clipper.stage_contracts import content_fingerprint

BASE = Path(__file__).resolve().parents[1] / "niches/book-source-candidates-v1.json"


def synthetic_candidate(
    tmp_path: Path, *, paragraphs: int = 129, with_image: bool = True
) -> tuple[acquisition.ReviewedCandidate, object]:
    """All prose and image bytes are original synthetic fixtures, never book extracts."""
    candidate = next(
        item
        for item in acquisition.load_registry(BASE)
        if item.book_id == "atomic-habits"
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


def ingested(
    tmp_path: Path, *, paragraphs: int = 129, with_image: bool = True
):
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
    assert runner.mine_candidate(
        acquired.analysis_ref_path, responses=replies,
        response_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    ) == summary
    assert export_path.read_bytes() == original_export
    assert progress_path.read_bytes() == original_progress


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
    stage_records = (
        tmp_path / "artifacts/book-dag/books-mining-unit-v1"
    ).glob("*/stage.json")
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
        "book-id", "stage-key", "source", "parser", "raw", "dossier",
        "extracted", "unsafe-path", "cross-book", "extra-field",
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
    monkeypatch.setattr(
        whole_book, "plan_for", lambda _book: pytest.fail("producer was invoked")
    )
    with pytest.raises(BookRunError, match="candidate analysis reference"):
        runner.mine_candidate(
            ref_path, responses={}, response_file_sha256="a" * 64,
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


def test_evidence_field_and_origin_never_conflate_research_ocr_and_source(
    tmp_path: Path
) -> None:
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
            digest, "image/png", 5,
            (
                ImageOccurrence(
                    "n-1", "fig-n.png", "",
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
            "3" * 64, "image/png", 5,
            (ImageOccurrence("m-1", "fig-m.png", ""),),
            machine_transcription="OCR synthetic ambiguous words",
            review_status="machine_unreviewed",
        ),
        ImageEvidence(
            "4" * 64, "image/png", 5,
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
    assert whole_book.candidate_for(synthetic, machine, response_for(machine)["lessons"][0])[
        "evidence_anchor"
    ]["field"] == "machine_ocr_text"

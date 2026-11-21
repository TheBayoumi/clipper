"""Title-only Archive acquisition -> faithful ingest through private versioned evidence."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper import cli
from clipper.books import acquisition
from clipper.books.contracts import OCRIdentity, SourceUnresolved
from clipper.books.orchestrator import BookOrchestrator, BookRunError


def _fixture(
    tmp_path: Path,
) -> tuple[
    acquisition.ReviewedCandidate,
    acquisition.BookTransport,
    Path,
]:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfile full-path="OPS/content.opf"/></container>',
        )
        archive.writestr(
            "OPS/content.opf",
            '<package xmlns:dc="http://purl.org/dc/elements/1.1/" unique-identifier="uid">'
            "<metadata><dc:title>Atomic Habits</dc:title>"
            "<dc:creator>James Clear</dc:creator>"
            '<dc:identifier id="uid">urn:isbn:9780735211308</dc:identifier>'
            "</metadata><manifest>"
            '<item id="a" href="chapter1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="b" href="chapter2.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="c" href="chapter3.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="pic" href="figure.png" media-type="image/png"/>'
            '</manifest><spine><itemref idref="a"/>'
            '<itemref idref="b"/><itemref idref="c"/></spine></package>',
        )
        archive.writestr(
            "OPS/chapter1.xhtml",
            "<html><body><h1>Front matter</h1><p>Short cover.</p></body></html>",
        )
        archive.writestr(
            "OPS/chapter2.xhtml",
            "<html><body><h1>Chapter</h1><p>Preserve <span>pro</span>"
            "<span>scribed</span> spelling.</p>"
            '<img src="figure.png" alt="40 10 60"/>'
            "</body></html>",
        )
        archive.writestr(
            "OPS/chapter3.xhtml",
            "<html><body><h2>Index</h2><p>Index and notes retained.</p></body></html>",
        )
        archive.writestr("OPS/figure.png", b"\x89PNG\r\n\x1a\nSOURCE")
    raw = stream.getvalue()
    baseline = next(
        x
        for x in acquisition.load_registry(
            Path(__file__).resolve().parents[1] / "niches/book-source-candidates-v1.json"
        )
        if x.book_id == "atomic-habits"
    )
    candidate = replace(
        baseline,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_bytes=len(raw),
    )
    registry = tmp_path / "reviewed.json"
    registry.write_text(
        json.dumps(
            {
                "schema": acquisition.REGISTRY_SCHEMA,
                "books": [asdict(candidate)],
            }
        )
    )

    class FakeTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.fail = False

        def get(self, url: str, *, limit: int) -> acquisition.HTTPResult:
            self.calls.append(url)
            if self.fail:
                raise AssertionError("offline cache unexpectedly used transport")
            if "/metadata/" in url:
                data = json.dumps(
                    {
                        "metadata": {"identifier": url.rsplit("/", 1)[-1]},
                        "files": [
                            {
                                "name": candidate.filename,
                                "size": str(candidate.expected_bytes),
                                "source": "original",
                            }
                        ],
                    }
                ).encode()
                return acquisition.HTTPResult(url, url, 200, "application/json", data)
            return acquisition.HTTPResult(
                url,
                "https://ia801234.us.archive.org/synthetic/fixture",
                200,
                "application/epub+zip",
                raw,
            )

    return candidate, FakeTransport(), registry


def test_title_only_ingest_preserves_private_raw_and_incomplete_image_evidence(
    tmp_path: Path,
) -> None:
    candidate, transport, _registry_path = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path / "artifacts")
    result = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    assert not isinstance(result, SourceUnresolved)
    assert result.acquired.source.authorization_status == "unverified"
    assert result.acquired.raw_path.is_file()
    assert result.book.recovery_status == "incomplete_recovery"
    assert len(result.book.images) == 1
    assert len(result.book.images[0].occurrences) == 1
    assert result.book.images[0].occurrences[0].alt_text == "40 10 60"
    assert result.book.total_word_count > 0
    assert result.extracted_path.is_file()
    assert result.dossier_path.is_file()
    assert result.extracted_path.stat().st_mode & 0o077 == 0
    dossier = json.loads(result.dossier_path.read_text())
    assert dossier["publish_eligible"] is False
    assert dossier["editorial_approved"] is False
    assert dossier["authorization_status"] == "unverified"
    assert dossier["unique_images"] == 1
    assert dossier["image_occurrences"] == 1
    assert dossier["recovery_status"] == "incomplete_recovery"
    assert dossier["raw_sha256"] == candidate.expected_sha256
    assert not (tmp_path / "artifacts/mined").exists()


def test_offline_content_addressed_resume_and_overlay_change_rekey(
    tmp_path: Path,
) -> None:
    candidate, transport, _registry = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path / "private")
    first = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    assert not isinstance(first, SourceUnresolved)
    second = runner.ingest_candidate(
        "Atomic Habits",
        registry=(candidate,),
        transport=Mock(side_effect=AssertionError("no transport")),
        offline=True,
    )
    assert not isinstance(second, SourceUnresolved)
    assert second.stage_key == first.stage_key
    assert second.book == first.book
    assert second.acquired.cache_hit is True
    assert len(list((tmp_path / "private/book-dag/books-ingest").glob("*/stage.json"))) == 1
    asset = first.book.images[0]
    overlay = {
        "schema": "clipper-book-image-overlay-v1",
        "source_sha256": candidate.expected_sha256,
        "entries": [
            {
                "asset_sha256": asset.asset_sha256,
                "anchors": [occ.anchor for occ in asset.occurrences],
                "original_transcription": "40 10 60",
                "normalized_transcription": "40 10 60",
                "normalization_notes": "",
                "reviewed_by": "offline-test-reviewer",
                "reviewed_at": "2026-11-21",
                "review_status": "reviewed",
            }
        ],
    }
    sidecar = tmp_path / "private-overlay.json"
    sidecar.write_text(json.dumps(overlay))
    reviewed = runner.ingest_candidate(
        "Atomic Habits",
        registry=(candidate,),
        offline=True,
        overlay_path=sidecar,
    )
    assert not isinstance(reviewed, SourceUnresolved)
    assert reviewed.stage_key != first.stage_key
    assert reviewed.book.images[0].review_status == "reviewed"
    assert reviewed.book.images[0].original_transcription == "40 10 60"
    assert len(list((tmp_path / "private/book-dag/books-ingest").glob("*/stage.json"))) == 2
    overlay["entries"][0]["normalization_notes"] = "kept exact unusual chart value"
    sidecar.write_text(json.dumps(overlay))
    updated = runner.ingest_candidate(
        "Atomic Habits", registry=(candidate,), offline=True, overlay_path=sidecar
    )
    assert not isinstance(updated, SourceUnresolved)
    assert updated.stage_key != reviewed.stage_key


def test_title_only_cli_registry_resolution_and_offline_ingest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, transport, registry = _fixture(tmp_path)
    result = cli.main(
        [
            "books",
            "candidates",
            "--title",
            "Atomic Habits",
            "--registry",
            str(registry),
        ]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out)["book_id"] == "atomic-habits"
    assert (
        cli.main(
            [
                "books",
                "candidates",
                "--title",
                "Not the same title",
                "--registry",
                str(registry),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["required_owner_decision"]
    monkeypatch.setattr(
        acquisition.ArchiveHTTPTransport,
        "get",
        lambda _self, url, *, limit: transport.get(url, limit=limit),
    )
    root = tmp_path / "artifacts"
    acquired = cli.main(
        [
            "books",
            "acquire",
            "--title",
            "Atomic Habits",
            "--registry",
            str(registry),
            "--artifact-root",
            str(root),
        ]
    )
    assert acquired == 0
    report = json.loads(capsys.readouterr().out)
    assert report["raw_sha256"] == candidate.expected_sha256
    assert report["authorization_status"] == "unverified"
    assert (
        cli.main(
            [
                "books",
                "ingest",
                "--title",
                "Atomic Habits",
                "--registry",
                str(registry),
                "--artifact-root",
                str(root),
                "--offline",
            ]
        )
        == 2
    )
    partial = json.loads(capsys.readouterr().out)
    assert partial["recovery_status"] == "incomplete_recovery"
    assert partial["dossier_path"]
    assert partial["unique_images"] == 1


def test_unknown_title_does_not_fetch_or_guess_source(tmp_path: Path) -> None:
    candidate, transport, _registry = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path)
    result = runner.ingest_candidate("Unreviewed Book", registry=(candidate,), transport=transport)
    assert isinstance(result, SourceUnresolved)
    assert "no exact reviewed registry match" in result.reason
    assert not transport.calls
    assert not (tmp_path / "book-candidates").exists()


def test_missing_sidecar_fails_after_raw_acquisition_without_forging_review(
    tmp_path: Path,
) -> None:
    candidate, transport, _registry = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path)
    with pytest.raises(BookRunError, match="reviewed image overlay missing"):
        runner.ingest_candidate(
            "Atomic Habits",
            registry=(candidate,),
            transport=transport,
            overlay_path=tmp_path / "missing.json",
        )
    assert list((tmp_path / "book-candidates/raw").glob("**/*.epub"))
    assert not list((tmp_path / "book-candidates/dossiers").glob("**/*.json"))


def test_ocr_producer_change_rekeys_and_identical_contract_resumes(tmp_path: Path) -> None:
    candidate, transport, _registry = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path / "private")
    first_ocr = Mock(return_value="synthetic original chart label")
    first = runner.ingest_candidate(
        "Atomic Habits",
        registry=(candidate,),
        transport=transport,
        ocr=first_ocr,
        ocr_identity=OCRIdentity("fixture-ocr", "1", (("language", "eng"),)),
    )
    assert not isinstance(first, SourceUnresolved)
    first_ocr.assert_called_once()
    changed_ocr = Mock(return_value="different synthetic observation")
    changed = runner.ingest_candidate(
        "Atomic Habits",
        registry=(candidate,),
        offline=True,
        ocr=changed_ocr,
        ocr_identity=OCRIdentity("fixture-ocr", "1", (("language", "deu"),)),
    )
    assert not isinstance(changed, SourceUnresolved)
    changed_ocr.assert_called_once()
    assert changed.stage_key != first.stage_key
    assert changed.book.images[0].original_transcription != first.book.images[0].original_transcription
    resumed_ocr = Mock(side_effect=AssertionError("unchanged OCR must not execute"))
    resumed = runner.ingest_candidate(
        "Atomic Habits",
        registry=(candidate,),
        offline=True,
        ocr=resumed_ocr,
        ocr_identity=OCRIdentity("fixture-ocr", "1", (("language", "deu"),)),
    )
    assert not isinstance(resumed, SourceUnresolved)
    assert resumed.stage_key == changed.stage_key
    resumed_ocr.assert_not_called()
    with pytest.raises(Exception, match="requires deterministic identity"):
        runner.ingest_candidate("Atomic Habits", registry=(candidate,), ocr=first_ocr)

"""Title-only Archive acquisition -> faithful ingest through private versioned evidence."""

from __future__ import annotations

import hashlib
import io
import json
import os
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
    tmp_path: Path, *, book_id: str = "atomic-habits"
) -> tuple[
    acquisition.ReviewedCandidate,
    acquisition.BookTransport,
    Path,
]:
    baseline = next(
        x
        for x in acquisition.load_registry(
            Path(__file__).resolve().parents[1] / "niches/book-source-candidates-v1.json"
        )
        if x.book_id == book_id
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfile full-path="OPS/content.opf"/></container>',
        )
        archive.writestr(
            "OPS/content.opf",
            '<package xmlns:dc="http://purl.org/dc/elements/1.1/" unique-identifier="uid">'
            f"<metadata><dc:title>{baseline.title}</dc:title>"
            f"<dc:creator>{baseline.author}</dc:creator>"
            f'<dc:identifier id="uid">urn:isbn:{baseline.isbn}</dc:identifier>'
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
    dag_output = tmp_path / "artifacts" / "book-dag" / "books-ingest" / result.stage_key
    assert (dag_output / "output.json").stat().st_mode & 0o777 == 0o600
    assert dag_output.stat().st_mode & 0o777 == 0o700
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
    assert (
        changed.book.images[0].original_transcription != first.book.images[0].original_transcription
    )
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


def test_cli_offline_private_observation_import_keeps_unreviewed_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate, transport, registry = _fixture(tmp_path)
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    extracted = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    assert not isinstance(extracted, SourceUnresolved)
    image = extracted.book.images[0]
    research = {
        "schema": "clipper-book-research-observations-v1",
        "source_sha256": extracted.book.content_sha256,
        "entries": [
            {
                "asset_sha256": image.asset_sha256,
                "occurrences": [
                    {"anchor": occ.anchor, "source_path": occ.source_path}
                    for occ in image.occurrences
                ],
                "original_transcription": "40 10 60",
                "normalized_transcription": "40 10 60",
                "normalization_notes": None,
                "reviewed_by": "fixture-reviewer",
                "reviewed_at": None,
                "review_status": "reviewed",
            }
        ],
    }
    input_file = tmp_path / "private-research.json"
    input_file.write_text(json.dumps(research))
    output_file = tmp_path / "private-reviewed.json"
    assert (
        cli.main(
            [
                "books",
                "overlay-import",
                "--title",
                "Atomic Habits",
                "--registry",
                str(registry),
                "--artifact-root",
                str(root),
                "--research",
                str(input_file),
                "--output",
                str(output_file),
            ]
        )
        == 0
    )
    reply = json.loads(capsys.readouterr().out)
    assert reply["unreviewed_assets"] == 0
    assert output_file.is_file()
    assert json.loads(output_file.read_text())["entries"][0]["reviewed_at"] is None
    report = json.loads(Path(reply["report_path"]).read_text())
    assert report["total_occurrences"] == 1
    assert report["assets"][0]["occurrences"][0]["source_path"]


def test_second_normal_candidate_ingest_reuses_pinned_extracted_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, transport, _registry = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path / "private")
    first = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    assert not isinstance(first, SourceUnresolved)
    monkeypatch.setattr(transport, "fail", True)
    second = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    assert not isinstance(second, SourceUnresolved)
    assert second.acquired.cache_hit is True
    assert second.stage_key == first.stage_key
    assert second.book == first.book
    assert second.extracted_path == first.extracted_path
    assert second.dossier_path == first.dossier_path
    assert len(list((tmp_path / "private/book-candidates/raw").glob("**/*.epub"))) == 1
    assert len(list((tmp_path / "private/book-candidates/extracted").glob("**/*.json"))) == 1


def test_synthetic_v2_cli_import_offline_reuse_and_changed_provenance_rekey(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate, transport, registry = _fixture(tmp_path)
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    first = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    assert not isinstance(first, SourceUnresolved)
    image = first.book.images[0]
    occ = image.occurrences[0]
    observation = {
        "schema": "clipper-book-research-observations-v2",
        "source_sha256": candidate.expected_sha256,
        "research_origin": "fixture historical ledger",
        "research_author": None,
        "occurrence_inventory": [
            {
                "asset_sha256": image.asset_sha256,
                "anchor": occ.anchor,
                "source_path": occ.source_path,
            }
        ],
        "observations": [
            {
                "observation_id": "fixture-1",
                "asset_sha256": image.asset_sha256,
                "anchor": occ.anchor,
                "source_path": occ.source_path,
                "original_transcription": None,
                "normalized_transcription": "Grouped chart text",
                "normalization_notes": "Editorial grouping only",
                "reviewed_by": None,
                "reviewed_at": None,
                "review_status": None,
            }
        ],
    }
    research_path = tmp_path / "research-v2.json"
    overlay_path = tmp_path / "overlay-v2.json"
    research_path.write_text(json.dumps(observation))
    cmd = [
        "books",
        "overlay-import",
        "--title",
        "Atomic Habits",
        "--registry",
        str(registry),
        "--artifact-root",
        str(root),
        "--research",
        str(research_path),
        "--output",
        str(overlay_path),
    ]
    assert cli.main(cmd) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["normalized_only_occurrences"] == 1
    assert result["linked_occurrences"] == 1
    report = json.loads(Path(result["report_path"]).read_text())
    assert report["unreviewed_assets"] == 1
    assert report["reviewed_occurrences"] == 0
    source_path = (
        root
        / "book-candidates"
        / "research-import"
        / "source"
        / f"{hashlib.sha256(research_path.read_bytes()).hexdigest()}.json"
    )
    assert source_path.read_bytes() == research_path.read_bytes()
    assert cli.main(cmd) == 2
    capsys.readouterr()
    linked = runner.ingest_candidate(
        "Atomic Habits", registry=(candidate,), offline=True, overlay_path=overlay_path
    )
    assert not isinstance(linked, SourceUnresolved)
    assert linked.stage_key != first.stage_key
    assert linked.book.recovery_status == "incomplete_recovery"
    assert linked.book.images[0].original_transcription is None
    assert linked.book.images[0].occurrences[0].review_status is None
    assert linked.book.images[0].occurrences[0].normalized_transcription == "Grouped chart text"
    dossier = json.loads(linked.dossier_path.read_text())
    assert dossier["normalized_only_occurrences"] == 1
    assert dossier["verified_research_links"] == 1
    assert linked.book.images[0].occurrences[0].research_link_status == "verified"
    assert dossier["reviewed_images"] == 0
    assert dossier["publish_eligible"] is False
    assert dossier["authorization_status"] == "unverified"
    repeated = runner.ingest_candidate(
        "Atomic Habits", registry=(candidate,), offline=True, overlay_path=overlay_path
    )
    assert not isinstance(repeated, SourceUnresolved)
    assert repeated.stage_key == linked.stage_key
    assert repeated.book == linked.book
    source_path.unlink()
    unverified = runner.ingest_candidate(
        "Atomic Habits", registry=(candidate,), offline=True, overlay_path=overlay_path
    )
    assert not isinstance(unverified, SourceUnresolved)
    assert unverified.stage_key != linked.stage_key
    assert unverified.book.images[0].occurrences[0].research_link_status == "unverified_claim"
    assert json.loads(unverified.dossier_path.read_text())["unverified_research_links"] == 1
    assert cli.main(cmd) == 2
    capsys.readouterr()
    restored = runner.ingest_candidate(
        "Atomic Habits", registry=(candidate,), offline=True, overlay_path=overlay_path
    )
    assert not isinstance(restored, SourceUnresolved)
    assert restored.stage_key == linked.stage_key
    altered = json.loads(overlay_path.read_text())
    altered["entries"][0]["normalized_transcription"] = "unlinked edit"
    tampered = tmp_path / "tampered-overlay.json"
    tampered.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="differs from deterministic"):
        runner.ingest_candidate(
            "Atomic Habits", registry=(candidate,), offline=True, overlay_path=tampered
        )
    observation["observations"][0]["normalization_notes"] = "Revised grouping note"
    research_path.write_text(json.dumps(observation))
    changed_path = tmp_path / "changed-overlay.json"
    updated = [*cmd[:-1], str(changed_path)]
    assert cli.main(updated) == 2
    capsys.readouterr()
    changed = runner.ingest_candidate(
        "Atomic Habits", registry=(candidate,), offline=True, overlay_path=changed_path
    )
    assert not isinstance(changed, SourceUnresolved)
    assert changed.stage_key != linked.stage_key
    assert changed.book.images[0].occurrences[0].normalization_notes == "Revised grouping note"
    assert len(list((root / "book-candidates" / "extracted").glob("**/*.json"))) == 4


def test_cli_bad_v2_research_preserves_exact_input_and_rejection_reason(
    tmp_path: Path,
) -> None:
    candidate, transport, registry = _fixture(tmp_path)
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    first = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    assert not isinstance(first, SourceUnresolved)
    image = first.book.images[0]
    occ = image.occurrences[0]
    payload = {
        "schema": "clipper-book-research-observations-v2",
        "source_sha256": first.book.content_sha256,
        "research_origin": "synthetic legacy observation",
        "research_author": None,
        "occurrence_inventory": [
            {
                "asset_sha256": image.asset_sha256,
                "anchor": occ.anchor,
                "source_path": occ.source_path,
            }
        ],
        "observations": [
            {
                "observation_id": "not-reviewed",
                "asset_sha256": image.asset_sha256,
                "anchor": occ.anchor,
                "source_path": occ.source_path,
                "original_transcription": None,
                "normalized_transcription": "synthetic line grouping",
                "normalization_notes": "groups words visually",
                "reviewed_by": None,
                "reviewed_at": None,
                "review_status": "reviewed",
            }
        ],
    }
    input_path = tmp_path / "bad-research.json"
    data = json.dumps(payload).encode()
    input_path.write_bytes(data)
    output_path = tmp_path / "must-not-exist.json"
    assert (
        cli.main(
            [
                "books",
                "overlay-import",
                "--title",
                "Atomic Habits",
                "--registry",
                str(registry),
                "--artifact-root",
                str(root),
                "--research",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )
        == 1
    )
    assert not output_path.exists()
    digest = hashlib.sha256(data).hexdigest()
    base = root / "book-candidates" / "research-import"
    assert (base / "source" / f"{digest}.json").read_bytes() == data
    rejection = base / "rejections" / candidate.fingerprint / f"{digest}.json"
    rejected = json.loads(rejection.read_text())
    assert rejected["status"] == "rejected"
    assert "review provenance invalid" in rejected["reason"]
    assert rejected["source_sha256"] == first.book.content_sha256


@pytest.mark.parametrize("offline", [False, True])
def test_empty_overlay_cannot_alias_omitted_overlay_cache(tmp_path: Path, offline: bool) -> None:
    candidate, transport, _ = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path / "private")
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    runner.acquire_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    for warmed in (False, True):
        if warmed:
            normal = runner.ingest_candidate("Atomic Habits", registry=(candidate,), offline=True)
            assert not isinstance(normal, SourceUnresolved)
        with pytest.raises(ValueError, match="explicit image overlay is empty"):
            runner.ingest_candidate(
                "Atomic Habits",
                registry=(candidate,),
                transport=transport,
                offline=offline,
                overlay_path=empty,
            )


def test_malformed_unicode_rejections_are_immutable_and_source_bound(tmp_path: Path) -> None:
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    input_path = tmp_path / "malformed.json"
    data = b'{"private":"secret\\ud800"}'
    input_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    reports: list[Path] = []
    previous = os.umask(0)
    try:
        for book_id in ("atomic-habits", "psychology-of-money"):
            fixture_root = tmp_path / book_id
            fixture_root.mkdir()
            candidate, transport, registry = _fixture(fixture_root, book_id=book_id)
            acquired = runner.ingest_candidate(
                candidate.title, registry=(candidate,), transport=transport
            )
            assert not isinstance(acquired, SourceUnresolved)
            output = fixture_root / "overlay.json"
            cmd = [
                "books",
                "overlay-import",
                "--title",
                candidate.title,
                "--registry",
                str(registry),
                "--artifact-root",
                str(root),
                "--research",
                str(input_path),
                "--output",
                str(output),
            ]
            assert cli.main(cmd) == 1
            base = root / "book-candidates" / "research-import"
            rejection = base / "rejections" / candidate.fingerprint / f"{digest}.json"
            saved = rejection.read_bytes()
            assert cli.main(cmd) == 1
            assert rejection.read_bytes() == saved
            assert b"secret" not in saved
            assert json.loads(saved)["source_sha256"] == candidate.expected_sha256
            assert json.loads(saved)["candidate_fingerprint"] == candidate.fingerprint
            assert rejection.stat().st_mode & 0o777 == 0o600
            assert rejection.parent.stat().st_mode & 0o777 == 0o700
            assert not output.exists()
            assert (base / "source" / f"{digest}.json").read_bytes() == data
            assert (base / "source" / f"{digest}.json").stat().st_mode & 0o777 == 0o600
            reports.append(rejection)
        assert len(set(reports)) == 2
    finally:
        os.umask(previous)

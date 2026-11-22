"""Title-only Archive acquisition -> faithful ingest through private versioned evidence."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from io import FileIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper import cli
from clipper.books import acquisition, faithful
from clipper.books.contracts import OCRIdentity, SourceUnresolved
from clipper.books.orchestrator import BookOrchestrator, BookRunError
from clipper.stage_contracts import content_fingerprint


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
    stage = runner.root / "book-dag" / "books-ingest"
    current_output = stage / first.stage_key / "output.json"
    old_directory = stage / "synthetic-predecessor-cache"
    old_directory.mkdir(mode=0o755)
    old_output = old_directory / "output.json"
    old_output.write_bytes(current_output.read_bytes())
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_output, current_output)}
    for path in hashes:
        path.chmod(0o644)
        path.parent.chmod(0o755)
    stage.chmod(0o755)
    stage.parent.chmod(0o755)
    monkeypatch.setattr(transport, "fail", True)
    second = runner.ingest_candidate("Atomic Habits", registry=(candidate,), transport=transport)
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
    assert stage.stat().st_mode & 0o777 == 0o700
    assert stage.parent.stat().st_mode & 0o777 == 0o700
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
    (rejection,) = (base / "rejections" / candidate.fingerprint).glob("diagnostic-*.json")
    rejected = json.loads(rejection.read_text())
    assert rejected["research_sha256"] == digest
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


@pytest.mark.parametrize(
    "data",
    [
        b'{"private":"secret\\ud800"}',
        b'{"nested":{"private":"secret","number":' + b"9" * 4301 + b"}}",
    ],
)
def test_malformed_rejections_are_immutable_and_source_bound(
    tmp_path: Path, data: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    input_path = tmp_path / "malformed.json"
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
            directory = base / "rejections" / candidate.fingerprint
            (rejection,) = directory.glob("diagnostic-*.json")
            saved = rejection.read_bytes()
            assert json.loads(saved)["research_sha256"] == digest
            assert cli.main(cmd) == 1
            assert rejection.read_bytes() == saved
            assert b"secret" not in saved
            assert json.loads(saved)["source_sha256"] == candidate.expected_sha256
            assert json.loads(saved)["candidate_fingerprint"] == candidate.fingerprint
            assert rejection.stat().st_mode & 0o777 == 0o600
            assert rejection.parent.stat().st_mode & 0o777 == 0o700
            assert rejection.parent.parent.stat().st_mode & 0o777 == 0o700
            assert base.stat().st_mode & 0o777 == 0o700
            assert not output.exists()
            assert (base / "source" / f"{digest}.json").read_bytes() == data
            assert (base / "source" / f"{digest}.json").stat().st_mode & 0o777 == 0o600
            assert (base / "source").stat().st_mode & 0o777 == 0o700
            assert "secret" not in caplog.text
            assert "9" * 4301 not in caplog.text
            reports.append(rejection)
        assert len(set(reports)) == 2
    finally:
        os.umask(previous)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_import_report_coexists_with_current_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], version: int
) -> None:
    candidate, transport, registry = _fixture(tmp_path)
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    first = runner.ingest_candidate(candidate.title, registry=(candidate,), transport=transport)
    assert not isinstance(first, SourceUnresolved)
    image = first.book.images[0]
    occ = image.occurrences[0]
    text_fields = {
        "original_transcription": "synthetic café",
        "normalized_transcription": None,
        "normalization_notes": None,
        "reviewed_by": "fixture reviewer",
        "reviewed_at": None,
        "review_status": "reviewed",
    }
    identity = {"asset_sha256": image.asset_sha256, "anchor": occ.anchor}
    occurrence = {**identity, "source_path": occ.source_path}
    research = {
        "schema": f"clipper-book-research-observations-v{version}",
        "source_sha256": first.book.content_sha256,
    }
    if version == 1:
        research["entries"] = [
            {
                "asset_sha256": image.asset_sha256,
                "occurrences": [{"anchor": occ.anchor, "source_path": occ.source_path}],
                **text_fields,
            }
        ]
    else:
        research.update(
            research_origin="synthetic ledger",
            research_author=None,
            occurrence_inventory=[occurrence],
            observations=[{**occurrence, "observation_id": "fixture", **text_fields}],
        )
    payload = json.dumps(research, ensure_ascii=False).encode()
    overlay, report = faithful.convert_research_observations(first.book, payload)
    overlay_digest = hashlib.sha256(overlay).hexdigest()
    raw_digest = hashlib.sha256(payload).hexdigest()
    # Reconstruct only the non-content v2 report contract, from synthetic evidence.
    legacy = {
        **report,
        "schema": "clipper-book-image-reconciliation-v2",
        "parser_version": "clipper-book-ordered-extract-v2",
    }
    for field in ("verified_research_links", "unverified_research_links", "machine_occurrences"):
        legacy.pop(field)
    for asset in legacy["assets"]:
        for record in asset["occurrences"]:
            record.pop("research_link_status")
    older_bytes = (json.dumps(legacy, sort_keys=True, indent=2) + "\n").encode()
    base = root / "book-candidates" / "research-import"
    older_path = base / candidate.book_id / f"{overlay_digest}.json"
    source = base / "source" / f"{raw_digest}.json"
    research_path = tmp_path / "research.json"
    output = tmp_path / "sidecar" / "overlay.json"
    research_path.write_bytes(payload)
    previous = os.umask(0)
    try:
        acquisition._private_write(older_path, older_bytes)
        acquisition._private_write(source, payload)
        acquisition._private_write(output, overlay)
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
            str(research_path),
            "--output",
            str(output),
        ]
        assert cli.main(cmd) == 0
        current_path = Path(json.loads(capsys.readouterr().out)["report_path"])
        current_bytes = current_path.read_bytes()
        current = json.loads(current_bytes)
        material = current["import_report_material"]
        report_id = content_fingerprint(material)
        assert current["import_report_identity"] == report_id
        assert current_path.name == f"report-{report_id}.json"
        assert material["schema"] == "clipper-book-research-import-report-v1"
        assert material["source_sha256"] == first.book.content_sha256
        assert material["research_sha256"] == raw_digest
        assert material["overlay_sha256"] == overlay_digest
        assert material["candidate_fingerprint"] == candidate.fingerprint
        assert current["schema"] == "clipper-book-image-reconciliation-v3"
        assert current["parser_version"] == faithful.PARSER_VERSION
        assert current["verified_research_links"] == (1 if version == 2 else 0)
        assert current_path != older_path
        assert older_path.read_bytes() == older_bytes
        assert output.read_bytes() == overlay
        assert source.read_bytes() == payload
        assert cli.main(cmd) == 0
        assert Path(json.loads(capsys.readouterr().out)["report_path"]) == current_path
        assert current_path.read_bytes() == current_bytes
        for path in (older_path, current_path, source, output):
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
        assert base.stat().st_mode & 0o777 == 0o700
        normal = runner.ingest_candidate(
            candidate.title, registry=(candidate,), transport=transport, overlay_path=output
        )
        offline = runner.ingest_candidate(
            candidate.title, registry=(candidate,), offline=True, overlay_path=output
        )
        assert not isinstance(normal, SourceUnresolved)
        assert not isinstance(offline, SourceUnresolved)
        assert normal.stage_key == offline.stage_key
        assert normal.book == offline.book
        # A genuinely conflicting current report must not be overwritten or bypassed.
        conflict = b'{"synthetic":"conflict"}'
        current_path.write_bytes(conflict)
        absent_output = output.with_name("must-not-exist.json")
        assert cli.main([*cmd[:-1], str(absent_output)]) == 1
        assert current_path.read_bytes() == conflict
        assert older_path.read_bytes() == older_bytes
        assert output.read_bytes() == overlay
        assert not absent_output.exists()
    finally:
        os.umask(previous)


@pytest.mark.parametrize("offline", [False, True])
def test_oversized_overlay_cannot_resume_cached_ingest(tmp_path: Path, offline: bool) -> None:
    candidate, transport, _ = _fixture(tmp_path)
    runner = BookOrchestrator(tmp_path / "private")
    runner.acquire_candidate(candidate.title, registry=(candidate,), transport=transport)
    overlay_path = tmp_path / "overlay.json"
    overlay = json.dumps(
        {
            "schema": "clipper-book-image-overlay-v1",
            "source_sha256": candidate.expected_sha256,
            "entries": [],
        }
    ).encode()
    valid = overlay + b" " * (faithful.MAX_EVIDENCE_BYTES - len(overlay))
    for warmed in (False, True):
        if warmed:
            overlay_path.write_bytes(valid)
            accepted = runner.ingest_candidate(
                candidate.title, registry=(candidate,), offline=True, overlay_path=overlay_path
            )
            assert not isinstance(accepted, SourceUnresolved)
        overlay_path.write_bytes(valid + b" ")
        with pytest.raises(faithful.ExtractionError, match="exceeds bounded size"):
            runner.ingest_candidate(
                candidate.title,
                registry=(candidate,),
                transport=transport,
                offline=offline,
                overlay_path=overlay_path,
            )


def test_growing_research_file_is_bounded_at_opened_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    candidate, transport, registry = _fixture(tmp_path)
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    recovered = runner.ingest_candidate(candidate.title, registry=(candidate,), transport=transport)
    assert not isinstance(recovered, SourceUnresolved)
    research = tmp_path / "growing.json"
    research.write_bytes(b"{}")
    output = tmp_path / "must-not-exist.json"
    original_open = Path.open
    read_sizes: list[int] = []
    raw_offsets: list[int] = []
    budget = faithful.MAX_EVIDENCE_BYTES + 1

    @contextmanager
    def open_grown(path: Path, *args, **kwargs):
        if path == research:
            with original_open(path, "ab") as growing:
                growing.write(b" " * (budget + 11000))
        with original_open(path, *args, **kwargs) as handle:
            if path == research:
                assert isinstance(handle, FileIO)
                assert kwargs["buffering"] == 0

                def bounded_read(size: int = -1) -> bytes:
                    offset = handle.tell()
                    read_sizes.append(size)
                    assert size == budget - offset
                    chunk = handle.read(min(size, 65536))
                    assert handle.tell() - offset == len(chunk)
                    raw_offsets.append(handle.tell())
                    return chunk

                yield Mock(wraps=handle, read=bounded_read)
            else:
                yield handle

    monkeypatch.setattr(Path, "open", open_grown)
    assert (
        cli.main(
            [
                "books",
                "overlay-import",
                "--title",
                candidate.title,
                "--registry",
                str(registry),
                "--artifact-root",
                str(root),
                "--research",
                str(research),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert read_sizes[0] == budget
    assert raw_offsets[-1] == budget
    assert all(offset <= budget for offset in raw_offsets)
    assert "exceeds bounded size" in caplog.text
    assert not (root / "book-candidates" / "research-import").exists()
    assert not output.exists()


def test_oversized_cached_research_is_rejected_before_hashing(tmp_path: Path) -> None:
    candidate, transport, _ = _fixture(tmp_path)
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    first = runner.ingest_candidate(candidate.title, registry=(candidate,), transport=transport)
    assert not isinstance(first, SourceUnresolved)
    research = {
        "schema": "clipper-book-research-observations-v2",
        "source_sha256": first.book.content_sha256,
        "research_origin": "fixture",
        "research_author": None,
        "occurrence_inventory": [],
        "observations": [],
    }
    payload = json.dumps(research).encode()
    overlay, _ = faithful.convert_research_observations(first.book, payload)
    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_bytes(overlay)
    path = (
        root
        / "book-candidates"
        / "research-import"
        / "source"
        / f"{hashlib.sha256(payload).hexdigest()}.json"
    )
    acquisition._private_write(path, payload)
    runner.ingest_candidate(
        candidate.title, registry=(candidate,), offline=True, overlay_path=overlay_path
    )
    path.write_bytes(b" " * (faithful.MAX_EVIDENCE_BYTES + 1))
    with pytest.raises(faithful.ExtractionError, match="exceeds bounded size"):
        runner.ingest_candidate(
            candidate.title, registry=(candidate,), offline=True, overlay_path=overlay_path
        )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("book_id", ["atomic-habits", "psychology-of-money"])
def test_legacy_rejection_coexists_with_changed_numeric_diagnostic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, version: int, book_id: str
) -> None:
    candidate, transport, registry = _fixture(tmp_path, book_id=book_id)
    root = tmp_path / "private"
    runner = BookOrchestrator(root)
    first = runner.ingest_candidate(candidate.title, registry=(candidate,), transport=transport)
    assert not isinstance(first, SourceUnresolved)
    image = first.book.images[0]
    occ = image.occurrences[0]
    fields = {
        "original_transcription": 137,
        "normalized_transcription": "synthetic-secret",
        "normalization_notes": "fixture correction note",
        "reviewed_by": "fixture reviewer",
        "reviewed_at": None,
        "review_status": "reviewed",
    }
    occurrence = {
        "asset_sha256": image.asset_sha256,
        "anchor": occ.anchor,
        "source_path": occ.source_path,
    }
    research = {
        "schema": f"clipper-book-research-observations-v{version}",
        "source_sha256": first.book.content_sha256,
    }
    if version == 1:
        research["entries"] = [
            {
                "asset_sha256": image.asset_sha256,
                "occurrences": [{"anchor": occ.anchor, "source_path": occ.source_path}],
                **fields,
            }
        ]
        old_reason = "reviewed image overlay text, anchors or review state invalid"
    else:
        research.update(
            research_origin="synthetic ledger",
            research_author=None,
            occurrence_inventory=[occurrence],
            observations=[{**occurrence, "observation_id": "fixture", **fields}],
        )
        old_reason = "v2 image overlay observation/review provenance invalid"
    data = json.dumps(research).encode()
    digest = hashlib.sha256(data).hexdigest()
    base = root / "book-candidates" / "research-import"
    source = base / "source" / f"{digest}.json"
    directory = base / "rejections" / candidate.fingerprint
    old_path = directory / f"{digest}.json"
    old_report = {
        "schema": "clipper-book-research-import-rejection-v2",
        "research_sha256": digest,
        "source_sha256": first.book.content_sha256,
        "candidate_fingerprint": candidate.fingerprint,
        "status": "rejected",
        "reason": old_reason,
    }
    old_bytes = (json.dumps(old_report, sort_keys=True, indent=2) + "\n").encode()
    old_hash = hashlib.sha256(old_bytes).hexdigest()
    input_path = tmp_path / "research.json"
    input_path.write_bytes(data)
    output = tmp_path / "overlay.json"
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
    previous = os.umask(0)
    try:
        acquisition._private_write(source, data)
        acquisition._private_write(old_path, old_bytes)
        assert cli.main(cmd) == 1
        (current_path,) = directory.glob("diagnostic-*.json")
        saved = current_path.read_bytes()
        current = json.loads(saved)
        diagnostic_id = current.pop("diagnostic_identity")
        assert current == {
            **old_report,
            "schema": "clipper-book-research-import-rejection-v3",
            "reason": "private evidence contains unsupported numeric token",
        }
        assert diagnostic_id == content_fingerprint(current)
        assert current_path.name == f"diagnostic-{diagnostic_id}.json"
        assert current_path != old_path
        assert "unsupported numeric token" in caplog.text
        assert "immutable artifact conflicts" not in caplog.text
        assert "synthetic-secret" not in caplog.text
        assert b"synthetic-secret" not in saved
        assert cli.main(cmd) == 1
        assert list(directory.glob("diagnostic-*.json")) == [current_path]
        assert current_path.read_bytes() == saved
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == old_hash
        assert old_path.read_bytes() == old_bytes
        assert source.read_bytes() == data
        assert input_path.read_bytes() == data
        for path in (source, old_path, current_path):
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.parent.stat().st_mode & 0o777 == 0o700
        assert base.stat().st_mode & 0o777 == 0o700
        assert directory.parent.stat().st_mode & 0o777 == 0o700
        assert not output.exists()
        conflict = b'{"synthetic":"conflict"}'
        current_path.write_bytes(conflict)
        caplog.clear()
        assert cli.main(cmd) == 1
        assert "immutable artifact conflicts" in caplog.text
        assert "synthetic-secret" not in caplog.text
        assert current_path.read_bytes() == conflict
        assert old_path.read_bytes() == old_bytes
        assert source.read_bytes() == data
        assert not output.exists()
    finally:
        os.umask(previous)

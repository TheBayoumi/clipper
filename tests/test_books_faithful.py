"""Original synthetic EPUB/PDF and overlay fixtures; no book files or network."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pymupdf
import pytest
from PIL import Image

from clipper.books.acquisition import ReviewedCandidate, load_registry
from clipper.books.contracts import BookSource, OCRIdentity
from clipper.books.faithful import (
    ExtractionError,
    _xml,
    convert_research_observations,
    cpu_tesseract,
    extract_candidate,
    image_reconciliation,
    ocr_contract,
    reading_blocks,
    tesseract_identity,
)

ROOT = Path(__file__).resolve().parents[1]
ATOMIC = next(
    x
    for x in load_registry(ROOT / "niches/book-source-candidates-v1.json")
    if x.book_id == "atomic-habits"
)
DEEP = next(
    x
    for x in load_registry(ROOT / "niches/book-source-candidates-v1.json")
    if x.book_id == "deep-work"
)
IMAGE = b"\x89PNG\r\n\x1a\nsynthetic-image-only-heading"


def _candidate(
    raw: bytes, base: ReviewedCandidate = ATOMIC, *, fmt: str = "epub"
) -> ReviewedCandidate:
    return replace(
        base,
        format=fmt,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_bytes=len(raw),
    )


def _source(raw: bytes, candidate: ReviewedCandidate) -> BookSource:
    return BookSource(
        candidate.title,
        candidate.author,
        candidate.edition_identity,
        "https://ia801234.us.archive.org/fixture/source",
        "archive_candidate",
        "reviewed_registry",
        "2026-11-21T00:00:00+00:00",
        hashlib.sha256(raw).hexdigest(),
        candidate.format,
    )


def _epub(
    *,
    title: str = "Atomic Habits",
    author: str = "James Clear",
    isbn: str = "9780735211308",
    changed_asset: bool = False,
    bad_image_path: bool = False,
    dtd: bool = False,
) -> bytes:
    manifest = "".join(
        f'<item id="s{i}" href="Text/{name}.xhtml" media-type="application/xhtml+xml"/>'
        for i, name in enumerate(("cover", "part-one", "part-two", "index"))
    )
    manifest += '<item id="chart" href="Images/chart.png" media-type="image/png"/>'
    spine = "".join(f'<itemref idref="s{i}"/>' for i in range(4))
    opf = (
        '<package xmlns:dc="http://purl.org/dc/elements/1.1/" unique-identifier="uid">'
        "<metadata>"
        f"<dc:title>{title}</dc:title>"
        "<dc:title>Atomic Habits: An Easy &amp; Proven Way to Build Good Habits "
        "&amp; Break Bad Ones</dc:title>"
        f"<dc:creator>{author}</dc:creator>"
        '<dc:identifier id="uid">urn:uuid:synthetic-file</dc:identifier>'
        f'<dc:identifier id="isbn">urn:isbn:{isbn}</dc:identifier>'
        "</metadata>"
        f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>"
    )
    image_path = "../../evil.png" if bad_image_path else "../Images/chart.png"
    pages = [
        "<html><body><h1>Cover</h1><p>Short front matter.</p></body></html>",
        (
            "<html><body><h1>Chapter One</h1>"
            "<p>The word is <span>pro</span><span>scribed</span>, not automatically "
            "corrected. The chapter explains ideas without marketing replacements.</p>"
            f'<figure><img src="{image_path}" alt="Automaticity 40 10 60" />'
            "<figcaption>Original chapter figure caption.</figcaption></figure>"
            "<ul><li>First list item</li><li>Second list item</li></ul>"
            "</body></html>"
        ),
        (
            '<html><body><h1><img src="../Images/chart.png" alt="" /></h1>'
            "<p>The source uses an image-only heading.</p>"
            "<table><tr><th>Label</th><td>Unaltered value</td></tr></table></body></html>"
        ),
        (
            "<html><body><h2>Notes and index</h2><p>Keep the index verbatim.</p>"
            '<img src="../Images/chart.png" alt="Reused chart"/></body></html>'
        ),
    ]
    if dtd:
        pages[0] = "<!DOCTYPE foo [<!ENTITY x SYSTEM 'file:///etc/passwd'>]>" + pages[0]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfile full-path="OPS/content.opf"/></container>',
        )
        archive.writestr("OPS/content.opf", opf)
        for index, name in enumerate(("cover", "part-one", "part-two", "index")):
            archive.writestr(f"OPS/Text/{name}.xhtml", pages[index])
        archive.writestr("OPS/Images/chart.png", IMAGE + (b"changed" if changed_asset else b""))
    return stream.getvalue()


def _pdf(*, with_raster: bool = False, author: str = "Cal Newport") -> bytes:
    document = pymupdf.open()
    document.set_metadata({"title": "Deep Work", "author": author, "creator": "original fixture"})
    page = document.new_page()
    page.insert_text((40, 50), "Deep Work - Cal Newport - ISBN 9780349411910")
    page.insert_text((40, 85), "Preserve this UK edition wording exactly: proscribed.")
    page = document.new_page()
    page.insert_text((40, 50), "NOTES: Preserve author's notes.")
    page.insert_text((40, 80), "INDEX: Entries retained. Proscribed is not prescribed.")
    if with_raster:
        png = io.BytesIO()
        Image.new("RGB", (50, 30), "white").save(png, format="PNG")
        page.insert_image(pymupdf.Rect(20, 120, 220, 240), stream=png.getvalue())
    page = document.new_page()
    page.insert_text((40, 50), "Third party distribution footer is not book content.")
    raw = document.tobytes()
    document.close()
    return raw


def test_epub_source_order_and_text_image_separation() -> None:
    raw = _epub()
    book = extract_candidate(_source(raw, _candidate(raw)), raw, _candidate(raw))
    assert book.parser_version == "clipper-book-ordered-extract-v1"
    assert len(book.sections) == 4
    assert book.sections[0].heading == "Cover"
    assert "proscribed" in "\n".join(block.text for block in book.blocks)
    assert "pro scribed" not in "\n".join(block.text for block in book.blocks)
    assert book.sections[-1].heading == "Notes and index"
    assert any(x.kind == "li" for x in book.blocks)
    assert any(x.kind == "th" for x in book.blocks)
    assert any(x.kind == "td" for x in book.blocks)
    assert any(x.kind == "figcaption" for x in book.blocks)
    assert len(book.images) == 1
    assert len(book.images[0].occurrences) == 3
    assert book.images[0].occurrences[0].alt_text == "Automaticity 40 10 60"
    assert book.images[0].occurrences[1].alt_text == ""
    assert len({x.anchor for x in book.images[0].occurrences}) == 3
    assert book.images[0].original_transcription is None
    assert book.recovery_status == "incomplete_recovery"
    assert book.source.authorization_status == "unverified"
    assert book.total_word_count < 1200


def test_epub_reuses_image_bytes_for_one_injected_cpu_ocr() -> None:
    raw = _epub()
    completion = Mock(return_value="40 10 60, do not autocorrect")
    book = extract_candidate(
        _source(raw, _candidate(raw)),
        raw,
        _candidate(raw),
        ocr=completion,
        ocr_identity=OCRIdentity("test-ocr", "v1"),
    )
    completion.assert_called_once_with(IMAGE, "image/png")
    assert len(book.images[0].occurrences) == 3
    assert book.images[0].original_transcription == "40 10 60, do not autocorrect"
    assert book.images[0].method == "local_cpu_ocr"
    assert book.images[0].review_status == "machine_unreviewed"
    assert book.recovery_status == "incomplete_recovery"
    assert not any(block.kind == "image_ocr" for block in book.blocks)


def test_reviewed_overlay_import_preserves_raw_transcription_and_rekeys() -> None:
    raw = _epub()
    selected = _candidate(raw)
    original = extract_candidate(_source(raw, selected), raw, selected)
    asset = original.images[0]
    entry = {
        "asset_sha256": asset.asset_sha256,
        "anchors": [item.anchor for item in asset.occurrences],
        "original_transcription": "40 10 60",
        "normalized_transcription": "40 10 60",
        "normalization_notes": "",
        "reviewed_by": "fixture-reviewer",
        "reviewed_at": "2026-11-21",
        "review_status": "reviewed",
    }
    overlay = {
        "schema": "clipper-book-image-overlay-v1",
        "source_sha256": original.content_sha256,
        "entries": [entry],
    }
    payload = json.dumps(overlay).encode()
    verified = extract_candidate(
        original.source,
        raw,
        selected,
        overlay=payload,
        ocr=Mock(side_effect=AssertionError("reviewed image must not be re-OCRed")),
        ocr_identity=OCRIdentity("test-ocr", "v1"),
    )
    assert verified.images[0].original_transcription == "40 10 60"
    assert verified.images[0].method == "reviewed_overlay"
    assert verified.images[0].review_status == "reviewed"
    assert verified.images[0].reviewed_by == "fixture-reviewer"
    assert verified.images[0].reviewed_at == "2026-11-21"
    assert verified.parser_config_sha256 != original.parser_config_sha256
    assert verified.images[0].occurrences == asset.occurrences
    assert not any(x.kind == "image_transcription" for x in verified.blocks)
    bad = {**overlay, "source_sha256": "0" * 64}
    with pytest.raises(ExtractionError, match="raw-book digest mismatch"):
        extract_candidate(original.source, raw, selected, overlay=json.dumps(bad).encode())
    bad = {**overlay, "entries": [{**entry, "anchors": ["wrong"]}]}
    with pytest.raises(ExtractionError, match="text, anchors or review state invalid"):
        extract_candidate(original.source, raw, selected, overlay=json.dumps(bad).encode())
    bad = {**overlay, "entries": [{**entry, "normalized_transcription": "40 50 60"}]}
    with pytest.raises(ExtractionError, match="text, anchors or review state invalid"):
        extract_candidate(original.source, raw, selected, overlay=json.dumps(bad).encode())
    altered = _epub(changed_asset=True)
    with pytest.raises(ExtractionError, match="raw-book digest mismatch"):
        extract_candidate(
            _source(altered, _candidate(altered)),
            altered,
            _candidate(altered),
            overlay=payload,
        )


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"title": "Wrong Book"}, "title mismatch"),
        ({"author": "Wrong Author"}, "author mismatch"),
        ({"isbn": "9780000000000"}, "ISBN mismatch"),
        ({"bad_image_path": True}, "not declared|path traversal"),
        ({"dtd": True}, "DTD/entity"),
    ],
)
def test_epub_title_edition_assets_and_xml_fail_closed(
    kwargs: dict[str, object], pattern: str
) -> None:
    raw = _epub(**kwargs)
    selected = _candidate(raw)
    with pytest.raises(ExtractionError, match=pattern):
        extract_candidate(_source(raw, selected), raw, selected)


def test_pdf_notes_index_exclusion_and_edition_wording() -> None:
    raw = _pdf()
    selected = replace(
        _candidate(raw, DEEP, fmt="pdf"),
        expected_pages=3,
        exclusions=(
            {
                "page": 3,
                "reason": "Synthetic appended third-party footer",
                "evidence": "recorded synthetic page description",
            },
        ),
    )
    book = extract_candidate(_source(raw, selected), raw, selected)
    assert len(book.sections) == 2
    assert len(book.excluded_pages) == 1
    assert book.excluded_pages[0].page_number == 3
    assert "third-party" in book.excluded_pages[0].reason
    text = "\n".join(x.text for x in book.blocks)
    assert "NOTES:" in text and "INDEX:" in text
    assert "proscribed" in text
    assert "distribution footer" not in text
    assert book.content_sha256 == selected.expected_sha256
    assert book.recovery_status == "incomplete_recovery"


def test_pdf_raster_image_and_sparse_text_require_review() -> None:
    raw = _pdf(with_raster=True)
    selected = replace(_candidate(raw, DEEP, fmt="pdf"), expected_pages=3, exclusions=())
    original = extract_candidate(_source(raw, selected), raw, selected)
    assert original.images
    assert "sparse or raster" in " ".join(original.warnings)
    assert original.recovery_status == "incomplete_recovery"
    ocr = Mock(return_value="source-visible label")
    recovered = extract_candidate(
        _source(raw, selected),
        raw,
        selected,
        ocr=ocr,
        ocr_identity=OCRIdentity("fixture-pdf-ocr", "v1"),
    )
    assert recovered.images[0].review_status == "machine_unreviewed"
    assert ocr.call_count >= 1


def test_pdf_page_count_and_wrong_author_fail_closed() -> None:
    raw = _pdf()
    wrong_count = replace(_candidate(raw, DEEP, fmt="pdf"), expected_pages=230)
    with pytest.raises(ExtractionError, match="physical page count changed"):
        extract_candidate(_source(raw, wrong_count), raw, wrong_count)
    changed = _pdf(author="Wrong Author")
    wrong_author = replace(_candidate(changed, DEEP, fmt="pdf"), expected_pages=3)
    with pytest.raises(ExtractionError, match="author metadata mismatch"):
        extract_candidate(_source(changed, wrong_author), changed, wrong_author)


def test_source_identity_and_pdf_signature_fail_closed() -> None:
    raw = _epub()
    selected = _candidate(raw)
    with pytest.raises(ExtractionError, match="pinned source and edition"):
        extract_candidate(replace(_source(raw, selected), content_sha256="a" * 64), raw, selected)
    with pytest.raises(ExtractionError, match="authorization"):
        extract_candidate(
            replace(_source(raw, selected), authorization_status="approved"), raw, selected
        )
    with pytest.raises(ExtractionError, match="explicit Archive candidate"):
        extract_candidate(replace(_source(raw, selected), source_kind="owner_file"), raw, selected)
    changed_format = replace(selected, format="pdf")
    with pytest.raises(ExtractionError, match="file signature/format mismatch"):
        extract_candidate(_source(raw, changed_format), raw, changed_format)


def test_cpu_tesseract_adapter_is_optional_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("subprocess.run", Mock(side_effect=OSError("not installed")))
    assert cpu_tesseract(b"image", "image/png") is None

    class Result:
        returncode = 0
        stdout = b"Uncorrected 40 10 60\n"

    monkeypatch.setattr("subprocess.run", Mock(return_value=Result()))
    assert cpu_tesseract(b"image", "image/png") == "Uncorrected 40 10 60"


def test_epub_xml_dtd_is_rejected_before_parsing() -> None:
    with pytest.raises(ExtractionError, match="DTD/entity"):
        _xml(b"<!DOCTYPE x [<!ENTITY p SYSTEM 'file:///private'>]><html/>")


def _mutate_epub(
    data: bytes,
    *,
    replacements: dict[str, tuple[str, str]] | None = None,
    deletions: tuple[str, ...] = (),
    additions: dict[str, bytes] | None = None,
) -> bytes:
    """Original synthetic archive rewrites only; never checked-in book bytes."""
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(stream, "w") as target:
        for item in source.infolist():
            if item.filename in deletions:
                continue
            value = source.read(item.filename)
            edit = (replacements or {}).get(item.filename)
            if edit is not None:
                value = value.replace(edit[0].encode(), edit[1].encode())
            target.writestr(item.filename, value)
        for name, payload in (additions or {}).items():
            target.writestr(name, payload)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("path", "old", "new", "error"),
    [
        (
            "META-INF/container.xml",
            '<rootfile full-path="OPS/content.opf"/>',
            '<rootfile full-path="../bad.opf"/>',
            "package path is unsafe",
        ),
        (
            "META-INF/container.xml",
            '<rootfile full-path="OPS/content.opf"/>',
            "",
            "exactly one package rootfile",
        ),
        (
            "OPS/content.opf",
            'unique-identifier="uid"',
            'unique-identifier="missing"',
            "primary identifier does not exist",
        ),
        (
            "OPS/content.opf",
            "urn:uuid:synthetic-file",
            "urn:isbn:9780000000000",
            "primary ISBN disagrees",
        ),
        (
            "OPS/content.opf",
            "<metadata>",
            "<missing>",
            "malformed EPUB",
        ),
        (
            "OPS/content.opf",
            "<spine><itemref",
            '<spine><itemref idref="s0"/><itemref',
            "duplicates reading-order",
        ),
        (
            "OPS/content.opf",
            'idref="s1"',
            'idref="unknown"',
            "spine idref missing",
        ),
        (
            "OPS/content.opf",
            'media-type="application/xhtml+xml"',
            'media-type="application/pdf"',
            "non-XHTML spine",
        ),
        (
            "OPS/Text/part-one.xhtml",
            "<body>",
            "<head>",
            "malformed EPUB",
        ),
    ],
)
def test_epub_package_metadata_and_spine_ambiguity_fail_closed(
    path: str, old: str, new: str, error: str
) -> None:
    raw = _mutate_epub(_epub(), replacements={path: (old, new)})
    candidate = _candidate(raw)
    with pytest.raises(ExtractionError, match=error):
        extract_candidate(_source(raw, candidate), raw, candidate)


def test_epub_missing_referenced_asset_is_an_error() -> None:
    raw = _mutate_epub(_epub(), deletions=("OPS/Images/chart.png",))
    with pytest.raises(ExtractionError, match="referenced spine/asset missing"):
        extract_candidate(_source(raw, _candidate(raw)), raw, _candidate(raw))


def test_epub_short_frontmatter_and_empty_spine_report_coverage_diagnostics() -> None:
    raw = _mutate_epub(
        _epub(),
        replacements={
            "OPS/content.opf": (
                '<itemref idref="s1"/><itemref idref="s2"/><itemref idref="s3"/>',
                "",
            ),
            "OPS/Text/cover.xhtml": (
                "<h1>Cover</h1><p>Short front matter.</p>",
                "",
            ),
        },
    )
    candidate = _candidate(raw)
    book = extract_candidate(_source(raw, candidate), raw, candidate)
    assert len(book.sections) == 1
    assert any("no source-visible blocks" in warning for warning in book.warnings)
    assert any("only 1 spine" in warning for warning in book.warnings)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b"<root", "malformed EPUB"),
        (b"<!DOCTYPE x><html/>", "DTD/entity"),
    ],
)
def test_xml_untrusted_markup_is_rejected(payload: bytes, error: str) -> None:
    from clipper.books.faithful import _xml

    with pytest.raises(ExtractionError, match=error):
        _xml(payload)


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("/absolute.png", "external or absolute"),
        ("https://host/image.png", "external or absolute"),
        ("../../../evil.png", "path traversal"),
    ],
)
def test_epub_relative_asset_paths_are_strictly_sandboxed(path: str, error: str) -> None:
    from clipper.books.faithful import _relative

    with pytest.raises(ExtractionError, match=error):
        _relative("OPS/Text/page.xhtml", path)


def test_epub_zip_bomb_and_untrusted_members_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clipper.books import faithful

    raw = _epub()
    with pytest.raises(ExtractionError, match="invalid EPUB ZIP"):
        faithful._zip(b"not a ZIP")
    monkeypatch.setattr(faithful, "MAX_MEMBERS", 1)
    with pytest.raises(ExtractionError, match="entry limit"):
        faithful._zip(raw)
    monkeypatch.setattr(faithful, "MAX_MEMBERS", 1500)
    monkeypatch.setattr(faithful, "MAX_TOTAL_UNCOMPRESSED", 100)
    with pytest.raises(ExtractionError, match="decompressed byte budget"):
        faithful._zip(raw)
    monkeypatch.setattr(faithful, "MAX_TOTAL_UNCOMPRESSED", 100_000_000)
    monkeypatch.setattr(faithful, "MAX_ITEM_BYTES", 2)
    with pytest.raises(ExtractionError, match="decompressed byte budget"):
        faithful._zip(raw)
    monkeypatch.setattr(faithful, "MAX_ITEM_BYTES", 20_000_000)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("../private", b"no")
    with pytest.raises(ExtractionError, match="unsafe EPUB archive member"):
        faithful._zip(archive.getvalue())
    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as out:
        out.writestr("highratio", b"\x00" * 4_000_000)
    with pytest.raises(ExtractionError, match="compression ratio"):
        faithful._zip(bomb.getvalue())


def test_epub_read_refuses_missing_member_and_individual_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clipper.books import faithful

    with zipfile.ZipFile(io.BytesIO(_epub())) as archive:
        with pytest.raises(ExtractionError, match="referenced spine/asset missing"):
            faithful._read(archive, "OPS/no-such-image.png")
        monkeypatch.setattr(faithful, "MAX_ITEM_BYTES", 1)
        with pytest.raises(ExtractionError, match="member exceeds byte limit"):
            faithful._read(archive, "OPS/content.opf")


def test_overlay_malformed_schema_missing_review_and_unknown_asset_fail_closed() -> None:
    raw = _epub()
    candidate = _candidate(raw)
    book = extract_candidate(_source(raw, candidate), raw, candidate)
    asset = book.images[0]
    record = {
        "schema": "clipper-book-image-overlay-v1",
        "source_sha256": book.content_sha256,
        "entries": [
            {
                "asset_sha256": asset.asset_sha256,
                "anchors": [x.anchor for x in asset.occurrences],
                "original_transcription": "40 10 60",
                "normalized_transcription": None,
                "normalization_notes": None,
                "reviewed_by": "fixture-human",
                "reviewed_at": "2026-11-21",
                "review_status": "reviewed",
            }
        ],
    }
    inputs: list[tuple[bytes, str]] = [
        (b"{broken JSON", "invalid JSON"),
        (json.dumps({**record, "schema": "unknown"}).encode(), "schema mismatch"),
        (json.dumps({**record, "entries": "invalid"}).encode(), "must be a list"),
        (
            json.dumps({**record, "entries": [{"missing": "required"}]}).encode(),
            "entry fields mismatch",
        ),
        (
            json.dumps(
                {
                    **record,
                    "entries": [
                        {**record["entries"][0], "asset_sha256": "0" * 64},
                    ],
                }
            ).encode(),
            "SHA-256 missing or duplicate",
        ),
        (
            json.dumps({**record, "entries": [*record["entries"], *record["entries"]]}).encode(),
            "SHA-256 missing or duplicate",
        ),
    ]
    for payload, error in inputs:
        with pytest.raises(ExtractionError, match=error):
            extract_candidate(book.source, raw, candidate, overlay=payload)


def test_missing_ocr_does_not_imply_complete_transcription() -> None:
    raw = _epub()
    candidate = _candidate(raw)
    book = extract_candidate(
        _source(raw, candidate),
        raw,
        candidate,
        ocr=lambda _p, _m: None,
        ocr_identity=OCRIdentity("test-no-words", "v1"),
    )
    assert book.recovery_status == "incomplete_recovery"
    assert all(image.original_transcription is None for image in book.images)
    assert "OCR absent/failed" in book.images[0].uncertainty


def test_pdf_rejects_corrupted_bytes_zero_pages_and_invalid_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"%PDF-not-a-valid-document"
    candidate = _candidate(raw, DEEP, fmt="pdf")
    with pytest.raises(ExtractionError, match="cannot be opened"):
        extract_candidate(_source(raw, candidate), raw, candidate)

    class EmptyPDF:
        needs_pass = False

        def __enter__(self) -> EmptyPDF:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __len__(self) -> int:
            return 0

    data = _pdf()
    candidate = replace(_candidate(data, DEEP, fmt="pdf"), expected_pages=None)
    with monkeypatch.context() as patch:
        patch.setattr("pymupdf.open", lambda **_kwargs: EmptyPDF())
        with pytest.raises(ExtractionError, match="page count unsupported"):
            extract_candidate(_source(data, candidate), data, candidate)
    original = _pdf()
    candidate = replace(
        _candidate(original, DEEP, fmt="pdf"),
        expected_pages=3,
        exclusions=({"page": 4, "reason": "unknown", "evidence": "bad"},),
    )
    with pytest.raises(ExtractionError, match="excluded page is outside"):
        extract_candidate(_source(original, candidate), original, candidate)


def test_pdf_frontmatter_identity_and_isbn_gap_are_diagnostic() -> None:
    document = pymupdf.open()
    document.new_page().insert_text((30, 50), "Deep Work by someone else")
    raw = document.tobytes()
    document.close()
    candidate = replace(_candidate(raw, DEEP, fmt="pdf"), expected_pages=1, exclusions=())
    with pytest.raises(ExtractionError, match="author absent"):
        extract_candidate(_source(raw, candidate), raw, candidate)

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((30, 50), "Deep Work by Cal Newport; no ISBN number given")
    raw = document.tobytes()
    document.close()
    candidate = replace(_candidate(raw, DEEP, fmt="pdf"), expected_pages=1, exclusions=())
    book = extract_candidate(_source(raw, candidate), raw, candidate)
    assert any("ISBN" in warning and "uncertain" in warning for warning in book.warnings)
    assert book.recovery_status == "incomplete_recovery"


def test_atomic_reviewed_two_title_metadata_exactly_matches_same_isbn() -> None:
    raw = _epub(title="Atomic Habits: Tiny Changes, Remarkable Results")
    raw = _mutate_epub(
        raw,
        replacements={
            "OPS/content.opf": (
                "<dc:title>Atomic Habits: An Easy &amp; Proven",
                "<dc:title>An Easy &amp; Proven",
            )
        },
    )
    selected = _candidate(raw)
    book = extract_candidate(_source(raw, selected), raw, selected)
    assert book.sections
    assert book.source.content_sha256 == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ExtractionError, match="ambiguous EPUB title"):
        bad = _mutate_epub(
            raw,
            replacements={
                "OPS/content.opf": (
                    "An Easy &amp; Proven Way to Build Good Habits",
                    "An Easy &amp; Proven Way to Build Different Habits",
                )
            },
        )
        extract_candidate(_source(bad, _candidate(bad)), bad, _candidate(bad))


def test_injected_ocr_requires_explicit_producer_identity() -> None:
    raw = _epub()
    candidate = _candidate(raw)
    with pytest.raises(ExtractionError, match="requires deterministic identity"):
        extract_candidate(_source(raw, candidate), raw, candidate, ocr=lambda _a, _m: "X")
    with pytest.raises(ExtractionError, match="without a producer"):
        extract_candidate(
            _source(raw, candidate),
            raw,
            candidate,
            ocr_identity=OCRIdentity("invalid", "v1"),
        )


def test_ordered_reading_blocks_keep_source_and_reviewed_images_distinct() -> None:
    raw = _epub()
    candidate = _candidate(raw)
    book = extract_candidate(_source(raw, candidate), raw, candidate)
    asset = book.images[0]
    observation = {
        "schema": "clipper-book-research-observations-v1",
        "source_sha256": candidate.expected_sha256,
        "entries": [
            {
                "asset_sha256": asset.asset_sha256,
                "occurrences": [
                    {"anchor": x.anchor, "source_path": x.source_path} for x in asset.occurrences
                ],
                "original_transcription": "40 10 60",
                "normalized_transcription": "40, 10, 60",
                "normalization_notes": "Editorial punctuation grouping only",
                "reviewed_by": "fixture-human",
                "reviewed_at": None,
                "review_status": "reviewed",
            }
        ],
    }
    overlay, report = convert_research_observations(book, json.dumps(observation).encode())
    assert report["unreviewed_assets"] == 0
    assert report["total_occurrences"] == 3
    converted = extract_candidate(_source(raw, candidate), raw, candidate, overlay=overlay)
    assert converted.images[0].reviewed_at is None
    assert converted.images[0].normalization_notes == "Editorial punctuation grouping only"
    view = list(reading_blocks(converted))
    assert len(view) == len(converted.blocks)
    assert [x.anchor for x in view] == [x.anchor for x in converted.blocks]
    image_items = [x for x in view if x.origin == "reviewed_image"]
    assert len(image_items) == 3
    assert all(x.original_image_text == "40 10 60" for x in image_items)
    assert all(x.text == "40, 10, 60" for x in image_items)
    assert not any(x.text == "Automaticity 40 10 60" for x in view)
    assert len(converted.blocks) == len(book.blocks)
    assert image_reconciliation(book)["unreviewed_occurrences"] == 3
    observation["entries"][0]["occurrences"][0]["source_path"] = "other.xhtml"
    with pytest.raises(ExtractionError, match="occurrence path/order/hash mismatch"):
        convert_research_observations(book, json.dumps(observation).encode())


def test_research_import_preserves_unmatched_image_uncertainty() -> None:
    raw = _epub()
    candidate = _candidate(raw)
    book = extract_candidate(_source(raw, candidate), raw, candidate)
    payload = {
        "schema": "clipper-book-research-observations-v1",
        "source_sha256": book.content_sha256,
        "entries": [],
    }
    overlay, report = convert_research_observations(book, json.dumps(payload).encode())
    assert report["unreviewed_assets"] == 1
    assert report["unreviewed_occurrences"] == 3
    assert report["assets"][0]["occurrences"][0]["source_path"]
    still_incomplete = extract_candidate(book.source, raw, candidate, overlay=overlay)
    assert still_incomplete.recovery_status == "incomplete_recovery"
    assert all(
        x.origin == "unrecovered_image" or x.origin == "source"
        for x in reading_blocks(still_incomplete)
    )


def test_local_ocr_contract_includes_version_and_cpu_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _command: "/installed/tesseract")
    recorded = Mock()
    recorded.returncode = 0
    recorded.stdout = b"synthetic-tesseract-v1"
    recorded.stderr = b"synthetic-dependencies"
    run = Mock(return_value=recorded)
    monkeypatch.setattr("subprocess.run", run)
    identity = tesseract_identity()
    assert identity.adapter == "local-tesseract"
    assert identity.version == hashlib.sha256(recorded.stdout + b"\n" + recorded.stderr).hexdigest()
    assert identity.settings == (
        ("executable", "/installed/tesseract"),
        ("language", "eng"),
        ("psm", "6"),
    )
    run.assert_called_once_with(
        ["/installed/tesseract", "--version"],
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert ocr_contract(lambda _p, _m: "source-image-words", identity) is not None
    changed = replace(identity, settings=(("language", "deu"),))
    assert ocr_contract(lambda _p, _m: "changed", changed) != ocr_contract(
        lambda _p, _m: "source-image-words", identity
    )
    monkeypatch.setattr("shutil.which", lambda _command: None)
    unavailable = tesseract_identity()
    assert unavailable.version == "not-installed"
    assert unavailable.settings[0] == ("executable", "missing")


def test_invalid_ocr_identity_rejected_before_processing() -> None:
    for identity in (
        OCRIdentity("", "1"),
        OCRIdentity("producer", ""),
        OCRIdentity("producer", "1", (("", "value"),)),
        OCRIdentity("producer", "1", (("lang", "eng"), ("lang", "deu"))),
    ):
        with pytest.raises(ExtractionError, match="requires deterministic identity"):
            ocr_contract(lambda _p, _m: "source words", identity)

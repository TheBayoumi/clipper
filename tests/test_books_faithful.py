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
from clipper.books.contracts import BookSource
from clipper.books.faithful import (
    ExtractionError,
    _parse_xml,
    cpu_tesseract,
    extract_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
ATOMIC = next(
    x for x in load_registry(ROOT / "niches/book-source-candidates-v1.json")
    if x.book_id == "atomic-habits"
)
DEEP = next(
    x for x in load_registry(ROOT / "niches/book-source-candidates-v1.json")
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
            "<html><body><h1><img src=\"../Images/chart.png\" alt=\"\" /></h1>"
            "<p>The source uses an image-only heading.</p>"
            "<table><tr><th>Label</th><td>Unaltered value</td></tr></table></body></html>"
        ),
        (
            "<html><body><h2>Notes and index</h2><p>Keep the index verbatim.</p>"
            "<img src=\"../Images/chart.png\" alt=\"Reused chart\"/></body></html>"
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
        archive.writestr(
            "OPS/Images/chart.png", IMAGE + (b"changed" if changed_asset else b"")
        )
    return stream.getvalue()


def _pdf(*, with_raster: bool = False, author: str = "Cal Newport") -> bytes:
    document = pymupdf.open()
    document.set_metadata(
        {"title": "Deep Work", "author": author, "creator": "original fixture"}
    )
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
        _source(raw, _candidate(raw)), raw, _candidate(raw), ocr=completion
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
        original.source, raw, selected, overlay=payload,
        ocr=Mock(side_effect=AssertionError("reviewed image must not be re-OCRed")),
    )
    assert verified.images[0].original_transcription == "40 10 60"
    assert verified.images[0].method == "reviewed_overlay"
    assert verified.images[0].review_status == "reviewed"
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
            _source(altered, _candidate(altered)), altered, _candidate(altered),
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
    recovered = extract_candidate(_source(raw, selected), raw, selected, ocr=ocr)
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
        extract_candidate(
            replace(_source(raw, selected), source_kind="owner_file"), raw, selected
        )
    changed_format = replace(selected, format="pdf")
    with pytest.raises(ExtractionError, match="file signature/format mismatch"):
        extract_candidate(_source(raw, changed_format), raw, changed_format)


def test_cpu_tesseract_adapter_is_optional_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "subprocess.run", Mock(side_effect=OSError("not installed"))
    )
    assert cpu_tesseract(b"image", "image/png") is None

    class Result:
        returncode = 0
        stdout = b"Uncorrected 40 10 60\n"

    monkeypatch.setattr("subprocess.run", Mock(return_value=Result()))
    assert cpu_tesseract(b"image", "image/png") == "Uncorrected 40 10 60"

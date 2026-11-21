"""Synthetic full-book fixtures and recorded Gutenberg snippet: no network or inference."""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books.contracts import BookSource, SourceUnresolved
from clipper.books.ingest import (
    BookIngestError,
    GutenbergResolver,
    LocalFileResolver,
    ingest_book,
    parse_book,
)

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "tests/fixtures/books/pride-and-prejudice-gutenberg-excerpt.txt"


def make_txt(
    title: str = "Synthetic Complete Book",
    author: str = "Fixture Author",
    *,
    chapters: int = 3,
    words: int = 500,
    gutenberg: bool = False,
    rights: bool = False,
) -> bytes:
    edition = "gutenberg-1342" if gutenberg else "test-edition"
    header = f"Title: {title}\nAuthor: {author}\nEdition: {edition}\n"
    if gutenberg:
        header += "EBook #1342\n"
        if rights:
            header += "Rights: Public domain in the USA\n"
        header += "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    body = "".join(
        f"\nCHAPTER {i + 1}\n"
        + " ".join(["An original synthetic passage provides only test data."] * words)
        + "\n"
        for i in range(chapters)
    )
    if gutenberg:
        body += "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    return (header + body).encode("utf-8")


def make_epub(*, chapters: int = 3, author: str = "Fixture Author") -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as target:
        target.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OPS/content.opf"/></rootfiles></container>',
        )
        refs = "".join(
            f'<item id="c{i}" href="chapter{i}.xhtml"/>' for i in range(chapters)
        )
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(chapters))
        target.writestr(
            "OPS/content.opf",
            '<package><metadata>'
            f'<title>Synthetic Complete Book</title><creator>{author}</creator>'
            '<identifier>test-edition</identifier></metadata>'
            f'<manifest>{refs}</manifest><spine>{spine}</spine></package>',
        )
        paragraph = " ".join(["Invented EPUB test content has no external source."] * 75)
        for i in range(chapters):
            target.writestr(
                f"OPS/chapter{i}.xhtml",
                f"<html><body><h1>Part {i + 1}</h1><p>{paragraph}</p></body></html>",
            )
    return result.getvalue()


def source_for(raw: bytes, *, format: str = "txt") -> BookSource:
    return BookSource(
        "Synthetic Complete Book", "Fixture Author", None, "file:///tmp/synthetic.txt",
        "owner_file", "owner", "", hashlib.sha256(raw).hexdigest(), format,
    )


def test_local_full_txt_and_epub_parse_without_ownership_upgrade(tmp_path: Path) -> None:
    for extension, raw in (("txt", make_txt()), ("epub", make_epub())):
        path = tmp_path / f"synthetic.{extension}"
        path.write_bytes(raw)
        result = ingest_book(
            "  SYNTHETIC COMPLETE BOOK ", "fixture AUTHOR", LocalFileResolver(path)
        )
        assert not isinstance(result, SourceUnresolved)
        assert len(result.sections) == 3
        assert result.total_word_count >= 1200
        assert result.source.authorization_status == "unverified"
        assert result.source.authorization_basis is None
        assert result.source.content_sha256 == hashlib.sha256(raw).hexdigest()
        assert result.content_sha256 == result.source.content_sha256
        assert result.source_fingerprint == result.source.fingerprint


def test_unresolved_missing_and_unsupported_local_book(tmp_path: Path) -> None:
    unresolved = ingest_book("Unknown Book", "Unknown Writer", LocalFileResolver(None))
    assert isinstance(unresolved, SourceUnresolved)
    assert "supply owner-placed book file" in unresolved.required_owner_decision
    with pytest.raises(BookIngestError, match="file missing"):
        ingest_book("X", "Y", LocalFileResolver(tmp_path / "missing.txt"))
    unknown = tmp_path / "unknown.pdf"
    unknown.write_bytes(b"%PDF")
    with pytest.raises(BookIngestError, match="unreadable owner source format"):
        ingest_book("X", "Y", LocalFileResolver(unknown))


@pytest.mark.parametrize(
    ("data", "error"),
    [
        (make_txt(title="Another Title"), "title mismatch"),
        (make_txt(author="Another Author"), "author mismatch"),
        (make_txt(chapters=1), "section completeness"),
        (make_txt(words=10), "word-count completeness"),
        (b"\xff", "UTF-8"),
    ],
)
def test_rejects_wrong_book_and_incomplete_contents(data: bytes, error: str) -> None:
    with pytest.raises(BookIngestError, match=error):
        parse_book(source_for(data), data)


def test_hash_format_edition_and_epub_corruption_fail_closed() -> None:
    good = make_txt()
    with pytest.raises(BookIngestError, match="SHA-256"):
        parse_book(source_for(good), good + b"tamper")
    with pytest.raises(BookIngestError, match="unsupported book format"):
        parse_book(replace(source_for(good), format="pdf"), good)
    with pytest.raises(BookIngestError, match="edition mismatch"):
        parse_book(replace(source_for(good), edition_identity="different"), good)
    with pytest.raises(BookIngestError, match="invalid EPUB"):
        raw = b"this is not an epub"
        parse_book(source_for(raw, format="epub"), raw)
    with pytest.raises(BookIngestError, match="author mismatch"):
        raw = make_epub(author="wrong")
        parse_book(source_for(raw, format="epub"), raw)


def test_gutenberg_fixture_transport_e2e_authorization_metadata_is_not_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access forbidden")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("clipper.books.ingest._https_get", forbidden)
    html = (
        b'<li class="booklink"><a href="/ebooks/1342">'
        b'<span class="title">Synthetic Complete Book</span>'
        b'<span class="subtitle">by Fixture Author</span></a></li>'
    )
    full_book = make_txt(gutenberg=True, rights=True)
    transport = Mock(side_effect=[html, full_book])
    book = ingest_book(
        "Synthetic Complete Book", "Fixture Author", GutenbergResolver(transport)
    )
    assert not isinstance(book, SourceUnresolved)
    assert book.source.source_kind == "public_domain"
    assert book.source.authorization_status == "unverified"
    assert book.source.authorization_basis == "public_domain"
    assert book.source.edition_identity == "gutenberg-1342"
    assert book.source.source_uri == "https://www.gutenberg.org/ebooks/1342.txt.utf-8"
    assert transport.call_count == 2


def test_gutenberg_rights_basis_requires_specific_edition_metadata() -> None:
    raw = make_txt(gutenberg=True, rights=False)
    source = replace(
        source_for(raw), source_kind="public_domain", edition_identity="gutenberg-1342"
    )
    book = parse_book(source, raw)
    assert book.source.authorization_status == "unverified"
    assert book.source.authorization_basis is None
    invalid = raw.replace(b"*** END OF", b"*** MISSING OF")
    with pytest.raises(BookIngestError, match="START/END"):
        parse_book(replace(source, content_sha256=hashlib.sha256(invalid).hexdigest()), invalid)


def test_gutenberg_missing_and_ambiguous_matches_no_fetch() -> None:
    item = (
        '<li class="booklink"><a href="/ebooks/1342">'
        '<span class="title">Synthetic Complete Book</span>'
        '<span class="subtitle">by Fixture Author</span></a></li>'
    )
    missing = Mock(return_value=b"<html><body></body></html>")
    resolved = ingest_book(
        "Synthetic Complete Book", "Fixture Author", GutenbergResolver(missing)
    )
    assert isinstance(resolved, SourceUnresolved)
    assert missing.call_count == 1
    multiple = Mock(return_value=(item + item).encode("utf-8"))
    with pytest.raises(BookIngestError, match="ambiguous Gutenberg edition"):
        ingest_book("Synthetic Complete Book", "Fixture Author", GutenbergResolver(multiple))
    assert multiple.call_count == 1


def test_gutenberg_refuses_other_hosts_and_resolver_ambiguity() -> None:
    resolver = GutenbergResolver(lambda _url: b"nothing")
    with pytest.raises(BookIngestError, match="invalid Gutenberg HTTPS ebook URI"):
        resolver.read(replace(source_for(b"x"), source_uri="https://example.org/evil"))
    class Ambiguous:
        def resolve(self, _title: str, _author: str) -> tuple[BookSource, ...]:
            return (source_for(b"x"), source_for(b"x"))
        def read(self, _source: BookSource) -> bytes:
            raise AssertionError("ambiguous resolver cannot read")
    with pytest.raises(BookIngestError, match="ambiguous source resolver"):
        ingest_book("X", "Y", Ambiguous())


def test_recorded_public_domain_excerpt_is_incomplete_by_full_book_rules() -> None:
    assert "Pride and Prejudice" in RECORDED.read_text()
    excerpt = RECORDED.read_bytes()
    source = BookSource(
        "Pride and Prejudice", "Jane Austen", "gutenberg-1342",
        "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
        "public_domain", "fixture", "", hashlib.sha256(excerpt).hexdigest(), "txt",
    )
    with pytest.raises(BookIngestError, match="completeness"):
        parse_book(source, excerpt)


def test_chapter_completeness_after_global_word_count_passes() -> None:
    raw = make_txt()
    prefix, later = raw.split(b"CHAPTER 2", 1)
    header = prefix.split(b"CHAPTER 1", 1)[0]
    damaged = header + b"CHAPTER 1\nshort\nCHAPTER 2" + later
    with pytest.raises(BookIngestError, match="chapter completeness"):
        parse_book(source_for(damaged), damaged)


def test_https_transport_denies_nonofficial_host_without_network() -> None:
    from clipper.books.ingest import _https_get

    with pytest.raises(BookIngestError, match="official HTTPS"):
        _https_get("file:///etc/passwd")
    with pytest.raises(BookIngestError, match="official HTTPS"):
        _https_get("https://other.example/ebooks/1")

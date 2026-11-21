"""Strict local-file and Gutenberg book ingestion, with injectable HTTPS transport."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import replace
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Protocol

from .contracts import BookSection, BookSource, BookText, SourceUnresolved

MIN_WORDS = 1200
MAX_WORDS = 500000
MAX_BYTES = 12000000


class BookIngestError(ValueError):
    """A source is missing, ambiguous, malformed or incomplete."""


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(".,;: ")


def _metadata(title: str, author: str, source: BookSource) -> None:
    if _norm(title) != _norm(source.title):
        raise BookIngestError(f"title mismatch: requested {source.title!r}, found {title!r}")
    if _norm(author) != _norm(source.author):
        raise BookIngestError(f"author mismatch: requested {source.author!r}, found {author!r}")


def _text_sections(text: str) -> tuple[BookSection, ...]:
    chapters: list[BookSection] = []
    heading = ""
    paragraphs: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^CHAPTER\s+\w+", line, re.IGNORECASE):
            if heading:
                chapters.append(
                    BookSection(f"chapter-{len(chapters) + 1:02d}", heading, tuple(paragraphs))
                )

            heading, paragraphs = line, []
        elif heading and line:
            paragraphs.append(line)
    if heading:
        chapters.append(BookSection(f"chapter-{len(chapters) + 1:02d}", heading, tuple(paragraphs)))
    return tuple(chapters)


def _txt(raw: bytes, source: BookSource) -> tuple[str, str, str | None, tuple[BookSection, ...]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BookIngestError("text source must be UTF-8") from exc

    def field(name: str) -> str:
        match = re.search(rf"^{name}:\s*(.+)$", text, flags=re.M | re.I)
        return match.group(1).strip() if match else ""

    body = text
    if source.source_kind == "public_domain":
        start = re.search(r"\*\*\* START OF (?:THE )?PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
        end = re.search(r"\*\*\* END OF (?:THE )?PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text)
        if start is None or end is None or end.start() <= start.end():
            raise BookIngestError("Gutenberg START/END completeness check failed")
        body = text[start.end() : end.start()]
    return field("Title"), field("Author"), field("Edition") or None, _text_sections(body)


def _parse_xml(data: bytes) -> ET.Element:
    """Reject DTD and entity declarations before bounded stdlib XML parsing."""
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise BookIngestError("EPUB XML DTD and external entities forbidden")
    return ET.fromstring(data)  # noqa: S314 - DTD/entity blocked and archive bounded


def _epub(raw: bytes) -> tuple[str, str, str | None, tuple[BookSection, ...]]:
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_BYTES:
                raise BookIngestError("EPUB uncompressed byte limit exceeded")
            container = _parse_xml(archive.read("META-INF/container.xml"))
            opf = next(
                node.attrib["full-path"]
                for node in container.iter()
                if node.tag.endswith("rootfile")
            )
            if ".." in Path(opf).parts or opf.startswith("/"):
                raise BookIngestError("unsafe EPUB package path")
            package = _parse_xml(archive.read(opf))
            metadata = next(node for node in package if node.tag.endswith("metadata"))

            def field(name: str) -> str:
                return next(
                    (
                        node.text.strip()
                        for node in metadata
                        if node.tag.endswith(name) and node.text
                    ),
                    "",
                )

            manifest = {
                node.attrib["id"]: node.attrib["href"]
                for node in package.iter()
                if node.tag.endswith("item")
            }
            order = [
                node.attrib["idref"] for node in package.iter() if node.tag.endswith("itemref")
            ]
            chapters: list[BookSection] = []
            for item in order:
                href = manifest[item]
                if ".." in Path(href).parts or href.startswith("/"):
                    raise BookIngestError("unsafe EPUB chapter path")
                page = _parse_xml(archive.read(str(Path(opf).parent / href)))
                body = next(node for node in page.iter() if node.tag.endswith("body"))
                paragraphs = tuple(
                    " ".join("".join(p.itertext()).split())
                    for p in body.iter()
                    if p.tag.endswith("p")
                )
                heading = next(
                    (
                        " ".join("".join(p.itertext()).split())
                        for p in body.iter()
                        if p.tag.endswith("h1")
                    ),
                    item,
                )
                chapters.append(
                    BookSection(f"chapter-{len(chapters) + 1:02d}", heading, paragraphs)
                )
            return field("title"), field("creator"), field("identifier") or None, tuple(chapters)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, StopIteration) as exc:
        raise BookIngestError("invalid EPUB ZIP, package, metadata or chapter XML") from exc


def parse_book(source: BookSource, raw: bytes) -> BookText:
    if len(raw) > MAX_BYTES:
        raise BookIngestError(f"book byte limit: {len(raw)} exceeds {MAX_BYTES}")
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != source.content_sha256:
        raise BookIngestError("source content SHA-256 mismatch")
    if source.format == "txt":
        title, author, edition, sections = _txt(raw, source)
    elif source.format == "epub":
        title, author, edition, sections = _epub(raw)
    else:
        raise BookIngestError(f"unsupported book format: {source.format}")
    _metadata(title, author, source)
    if source.edition_identity and edition and edition != source.edition_identity:
        raise BookIngestError(
            f"edition mismatch: expected {source.edition_identity!r}, found {edition!r}"
        )
    if len(sections) < 3:
        raise BookIngestError(f"section completeness failed: {len(sections)} chapters; minimum 3")
    words = sum(len(p.split()) for section in sections for p in section.paragraphs)
    if not MIN_WORDS <= words <= MAX_WORDS:
        raise BookIngestError(
            f"word-count completeness failed: {words} words; expected {MIN_WORDS}-{MAX_WORDS}"
        )
    for section in sections:
        count = sum(len(p.split()) for p in section.paragraphs)
        if count < 100:
            raise BookIngestError(
                f"chapter completeness failed: {section.chapter_id} has {count} words; minimum 100"
            )
    if source.source_kind == "public_domain":
        header = raw[:5000].decode("utf-8-sig")
        edition_id = (source.edition_identity or "").removeprefix("gutenberg-")
        if (
            re.search(r"^Rights:\s*Public domain in the USA\s*$", header, re.M)
            and f"EBook #{edition_id}" in header
            and edition_id
        ):
            source = replace(source, authorization_basis="public_domain")
    return BookText(
        source, source.edition_identity or edition, sections, words, checksum, source.fingerprint
    )


class SourceResolver(Protocol):
    def resolve(self, title: str, author: str) -> tuple[BookSource, ...] | SourceUnresolved: ...
    def read(self, source: BookSource) -> bytes: ...


class LocalFileResolver:
    def __init__(self, path: Path | None) -> None:
        self.path = path

    def resolve(self, title: str, author: str) -> tuple[BookSource, ...] | SourceUnresolved:
        if self.path is None:
            return SourceUnresolved(
                title, author, "supply owner-placed book file", "no source declared"
            )
        if not self.path.is_file():
            raise BookIngestError(f"owner source file missing: {self.path}")
        if self.path.suffix.lower() not in {".txt", ".epub"}:
            raise BookIngestError(f"unreadable owner source format: {self.path.suffix}")
        data = self.path.read_bytes()
        if len(data) > MAX_BYTES:
            raise BookIngestError(f"owner source exceeds byte limit: {len(data)}")
        return (
            BookSource(
                title,
                author,
                None,
                self.path.resolve().as_uri(),
                "owner_file",
                "owner",
                "",
                hashlib.sha256(data).hexdigest(),
                self.path.suffix.lower()[1:],
            ),
        )

    def read(self, source: BookSource) -> bytes:
        if self.path is None or source.source_uri != self.path.resolve().as_uri():
            raise BookIngestError("owner source URI does not match declared file")
        return self.path.read_bytes()


class _BookIndex(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[tuple[str, str, str]] = []
        self.active = False
        self.field = ""
        self.number = ""
        self.title = ""
        self.author = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if tag == "li" and "booklink" in (data.get("class") or "").split():
            self.active = True
            self.number = self.title = self.author = ""
        if self.active and tag == "a":
            match = re.fullmatch(r"/ebooks/(\d+)", data.get("href") or "")
            if match:
                self.number = match.group(1)
        if self.active and tag == "span":
            styles = (data.get("class") or "").split()
            self.field = (
                "title" if "title" in styles else ("author" if "subtitle" in styles else "")
            )

    def handle_data(self, data: str) -> None:
        if self.field == "title":
            self.title += data
        if self.field == "author":
            self.author += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "span":
            self.field = ""
        if tag == "li" and self.active:
            self.items.append((self.number, self.title, self.author.removeprefix("by ")))
            self.active = False


def _https_get(url: str) -> bytes:
    """Production opt-in transport; never used by test or CLI default paths."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "www.gutenberg.org":
        raise BookIngestError("Gutenberg transport accepts only official HTTPS URLs")
    with urllib.request.urlopen(url, timeout=15) as response:  # noqa: S310 - HTTPS host checked
        redirected = urllib.parse.urlsplit(response.geturl())
        if redirected.scheme != "https" or redirected.hostname != "www.gutenberg.org":
            raise BookIngestError("Gutenberg redirect must remain on official HTTPS host")
        result: bytes = response.read(MAX_BYTES + 1)
    if len(result) > MAX_BYTES:
        raise BookIngestError("Gutenberg transport byte limit exceeded")
    return result


class GutenbergResolver:
    def __init__(self, transport: Callable[[str], bytes] | None = None) -> None:
        self.transport = transport or _https_get

    def resolve(self, title: str, author: str) -> tuple[BookSource, ...] | SourceUnresolved:
        query = urllib.parse.quote(f"{title} {author}")
        data = self.transport(f"https://www.gutenberg.org/ebooks/search/?query={query}")
        index = _BookIndex()
        index.feed(data.decode("utf-8"))
        found = [
            item
            for item in index.items
            if item[0] and _norm(item[1]) == _norm(title) and _norm(item[2]) == _norm(author)
        ]
        if len(found) > 1:
            raise BookIngestError(f"ambiguous Gutenberg edition matches: {len(found)}")
        if not found:
            return SourceUnresolved(
                title,
                author,
                "declare exact book edition or supply owner file",
                "no exact title and author match in Gutenberg; no fallback",
            )
        number = found[0][0]
        return (
            BookSource(
                title,
                author,
                f"gutenberg-{number}",
                f"https://www.gutenberg.org/ebooks/{number}.txt.utf-8",
                "public_domain",
                "resolver:gutenberg",
                "",
                "",
                "txt",
            ),
        )

    def read(self, source: BookSource) -> bytes:
        if not re.fullmatch(
            r"https://www\.gutenberg\.org/ebooks/\d+\.txt\.utf-8", source.source_uri
        ):
            raise BookIngestError("invalid Gutenberg HTTPS ebook URI")
        return self.transport(source.source_uri)


def ingest_book(title: str, author: str, resolver: SourceResolver) -> BookText | SourceUnresolved:
    matches = resolver.resolve(title, author)
    if isinstance(matches, SourceUnresolved):
        return matches
    if len(matches) != 1:
        raise BookIngestError(f"ambiguous source resolver results: {len(matches)}")
    source = matches[0]
    raw = resolver.read(source)
    if source.source_kind == "public_domain":
        source = replace(source, content_sha256=hashlib.sha256(raw).hexdigest())
    return parse_book(source, raw)

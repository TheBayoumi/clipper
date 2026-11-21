"""Faithful, ordered, source-anchored EPUB/PDF extraction with explicit loss diagnostics.


Never silently rewrite spelling, text or a chart axis. Images and overlays are
separate evidence; acquisition authorization stays independently unverified.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import replace
from io import BytesIO
from typing import Any

from clipper.stage_contracts import content_fingerprint

from .acquisition import ReviewedCandidate
from .contracts import (
    BookSection,
    BookSource,
    BookText,
    ExcludedPage,
    ImageEvidence,
    ImageOccurrence,
    SourceBlock,
)

PARSER_VERSION = "clipper-book-ordered-extract-v1"
MAX_MEMBERS = 1500
MAX_TOTAL_UNCOMPRESSED = 100_000_000
MAX_ITEM_BYTES = 20_000_000
BLOCK_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "td",
    "th",
    "figcaption",
    "blockquote",
    "pre",
    "dt",
    "dd",
    "caption",
}
IMAGE_TAGS = {"img", "image"}


class ExtractionError(ValueError):
    """Reject a malformed or conflicting edition rather than adjusting its text."""


OCR = Callable[[bytes, str], str | None]


def _tag(value: str) -> str:
    return value.rsplit("}", 1)[-1].lower()


def _xml(payload: bytes) -> ET.Element:
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ExtractionError("EPUB XML DTD/entity declaration is forbidden")
    try:
        return ET.fromstring(payload)  # noqa: S314 - declarations rejected, input bounded
    except ET.ParseError as exc:
        raise ExtractionError("malformed EPUB package or XHTML XML") from exc


def _relative(base: str, path: str) -> str:
    if (
        not path
        or path.startswith(("/", "\\"))
        or "://" in path
        or "\\" in path
        or "#" in path
        or "?" in path
    ):
        raise ExtractionError("unsafe EPUB external or absolute asset reference")
    normalized = posixpath.normpath(
        posixpath.join(posixpath.dirname(base), urllib.parse.unquote(path))
    )
    if normalized in {".", ".."} or normalized.startswith("../"):
        raise ExtractionError("EPUB archive path traversal blocked")
    return normalized


def _zip(raw: bytes) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(BytesIO(raw))
        members = archive.infolist()
        if len(members) > MAX_MEMBERS:
            raise ExtractionError(f"EPUB entry limit exceeded: {len(members)}")
        total = 0
        for info in members:
            path = info.filename
            if (
                not path
                or path.startswith(("/", "\\"))
                or "\\" in path
                or posixpath.normpath(path.rstrip("/")) != path.rstrip("/")
                or path.startswith("../")
                or (info.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ExtractionError(f"unsafe EPUB archive member: {path}")
            total += info.file_size
            if info.file_size > MAX_ITEM_BYTES or total > MAX_TOTAL_UNCOMPRESSED:
                raise ExtractionError("EPUB decompressed byte budget exceeded")
            if info.compress_size and info.file_size > 1000 * info.compress_size:
                raise ExtractionError("EPUB compression ratio indicates an archive bomb")
        return archive
    except zipfile.BadZipFile as exc:
        raise ExtractionError("invalid EPUB ZIP container") from exc


def _read(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        item = archive.getinfo(name)
        if item.file_size > MAX_ITEM_BYTES:
            raise ExtractionError(f"EPUB member exceeds byte limit: {name}")
        content = archive.read(name)
        if len(content) != item.file_size:
            raise ExtractionError(f"EPUB member size mismatch: {name}")
        return content
    except KeyError as exc:
        raise ExtractionError(f"EPUB referenced spine/asset missing: {name}") from exc


def _isbn(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit() or ch.upper() == "X")


def _equal(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" ,.;:")


def _verify_epub_metadata(
    package: ET.Element, candidate: ReviewedCandidate
) -> tuple[str, str, tuple[str, ...]]:
    metadata = next((x for x in package if _tag(x.tag) == "metadata"), None)
    if metadata is None:
        raise ExtractionError("EPUB package metadata missing")
    titles = [
        (x.text or "").strip()
        for x in metadata
        if _tag(x.tag) == "title" and (x.text or "").strip()
    ]
    authors = [
        (x.text or "").strip()
        for x in metadata
        if _tag(x.tag) == "creator" and (x.text or "").strip()
    ]
    allowed = {_equal(candidate.title), *(_equal(x) for x in candidate.title_aliases)}
    if not titles or not any(_equal(x) in allowed for x in titles):
        raise ExtractionError(
            f"EPUB title mismatch: expected {candidate.title!r}, found {titles!r}"
        )
    if any(
        _equal(x) not in allowed
        and not any(len(_equal(x)) >= 10 and _equal(x) in name for name in allowed)
        for x in titles
    ):
        raise ExtractionError(f"ambiguous EPUB title/edition metadata: {titles!r}")
    if not authors or _equal(candidate.author) not in {_equal(x) for x in authors}:
        raise ExtractionError(
            f"EPUB author mismatch: expected {candidate.author!r}, found {authors!r}"
        )
    identifiers = [
        (x.attrib.get("id", ""), (x.text or "").strip())
        for x in metadata
        if _tag(x.tag) == "identifier"
    ]
    if not any(candidate.isbn == _isbn(value) for _, value in identifiers):
        raise ExtractionError(
            f"EPUB ISBN mismatch: expected {candidate.isbn}, "
            f"observed {[value for _, value in identifiers]!r}"
        )
    primary_id = package.attrib.get("unique-identifier")
    if primary_id:
        primary = next((v for k, v in identifiers if k == primary_id), None)
        if primary is None:
            raise ExtractionError("EPUB primary identifier does not exist")
        if len(_isbn(primary)) in (10, 13) and _isbn(primary) != candidate.isbn:
            raise ExtractionError(f"EPUB primary ISBN disagrees with reviewed edition: {primary}")
    return titles[0], authors[0], tuple(value for _, value in identifiers)


def _asset_evidence(
    data: bytes,
    media: str,
    path: str,
    anchor: str,
    alt: str,
    current: dict[str, ImageEvidence],
    payloads: dict[str, bytes],
) -> str:
    digest = hashlib.sha256(data).hexdigest()
    payloads[digest] = data
    occurrence = ImageOccurrence(anchor, path, alt)
    old = current.get(digest)
    if old is None:
        current[digest] = ImageEvidence(digest, media, len(data), (occurrence,))
    else:
        current[digest] = replace(old, occurrences=(*old.occurrences, occurrence))
    return digest


class _SpineWalker:
    """Walk a single spine item without duplicate nested text or late-bound closures."""

    def __init__(
        self,
        archive: zipfile.ZipFile,
        chapter_path: str,
        section_id: str,
        ref: str,
        manifest: dict[str, tuple[str, str]],
        blocks: list[SourceBlock],
        assets: dict[str, ImageEvidence],
        payloads: dict[str, bytes],
    ) -> None:
        self.archive = archive
        self.chapter_path = chapter_path
        self.section_id = section_id
        self.heading = ref
        self.ref = ref
        self.manifest = manifest
        self.blocks = blocks
        self.assets = assets
        self.payloads = payloads
        self.before = len(blocks)
        self.buffer: list[str] = []

    def emit(self, kind: str) -> None:
        text = "".join(self.buffer)
        self.buffer.clear()
        if not text.strip():
            return
        anchor = self.anchor()
        self.blocks.append(SourceBlock(anchor, self.section_id, self.chapter_path, kind, text))
        if kind.startswith("h") and self.heading == self.ref:
            self.heading = text.strip()

    def anchor(self) -> str:
        return f"{self.section_id}:b{len(self.blocks) - self.before + 1:04d}"

    def walk(self, node: ET.Element, parent_kind: str = "text") -> None:
        tag = _tag(node.tag)
        kind = tag if tag in BLOCK_TAGS else parent_kind
        if tag in IMAGE_TAGS:
            self.emit(parent_kind)
            path = (
                node.attrib.get("src")
                or node.attrib.get("href")
                or node.attrib.get("{http://www.w3.org/1999/xlink}href")
                or ""
            )
            target = _relative(self.chapter_path, path)
            media = next((mime for value, mime in self.manifest.values() if value == target), None)
            if media is None:
                raise ExtractionError(f"EPUB image not declared in manifest: {target}")
            data = _read(self.archive, target)
            alt = node.attrib.get("alt", "")
            anchor = self.anchor()
            digest = _asset_evidence(data, media, target, anchor, alt, self.assets, self.payloads)
            self.blocks.append(
                SourceBlock(
                    anchor,
                    self.section_id,
                    self.chapter_path,
                    "image",
                    "",
                    asset_sha256=digest,
                )
            )
            return
        if tag in BLOCK_TAGS:
            self.emit(parent_kind)
        if node.text:
            self.buffer.append(node.text)
        for child in node:
            self.walk(child, kind)
            if child.tail:
                self.buffer.append(child.tail)
        if tag in BLOCK_TAGS:
            self.emit(kind)

    def finish(self) -> BookSection:
        self.emit("text")
        selected = self.blocks[self.before :]
        paragraphs = tuple(
            block.text for block in selected if block.kind != "image" and block.text.strip()
        )
        return BookSection(self.section_id, self.heading, paragraphs)


def _epub(
    source: BookSource, raw: bytes, candidate: ReviewedCandidate
) -> tuple[
    tuple[BookSection, ...],
    tuple[SourceBlock, ...],
    tuple[ImageEvidence, ...],
    tuple[str, ...],
    dict[str, bytes],
]:
    chapters: list[BookSection] = []
    blocks: list[SourceBlock] = []
    assets: dict[str, ImageEvidence] = {}
    payloads: dict[str, bytes] = {}
    warnings: list[str] = []
    with _zip(raw) as archive:
        container = _xml(_read(archive, "META-INF/container.xml"))
        roots = [x.attrib.get("full-path") for x in container.iter() if _tag(x.tag) == "rootfile"]
        if len(roots) != 1 or not roots[0]:
            raise ExtractionError("EPUB must declare exactly one package rootfile")
        package_name = roots[0]
        if (
            package_name.startswith(("/", "\\"))
            or posixpath.normpath(package_name) != package_name
            or package_name.startswith("../")
        ):
            raise ExtractionError("EPUB package path is unsafe")
        package = _xml(_read(archive, package_name))
        _verify_epub_metadata(package, candidate)
        manifest: dict[str, tuple[str, str]] = {
            item.attrib["id"]: (
                _relative(package_name, item.attrib["href"]),
                item.attrib.get("media-type", ""),
            )
            for item in package.iter()
            if _tag(item.tag) == "item"
        }
        ordered = [
            item.attrib.get("idref", "") for item in package.iter() if _tag(item.tag) == "itemref"
        ]
        if not ordered or len(ordered) != len(set(ordered)):
            raise ExtractionError("EPUB spine missing or duplicates reading-order entries")
        for index, ref in enumerate(ordered, start=1):
            if ref not in manifest:
                raise ExtractionError(f"EPUB spine idref missing from manifest: {ref}")
            chapter_path, media = manifest[ref]
            if media not in {"application/xhtml+xml", "text/html"}:
                raise ExtractionError(f"EPUB non-XHTML spine item: {ref} ({media})")
            root = _xml(_read(archive, chapter_path))
            body = next((node for node in root.iter() if _tag(node.tag) == "body"), None)
            if body is None:
                raise ExtractionError(f"EPUB XHTML body missing: {chapter_path}")
            section_id = f"spine-{index:04d}"
            reader = _SpineWalker(
                archive, chapter_path, section_id, ref, manifest, blocks, assets, payloads
            )
            reader.walk(body)
            section = reader.finish()
            if not section.paragraphs and not any(
                block.kind == "image" for block in blocks[reader.before :]
            ):
                warnings.append(f"spine item {section_id} contains no source-visible blocks")
            chapters.append(section)
    if len(chapters) < 3:
        warnings.append(f"only {len(chapters)} spine items recovered; completeness uncertain")
    return tuple(chapters), tuple(blocks), tuple(assets.values()), tuple(warnings), payloads


def _pdf(
    source: BookSource, raw: bytes, candidate: ReviewedCandidate
) -> tuple[
    tuple[BookSection, ...],
    tuple[SourceBlock, ...],
    tuple[ImageEvidence, ...],
    tuple[ExcludedPage, ...],
    tuple[str, ...],
    dict[str, bytes],
]:
    try:
        import pymupdf
    except ImportError as exc:
        raise ExtractionError(
            "PDF extraction requires optional pymupdf book-pdf dependency"
        ) from exc
    chapters: list[BookSection] = []
    blocks: list[SourceBlock] = []
    assets: dict[str, ImageEvidence] = {}
    payloads: dict[str, bytes] = {}
    excluded: list[ExcludedPage] = []
    warnings: list[str] = []
    try:
        document: Any = pymupdf.open(stream=raw, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise ExtractionError("PDF cannot be opened as a valid document") from exc
    with document:
        if document.needs_pass:
            raise ExtractionError("encrypted PDF is not supported; no DRM bypass")
        count = len(document)
        if not 1 <= count <= 1200:
            raise ExtractionError(f"PDF page count unsupported: {count}")
        if candidate.expected_pages is not None and count != candidate.expected_pages:
            raise ExtractionError(
                f"PDF physical page count changed: expected {candidate.expected_pages}, got {count}"
            )
        metadata = document.metadata or {}
        pdf_title = str(metadata.get("title") or "")
        pdf_author = str(metadata.get("author") or "")
        if pdf_title and _equal(pdf_title) not in {
            _equal(candidate.title),
            *(_equal(x) for x in candidate.title_aliases),
        }:
            raise ExtractionError(f"PDF title metadata mismatch: {pdf_title!r}")
        if pdf_author and _equal(pdf_author) != _equal(candidate.author):
            raise ExtractionError(f"PDF author metadata mismatch: {pdf_author!r}")
        excluded_pages = {entry["page"]: entry for entry in candidate.exclusions}
        if any(not isinstance(page, int) or not 1 <= page <= count for page in excluded_pages):
            raise ExtractionError("PDF excluded page is outside the selected edition")
        first_pages: list[str] = []
        for number in range(1, count + 1):
            page: Any = document[number - 1]
            section_id = f"page-{number:04d}"
            raw_text = page.get_text("text")
            if number <= 20:
                first_pages.append(raw_text)
            if number in excluded_pages:
                rule = excluded_pages[number]
                excluded.append(
                    ExcludedPage(
                        number,
                        str(rule["reason"]),
                        str(rule["evidence"]),
                        hashlib.sha256(raw_text.encode()).hexdigest(),
                    )
                )
                continue
            pdf_blocks = page.get_text("dict", sort=False).get("blocks", [])
            before = len(blocks)
            for item in pdf_blocks:
                anchor = f"{section_id}:b{len(blocks) - before + 1:04d}"
                if item.get("type") == 0:
                    lines = [
                        "".join(span.get("text", "") for span in line.get("spans", []))
                        for line in item.get("lines", [])
                    ]
                    text = "\n".join(lines)
                    if text.strip():
                        blocks.append(SourceBlock(anchor, section_id, section_id, "pdf_text", text))
                elif item.get("type") == 1 and item.get("image"):
                    data = item["image"]
                    digest = _asset_evidence(
                        data,
                        str(item.get("ext", "image")),
                        section_id,
                        anchor,
                        "",
                        assets,
                        payloads,
                    )
                    blocks.append(
                        SourceBlock(
                            anchor, section_id, section_id, "image", "", asset_sha256=digest
                        )
                    )
            selected = blocks[before:]
            paragraphs = tuple(block.text for block in selected if block.kind == "pdf_text")
            words = sum(len(item.split()) for item in paragraphs)
            if words < 10:
                warnings.append(
                    f"sparse or raster PDF page {number}: {words} extracted words; "
                    "OCR/review required"
                )
            chapters.append(BookSection(section_id, f"PDF page {number}", paragraphs))
        front = "\n".join(first_pages)
        if candidate.title.casefold() not in front.casefold() and not pdf_title:
            raise ExtractionError("PDF title absent from source front matter")
        if candidate.author.casefold() not in front.casefold() and not pdf_author:
            raise ExtractionError("PDF author absent from source front matter")
        if candidate.isbn not in re.sub(r"[\s\-]", "", front):
            warnings.append(
                f"reviewed ISBN {candidate.isbn} not visible in first 20 pages; edition uncertain"
            )
    return (
        tuple(chapters),
        tuple(blocks),
        tuple(assets.values()),
        tuple(excluded),
        tuple(warnings),
        payloads,
    )


def cpu_tesseract(image: bytes, _media_type: str) -> str | None:
    """Explicit opt-in local installed binary; never downloads models or calls a provider."""
    try:
        process = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "6"],
            input=image,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    result = process.stdout.decode("utf-8", errors="replace").strip()
    return result or None


def _overlay(
    raw_sha: str,
    blocks: tuple[SourceBlock, ...],
    images: tuple[ImageEvidence, ...],
    payload: bytes,
) -> tuple[tuple[SourceBlock, ...], tuple[ImageEvidence, ...]]:
    try:
        document: object = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionError("reviewed image overlay is invalid JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "source_sha256", "entries"}
        or document["schema"] != "clipper-book-image-overlay-v1"
    ):
        raise ExtractionError("reviewed image overlay schema mismatch")
    if document["source_sha256"] != raw_sha:
        raise ExtractionError("reviewed image overlay raw-book digest mismatch")
    entries = document["entries"]
    if not isinstance(entries, list):
        raise ExtractionError("reviewed image overlay entries must be a list")
    by_hash = {image.asset_sha256: image for image in images}
    updated: dict[str, ImageEvidence] = {}
    seen: set[str] = set()
    for entry in entries:
        expected = {
            "asset_sha256",
            "anchors",
            "original_transcription",
            "normalized_transcription",
            "normalization_notes",
            "reviewed_by",
            "reviewed_at",
            "review_status",
        }
        if not isinstance(entry, dict) or set(entry) != expected:
            raise ExtractionError("reviewed image overlay entry fields mismatch")
        digest = entry["asset_sha256"]
        if not isinstance(digest, str) or digest not in by_hash or digest in seen:
            raise ExtractionError("reviewed image overlay asset SHA-256 missing or duplicate")
        source_image = by_hash[digest]
        original = entry["original_transcription"]
        normalized = entry["normalized_transcription"]
        notes = entry["normalization_notes"]
        anchors = entry["anchors"]
        if (
            not isinstance(original, str)
            or not original.strip()
            or not isinstance(normalized, (str, type(None)))
            or not isinstance(notes, (str, type(None)))
            or (normalized is not None and normalized != original and not notes)
            or not isinstance(anchors, list)
            or set(anchors) != {x.anchor for x in source_image.occurrences}
            or not all(isinstance(x, str) for x in anchors)
            or not isinstance(entry["reviewed_by"], str)
            or not entry["reviewed_by"].strip()
            or not isinstance(entry["reviewed_at"], str)
            or not entry["reviewed_at"].strip()
            or entry["review_status"] != "reviewed"
        ):
            raise ExtractionError("reviewed image overlay text, anchors or review state invalid")
        updated[digest] = replace(
            source_image,
            original_transcription=original,
            normalized_transcription=normalized,
            normalization_notes=notes,
            method="reviewed_overlay",
            uncertainty="transcription still requires edition review",
            review_status="reviewed",
        )
        seen.add(digest)
    reordered = tuple(updated.get(x.asset_sha256, x) for x in images)
    # Transcriptions are not inserted into prose. Keep them in image evidence only.
    return blocks, reordered


def extract_candidate(
    source: BookSource,
    raw: bytes,
    candidate: ReviewedCandidate,
    *,
    overlay: bytes | None = None,
    ocr: OCR | None = None,
) -> BookText:
    if source.source_kind != "archive_candidate":
        raise ExtractionError("faithful extraction requires an explicit Archive candidate")
    digest = hashlib.sha256(raw).hexdigest()
    if (
        digest != source.content_sha256
        or digest != candidate.expected_sha256
        or len(raw) != candidate.expected_bytes
    ):
        raise ExtractionError("candidate raw bytes do not match pinned source and edition")
    if source.authorization_status != "unverified":
        raise ExtractionError("candidate source authorization cannot be auto-upgraded")
    if candidate.format == "epub" and raw[:4] == b"PK\x03\x04":
        sections, blocks, images, warnings, payloads = _epub(source, raw, candidate)
        excluded: tuple[ExcludedPage, ...] = ()
    elif candidate.format == "pdf" and raw[:5] == b"%PDF-":
        sections, blocks, images, excluded, warnings, payloads = _pdf(source, raw, candidate)
    else:
        raise ExtractionError("candidate file signature/format mismatch")
    if overlay is not None:
        blocks, images = _overlay(digest, blocks, images, overlay)
    if ocr is not None:
        resolved: list[ImageEvidence] = []
        for image in images:
            if image.method == "reviewed_overlay":
                resolved.append(image)
                continue
            # OCR is never source truth and is kept separately as unreviewed evidence.
            warning = "OCR absent/failed; source image words not recovered"
            value = ocr(payloads[image.asset_sha256], image.media_type)
            if value and value.strip():
                resolved.append(
                    replace(
                        image,
                        original_transcription=value,
                        method="local_cpu_ocr",
                        review_status="machine_unreviewed",
                        uncertainty="machine OCR requires manual verification",
                    )
                )
            else:
                resolved.append(replace(image, uncertainty=warning))
        images = tuple(resolved)
    gaps = list(warnings)
    if any(image.review_status != "reviewed" for image in images):
        gaps.append(
            f"image transcription incomplete: "
            f"{sum(x.review_status != 'reviewed' for x in images)} unique assets unreviewed"
        )
    words = sum(len(p.split()) for section in sections for p in section.paragraphs)
    if words < 1200:
        gaps.append(
            f"low extracted source word count: {words}; full-edition completeness uncertain"
        )
    status = "incomplete_recovery" if gaps else "mechanically_extracted_review_required"
    config = {
        "parser": PARSER_VERSION,
        "candidate": candidate.fingerprint,
        "format": candidate.format,
        "ocr": ocr is not None,
        "overlay_sha256": hashlib.sha256(overlay).hexdigest() if overlay else None,
    }
    return BookText(
        source=source,
        edition_identity=candidate.edition_identity,
        sections=sections,
        total_word_count=words,
        content_sha256=digest,
        source_fingerprint=source.fingerprint,
        blocks=blocks,
        images=images,
        excluded_pages=excluded,
        warnings=tuple(gaps),
        recovery_status=status,
        parser_version=PARSER_VERSION,
        parser_config_sha256=content_fingerprint(config),
    )

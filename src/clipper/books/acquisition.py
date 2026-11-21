"""Reviewed Archive candidate discovery and bounded anonymous HTTPS acquisition.

The registry identifies candidates, not copyright permission or publisher authenticity.
Network I/O is isolated in ArchiveHTTPTransport; tests supply recorded responses.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from clipper.stage_contracts import content_fingerprint

from .contracts import BookSource, SourceUnresolved

REGISTRY = Path("niches/book-source-candidates-v1.json")
REGISTRY_SCHEMA = "clipper-reviewed-book-candidates-v1"
MAX_METADATA_BYTES = 2_000_000
MAX_ASSET_BYTES = 12_000_000
_HEX = re.compile(r"^[0-9a-f]{64}$")
_CDN = re.compile(r"^(?:ia[0-9]{2,7}\.(?:us|eu)|dn[0-9]{6}\.ca)\.archive\.org$")


class AcquisitionError(ValueError):
    """An unsupported, changed, restricted, or unsafe candidate fails closed."""


@dataclass(frozen=True, slots=True)
class ReviewedCandidate:
    book_id: str
    title: str
    author: str
    record_url: str
    filename: str
    format: str
    expected_bytes: int
    expected_sha256: str
    isbn: str
    edition_identity: str
    title_aliases: tuple[str, ...]
    exclusions: tuple[dict[str, Any], ...]
    expected_pages: int | None = None

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class HTTPResult:
    requested_url: str
    final_url: str
    status: int
    content_type: str
    body: bytes


class BookTransport(Protocol):
    def get(self, url: str, *, limit: int) -> HTTPResult: ...


@dataclass(frozen=True, slots=True)
class AcquiredBook:
    source: BookSource
    candidate: ReviewedCandidate
    record_url: str
    asset_url: str
    fetched_at: str
    http_status: int
    raw_path: Path
    observed_bytes: int
    observed_sha256: str
    metadata_sha256: str
    cache_hit: bool
    warnings: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(
            {
                "candidate": asdict(self.candidate),
                "record_url": self.record_url,
                "source_kind": self.source.source_kind,
                "authorization_status": self.source.authorization_status,
                "metadata_sha256": self.metadata_sha256,
                "observed_bytes": self.observed_bytes,
                "observed_sha256": self.observed_sha256,
            }
        )


def _norm(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" ,.;:")


def load_registry(path: Path = REGISTRY) -> tuple[ReviewedCandidate, ...]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AcquisitionError(f"reviewed source registry unreadable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != REGISTRY_SCHEMA:
        raise AcquisitionError("unknown reviewed source registry schema")
    books = payload.get("books")
    if not isinstance(books, list) or not books:
        raise AcquisitionError("reviewed source registry has no candidates")
    output: list[ReviewedCandidate] = []
    keys: set[tuple[str, str, str]] = set()
    for row in books:
        if not isinstance(row, dict):
            raise AcquisitionError("invalid reviewed candidate record")
        if not isinstance(row.get("title_aliases"), list) or not isinstance(
            row.get("exclusions"), list
        ):
            raise AcquisitionError("reviewed candidate aliases/exclusions must be arrays")
        try:
            candidate = ReviewedCandidate(
                **{
                    **row,
                    "title_aliases": tuple(row["title_aliases"]),
                    "exclusions": tuple(row["exclusions"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionError("incomplete reviewed candidate record") from exc
        identity = (candidate.book_id, candidate.edition_identity, candidate.filename)
        expected_record = re.fullmatch(
            r"https://archive\.org/metadata/[a-zA-Z0-9_.-]+", candidate.record_url
        )
        if (
            identity in keys
            or not expected_record
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", candidate.book_id)
            or not all(
                isinstance(v, str) and v.strip()
                for v in (candidate.title, candidate.author, candidate.isbn)
            )
            or not isinstance(candidate.expected_bytes, int)
            or not 0 < candidate.expected_bytes <= MAX_ASSET_BYTES
            or not _HEX.fullmatch(candidate.expected_sha256)
            or candidate.format not in {"epub", "pdf"}
            or Path(candidate.filename).name != candidate.filename
            or "/" in candidate.filename
            or "\\" in candidate.filename
            or candidate.filename in {".", ".."}
            or not isinstance(candidate.title_aliases, tuple)
            or any(not isinstance(alias, str) for alias in candidate.title_aliases)
        ):
            raise AcquisitionError(f"invalid or duplicate reviewed candidate: {identity}")
        keys.add(identity)
        output.append(candidate)
    return tuple(output)


def discover(
    title: str,
    candidates: tuple[ReviewedCandidate, ...],
    *,
    author: str | None = None,
    edition: str | None = None,
) -> ReviewedCandidate | SourceUnresolved:
    matches = [
        item
        for item in candidates
        if _norm(title) in {_norm(item.title), *(_norm(x) for x in item.title_aliases)}
        and (author is None or _norm(author) == _norm(item.author))
        and (edition is None or edition == item.edition_identity)
    ]
    if len(matches) > 1:
        options = "; ".join(f"{x.title} / {x.author} / {x.edition_identity}" for x in matches)
        return SourceUnresolved(
            title, author or "", "select one reviewed author/edition", f"ambiguous: {options}"
        )
    if not matches:
        return SourceUnresolved(
            title,
            author or "",
            "review an exact source candidate or specify supported edition",
            "no exact reviewed registry match; no fuzzy or restricted-source fallback",
        )
    return matches[0]


def _validate_url(url: str, *, redirect: bool = False) -> str:
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    valid_host = host in {"archive.org", "www.archive.org"} or (
        redirect and bool(_CDN.fullmatch(host))
    )
    if (
        parts.scheme != "https"
        or not valid_host
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or "\\" in url
        or any(part in {".", ".."} for part in urllib.parse.unquote(parts.path).split("/"))
    ):
        raise AcquisitionError(f"unsafe Archive HTTPS URL or redirect host: {host}")
    return host


def _check_public_dns(host: str) -> None:
    try:
        records = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise AcquisitionError(f"cannot resolve Archive host: {host}") from exc
    if not records or any(not ipaddress.ip_address(record[4][0]).is_global for record in records):
        raise AcquisitionError(f"Archive host resolves to unsafe/private address: {host}")


class _StrictRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 3
    max_repeats = 1

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        host = _validate_url(newurl, redirect=True)
        _check_public_dns(host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class ArchiveHTTPTransport:
    """Opt-in real transport; redirect and DNS policies are rechecked per hop."""

    def get(self, url: str, *, limit: int) -> HTTPResult:
        host = _validate_url(url)
        _check_public_dns(host)
        request = urllib.request.Request(  # noqa: S310 - host and scheme validated
            url, headers={"User-Agent": "ClipperBookIngest/1"}
        )
        opener = urllib.request.build_opener(_StrictRedirect())
        try:
            with opener.open(request, timeout=20) as response:
                final = response.geturl()
                _validate_url(final, redirect=True)
                status = int(response.status)
                kind = response.headers.get("Content-Type", "").split(";")[0].lower()
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > limit:
                    raise AcquisitionError("Archive Content-Length exceeds bounded file limit")
                data: bytes = response.read(limit + 1)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise AcquisitionError(f"Archive HTTP transport failed: {exc}") from exc
        if len(data) > limit:
            raise AcquisitionError(f"Archive response exceeds {limit}-byte limit")
        return HTTPResult(url, final, status, kind, data)


def _response(result: HTTPResult, *, limit: int, kind: str) -> None:
    _validate_url(result.requested_url)
    _validate_url(result.final_url, redirect=True)
    if result.status != 200:
        raise AcquisitionError(f"Archive {kind} HTTP status {result.status}; no retry/fallback")
    if len(result.body) > limit:
        raise AcquisitionError(f"Archive {kind} response exceeds {limit}-byte limit")
    if "html" in result.content_type or result.body[:64].lstrip().lower().startswith(
        (b"<!doctype html", b"<html")
    ):
        raise AcquisitionError(f"Archive {kind} returned HTML rather than expected data")


def _private_write(path: Path, data: bytes) -> None:
    """Publish complete immutable bytes with an exclusive atomic link, never replace."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists():
        if path.read_bytes() != data:
            raise AcquisitionError(f"immutable artifact conflicts with recorded bytes: {path.name}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise AcquisitionError(
                    f"immutable artifact conflicts with recorded bytes: {path.name}"
                )
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


class ArchiveAcquirer:
    """Immutable first acquisition plus explicit, separately recorded refresh checks."""

    def __init__(self, root: Path, transport: BookTransport | None = None) -> None:
        self.root = root
        self.transport = transport if transport is not None else ArchiveHTTPTransport()

    def _paths(self, candidate: ReviewedCandidate) -> tuple[Path, Path]:
        key = candidate.fingerprint
        return (
            self.root / "manifests" / f"{key}.json",
            self.root / "raw" / candidate.expected_sha256[:2]
            / f"{candidate.expected_sha256}.{candidate.format}",
        )

    def _cached(self, candidate: ReviewedCandidate) -> AcquiredBook:
        manifest_path, raw_path = self._paths(candidate)
        if not manifest_path.is_file() or not raw_path.is_file():
            raise AcquisitionError("offline candidate cache is absent; no network allowed")
        try:
            saved: object = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("schema") != "clipper-book-acquisition-v1":
                raise AcquisitionError("offline acquisition manifest schema mismatch")
            if saved.get("candidate_fingerprint") != candidate.fingerprint:
                raise AcquisitionError("offline acquisition manifest identity mismatch")
            data = raw_path.read_bytes()
            if len(data) != candidate.expected_bytes or (
                hashlib.sha256(data).hexdigest() != candidate.expected_sha256
            ):
                raise AcquisitionError("offline candidate raw artifact changed; review required")
            source = BookSource(**saved["source"])
            if (
                saved.get("record_url") != candidate.record_url
                or saved.get("observed_sha256") != candidate.expected_sha256
                or saved.get("observed_bytes") != candidate.expected_bytes
                or saved.get("http_status") != 200
                or source.source_kind != "archive_candidate"
                or source.content_sha256 != candidate.expected_sha256
                or source.authorization_status != "unverified"
                or source.edition_identity != candidate.edition_identity
                or source.title != candidate.title
                or source.author != candidate.author
                or source.source_uri != saved.get("asset_url")
                or not isinstance(saved.get("metadata_sha256"), str)
                or not _HEX.fullmatch(saved["metadata_sha256"])
            ):
                raise AcquisitionError("offline acquisition manifest provenance mismatch")
            _validate_url(saved["asset_url"], redirect=True)
            return AcquiredBook(
                source,
                candidate,
                saved["record_url"],
                saved["asset_url"],
                saved["fetched_at"],
                saved["http_status"],
                raw_path,
                len(data),
                candidate.expected_sha256,
                saved["metadata_sha256"],
                True,
                tuple(saved["warnings"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AcquisitionError):
                raise
            raise AcquisitionError("offline acquisition manifest malformed") from exc

    def acquire(
        self,
        candidate: ReviewedCandidate,
        *,
        offline: bool = False,
        refresh: bool = False,
    ) -> AcquiredBook:
        if offline and refresh:
            raise AcquisitionError("--offline and --refresh are mutually exclusive")
        manifest_path, raw_path = self._paths(candidate)
        if offline or (manifest_path.is_file() and not refresh):
            return self._cached(candidate)
        key = candidate.fingerprint
        try:
            metadata = self.transport.get(candidate.record_url, limit=MAX_METADATA_BYTES)
            _response(metadata, limit=MAX_METADATA_BYTES, kind="metadata")
            if "json" not in metadata.content_type:
                raise AcquisitionError("Archive metadata is not JSON")
            record: object = json.loads(metadata.body)
            if not isinstance(record, dict):
                raise AcquisitionError("Archive metadata object malformed")
            if record.get("is_dark") or record.get("is_restricted"):
                raise AcquisitionError("Archive item is restricted or unavailable")
            details = record.get("metadata")
            if not isinstance(details, dict):
                raise AcquisitionError("Archive metadata edition record is malformed")
            identifier = candidate.record_url.rsplit("/", 1)[-1]
            if details.get("is_restricted"):
                raise AcquisitionError("Archive record is restricted")
            if details.get("identifier") not in (None, identifier):
                raise AcquisitionError("Archive record identifier mismatch")
            files = record.get("files")
            if not isinstance(files, list):
                raise AcquisitionError("Archive file listing missing")
            matches = [
                x for x in files if isinstance(x, dict) and x.get("name") == candidate.filename
            ]
            if len(matches) != 1:
                raise AcquisitionError(
                    f"Archive selected asset missing or ambiguous: {candidate.filename}; "
                    f"matches={len(matches)}"
                )
            entry = matches[0]
            if entry.get("private") in (True, "true", "1") or entry.get("no_download"):
                raise AcquisitionError("Archive selected asset is private or restricted")
            if str(entry.get("size")) != str(candidate.expected_bytes):
                raise AcquisitionError(
                    f"Archive selected asset size changed: expected {candidate.expected_bytes}, "
                    f"listed {entry.get('size')}"
                )
            asset_url = f"https://archive.org/download/{identifier}/" + urllib.parse.quote(
                entry["name"], safe=""
            )
            download = self.transport.get(asset_url, limit=MAX_ASSET_BYTES)
            _response(download, limit=MAX_ASSET_BYTES, kind="asset")
            magic = b"PK\x03\x04" if candidate.format == "epub" else b"%PDF-"
            if not download.body.startswith(magic):
                raise AcquisitionError("downloaded candidate is not the declared EPUB/PDF format")
            actual_bytes = len(download.body)
            actual_hash = hashlib.sha256(download.body).hexdigest()
            if actual_bytes != candidate.expected_bytes or actual_hash != candidate.expected_sha256:
                raise AcquisitionError(
                    "selected candidate bytes/hash changed: "
                    f"expected {candidate.expected_bytes}/{candidate.expected_sha256}, "
                    f"observed {actual_bytes}/{actual_hash}"
                )
            timestamp = datetime.now(UTC).isoformat()
            metadata_digest = hashlib.sha256(metadata.body).hexdigest()
            source = BookSource(
                candidate.title,
                candidate.author,
                candidate.edition_identity,
                download.final_url,
                "archive_candidate",
                "reviewed_registry",
                timestamp,
                actual_hash,
                candidate.format,
            )
            warnings = (
                "community-uploaded candidate; publisher authenticity and permissions unverified",
            )
            manifest = {
                "schema": "clipper-book-acquisition-v1",
                "candidate_fingerprint": key,
                "record_url": candidate.record_url,
                "asset_url": download.final_url,
                "asset_requested_url": asset_url,
                "http_status": download.status,
                "fetched_at": timestamp,
                "metadata_sha256": metadata_digest,
                "observed_bytes": actual_bytes,
                "observed_sha256": actual_hash,
                "source": asdict(source),
                "authorization_status": "unverified",
                "edition_authenticated": False,
                "warnings": list(warnings),
            }
            _private_write(raw_path, download.body)
            if not manifest_path.is_file():
                try:
                    _private_write(
                        manifest_path,
                        (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(),
                    )
                except AcquisitionError:
                    # A concurrent identical fetch may have committed its own first timestamp.
                    if not manifest_path.is_file():
                        raise
            canonical = self._cached(candidate)
            if canonical.metadata_sha256 != metadata_digest:
                raise AcquisitionError(
                    "Archive metadata changed relative to immutable acquisition; review required"
                )
            if refresh or canonical.fetched_at != timestamp:
                event = {
                    "schema": "clipper-book-acquisition-refresh-v1",
                    "candidate_fingerprint": key,
                    "fetched_at": timestamp,
                    "record_url": metadata.final_url,
                    "asset_requested_url": asset_url,
                    "asset_url": download.final_url,
                    "metadata_sha256": metadata_digest,
                    "observed_bytes": actual_bytes,
                    "observed_sha256": actual_hash,
                    "http_status": download.status,
                }
                event_bytes = (
                    json.dumps(event, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
                ).encode()
                event_key = hashlib.sha256(event_bytes).hexdigest()
                _private_write(self.root / "events" / key / f"{event_key}.json", event_bytes)
            return AcquiredBook(
                canonical.source,
                candidate,
                canonical.record_url,
                canonical.asset_url,
                canonical.fetched_at,
                canonical.http_status,
                raw_path,
                actual_bytes,
                actual_hash,
                canonical.metadata_sha256,
                False,
                canonical.warnings,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            report = (
                json.dumps(
                    {
                        "candidate": candidate.book_id,
                        "expected_sha256": candidate.expected_sha256,
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            incident = hashlib.sha256(report).hexdigest()
            _private_write(self.root / "errors" / f"{key}-{incident}.json", report)
            if isinstance(exc, AcquisitionError):
                raise
            raise AcquisitionError(f"Archive candidate metadata/acquisition failed: {exc}") from exc

"""No-network reviewed candidate discovery and anonymous Archive transport tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books.acquisition import (
    AcquisitionError,
    ArchiveAcquirer,
    ArchiveHTTPTransport,
    HTTPResult,
    ReviewedCandidate,
    _check_public_dns,
    _validate_url,
    discover,
    load_registry,
)
from clipper.books.contracts import SourceUnresolved

ROOT = Path(__file__).resolve().parents[1]


def _sample() -> tuple[ReviewedCandidate, bytes, dict[str, object]]:
    source = next(x for x in load_registry(ROOT / "niches/book-source-candidates-v1.json")
                  if x.book_id == "atomic-habits")
    raw = b"PK\x03\x04" + b"original tiny synthetic EPUB fixture"
    candidate = replace(
        source,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_bytes=len(raw),
    )
    record = {
        "metadata": {"identifier": candidate.record_url.rsplit("/", 1)[-1]},
        "files": [{"name": candidate.filename, "size": str(len(raw)), "source": "original"}],
    }
    return candidate, raw, record


class FixtureTransport:
    def __init__(self, metadata: dict[str, object], asset: bytes) -> None:
        self.metadata = metadata
        self.asset = asset
        self.calls: list[tuple[str, int]] = []
        self.asset_status = 200
        self.asset_type = "application/epub+zip"
        self.asset_final: str | None = None

    def get(self, url: str, *, limit: int) -> HTTPResult:
        self.calls.append((url, limit))
        if "/metadata/" in url:
            return HTTPResult(
                url, url, 200, "application/json",
                json.dumps(self.metadata).encode("utf-8"),
            )
        return HTTPResult(
            url, self.asset_final or url, self.asset_status, self.asset_type,
            self.asset,
        )


def test_reviewed_registry_title_only_and_explicit_edition() -> None:
    items = load_registry(ROOT / "niches/book-source-candidates-v1.json")
    assert len(items) == 3
    for title, book_id in [
        ("Atomic Habits", "atomic-habits"),
        ("The Psychology of Money", "psychology-of-money"),
        ("Deep Work", "deep-work"),
    ]:
        selected = discover(title.upper(), items)
        assert isinstance(selected, ReviewedCandidate)
        assert selected.book_id == book_id
        assert selected.expected_sha256
        assert selected.fingerprint == selected.fingerprint
        assert selected.edition_identity
    assert isinstance(discover("Atomic Habit", items), SourceUnresolved)
    assert isinstance(discover("Unknown Book", items), SourceUnresolved)
    assert isinstance(discover("Atomic Habits", items, author="Other"), SourceUnresolved)
    assert isinstance(discover("Deep Work", items, edition="US edition"), SourceUnresolved)
    collision = replace(items[0], edition_identity="alternative-edition")
    ambiguous = discover("Atomic Habits", (*items, collision))
    assert isinstance(ambiguous, SourceUnresolved)
    assert "alternative-edition" in ambiguous.reason
    assert "select one" in ambiguous.required_owner_decision


def test_registry_rejects_unsupported_schema_and_unsafe_selected_file(tmp_path: Path) -> None:
    original = json.loads((ROOT / "niches/book-source-candidates-v1.json").read_text())
    for mutation, error in [
        ({"schema": "wrong", "books": original["books"]}, "unknown"),
        ({"schema": original["schema"], "books": []}, "no candidates"),
        ({"schema": original["schema"], "books": ["corrupt"]}, "invalid"),
        ({"schema": original["schema"], "books": [{**original["books"][0], "filename": "../x"}]},
         "invalid or duplicate"),
        ({"schema": original["schema"], "books": [{**original["books"][0], "expected_sha256": "0"}]},
         "invalid or duplicate"),
        ({"schema": original["schema"], "books": [original["books"][0], original["books"][0]]},
         "invalid or duplicate"),
    ]:
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(mutation))
        with pytest.raises(AcquisitionError, match=error):
            load_registry(path)
    with pytest.raises(AcquisitionError, match="unreadable"):
        load_registry(tmp_path / "absent.json")


def test_acquisition_records_actual_url_and_immutable_private_cache(tmp_path: Path) -> None:
    candidate, raw, metadata = _sample()
    transport = FixtureTransport(metadata, raw)
    transport.asset_final = "https://ia801234.us.archive.org/asset-source/epub"
    service = ArchiveAcquirer(tmp_path, transport)
    acquired = service.acquire(candidate)
    assert acquired.source.source_kind == "archive_candidate"
    assert acquired.source.authorization_status == "unverified"
    assert acquired.source.authorization_basis is None
    assert acquired.source.edition_identity == candidate.edition_identity
    assert acquired.observed_sha256 == candidate.expected_sha256
    assert acquired.asset_url == transport.asset_final
    assert acquired.http_status == 200
    assert acquired.raw_path.read_bytes() == raw
    assert acquired.raw_path.stat().st_mode & 0o077 == 0
    assert len(transport.calls) == 2
    assert transport.calls[0][0] == candidate.record_url
    assert "Atomic%20Habits%20by%20James%20Clear.epub" in transport.calls[1][0]
    assert "not_authenticated" not in acquired.source.source_kind
    assert acquired.fingerprint == acquired.fingerprint
    offline = ArchiveAcquirer(tmp_path, Mock())
    reused = offline.acquire(candidate, offline=True)
    assert reused.cache_hit is True
    assert reused.observed_sha256 == acquired.observed_sha256
    assert "not revalidated" in reused.warnings[0]
    acquired.raw_path.write_bytes(raw + b"x")
    with pytest.raises(AcquisitionError, match="changed"):
        offline.acquire(candidate, offline=True)


def test_acquisition_rejects_metadata_disappearance_size_and_restrictions(tmp_path: Path) -> None:
    candidate, raw, metadata = _sample()
    malformed: list[tuple[dict[str, object], str]] = [
        ({**metadata, "is_dark": True}, "restricted"),
        ({**metadata, "is_restricted": True}, "restricted"),
        ({**metadata, "files": []}, "missing or ambiguous"),
        ({**metadata, "files": metadata["files"] * 2}, "missing or ambiguous"),
        ({**metadata, "files": [{**metadata["files"][0], "size": "3"}]}, "size changed"),
        ({**metadata, "files": [{**metadata["files"][0], "private": "true"}]}, "private"),
        ({**metadata, "metadata": {"identifier": "other"}}, "identifier mismatch"),
        ({**metadata, "files": None}, "listing missing"),
    ]
    for index, (payload, error) in enumerate(malformed):
        transport = FixtureTransport(payload, raw)
        with pytest.raises(AcquisitionError, match=error):
            ArchiveAcquirer(tmp_path / str(index), transport).acquire(candidate)
        assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("body", "kind", "status", "error"),
    [
        (b"<html>an error</html>", "text/html", 200, "HTML"),
        (b"not an EPUB", "application/epub+zip", 200, "declared EPUB"),
        (b"PK\x03\x04wrong bytes", "application/zip", 200, "bytes/hash changed"),
        (b"PK\x03\x04gone", "application/octet-stream", 404, "HTTP status 404"),
    ],
)
def test_asset_errors_stop_without_substitution(
    tmp_path: Path, body: bytes, kind: str, status: int, error: str
) -> None:
    candidate, _raw, metadata = _sample()
    transport = FixtureTransport(metadata, body)
    transport.asset_type = kind
    transport.asset_status = status
    with pytest.raises(AcquisitionError, match=error):
        ArchiveAcquirer(tmp_path, transport).acquire(candidate)
    assert len(transport.calls) == 2
    assert not list((tmp_path / "raw").glob("**/*.epub"))
    errors = list((tmp_path / "errors").glob("*.json"))
    assert len(errors) == 1
    assert "error_type" in json.loads(errors[0].read_text())


def test_bad_metadata_http_html_and_offline_cache_absence(tmp_path: Path) -> None:
    candidate, raw, metadata = _sample()
    with pytest.raises(AcquisitionError, match="offline candidate cache is absent"):
        ArchiveAcquirer(tmp_path, Mock()).acquire(candidate, offline=True)
    transport = FixtureTransport(metadata, raw)
    transport.get = Mock(return_value=HTTPResult(
        candidate.record_url, candidate.record_url, 429, "text/html",
        b"<html>throttled</html>",
    ))
    with pytest.raises(AcquisitionError, match="HTTP status 429"):
        ArchiveAcquirer(tmp_path / "limited", transport).acquire(candidate)
    assert transport.get.call_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://archive.org/metadata/x",
        "file:///etc/passwd",
        "https://localhost/metadata/x",
        "https://127.0.0.1/metadata/x",
        "https://evil.org/metadata/x",
        "https://archive.org:444/metadata/x",
        "https://u:p@archive.org/metadata/x",
        "https://archive.org/a/../metadata/x",
        "https://archive.org/a/%2e%2e/b",
        "https://archive.org/file#fragment",
    ],
)
def test_archive_https_allowlist_rejects_unsafe_input(url: str) -> None:
    with pytest.raises(AcquisitionError):
        _validate_url(url)


def test_redirect_hosts_and_private_dns_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _validate_url("https://ia800123.us.archive.org/download/item", redirect=True)
    for url in (
        "https://assets.evil.org/x",
        "https://ia800123.evil.archive.org/x",
        "http://ia800123.us.archive.org/x",
    ):
        with pytest.raises(AcquisitionError):
            _validate_url(url, redirect=True)
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda *_args, **_kwargs: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(AcquisitionError, match="unsafe/private address"):
        _check_public_dns("archive.org")


def test_real_transport_adapter_only_uses_mocked_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise transport and redirect policies without a socket or live HTTP."""

    class FakeReply:
        status = 200

        def __enter__(self) -> FakeReply:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://ia801234.us.archive.org/download/test"

        headers = {"Content-Type": "application/epub+zip", "Content-Length": "5"}

        def read(self, count: int) -> bytes:
            return b"PK\x03\x04X"[:count]

    class FakeOpener:
        def open(self, request: object, *, timeout: int) -> FakeReply:
            assert getattr(request, "full_url") == "https://archive.org/download/test"
            assert timeout == 20
            return FakeReply()

    monkeypatch.setattr("clipper.books.acquisition._check_public_dns", lambda _host: None)
    monkeypatch.setattr("urllib.request.build_opener", lambda *_handlers: FakeOpener())
    result = ArchiveHTTPTransport().get("https://archive.org/download/test", limit=16)
    assert result.status == 200
    assert result.body == b"PK\x03\x04X"
    assert result.final_url.startswith("https://ia801234.us.archive.org/")

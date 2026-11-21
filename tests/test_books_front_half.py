"""Offline Gutenberg -> BookText -> mined candidate catalog -> legacy selector."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper import cli
from clipper.books import ingest, mining, scripting, selection
from clipper.books.contracts import BookSection, BookSource, BookText
from clipper.books.orchestrator import BookOrchestrator, BookRunError
from clipper.stage_contracts import content_fingerprint

HTML = (
    b'<li class="booklink"><a href="/ebooks/1342">'
    b'<span class="title">Synthetic Complete Book</span>'
    b'<span class="subtitle">by Fixture Author</span></a></li>'
)
TITLE = "Synthetic Complete Book"
AUTHOR = "Fixture Author"


def _complete_book(*, altered: bool = False, rights: bool = True) -> bytes:
    header = f"Title: {TITLE}\nAuthor: {AUTHOR}\nEdition: gutenberg-1342\nEBook #1342\n"
    if rights:
        header += "Rights: Public domain in the USA\n"
    header += "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    paragraph = " ".join(
        ["A fabricated line about everyday choices serves only as a test source."] * 75
    )
    if altered:
        paragraph += " Changed source file identity."
    for index in range(1, 4):
        header += f"\nCHAPTER {index}\n{paragraph}\n"
    return (header + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n").encode("utf-8")


def _transport(book: bytes) -> Mock:
    def responder(url: str) -> bytes:
        if url.startswith("https://www.gutenberg.org/ebooks/search/?query="):
            return HTML
        if url == "https://www.gutenberg.org/ebooks/1342.txt.utf-8":
            return book
        raise AssertionError(f"unexpected network URL: {url}")

    return Mock(side_effect=responder)


def _completion(request: dict[str, object]) -> str:
    assert request["model"] == scripting.SCRIPT_MODEL
    assert request["revision"] == scripting.SCRIPT_REVISION
    messages = request["messages"]
    assert isinstance(messages, list)
    payload = json.loads(messages[1]["content"])
    chapter, offset = payload["chapter_id"], payload["word_offset"]
    return json.dumps(
        {
            "lessons": [
                {
                    "id": f"lesson-{chapter}-{offset}",
                    "topic": f"One original idea in {chapter}",
                    "claim": "An unreviewed claim about a source-backed example",
                    "hook_angle": "An original short hook",
                    "visual_action": "A character demonstrates a careful choice",
                    "evidence": payload["text"][:48],
                }
            ]
        }
    )


def test_full_offline_gutenberg_ingest_mine_select_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no live network")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    transport = _transport(_complete_book())
    runner = BookOrchestrator(tmp_path)
    book = runner.ingest(TITLE, AUTHOR, ingest.GutenbergResolver(transport))
    assert isinstance(book, BookText)
    assert book.total_word_count >= ingest.MIN_WORDS
    assert book.source.authorization_status == "unverified"
    assert book.source.authorization_basis == "public_domain"
    assert len(book.sections) == 3
    assert runner.load_text("synthetic-complete-book") == book
    assert transport.call_count == 2
    expected = mining.mine_book(book, _completion)
    assert expected == mining.mine_book(book, _completion)
    result = runner.mine(book, completion=_completion, response_sha256="a" * 64)
    assert result == expected
    assert selection._validate_catalog(result)
    assert result["input_kind"] == "mined_separate_unreviewed"
    assert result["review_status"] == "MINED_UNREVIEWED"
    assert result["publish_eligible"] is False
    assert result["books"][0]["authorization_status"] == "unverified"
    assert all(
        item["review_status"] == "MINED_UNREVIEWED" for item in result["books"][0]["lessons"]
    )
    assert all(item["publish_eligible"] is False for item in result["books"][0]["lessons"])
    assert all(
        item["claim_source_url"].startswith(book.source.source_uri + "#chapter-")
        for item in result["books"][0]["lessons"]
    )
    picked = selection.choose_lesson(result, book_id="synthetic-complete-book")
    assert picked["publish_eligible"] is False
    assert picked["lesson"]["review_status"] == "MINED_UNREVIEWED"
    assert picked["lesson"]["publish_eligible"] is False
    resumed = runner.mine(
        book,
        completion=Mock(side_effect=AssertionError("mining producer cannot replay")),
        response_sha256="a" * 64,
    )
    assert resumed == result
    assert len(list((tmp_path / "book-dag/books-mining").glob("*/stage.json"))) == 1


def test_cli_ingest_owner_file_and_reject_unsourced_mining(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "owner.txt"
    text = _complete_book().decode()
    text = text.split("*** START", 1)[0] + text.split("*** START", 1)[1]
    path.write_text(text)
    assert (
        cli.main(
            [
                "books",
                "ingest",
                "--title",
                TITLE,
                "--author",
                AUTHOR,
                "--source",
                str(path),
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["authorization_status"] == "unverified"
    assert report["book_id"] == "synthetic-complete-book"
    assert report["total_words"] >= ingest.MIN_WORDS
    runner = BookOrchestrator(tmp_path / "artifacts")
    local = runner.load_text("synthetic-complete-book")
    with pytest.raises(mining.BookMiningError, match="no HTTPS citation URI"):
        mining.mine_book(local, _completion)
    responses = tmp_path / "responses.json"
    responses.write_text("{}")
    assert (
        cli.main(
            [
                "books",
                "mine",
                "--book-id",
                "synthetic-complete-book",
                "--responses",
                str(responses),
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ]
        )
        == 1
    )
    assert "no HTTPS citation URI" in caplog.text


def test_cli_gutenberg_fixture_mining_uses_recorded_responses_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = BookOrchestrator(tmp_path)
    book = runner.ingest(TITLE, AUTHOR, ingest.GutenbergResolver(_transport(_complete_book())))
    assert isinstance(book, BookText)
    replies = {
        content_fingerprint(mining.mining_request(book, chunk)): json.loads(
            _completion(mining.mining_request(book, chunk))
        )
        for chunk in mining.chunks_for(book)
    }
    file = tmp_path / "recorded.json"
    file.write_text(json.dumps(replies), encoding="utf-8")
    assert (
        cli.main(
            [
                "books",
                "mine",
                "--book-id",
                "synthetic-complete-book",
                "--responses",
                str(file),
                "--artifact-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["review_status"] == "MINED_UNREVIEWED"
    saved = json.loads((tmp_path / "mined/synthetic-complete-book.json").read_text())
    assert saved["publish_eligible"] is False
    assert selection.choose_lesson(saved)["lesson"]["review_status"] == "MINED_UNREVIEWED"


def test_response_validation_fails_closed_and_does_not_invent_evidence() -> None:
    chunk = mining.BookChunk("chapter-01", 1, 0, "This is an exact evidence sentence.")
    original = {
        "id": "one-idea",
        "topic": "Useful lesson",
        "claim": "A candidate claim",
        "hook_angle": "Start here",
        "visual_action": "Host opens a book",
        "evidence": "exact evidence",
    }
    assert mining._validate_response({"lessons": [original]}, chunk)[0] == original
    cases = [
        ({"wrong": []}, "only lessons"),
        ({"lessons": "bad"}, "0-3 lessons"),
        ({"lessons": [{}]}, "fields"),
        ({"lessons": [{**original, "claim": ""}]}, "bounded nonempty"),
        ({"lessons": [{**original, "id": "not a slug"}]}, "safe slug"),
        ({"lessons": [{**original, "evidence": "fabricated"}]}, "exact anchored"),
    ]
    for raw, reason in cases:
        with pytest.raises(mining.BookMiningError, match=reason):
            mining._validate_response(raw, chunk)


def test_mining_rejects_bad_completions_duplicates_and_unauthorized_sources() -> None:
    raw = _complete_book()
    book = ingest.ingest_book(TITLE, AUTHOR, ingest.GutenbergResolver(_transport(raw)))
    assert isinstance(book, BookText)
    with pytest.raises(mining.BookMiningError, match="valid JSON"):
        mining.mine_book(book, lambda _request: "{broken")
    with pytest.raises(mining.BookMiningError, match="no candidate lessons"):
        mining.mine_book(book, lambda _request: '{"lessons":[]}')
    duplicate = json.loads(_completion(mining.mining_request(book, mining.chunks_for(book)[0])))
    duplicate["lessons"][0]["id"] = "duplicate"
    with pytest.raises(mining.BookMiningError, match="duplicate mined lesson"):
        mining.mine_book(book, lambda _request: json.dumps(duplicate))
    bad = replace(book, source=replace(book.source, authorization_status="auto_approved"))
    with pytest.raises(mining.BookMiningError, match="upgrade or accept"):
        mining.mine_book(bad, _completion)


def test_chunk_capacity_and_empty_book_fail_closed() -> None:
    source = BookSource(
        TITLE,
        AUTHOR,
        None,
        "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
        "public_domain",
        "fixture",
        "",
        "e" * 64,
        "txt",
    )
    empty = BookText(source, None, (), 0, "e" * 64, source.fingerprint)
    with pytest.raises(mining.BookMiningError, match="nonempty ordered"):
        mining.chunks_for(empty)
    huge = BookText(
        source,
        None,
        (BookSection("chapter-01", "Long", ("long " * (350 * 130),)),),
        350 * 130,
        "e" * 64,
        source.fingerprint,
    )
    with pytest.raises(mining.BookMiningError, match="chunk capacity exceeded"):
        mining.chunks_for(huge)


def test_ingest_changes_rekey_dag_and_mining_without_provider_use(tmp_path: Path) -> None:
    runner = BookOrchestrator(tmp_path)
    path = tmp_path / "synthetic.txt"
    path.write_bytes(_complete_book())
    first = runner.ingest(TITLE, AUTHOR, ingest.LocalFileResolver(path))
    path.write_bytes(_complete_book(altered=True))
    second = runner.ingest(TITLE, AUTHOR, ingest.LocalFileResolver(path))
    assert isinstance(first, BookText) and isinstance(second, BookText)
    assert first.content_sha256 != second.content_sha256
    assert first.fingerprint != second.fingerprint
    records = list((tmp_path / "book-dag/books-ingest").glob("*/stage.json"))
    assert len(records) == 2
    assert all(json.loads(path.read_text())["status"] == "PASS" for path in records)


def test_mining_refuses_unrecorded_response_and_bad_catalog_snapshot(tmp_path: Path) -> None:
    runner = BookOrchestrator(tmp_path)
    book = runner.ingest(TITLE, AUTHOR, ingest.GutenbergResolver(_transport(_complete_book())))
    assert isinstance(book, BookText)
    with pytest.raises(BookRunError, match="exact recorded-response"):
        runner.mine(book, completion=_completion, response_sha256="not-a-digest")
    path = tmp_path / "book-text/synthetic-complete-book.json"
    document = json.loads(path.read_text())
    document["total_word_count"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(BookRunError, match="word-count mismatch"):
        runner.load_text("synthetic-complete-book")


def test_mining_chunk_fingerprint_and_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    book = ingest.ingest_book(TITLE, AUTHOR, ingest.GutenbergResolver(_transport(_complete_book())))
    assert isinstance(book, BookText)
    chunk = mining.chunks_for(book)[0]
    assert len(chunk.fingerprint) == 64
    monkeypatch.setattr(mining, "MAX_LESSONS", 2)
    with pytest.raises(mining.BookMiningError, match="mined lesson limit exceeded"):
        mining.mine_book(book, _completion)

"""Deterministic, provenance-anchored lesson mining; injected local completion only."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from clipper.stage_contracts import content_fingerprint

from . import scripting, selection
from .contracts import BookText

MAX_CHUNKS = 128
CHUNK_WORDS = 350
MAX_LESSONS = 64


class BookMiningError(ValueError):
    """Candidate extraction or review-boundary validation failed closed."""


@dataclass(frozen=True, slots=True)
class BookChunk:
    chapter_id: str
    paragraph_index: int
    word_offset: int
    text: str

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(
            {
                "chapter": self.chapter_id,
                "paragraph": self.paragraph_index,
                "offset": self.word_offset,
                "text": self.text,
            }
        )


def chunks_for(text: BookText) -> tuple[BookChunk, ...]:
    chunks: list[BookChunk] = []
    for chapter in text.sections:
        for index, paragraph in enumerate(chapter.paragraphs, start=1):
            words = paragraph.split()
            for offset in range(0, len(words), CHUNK_WORDS):
                chunks.append(
                    BookChunk(chapter.chapter_id, index, offset, " ".join(words[offset:offset + CHUNK_WORDS]))
                )
                if len(chunks) > MAX_CHUNKS:
                    raise BookMiningError(
                        f"mining chunk capacity exceeded: {len(chunks)} > {MAX_CHUNKS}"
                    )
    if not chunks:
        raise BookMiningError("mining requires nonempty ordered book chapters")
    return tuple(chunks)


def mining_request(book: BookText, chunk: BookChunk) -> dict[str, object]:
    """Use the same pinned request-envelope abstraction as the Qwen script stage."""
    payload = {
        "book_title": book.source.title,
        "book_author": book.source.author,
        "edition": book.edition_identity,
        "book_sha256": book.content_sha256,
        "chapter_id": chunk.chapter_id,
        "paragraph_index": chunk.paragraph_index,
        "word_offset": chunk.word_offset,
        "text": chunk.text,
    }
    return {
        "model": scripting.SCRIPT_MODEL,
        "revision": scripting.SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 960,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return JSON only: {\\\"lessons\\\": [{\\\"id\\\": str, \\\"topic\\\": str, "
                    "\\\"claim\\\": str, \\\"hook_angle\\\": str, \\\"visual_action\\\": str, "
                    "\\\"evidence\\\": str}]}. No invented quotes or statistics. "
                    "Evidence MUST be an exact continuous substring of the supplied text. "
                    "Propose original, unreviewed candidate lessons only, never approved facts."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
    }


def _validate_response(
    payload: object, chunk: BookChunk
) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or set(payload) != {"lessons"}:
        raise BookMiningError("mining response must contain only lessons")
    lessons = payload["lessons"]
    if not isinstance(lessons, list) or len(lessons) > 3:
        raise BookMiningError("mining response requires 0-3 lessons per chunk")
    results: list[dict[str, str]] = []
    for item in lessons:
        keys = {"id", "topic", "claim", "hook_angle", "visual_action", "evidence"}
        if not isinstance(item, dict) or set(item) != keys:
            raise BookMiningError("mined lesson fields are incomplete or unexpected")
        if any(
            not isinstance(item[key], str) or not item[key].strip()
            or len(item[key]) > 400
            for key in keys
        ):
            raise BookMiningError("mined lesson fields must be bounded nonempty text")
        if not selection.IDENTIFIER.fullmatch(item["id"]):
            raise BookMiningError("mined lesson id must be a safe slug")
        if item["evidence"] not in chunk.text:
            raise BookMiningError("mined evidence is not an exact anchored book substring")
        results.append(item)
    return results


def mine_book(
    text: BookText,
    completion: Callable[[dict[str, object]], str],
) -> dict[str, Any]:
    """Produce separately marked unreviewed catalog; never merge into curated input."""
    source = text.source
    if source.authorization_status != "unverified":
        raise BookMiningError("mining cannot upgrade or accept asserted source authorization")
    if not selection._source_url(source.source_uri):
        raise BookMiningError(
            "local book has no HTTPS citation URI; owner must supply independently verified "
            "public evidence before catalog export"
        )
    parts = re.findall(r"[a-z0-9]+", source.title.casefold())
    book_id = "-".join(parts)
    if not book_id or not selection.IDENTIFIER.fullmatch(book_id):
        raise BookMiningError("source book title cannot form a stable catalog book id")
    mined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks_for(text):
        request = mining_request(text, chunk)
        raw = completion(request)
        try:
            candidate: object = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise BookMiningError("mining completion is not valid JSON") from exc
        for item in _validate_response(candidate, chunk):
            if item["id"] in seen:
                raise BookMiningError(f"duplicate mined lesson id: {item['id']}")
            seen.add(item["id"])
            anchor = f"{chunk.chapter_id}-p{chunk.paragraph_index:04d}"
            mined.append(
                {
                    "id": item["id"],
                    "topic": item["topic"],
                    "claim": item["claim"],
                    "claim_source_url": f"{source.source_uri}#{anchor}",
                    "visual_action": item["visual_action"],
                    "hook_angle": item["hook_angle"],
                    "review_status": "MINED_UNREVIEWED",
                    "publish_eligible": False,
                    "evidence_anchor": {
                        "chapter_id": chunk.chapter_id,
                        "paragraph_index": chunk.paragraph_index,
                        "word_offset": chunk.word_offset,
                        "evidence": item["evidence"],
                        "source_sha256": text.content_sha256,
                    },
                }
            )
            if len(mined) > MAX_LESSONS:
                raise BookMiningError(f"mined lesson limit exceeded: {len(mined)} > {MAX_LESSONS}")
    if not mined:
        raise BookMiningError("mining returned no candidate lessons; human review required")
    catalog = {
        "schema": "clipper-book-editorial-catalog-v1",
        "input_kind": "mined_separate_unreviewed",
        "review_status": "MINED_UNREVIEWED",
        "publish_eligible": False,
        "ingest_fingerprint": text.fingerprint,
        "books": [
            {
                "id": book_id,
                "title": source.title,
                "author": source.author,
                "pillar": "self_improvement",
                "source_url": source.source_uri,
                "status": "source_verified_candidate",
                "review_status": "MINED_UNREVIEWED",
                "publish_eligible": False,
                "source_fingerprint": source.fingerprint,
                "authorization_status": source.authorization_status,
                "lessons": mined,
            }
        ],
    }
    selection._validate_catalog(catalog)
    return catalog

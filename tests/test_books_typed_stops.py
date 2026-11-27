"""Lane-2 untrusted-input probes (synthetic text only)."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pytest
from test_books_chapter import _adapt, _plan, _script
from test_books_chapter_hybrid import _ledger, _original

from clipper.books import chapter
from clipper.books import chapter_hybrid as hybrid
from clipper.books.model_profile import QWEN3_1_7B, ModelProfileError


def _req(payload: dict[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": json.dumps({"input": payload})},
        ]
    }


def _valid() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    beats = {"beats": [{"beat_id": f"beat-{i:02d}"} for i in range(1, 6)]}
    script = json.loads(_script(_req({"adaptation": beats})))
    extra = {
        "source": {"authorization_status": "unverified"},
        "human_review": {"reviewer": "synthetic-human-reviewer"},
        "brief": beats,
    }
    adaptation = json.loads(_adapt(_req(extra)))
    reviews = {f"beat-{i:02d}": {"qualifier_quote": ""} for i in range(1, 6)}
    return script, adaptation, reviews


@pytest.mark.parametrize("bad", [None, 1, "x", []])
def test_freeze_rejects_non_object_source_entry_with_typed_stop(bad: object) -> None:
    with pytest.raises(hybrid.HybridStop):
        hybrid.freeze(units=[bad], gaps=[], reading_order=["synthetic-1"], identity={})  # type: ignore[list-item]


def test_freeze_rejects_unhashable_traversal_id_with_typed_stop() -> None:
    units, gaps, _ = _original(2)
    with pytest.raises(hybrid.HybridStop):
        hybrid.freeze(units, gaps, [["a"], "b", "c"], {})  # type: ignore[list-item]


@pytest.mark.parametrize("field", ["kind", "origin", "review_status", "research_link_status"])
def test_validate_ledger_rejects_entry_missing_field_with_typed_stop(field: str) -> None:
    ledger = copy.deepcopy(_ledger(3))
    del ledger["entries"][0][field]  # type: ignore[index]
    ledger["ledger_sha256"] = hybrid.digest(
        {k: v for k, v in ledger.items() if k != "ledger_sha256"}
    )
    with pytest.raises(hybrid.HybridStop):
        entries = hybrid.validate_ledger(ledger)
        hybrid.projection(entries)


def test_script_validate_probe_valid_baseline() -> None:
    script, adaptation, reviews = _valid()
    chapter._script_validate(script, adaptation, reviews, _plan())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("environment_id", []),
        ("illustration_label", {}),
        ("beat_ids", [[]]),
        ("speaker_id", []),
    ],
)
def test_script_validate_unhashable_ids_raise_chapter_stop(path: str, value: object) -> None:
    script, adaptation, reviews = _valid()
    scene = script["scenes"][0]
    if path == "speaker_id":
        for s in script["scenes"]:
            s["illustration_label"] = "INVENTED_ILLUSTRATION"
        scene["dialogue"] = [
            {
                "speaker_id": value,
                "text": "synthetic words here",
                "fiction_label": "INVENTED_DIALOGUE",
            }
        ]
    else:
        scene[path] = value
    with pytest.raises(chapter.ChapterStop):
        chapter._script_validate(script, adaptation, reviews, _plan())


@pytest.mark.parametrize("field", ["name", "revision", "model", "tokenizer_json_sha256"])
@pytest.mark.parametrize("bad", [None, [], 5])
def test_profile_rejects_wrong_type_with_model_profile_error(field: str, bad: object) -> None:
    with pytest.raises(ModelProfileError):
        replace(QWEN3_1_7B, **{field: bad})


def _synthetic_epub_page_sections() -> tuple[Any, ...]:
    import zipfile
    from io import BytesIO

    from clipper.books import ingest

    page = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Synthetic chapter</h1>'
        "<p>" + "Synthetic paragraph. " * 100 + "</p>"
        "<ol><li>Critical synthetic exception</li></ol>"
        "<figure><figcaption>Synthetic caption text</figcaption></figure>"
        "<table><tr><td>Synthetic cell value</td></tr></table></body></html>"
    )
    container = (
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="content.opf"/></rootfiles></container>'
    )
    opf = (
        '<package xmlns="http://www.idpf.org/2007/opf"><metadata>'
        "<title>T</title><creator>A</creator></metadata>"
        '<manifest><item id="c1" href="c1.xhtml"/></manifest>'
        '<spine><itemref idref="c1"/></spine></package>'
    )
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("content.opf", opf)
        zf.writestr("c1.xhtml", page)
    return ingest._epub(buf.getvalue())[3]

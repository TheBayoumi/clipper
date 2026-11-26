"""Test-only public synthetic source and hybrid producer fixtures.

No imports from broad test modules, CLI, pipeline, media, host or model
entrypoints. This module is intentionally outside the production package.
"""

from __future__ import annotations

import hashlib
import io
import json
import wave
import zipfile
from dataclasses import replace
from pathlib import Path

from clipper.books import acquisition, chapter_hybrid, evaluation, selected_scope, whole_book
from clipper.books.orchestrator import BookOrchestrator

BASE = Path(__file__).resolve().parents[1] / "niches/book-source-candidates-v1.json"


def synthetic_candidate(
    tmp_path: Path, *, paragraphs: int = 129, with_image: bool = True
) -> tuple[acquisition.ReviewedCandidate, object]:
    """All prose and image bytes are original synthetic fixtures, never book extracts."""
    candidate = next(
        item for item in acquisition.load_registry(BASE) if item.book_id == "atomic-habits"
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<container><rootfile full-path="OPS/content.opf"/></container>',
        )
        manifest = "".join(
            f'<item id="c{i}" href="chapter{i}.xhtml" media-type="application/xhtml+xml"/>'
            for i in range(3)
        )
        z.writestr(
            "OPS/content.opf",
            '<package xmlns:dc="http://purl.org/dc/elements/1.1/" unique-identifier="uid">'
            f"<metadata><dc:title>{candidate.title}</dc:title>"
            f"<dc:creator>{candidate.author}</dc:creator>"
            f'<dc:identifier id="uid">urn:isbn:{candidate.isbn}</dc:identifier>'
            "</metadata><manifest>"
            + manifest
            + ('<item id="pic" href="figure.png" media-type="image/png"/>' if with_image else "")
            + "</manifest><spine>"
            + "".join(f'<itemref idref="c{i}"/>' for i in range(3))
            + "</spine></package>",
        )
        for chapter in range(3):
            parts = [f"<h2>Fabricated heading {chapter}</h2>"]
            for index in range(chapter, paragraphs, 3):
                parts.append(
                    f"<p>Invented paragraph {index} describes a reproducible "
                    "synthetic learning example, without borrowing text.</p>"
                )
            if with_image and chapter == 1:
                parts.append('<img src="figure.png" alt="diagram is not extracted words"/>')
            z.writestr(
                f"OPS/chapter{chapter}.xhtml",
                "<html><body>" + "".join(parts) + "</body></html>",
            )
        if with_image:
            z.writestr("OPS/figure.png", b"\x89PNG\r\n\x1a\nSYNTHETIC-ONLY")
    raw = stream.getvalue()
    selected = replace(
        candidate,
        expected_bytes=len(raw),
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )

    class RecordedTransport:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str, *, limit: int) -> acquisition.HTTPResult:
            self.calls += 1
            if "/metadata/" in url:
                metadata = {
                    "metadata": {"identifier": selected.record_url.rsplit("/", 1)[-1]},
                    "files": [
                        {
                            "name": selected.filename,
                            "size": str(selected.expected_bytes),
                            "source": "original",
                        }
                    ],
                }
                return acquisition.HTTPResult(
                    url, url, 200, "application/json", json.dumps(metadata).encode()
                )
            assert url.startswith("https://archive.org/download/")
            assert limit >= len(raw)
            return acquisition.HTTPResult(
                url,
                "https://ia801234.us.archive.org/synthetic/whole-epub",
                200,
                "application/epub+zip",
                raw,
            )

    return selected, RecordedTransport()


def ingested(tmp_path: Path, *, paragraphs: int = 129, with_image: bool = True):
    candidate, transport = synthetic_candidate(
        tmp_path, paragraphs=paragraphs, with_image=with_image
    )
    runner = BookOrchestrator(tmp_path / "artifacts")
    ingest_result = runner.ingest_candidate(
        candidate.title, registry=(candidate,), transport=transport
    )
    assert not isinstance(ingest_result, tuple)
    assert transport.calls == 2
    return runner, ingest_result, transport


def _bundle(
    tmp_path: Path, *, with_image: bool = True, section_index: int = 0
) -> tuple[BookOrchestrator, Path, dict]:
    owner = tmp_path / "owner"
    owner.mkdir()
    exporter, acquired, _ = ingested(owner, paragraphs=8, with_image=with_image)
    section = acquired.book.sections[section_index].chapter_id
    original_plan = whole_book.plan_for(acquired.book, section_ids=(section,))
    filename, digest = selected_scope.export_chapter(
        exporter,
        acquired.analysis_ref_path,
        chapter_number=section_index + 1,
        chapter_section_ids=(section,),
    )
    assert hashlib.sha256(filename.read_bytes()).hexdigest() == digest
    target_root = tmp_path / "isolated-host"
    target = target_root / "book-candidates" / "selected-scopes" / f"{digest}.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(filename.read_bytes())
    runner = BookOrchestrator(target_root)
    imported = selected_scope.import_chapter(runner.root, target)
    assert imported.work == original_plan
    assert imported.book.fingerprint == acquired.book.fingerprint
    assert imported.book.parent_original_reverified_on_host is False
    assert imported.bundle_sha256 == digest
    assert not list(target_root.rglob("*.epub"))
    assert not list(target_root.rglob("analysis-refs"))
    assert not list(target_root.rglob("extracted"))
    return runner, target, json.loads(target.read_bytes())


def _selected_hybrid_source(
    tmp_path: Path, *, section_index: int = 0
) -> tuple[BookOrchestrator, Path, Path]:
    runner, selected, _data = _bundle(tmp_path, with_image=True, section_index=section_index)
    verified = selected_scope.import_chapter(runner.root, selected)
    section = selected_scope._section_manifest(verified.ref, verified.book, verified.work)
    path = runner.root / "book-candidates" / "section-scopes" / "chapter.json"
    evaluation._store(runner.root / "book-candidates", path, section)
    return runner, selected, path


def _injected_hybrid_response(kind: str, request: dict[str, object]) -> str:
    context = chapter_hybrid.input_payload(request["messages"])  # type: ignore[arg-type]
    if kind == "argument":
        source = next(item for item in context["original_source"] if item[1] == "text")
        return json.dumps(
            {
                "hypothesis": "The entire chapter remains an editable hypothesis.",
                "questions": ["Does the local evidence narrow the claim?"],
                "source_links": [{"id": source[0], "start": 0, "end": 10, "role": "ASSERTION"}],
            }
        )
    if kind == "audit":
        window = context["window"]
        return json.dumps(
            {
                "audit_id": window["audit_id"],
                "core_ids": window["core_ids"],
                "decisions": [
                    {
                        "id": item[0],
                        "decision": "keep" if item[1] == "text" else "omit",
                        "reason": "Explicit model-free original source or gap decision.",
                        "evidence": [{"id": item[0], "start": 0, "end": 10, "role": "ASSERTION"}]
                        if item[1] == "text"
                        else [],
                    }
                    for item in window["core"]
                ],
                "disagreements": [],
            }
        )
    if kind == "reconcile":
        entries = [item for item in context["complete_original_source"] if item[1] == "text"]
        halfway = max(1, len(entries) // 2)
        groups = [entries[:halfway], entries[halfway:]]
        if not groups[1]:
            groups = [[entries[0]], [entries[0]]]
        return json.dumps(
            {
                "argument": "The whole source is preserved as one revisable causal argument.",
                "amendments": [],
                "resolutions": [],
                "beats": [
                    {
                        "beat_id": f"beat-{index:03d}",
                        "spans": [
                            {"id": entry[0], "start": 0, "end": 10, "role": "ASSERTION"}
                            for entry in group
                        ],
                        "meaning": "A source-grounded, still unreviewed argument.",
                        "qualification": "A limitation expressed as a faithful paraphrase.",
                        "causal_link": "Connect this source with the next original passage.",
                        "essential": True,
                    }
                    for index, group in enumerate(groups, 1)
                ],
            }
        )
    raise AssertionError("unexpected synthetic stage")


def _hybrid_creative_output(kind: str, request: dict[str, object]) -> str:
    data = chapter_hybrid.input_payload(request["messages"])  # type: ignore[arg-type]
    if kind == "adaptation":
        return json.dumps(
            {
                "throughline": "An invented person tests, learns and changes a physical procedure.",
                "narrative_form": "DRAMATIZED_ILLUSTRATION",
                "choice_reason": "Fiction is never mistaken for a source-authenticated author example.",
                "beats": [
                    {
                        "beat_id": beat["beat_id"],
                        "decision": "keep",
                        "reason": "Retain the source-grounded link to the whole argument.",
                        "qualification": "A faithful paraphrase that omits literal source terms.",
                        "illustration": "INVENTED_ILLUSTRATION",
                    }
                    for beat in data["brief"]["beats"]
                ],
                "characters": [
                    {
                        "id": "worker",
                        "name": "Invented worker",
                        "initial_state": "examining the unchanged instructions",
                    }
                ],
                "locations": [
                    {
                        "id": "workroom",
                        "name": "Imaginary workroom",
                        "initial_state": "a bench containing unmodified tools",
                    }
                ],
                "props": [
                    {
                        "id": "card",
                        "name": "Paper checklist",
                        "initial_state": "blank",
                    }
                ],
            }
        )
    if kind == "chapter_script":
        beat_ids = [beat["beat_id"] for beat in data["adaptation"]["beats"]]
        narration = (
            "A worker sees that the shared instruction assumes a result without a test. "
            "She records the original observation before attempting a small change. "
            "The next limited experiment produces a different observation, which narrows "
            "the original claim instead of proving that every circumstance is identical. "
            "The group can now revise its procedure while admitting that other conditions "
            "might require additional verification."
        )
        scenes = []
        for index in range(3):
            scenes.append(
                {
                    "scene_id": f"scene-{index + 1:02d}",
                    "previous_id": f"scene-{index:02d}" if index else None,
                    "next_id": f"scene-{index + 2:02d}" if index < 2 else None,
                    "purpose": "Show one limited example and connect it to the next claim.",
                    "action": "The worker marks a revised step on the checklist.",
                    "environment_id": "workroom",
                    "narration": narration,
                    "dialogue": [],
                    "beat_ids": [beat_ids[index % len(beat_ids)]],
                    "illustration_label": "INVENTED_ILLUSTRATION",
                    "transition": "The prior observation motivates the next test.",
                    "consequence": "The next step changes without asserting universal proof.",
                }
            )
        return json.dumps({"title": "The revised checklist", "scenes": scenes})
    if kind == "chapter_direction":
        shots = []
        before = "blank"
        for index, timing in enumerate(data["timings"]):
            after = f"marked step {index + 1}"
            shots.append(
                {
                    "shot_id": f"shot-{index + 1:03d}",
                    "scene_id": timing["scene_id"],
                    "purpose": "Show a measured revision of the invented checklist.",
                    "action": "The character marks a practical change.",
                    "environment_id": "workroom",
                    "camera": "Medium close shot on the same card and hand.",
                    "transition": "Continue following the checklist into the next test.",
                    "entity_ids": ["card"],
                    "state_before": {"card": before},
                    "state_after": {"card": after},
                    "start_seconds": timing["start_seconds"],
                    "end_seconds": timing["end_seconds"],
                }
            )
            before = after
        return json.dumps({"shots": shots})
    raise AssertionError("unrecognized creative stage")


def _hybrid_wav(_text: str, voice: str) -> bytes:
    assert voice == "af_heart"
    file = io.BytesIO()
    with wave.open(file, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 24000)
    return file.getvalue()

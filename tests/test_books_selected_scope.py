"""Selected original chapter exporter/importer, no private or whole-book transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_books_whole_book import ingested

from clipper.books import chapter, evaluation, selected_scope, whole_book
from clipper.books.orchestrator import BookOrchestrator


def _bundle(tmp_path: Path, *, with_image: bool = True) -> tuple[BookOrchestrator, Path, dict]:
    owner = tmp_path / "owner"
    owner.mkdir()
    exporter, acquired, _ = ingested(owner, paragraphs=8, with_image=with_image)
    section = acquired.book.sections[0].chapter_id
    original_plan = whole_book.plan_for(acquired.book, section_ids=(section,))
    filename, digest = selected_scope.export_chapter(
        exporter,
        acquired.analysis_ref_path,
        chapter_number=1,
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


def _edited(runner: BookOrchestrator, data: dict) -> Path:
    encoded = selected_scope._json(data)
    digest = hashlib.sha256(encoded).hexdigest()
    path = runner.root / "book-candidates" / "selected-scopes" / f"{digest}.json"
    path.write_bytes(encoded)
    return path


def test_selected_chapter_is_private_full_source_order_and_host_requires_no_parent(
    tmp_path: Path,
) -> None:
    runner, path, _ = _bundle(tmp_path)
    verified = selected_scope.import_chapter(runner.root, path)
    assert verified.work.units
    assert all(u.section_id in verified.work.section_ids for u in verified.work.units)
    assert all(
        b.section_id in verified.work.section_ids for b in verified.book.selected_reading_blocks
    )
    assert tuple(u.ordinal for u in verified.work.units) == tuple(
        sorted(u.ordinal for u in verified.work.units)
    )
    assert verified.ref.raw_sha256 == verified.book.content_sha256
    assert verified.book.source_boundary == "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT"
    assert verified.work.parent_total_units > len(verified.work.units)
    assert (
        whole_book.plan_for(verified.book, section_ids=verified.work.section_ids) == verified.work
    )
    with pytest.raises(ValueError):
        whole_book.plan_for(verified.book, section_ids=("other-chapter",))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_file",
        "wrong_digest",
        "noncanonical",
        "raw_parent",
        "source_text",
        "missing_block",
        "reordered_blocks",
        "duplicated_block",
        "out_of_scope_block",
        "missing_unit",
        "reordered_units",
        "bool_ordinal",
        "parent_unit_count",
        "changed_gap",
        "scope_digest",
        "false_parent_verification",
        "fake_approval",
        "invalid_chapter_number",
        "different_parent_fingerprint",
        "duplicate_declared_section",
    ],
)
def test_selected_scope_rejects_canonical_and_rehashed_tampering(
    tmp_path: Path, mutation: str
) -> None:
    runner, path, data = _bundle(tmp_path)
    if mutation == "missing_file":
        path.unlink()
        with pytest.raises(selected_scope.SelectedSourceStop):
            selected_scope.import_chapter(runner.root, path)
        return
    if mutation in {"wrong_digest", "noncanonical"}:
        if mutation == "wrong_digest":
            path.write_bytes(
                path.read_bytes().replace(b"SELECTED_SCOPE_SOURCE_UNREVIEWED", b"UNREVIEWED")
            )
        else:
            path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(selected_scope.SelectedSourceStop):
            selected_scope.import_chapter(runner.root, path)
        return
    if mutation == "duplicate_declared_section":
        data["linked_section_ids"] = [data["chapter_section_ids"][0]]
    elif mutation == "raw_parent":
        data["parent_ref"]["raw_sha256"] = "f" * 64
    elif mutation == "source_text":
        data["reading_blocks"][0]["text"] = "Invented changed source"
    elif mutation == "missing_block":
        del data["reading_blocks"][0]
    elif mutation == "reordered_blocks":
        data["reading_blocks"][0], data["reading_blocks"][1] = (
            data["reading_blocks"][1],
            data["reading_blocks"][0],
        )
    elif mutation == "duplicated_block":
        data["reading_blocks"].insert(1, data["reading_blocks"][0])
    elif mutation == "out_of_scope_block":
        data["reading_blocks"][0]["section_id"] = "chapter-not-selected"
    elif mutation == "missing_unit":
        data["work_plan"]["units"].pop()
    elif mutation == "reordered_units":
        data["work_plan"]["units"][0], data["work_plan"]["units"][1] = (
            data["work_plan"]["units"][1],
            data["work_plan"]["units"][0],
        )
    elif mutation == "bool_ordinal":
        data["work_plan"]["units"][0]["ordinal"] = True
    elif mutation == "parent_unit_count":
        data["work_plan"]["parent_total_units"] = 1
    elif mutation == "changed_gap":
        if data["work_plan"]["skipped"]:
            data["work_plan"]["skipped"][0]["reason"] = "fabricated"
        else:
            data["work_plan"]["skipped"].append({"anchor": "fabricated"})
    elif mutation == "scope_digest":
        data["section_manifest_sha256"] = "f" * 64
    elif mutation == "false_parent_verification":
        data["parent_original_reverified_on_host"] = True
    elif mutation == "fake_approval":
        data["publish_eligible"] = True
    elif mutation == "invalid_chapter_number":
        data["chapter_number"] = 0
    else:
        data["source_identity"]["parent_book_fingerprint"] = "f" * 64
    tampered = _edited(runner, data)
    with pytest.raises(selected_scope.SelectedSourceStop):
        selected_scope.import_chapter(runner.root, tampered)


def test_selected_chapter_mining_uses_real_canonical_validation_and_raw_cache(
    tmp_path: Path,
) -> None:
    runner, bundle, data = _bundle(tmp_path, with_image=False)
    verified = selected_scope.import_chapter(runner.root, bundle)
    manifest = selected_scope._json(data["section_manifest"])
    section = (
        runner.root / "book-candidates/section-scopes" / (data["section_manifest_sha256"] + ".json")
    )
    section.parent.mkdir(parents=True)
    section.write_bytes(manifest)

    def fake_miner(request: dict[str, object]) -> str:
        assert request["model"] == "Qwen/Qwen3-1.7B"
        return '{"lessons": []}'

    result = runner.mine_candidate(
        bundle,
        local_cpu=True,
        section_ids=verified.work.section_ids,
        section_manifest_sha256=verified.section_manifest_sha256,
        selected_scope_path=bundle,
        max_new_units=len(verified.work.units),
        local_completion=fake_miner,
    )
    assert result["status"] == "complete"
    assert result["candidate_count"] == 0
    progress = Path(result["progress_path"])
    ledger, candidates, book = evaluation._mined_export(
        runner,
        bundle,
        progress,
        section_plan=verified.work,
        section_manifest_sha256=verified.section_manifest_sha256,
        selected_scope_path=bundle,
    )
    assert not candidates
    assert book.fingerprint == verified.book.fingerprint
    assert ledger["counts"]["zero_candidate"] == len(verified.work.units)
    assert ledger["execution_provenance"] == "INJECTED_TEST_PRODUCER"
    assert ledger["source_boundary"] == "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT"
    assert ledger["parent_original_reverified_here"] is False
    assert ledger["selected_scope_bundle_sha256"] == verified.bundle_sha256
    assert progress.is_file()
    assert result["export_path"] is not None
    with pytest.raises(chapter.ChapterStop):
        chapter.evaluate_chapter(
            runner,
            bundle,
            progress,
            section,
            selected_scope_path=bundle,
            plan=chapter.ChapterPlan(brief_cpu=True),
            brief_completion=lambda _req: (
                '{"argument":"cannot omit source","beats":[],"coverage":[]}'
            ),
        )


def test_selected_source_minable_chapter_brief_replays_without_parent_or_human_approval(
    tmp_path: Path,
) -> None:
    from unittest.mock import Mock

    runner, bundle, data = _bundle(tmp_path, with_image=True)
    chosen = selected_scope.import_chapter(runner.root, bundle)
    original = data["section_manifest"]
    section = (
        runner.root / "book-candidates/section-scopes" / (data["section_manifest_sha256"] + ".json")
    )
    section.parent.mkdir(parents=True)
    section.write_bytes(selected_scope._json(original))
    result = runner.mine_candidate(
        bundle,
        local_cpu=True,
        section_ids=chosen.work.section_ids,
        section_manifest_sha256=chosen.section_manifest_sha256,
        selected_scope_path=bundle,
        max_new_units=len(chosen.work.units),
        local_completion=lambda _request: '{"lessons":[]}',
    )
    assert result["status"] == "complete"

    def brief_completion(request: dict[str, object]) -> str:
        input_data = json.loads(request["messages"][1]["content"])["input"]
        source = input_data["source"]
        units = source["units"]
        gaps = source["gaps"]
        assert len(units) == len(chosen.work.units)
        assert len(gaps) == len(chosen.work.skipped)
        assert len(source["reading_order"]) == len(units) + len(gaps)
        assert source["parent_original_reverified_on_host"] is False
        assert source["source_boundary"] == (
            "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT"
        )
        return json.dumps(
            {
                "argument": "Synthetic original chapter material is reviewed unit by unit.",
                "beats": [
                    {
                        "beat_id": f"beat-{index:02d}",
                        "unit_id": unit["source_id"],
                        "span_start": 0,
                        "span_end": min(120, len(unit["text"])),
                        "quote": unit["text"][:120],
                        "candidate_id": None,
                        "meaning": "Original synthetic passage retained without invented support.",
                        "qualification": "",
                        "essential": True,
                        "source_role": "BOOK_ASSERTION",
                    }
                    for index, unit in enumerate(units, 1)
                ],
                "coverage": [
                    {
                        "source_id": item["source_id"],
                        "decision": "omit" if item["kind"] != "CANONICAL_WORK_UNIT" else "keep",
                        "reason": "Retain the exact original unit or acknowledge its gap.",
                    }
                    for item in [*units, *gaps]
                ],
            }
        )

    completion = Mock(side_effect=brief_completion)
    args = (runner, bundle, Path(result["progress_path"]), section)
    first = chapter.evaluate_chapter(
        *args,
        selected_scope_path=bundle,
        plan=chapter.ChapterPlan(brief_cpu=True),
        brief_completion=completion,
    )
    assert first["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert first["parent_original_reverified_on_host"] is False
    assert first["publish_eligible"] is False
    assert first["brief_provenance"] == "INJECTED_TEST_PRODUCER"
    second = chapter.evaluate_chapter(
        *args,
        selected_scope_path=bundle,
        plan=chapter.ChapterPlan(brief_cpu=True),
        brief_completion=completion,
    )
    assert second["brief_identity"] == first["brief_identity"]
    assert second["selected_source_bundle_sha256"] == chosen.bundle_sha256
    assert second["brief_attempts"] == 0
    assert completion.call_count == 1
    assert not list(runner.root.rglob("analysis-refs"))
    assert not list(runner.root.rglob("*.epub"))


def test_selected_scope_exporter_cli_sanitizes_private_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from unittest.mock import Mock

    calls: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []

    def safe_export(
        _runner: object,
        _candidate: Path,
        *,
        chapter_number: int,
        chapter_section_ids: tuple[str, ...],
        linked_section_ids: tuple[str, ...],
    ) -> tuple[Path, str]:
        calls.append((chapter_number, chapter_section_ids, linked_section_ids))
        return tmp_path / "synthetic.json", "a" * 64

    monkeypatch.setattr(selected_scope, "export_chapter", safe_export)
    args = [
        "--artifact-root",
        str(tmp_path),
        "--candidate-ref",
        str(tmp_path / "private-candidate.json"),
        "--chapter-number",
        "1",
        "--chapter-sections",
        "chapter-one",
        "--linked-sections",
        "footnote-one",
    ]
    assert selected_scope.main(args) == 0
    standard, errors = capsys.readouterr()
    result = json.loads(standard)
    assert result["status"] == "SELECTED_SCOPE_EXPORTED"
    assert result["sha256"] == "a" * 64
    assert not errors
    assert calls == [(1, ("chapter-one",), ("footnote-one",))]
    failure = Mock(side_effect=RuntimeError("PRIVATE_SYNTHETIC_SOURCE_TEXT_CANARY"))
    monkeypatch.setattr(selected_scope, "export_chapter", failure)
    assert selected_scope.main(args) == 2
    standard, errors = capsys.readouterr()
    assert json.loads(standard)["status"] == "SELECTED_SCOPE_EXPORT_FAILED_CLOSED"
    assert not errors
    assert "PRIVATE_SYNTHETIC_SOURCE_TEXT_CANARY" not in standard
    failure.assert_called_once()


@pytest.mark.parametrize(
    "selection",
    ["zero_chapter", "empty", "same_linked", "missing_id", "reordered_chapters"],
)
def test_exporter_rejects_unverified_scope_before_creating_bundle(
    tmp_path: Path, selection: str
) -> None:
    owner = tmp_path / "retained"
    owner.mkdir()
    runner, acquired, _ = ingested(owner, paragraphs=8, with_image=True)
    ids = tuple(section.chapter_id for section in acquired.book.sections)
    chapter_number = 1
    chapters, linked = (ids[0],), ()
    if selection == "zero_chapter":
        chapter_number = 0
    elif selection == "empty":
        chapters = ()
    elif selection == "same_linked":
        linked = chapters
    elif selection == "missing_id":
        chapters = ("missing-original-section",)
    else:
        assert selection == "reordered_chapters"
        chapters = (ids[1], ids[0])
    with pytest.raises(selected_scope.SelectedSourceStop):
        selected_scope.export_chapter(
            runner,
            acquired.analysis_ref_path,
            chapter_number=chapter_number,
            chapter_section_ids=chapters,
            linked_section_ids=linked,
        )
    assert not list((runner.root / "book-candidates").rglob("selected-scopes/*.json"))


def test_importer_retains_explicit_linked_section_and_original_parent_order(
    tmp_path: Path,
) -> None:
    owner = tmp_path / "retained"
    owner.mkdir()
    runner, acquired, _ = ingested(owner, paragraphs=8, with_image=True)
    ids = tuple(section.chapter_id for section in acquired.book.sections)
    path, _sha = selected_scope.export_chapter(
        runner,
        acquired.analysis_ref_path,
        chapter_number=1,
        chapter_section_ids=(ids[0],),
        linked_section_ids=(ids[1],),
    )
    chosen = selected_scope.import_chapter(runner.root, path)
    assert chosen.linked_section_ids == (ids[1],)
    assert chosen.work.section_ids == ids[:2]
    assert chosen.book.parent_original_reverified_on_host is False
    assert len(chosen.book.selected_reading_blocks) > 1


@pytest.mark.parametrize(
    "malformed",
    ["source_schema", "identity_schema", "empty_blocks", "missing_work_units"],
)
def test_importer_rejects_internally_rehashed_incomplete_selected_evidence(
    tmp_path: Path, malformed: str
) -> None:
    runner, _path, evidence = _bundle(tmp_path)
    if malformed == "source_schema":
        evidence["source"].pop("title")
    elif malformed == "identity_schema":
        evidence["source_identity"].pop("parser_version")
    elif malformed == "empty_blocks":
        evidence["reading_blocks"] = []
    else:
        evidence["work_plan"]["units"] = []
    with pytest.raises(selected_scope.SelectedSourceStop):
        selected_scope.import_chapter(runner.root, _edited(runner, evidence))


@pytest.mark.parametrize("boundary", ["missing_json", "oversize", "outside", "symlink"])
def test_selected_importer_requires_bounded_content_addressed_private_file(
    tmp_path: Path, boundary: str
) -> None:
    runner, original, _ = _bundle(tmp_path)
    location = runner.root / "book-candidates/selected-scopes"
    if boundary == "missing_json":
        raw = b"{"
        target = location / f"{hashlib.sha256(raw).hexdigest()}.json"
        target.write_bytes(raw)
    elif boundary == "oversize":
        raw = b" " * (selected_scope.MAX_BUNDLE_BYTES + 1)
        target = location / f"{hashlib.sha256(raw).hexdigest()}.json"
        target.write_bytes(raw)
    elif boundary == "outside":
        target = runner.root / "outside-selected.json"
        target.write_bytes(original.read_bytes())
    else:
        target = location / "symlink-selected.json"
        target.symlink_to(original)
    with pytest.raises(selected_scope.SelectedSourceStop):
        selected_scope.import_chapter(runner.root, target)


@pytest.mark.parametrize(
    "damage",
    [
        "parent_book_id",
        "parent_hash_format",
        "missing_parent_ref",
        "missing_source_metadata",
        "missing_source_identity",
        "wrong_source_hash",
        "unreviewed_policy",
        "reading_block_type",
        "reading_block_origin",
        "missing_plan_shape",
        "noncanonical_status",
    ],
)
def test_selected_importer_rejects_corrupt_parent_and_source_evidence(
    tmp_path: Path, damage: str
) -> None:
    runner, _original, data = _bundle(tmp_path)
    if damage == "parent_book_id":
        data["parent_ref"]["book_id"] = ""
    elif damage == "parent_hash_format":
        data["parent_ref"]["ingest_stage_key"] = True
    elif damage == "missing_parent_ref":
        data["parent_ref"].pop("dossier_sha256")
    elif damage == "missing_source_metadata":
        data["source"].pop("source_uri")
    elif damage == "missing_source_identity":
        data["source_identity"].pop("recovery_status")
    elif damage == "wrong_source_hash":
        data["source_identity"]["source_fingerprint"] = "f" * 64
    elif damage == "unreviewed_policy":
        data["review_status"] = "APPROVED"
    elif damage == "reading_block_type":
        data["reading_blocks"][0] = 123
    elif damage == "reading_block_origin":
        data["reading_blocks"][0]["origin"] = "unknown_origin"
    elif damage == "missing_plan_shape":
        data["work_plan"].pop("parent_plan_fingerprint")
    else:
        data["schema"] = "unverified-selected"
    with pytest.raises(selected_scope.SelectedSourceStop):
        selected_scope.import_chapter(runner.root, _edited(runner, data))

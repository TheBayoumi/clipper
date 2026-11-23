"""Selected original chapter exporter/importer, no private or whole-book transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_books_whole_book import ingested

from clipper.books import chapter, evaluation, selected_scope, whole_book
from clipper.books.orchestrator import BookOrchestrator


def _bundle(tmp_path: Path, *, with_image: bool = True) -> tuple[
    BookOrchestrator, Path, dict
]:
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
    assert all(b.section_id in verified.work.section_ids for b in verified.book.selected_reading_blocks)
    assert tuple(u.ordinal for u in verified.work.units) == tuple(
        sorted(u.ordinal for u in verified.work.units)
    )
    assert verified.ref.raw_sha256 == verified.book.content_sha256
    assert verified.book.source_boundary == "HOST_VERIFIED_SELECTED_BUNDLE_EXPORTER_ATTESTED_PARENT"
    assert verified.work.parent_total_units > len(verified.work.units)
    assert whole_book.plan_for(
        verified.book, section_ids=verified.work.section_ids
    ) == verified.work
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
            path.write_bytes(path.read_bytes().replace(b"SELECTED_SCOPE_SOURCE_UNREVIEWED", b"UNREVIEWED"))
        else:
            path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(selected_scope.SelectedSourceStop):
            selected_scope.import_chapter(runner.root, path)
        return
    if mutation == "raw_parent":
        data["parent_ref"]["raw_sha256"] = "f" * 64
    elif mutation == "source_text":
        data["reading_blocks"][0]["text"] = "Invented changed source"
    elif mutation == "missing_block":
        del data["reading_blocks"][0]
    elif mutation == "reordered_blocks":
        data["reading_blocks"][0], data["reading_blocks"][1] = (
            data["reading_blocks"][1], data["reading_blocks"][0]
        )
    elif mutation == "duplicated_block":
        data["reading_blocks"].insert(1, data["reading_blocks"][0])
    elif mutation == "out_of_scope_block":
        data["reading_blocks"][0]["section_id"] = "chapter-not-selected"
    elif mutation == "missing_unit":
        data["work_plan"]["units"].pop()
    elif mutation == "reordered_units":
        data["work_plan"]["units"][0], data["work_plan"]["units"][1] = (
            data["work_plan"]["units"][1], data["work_plan"]["units"][0]
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
    section = runner.root / "book-candidates/section-scopes" / (
        data["section_manifest_sha256"] + ".json"
    )
    section.parent.mkdir(parents=True)
    section.write_bytes(manifest)

    def fake_miner(request: dict[str, object]) -> str:
        assert request["model"] == "Qwen/Qwen3-1.7B"
        return '{"lessons": []}'

    result = runner.mine_candidate(
        bundle, local_cpu=True, section_ids=verified.work.section_ids,
        section_manifest_sha256=verified.section_manifest_sha256,
        selected_scope_path=bundle, max_new_units=len(verified.work.units),
        local_completion=fake_miner,
    )
    assert result["status"] == "complete"
    assert result["candidate_count"] == 0
    progress = Path(result["progress_path"])
    ledger, candidates, book = evaluation._mined_export(
        runner, bundle, progress,
        section_plan=verified.work,
        section_manifest_sha256=verified.section_manifest_sha256,
        selected_scope_path=bundle,
    )
    assert not candidates
    assert book.fingerprint == verified.book.fingerprint
    assert ledger["counts"]["zero_candidate"] == len(verified.work.units)
    assert ledger["execution_provenance"] == "INJECTED_TEST_PRODUCER"
    assert progress.is_file()
    assert result["export_path"] is not None
    with pytest.raises(chapter.ChapterStop):
        chapter.evaluate_chapter(
            runner, bundle, progress, section,
            selected_scope_path=bundle,
            plan=chapter.ChapterPlan(brief_cpu=True),
            brief_completion=lambda _req: '{"argument":"cannot omit source","beats":[],"coverage":[]}',
        )

"""Book stage DAG: deterministic selection, pinned Qwen script, unported stages blocked."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, NoReturn, cast

from clipper.dag import DagStore
from clipper.stage_contracts import (
    StageContract,
    StageIdentity,
    content_fingerprint,
    stage_identity,
)

from . import (
    acquisition,
    constrained,
    evaluation,
    faithful,
    ingest,
    mining,
    narration,
    scripting,
    selection,
    storyboards,
    whole_book,
)
from .contracts import (
    BOOK_STAGE_CONTRACTS,
    BookRef,
    BookSection,
    BookSource,
    BookText,
    EpisodeScript,
    ExcludedPage,
    ImageEvidence,
    ImageOccurrence,
    LessonSelection,
    NarrationTrack,
    OCRIdentity,
    ScriptScene,
    SourceBlock,
    SourceUnresolved,
)
from .storyboards import StoryboardProvenance


class BookRunError(ValueError):
    """Book run input or stored stage output is inconsistent."""


def _bounded_candidate_json(path: Path, *, max_bytes: int) -> object:
    """Parse a complete, byte-bounded JSON file; never accept a truncated prefix."""
    payload = acquisition._read_bounded_file(path, max_bytes + 1)
    if len(payload) > max_bytes:
        raise BookRunError("candidate evidence JSON exceeds bounded input")
    return json.loads(payload)


class NotPortedStage(BookRunError):
    """Refuse unimplemented book work before any external producer starts."""


@dataclass(frozen=True, slots=True)
class BookPlan:
    """Typed selection plus byte-compatible legacy output documents."""

    selected: LessonSelection
    selection_json: str
    niche_config_json: str
    outline_json: str
    storyboard_provenance: StoryboardProvenance


@dataclass(frozen=True, slots=True)
class CandidateIngestResult:
    candidate: acquisition.ReviewedCandidate
    acquired: acquisition.AcquiredBook
    book: BookText
    stage_key: str
    extracted_path: Path
    dossier_path: Path
    analysis_ref_path: Path


def _book_text_cached(value: object) -> BookText:
    if not isinstance(value, dict) or not isinstance(value.get("source"), dict):
        raise BookRunError("invalid cached book-ingest stage output")
    record = cast(dict[str, Any], value)
    sections = tuple(
        BookSection(
            chapter_id=item["chapter_id"],
            heading=item["heading"],
            paragraphs=tuple(item["paragraphs"]),
        )
        for item in record["sections"]
    )
    return BookText(
        source=BookSource(**record["source"]),
        edition_identity=record["edition_identity"],
        sections=sections,
        total_word_count=record["total_word_count"],
        content_sha256=record["content_sha256"],
        source_fingerprint=record["source_fingerprint"],
        blocks=tuple(SourceBlock(**x) for x in record.get("blocks", [])),
        images=tuple(
            ImageEvidence(
                **{
                    **item,
                    "occurrences": tuple(ImageOccurrence(**occ) for occ in item["occurrences"]),
                }
            )
            for item in record.get("images", [])
        ),
        excluded_pages=tuple(ExcludedPage(**x) for x in record.get("excluded_pages", [])),
        warnings=tuple(record.get("warnings", [])),
        recovery_status=record.get("recovery_status", "legacy_unassessed"),
        parser_version=record.get("parser_version", "legacy-v1"),
        parser_config_sha256=record.get("parser_config_sha256", ""),
    )


def _book_slug(title: str) -> str:
    slug = "-".join(re.findall(r"[a-z0-9]+", title.casefold()))
    if not selection.IDENTIFIER.fullmatch(slug):
        raise BookRunError("book title cannot form a stable catalog identifier")
    return slug


def _selected_payload(raw: dict[str, Any]) -> LessonSelection:
    book, lesson = raw["book"], raw["lesson"]
    return LessonSelection(
        book=BookRef(
            book_id=book["id"],
            title=book["title"],
            author=book["author"],
            pillar=book["pillar"],
            source_url=book["source_url"],
        ),
        lesson_id=raw["lesson_id"],
        angle_id=raw["angle_id"],
        episode_id=raw["episode_id"],
        topic=lesson["topic"],
        claim=lesson["claim"],
        claim_source_url=lesson["claim_source_url"],
        hook_angle=lesson["hook_angle"],
        visual_action=lesson["visual_action"],
        script_version=raw["script_version"],
        legacy_fingerprint=raw["fingerprint"],
    )


def _selected_cached(value: object) -> LessonSelection:
    if not isinstance(value, dict) or not isinstance(value.get("book"), dict):
        raise BookRunError("invalid cached book-selection stage output")
    record = cast(dict[str, Any], value)
    return LessonSelection(
        book=BookRef(**record["book"]),
        **{key: item for key, item in record.items() if key != "book"},
    )


def _script_cached(value: object) -> EpisodeScript:
    if not isinstance(value, dict) or not isinstance(value.get("scenes"), (list, tuple)):
        raise BookRunError("invalid cached book-script stage output")
    raw = cast(dict[str, Any], value)
    scenes = tuple(ScriptScene(**scene) for scene in raw["scenes"])
    return EpisodeScript(
        episode_id=raw["episode_id"],
        selection_fingerprint=raw["selection_fingerprint"],
        script_cache_key=raw["script_cache_key"],
        title=raw["title"],
        scenes=scenes,
    )


def _narration_cached(value: object, count: int) -> tuple[NarrationTrack, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise BookRunError("invalid cached book-narration stage output")
    if any(not isinstance(item, dict) for item in value):
        raise BookRunError("invalid cached book-narration stage output")
    return tuple(NarrationTrack(**cast(dict[str, Any], item)) for item in value)


def _niche_config(plan: BookPlan) -> scripting.NicheConfig:
    raw = json.loads(plan.niche_config_json)
    return scripting.NicheConfig(
        niche=raw["niche"],
        audience=raw["audience"],
        brand=raw["brand"],
        voice=raw["voice"],
        facts=tuple(raw["facts"]),
        topic=raw["topic"],
        format=raw["format"],
        pillar=raw["pillar"],
        book_title=raw["book_title"],
        book_author=raw["book_author"],
        source_url=raw["source_url"],
        book_id=raw["book_id"],
        lesson_id=raw["lesson_id"],
        angle_id=raw["angle_id"],
        episode_id=raw["episode_id"],
        script_version=raw["script_version"],
        selection_fingerprint=raw["selection_fingerprint"],
        hook_angle=raw["hook_angle"],
        visual_action=raw["visual_action"],
    )


class BookOrchestrator:
    """Reuse DagStore's exact stage identity, locking, evidence and cached resume."""

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root
        self.dag = DagStore(
            artifact_root / "book-dag",
            private_stages=frozenset(
                {
                    BOOK_STAGE_CONTRACTS["ingest"].name,
                    "books.mining.unit.v1",
                    "books.mining.local-completion.v1",
                }
            ),
        )
        self.script_cache_root = artifact_root / "book-script-cache"

    def acquire_candidate(
        self,
        title: str,
        *,
        author: str | None = None,
        edition: str | None = None,
        registry: tuple[acquisition.ReviewedCandidate, ...] | None = None,
        transport: acquisition.BookTransport | None = None,
        offline: bool = False,
        refresh: bool = False,
    ) -> acquisition.AcquiredBook | SourceUnresolved:
        """Discover the reviewed edition before any network or parser work."""
        available = registry if registry is not None else acquisition.load_registry()
        choice = acquisition.discover(title, available, author=author, edition=edition)
        if isinstance(choice, SourceUnresolved):
            return choice
        store = acquisition.ArchiveAcquirer(self.root / "book-candidates", transport)
        return store.acquire(choice, offline=offline, refresh=refresh)

    def ingest_candidate(
        self,
        title: str,
        *,
        author: str | None = None,
        edition: str | None = None,
        registry: tuple[acquisition.ReviewedCandidate, ...] | None = None,
        transport: acquisition.BookTransport | None = None,
        offline: bool = False,
        refresh: bool = False,
        overlay_path: Path | None = None,
        ocr: faithful.OCR | None = None,
        ocr_identity: OCRIdentity | None = None,
    ) -> CandidateIngestResult | SourceUnresolved:
        """Preserve separate immutable raw, extracted and evidence dossier identities."""
        producer_fingerprint = faithful.ocr_contract(ocr, ocr_identity)
        acquired = self.acquire_candidate(
            title,
            author=author,
            edition=edition,
            registry=registry,
            transport=transport,
            offline=offline,
            refresh=refresh,
        )
        if isinstance(acquired, SourceUnresolved):
            return acquired
        raw = acquired.raw_path.read_bytes()
        if (
            hashlib.sha256(raw).hexdigest() != acquired.observed_sha256
            or len(raw) != acquired.observed_bytes
        ):
            raise BookRunError("acquired raw artifact hash or length mismatch")
        overlay = None
        research_payload = None
        if overlay_path is not None:
            if not overlay_path.is_file():
                raise BookRunError(f"reviewed image overlay missing: {overlay_path}")
            overlay = faithful.read_evidence(overlay_path)
            if not overlay:
                raise faithful.ExtractionError("explicit image overlay is empty")
            declared_digest = faithful.research_digest(overlay)
            if declared_digest is not None:
                research_path = (
                    self.root
                    / "book-candidates"
                    / "research-import"
                    / "source"
                    / f"{declared_digest}.json"
                )
                if research_path.exists():
                    research_payload = faithful.read_evidence(research_path)
                    if hashlib.sha256(research_payload).hexdigest() != declared_digest:
                        raise faithful.ExtractionError("cached research bytes do not match digest")
        configuration = {
            "parser_version": faithful.PARSER_VERSION,
            "candidate_fingerprint": acquired.candidate.fingerprint,
            "raw_sha256": acquired.observed_sha256,
            "acquisition_fingerprint": acquired.fingerprint,
            "overlay_sha256": hashlib.sha256(overlay).hexdigest() if overlay is not None else None,
            "research_sha256": (
                hashlib.sha256(research_payload).hexdigest()
                if research_payload is not None
                else None
            ),
            "ocr_producer": producer_fingerprint,
        }
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS["ingest"],
            source_hash=content_fingerprint(configuration),
            dependency_output_hashes=(acquired.fingerprint,),
            decoding_parameters={
                "parser": faithful.PARSER_VERSION,
                "ocr": producer_fingerprint or "none",
            },
        )

        def extract() -> object:
            return asdict(
                faithful.extract_candidate(
                    acquired.source,
                    raw,
                    acquired.candidate,
                    overlay=overlay,
                    research_payload=research_payload,
                    ocr=ocr,
                    ocr_identity=ocr_identity,
                )
            )

        self.dag.protect_stage(identity.stage_name)
        stored, _reused = self.dag.execute(identity, extract)
        book = _book_text_cached(stored)
        if (
            book.content_sha256 != acquired.observed_sha256
            or book.source_fingerprint != acquired.source.fingerprint
            or book.source.authorization_status != "unverified"
            or book.parser_version != faithful.PARSER_VERSION
        ):
            raise BookRunError("cached candidate extraction disagrees with pinned source")
        extracted_path = (
            self.root
            / "book-candidates"
            / "extracted"
            / faithful.PARSER_VERSION
            / f"{identity.cache_key}.json"
        )
        dossier_path = (
            self.root
            / "book-candidates"
            / "dossiers"
            / acquired.candidate.book_id
            / f"{identity.cache_key}.json"
        )
        acquisition._private_write(
            extracted_path,
            (json.dumps(asdict(book), ensure_ascii=False, sort_keys=True) + "\n").encode(),
            private_root=self.root / "book-candidates",
        )
        reconciliation = faithful.image_reconciliation(book)
        dossier = {
            "schema": "clipper-book-candidate-ingest-dossier-v3",
            "book_id": acquired.candidate.book_id,
            "author": acquired.candidate.author,
            "edition_identity": acquired.candidate.edition_identity,
            "raw_sha256": acquired.observed_sha256,
            "raw_bytes": acquired.observed_bytes,
            "record_url": acquired.record_url,
            "actual_asset_url": acquired.asset_url,
            "http_status": acquired.http_status,
            "fetched_at": acquired.fetched_at,
            "authorization_status": book.source.authorization_status,
            "publisher_authenticity": "unverified",
            "source_fingerprint": acquired.source.fingerprint,
            "acquisition_fingerprint": acquired.fingerprint,
            "extract_stage_key": identity.cache_key,
            "extracted_sha256": hashlib.sha256(extracted_path.read_bytes()).hexdigest(),
            "parser_version": book.parser_version,
            "parser_config_sha256": book.parser_config_sha256,
            "ocr_producer_fingerprint": producer_fingerprint,
            "ocr_evidence_sha256": content_fingerprint(
                [
                    {
                        "asset_sha256": x.asset_sha256,
                        "method": x.method,
                        "review_status": x.review_status,
                        "original_transcription": x.original_transcription,
                        "machine_transcription": x.machine_transcription,
                    }
                    for x in book.images
                ]
            ),
            "text_blocks": len(book.blocks),
            "sections": len(book.sections),
            "extracted_words": book.total_word_count,
            "unique_images": len(book.images),
            "image_occurrences": sum(len(x.occurrences) for x in book.images),
            "reviewed_images": reconciliation["reviewed_assets"],
            "excluded_pages": [asdict(x) for x in book.excluded_pages],
            "recovery_status": book.recovery_status,
            "warnings": [*acquired.warnings, *book.warnings],
            "editorial_approved": False,
            "publish_eligible": False,
        }
        dossier["reviewed_occurrences"] = reconciliation["reviewed_occurrences"]
        dossier["normalized_only_occurrences"] = reconciliation["normalized_only_occurrences"]
        dossier["linked_occurrences"] = reconciliation["linked_occurrences"]
        dossier["unresolved_occurrences"] = reconciliation["unresolved_occurrences"]
        dossier["verified_research_links"] = reconciliation["verified_research_links"]
        dossier["unverified_research_links"] = reconciliation["unverified_research_links"]
        dossier["machine_occurrences"] = reconciliation["machine_occurrences"]
        reconciliation_path = (
            self.root
            / "book-candidates"
            / "reconciliation"
            / acquired.candidate.book_id
            / f"{identity.cache_key}.json"
        )
        acquisition._private_write(
            reconciliation_path,
            (json.dumps(reconciliation, sort_keys=True, indent=2) + "\n").encode(),
            private_root=self.root / "book-candidates",
        )
        dossier["reconciliation_path"] = str(reconciliation_path)
        acquisition._private_write(
            dossier_path,
            (json.dumps(dossier, sort_keys=True, indent=2) + "\n").encode(),
            private_root=self.root / "book-candidates",
        )
        analysis_ref_path = (
            self.root
            / "book-candidates"
            / "analysis-refs"
            / acquired.candidate.book_id
            / f"{identity.cache_key}.json"
        )
        analysis_ref = whole_book.CandidateAnalysisRef(
            schema=whole_book.REF_SCHEMA,
            book_id=acquired.candidate.book_id,
            candidate_fingerprint=acquired.candidate.fingerprint,
            candidate=asdict(acquired.candidate),
            acquisition_fingerprint=acquired.fingerprint,
            source_fingerprint=book.source_fingerprint,
            raw_sha256=acquired.observed_sha256,
            ingest_stage_key=identity.cache_key,
            extracted_sha256=hashlib.sha256(extracted_path.read_bytes()).hexdigest(),
            dossier_sha256=hashlib.sha256(dossier_path.read_bytes()).hexdigest(),
            book_fingerprint=book.fingerprint,
            parser_version=book.parser_version,
            parser_config_sha256=book.parser_config_sha256,
            recovery_status=book.recovery_status,
        )
        acquisition._private_write(
            analysis_ref_path,
            (json.dumps(asdict(analysis_ref), ensure_ascii=False, sort_keys=True) + "\n").encode(),
            private_root=self.root / "book-candidates",
        )
        return CandidateIngestResult(
            acquired.candidate,
            acquired,
            book,
            identity.cache_key,
            extracted_path,
            dossier_path,
            analysis_ref_path,
        )

    def load_candidate_for_analysis(
        self, ref_path: Path, *, book_id: str | None = None
    ) -> tuple[whole_book.CandidateAnalysisRef, BookText]:
        """Load only an exact immutable title-only extraction; never discover or fetch."""
        if book_id is not None and not selection.IDENTIFIER.fullmatch(book_id):
            raise BookRunError("invalid requested candidate book id")
        try:
            raw_ref = _bounded_candidate_json(ref_path, max_bytes=131_072)
            expected_fields = {field.name for field in fields(whole_book.CandidateAnalysisRef)}
            if not isinstance(raw_ref, dict) or set(raw_ref) != expected_fields:
                raise ValueError("invalid candidate analysis handle schema")
            if any(
                not isinstance(value, str) or not value
                for key, value in raw_ref.items()
                if key != "candidate"
            ):
                raise ValueError("candidate analysis handle field type mismatch")
            ref = whole_book.CandidateAnalysisRef(**raw_ref)
            if (
                ref.schema != whole_book.REF_SCHEMA
                or not selection.IDENTIFIER.fullmatch(ref.book_id)
                or (book_id is not None and ref.book_id != book_id)
            ):
                raise ValueError("candidate analysis handle edition/book mismatch")
            for digest in (
                ref.candidate_fingerprint,
                ref.acquisition_fingerprint,
                ref.source_fingerprint,
                ref.raw_sha256,
                ref.ingest_stage_key,
                ref.extracted_sha256,
                ref.dossier_sha256,
                ref.book_fingerprint,
                ref.parser_config_sha256,
            ):
                if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                    raise ValueError("invalid candidate analysis digest")
            if (
                ref_path.is_symlink()
                or ref_path.resolve(strict=True)
                != (
                    self.root
                    / "book-candidates"
                    / "analysis-refs"
                    / ref.book_id
                    / f"{ref.ingest_stage_key}.json"
                ).resolve()
            ):
                raise ValueError("candidate analysis handle path is not canonical")
            candidate_values = raw_ref["candidate"]
            candidate_keys = {field.name for field in fields(acquisition.ReviewedCandidate)}
            if not isinstance(candidate_values, dict) or set(candidate_values) != candidate_keys:
                raise ValueError("candidate analysis source metadata malformed")
            candidate = acquisition.ReviewedCandidate(**candidate_values)
            if (
                candidate.fingerprint != ref.candidate_fingerprint
                or candidate.book_id != ref.book_id
                or candidate.expected_sha256 != ref.raw_sha256
                or candidate.format not in {"epub", "pdf"}
                or not isinstance(candidate.expected_bytes, int)
                or isinstance(candidate.expected_bytes, bool)
                or not 0 < candidate.expected_bytes <= acquisition.MAX_ASSET_BYTES
            ):
                raise ValueError("candidate analysis source pin mismatch")
            base = self.root / "book-candidates"
            extracted_path = (
                base / "extracted" / ref.parser_version / f"{ref.ingest_stage_key}.json"
            )
            dossier_path = base / "dossiers" / ref.book_id / f"{ref.ingest_stage_key}.json"
            extracted = acquisition._read_bounded_file(extracted_path, 32_000_001)
            dossier_bytes = acquisition._read_bounded_file(dossier_path, 2_000_001)
            if (
                len(extracted) > 32_000_000
                or hashlib.sha256(extracted).hexdigest() != ref.extracted_sha256
                or len(dossier_bytes) > 2_000_000
                or hashlib.sha256(dossier_bytes).hexdigest() != ref.dossier_sha256
            ):
                raise ValueError("candidate extracted/dossier evidence changed")
            dossier = json.loads(dossier_bytes)
            if not isinstance(dossier, dict):
                raise ValueError("candidate dossier malformed")
            checks = {
                "schema": "clipper-book-candidate-ingest-dossier-v3",
                "book_id": ref.book_id,
                "edition_identity": candidate.edition_identity,
                "raw_sha256": ref.raw_sha256,
                "raw_bytes": candidate.expected_bytes,
                "acquisition_fingerprint": ref.acquisition_fingerprint,
                "source_fingerprint": ref.source_fingerprint,
                "extract_stage_key": ref.ingest_stage_key,
                "extracted_sha256": ref.extracted_sha256,
                "parser_version": ref.parser_version,
                "parser_config_sha256": ref.parser_config_sha256,
                "recovery_status": ref.recovery_status,
                "authorization_status": "unverified",
                "publish_eligible": False,
                "editorial_approved": False,
            }
            if any(dossier.get(k) != v for k, v in checks.items()):
                raise ValueError("candidate dossier identity mismatch")
            book = _book_text_cached(json.loads(extracted))
            if (
                book.fingerprint != ref.book_fingerprint
                or book.source_fingerprint != ref.source_fingerprint
                or book.source_fingerprint != book.source.fingerprint
                or book.content_sha256 != ref.raw_sha256
                or book.source.title != candidate.title
                or book.source.author != candidate.author
                or book.edition_identity != candidate.edition_identity
                or book.parser_version != ref.parser_version
                or book.parser_config_sha256 != ref.parser_config_sha256
                or book.recovery_status != ref.recovery_status
                or book.source.authorization_status != "unverified"
                or sum(len(p.split()) for section in book.sections for p in section.paragraphs)
                != book.total_word_count
            ):
                raise ValueError("candidate BookText evidence identity mismatch")
            image_digest = content_fingerprint(
                [
                    {
                        "asset_sha256": item.asset_sha256,
                        "method": item.method,
                        "review_status": item.review_status,
                        "original_transcription": item.original_transcription,
                        "machine_transcription": item.machine_transcription,
                    }
                    for item in book.images
                ]
            )
            if dossier.get("ocr_evidence_sha256") != image_digest:
                raise ValueError("candidate image evidence fingerprint mismatch")
            manifest = _bounded_candidate_json(
                base / "manifests" / f"{ref.candidate_fingerprint}.json",
                max_bytes=2_000_000,
            )
            if (
                not isinstance(manifest, dict)
                or manifest.get("schema") != "clipper-book-acquisition-v1"
                or manifest.get("candidate_fingerprint") != ref.candidate_fingerprint
                or manifest.get("observed_sha256") != ref.raw_sha256
                or manifest.get("observed_bytes") != candidate.expected_bytes
                or manifest.get("record_url") != candidate.record_url
                or manifest.get("source") != asdict(book.source)
            ):
                raise ValueError("candidate acquisition manifest identity mismatch")
            acquisition_digest = content_fingerprint(
                {
                    "candidate": asdict(candidate),
                    "record_url": candidate.record_url,
                    "source_kind": book.source.source_kind,
                    "authorization_status": book.source.authorization_status,
                    "metadata_sha256": manifest["metadata_sha256"],
                    "observed_bytes": candidate.expected_bytes,
                    "observed_sha256": ref.raw_sha256,
                }
            )
            if acquisition_digest != ref.acquisition_fingerprint:
                raise ValueError("candidate acquisition fingerprint mismatch")
            raw_path = base / "raw" / ref.raw_sha256[:2] / f"{ref.raw_sha256}.{candidate.format}"
            raw_bytes = acquisition._read_bounded_file(raw_path, candidate.expected_bytes + 1)
            if (
                len(raw_bytes) != candidate.expected_bytes
                or hashlib.sha256(raw_bytes).hexdigest() != ref.raw_sha256
            ):
                raise ValueError("candidate pinned raw evidence changed")
            stage_path = (
                self.root / "book-dag" / "books-ingest" / ref.ingest_stage_key / "stage.json"
            )
            record = _bounded_candidate_json(stage_path, max_bytes=2_000_000)
            if not isinstance(record, dict):
                raise ValueError("candidate ingest DAG record malformed")
            identity_data = record["identity"]
            if not isinstance(identity_data, dict):
                raise ValueError("candidate ingest DAG identity missing")
            parameters = identity_data["decoding_parameters"]
            dependencies = identity_data["dependency_output_hashes"]
            if not isinstance(parameters, dict) or not isinstance(dependencies, list):
                raise ValueError("candidate ingest DAG material malformed")
            identity = StageIdentity(
                identity_data["stage_name"],
                identity_data["source_hash"],
                identity_data["contract_hash"],
                tuple(dependencies),
                identity_data["model_revision"],
                parameters,
                identity_data["relevant_policy_hash"],
            )
            if (
                identity.stage_name != BOOK_STAGE_CONTRACTS["ingest"].name
                or identity.contract_hash != BOOK_STAGE_CONTRACTS["ingest"].contract_hash
                or identity.cache_key != ref.ingest_stage_key
                or tuple(dependencies) != (ref.acquisition_fingerprint,)
                or self.dag._validated_pass(identity, record) is None
                or content_fingerprint(self.dag._cache_hit(identity)[1])
                != content_fingerprint(asdict(book))
            ):
                raise ValueError("candidate ingest DAG publication is inconsistent")
            return ref, book
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
            raise BookRunError(
                "candidate analysis reference or pinned evidence is invalid"
            ) from exc

    def _produce_local_unit(
        self,
        *,
        ref: whole_book.CandidateAnalysisRef,
        identity: StageIdentity,
        request: dict[str, object],
        request_sha256: str,
        unit: whole_book.WorkUnit,
        producer_fingerprint: str,
        completion: Callable[[dict[str, object]], str],
    ) -> dict[str, object]:
        """Store bounded raw model evidence before validating; failed DAG stages can retry."""
        truncated = False
        try:
            response = completion(request)
        except scripting.IncompleteConstrainedResponse as exc:
            # Preserve the raw truncated generation before rejecting it.
            response = exc.raw
            truncated = True
        except constrained.NoEligibleSpans as exc:
            raise whole_book.LocalReplyFailure("no_eligible_source_span") from exc
        except constrained.DecoderUnavailable as exc:
            raise whole_book.LocalReplyFailure("constrained_decoder_unavailable") from exc
        except constrained.ConstraintError as exc:
            raise whole_book.LocalReplyFailure("invalid_constrained_request") from exc
        except scripting.PinnedQwenSnapshotError as exc:
            raise whole_book.LocalReplyFailure("pinned_model_cache_unavailable") from exc
        except Exception as exc:
            raise whole_book.LocalReplyFailure("runtime_error") from exc
        if not isinstance(response, str):
            raise whole_book.LocalReplyFailure("non_text_reply")
        try:
            payload = response.encode("utf-8")
        except UnicodeError as exc:
            raise whole_book.LocalReplyFailure("invalid_utf8") from exc
        digest = hashlib.sha256(payload).hexdigest()
        path = (
            self.root
            / "book-candidates"
            / "analysis"
            / "completions"
            / ref.book_id
            / identity.cache_key
            / f"{digest}.txt"
        )
        acquisition._private_write(
            path,
            payload[: whole_book.MAX_RECORDED_REPLY_BYTES],
            private_root=self.root / "book-candidates",
        )
        if len(payload) > whole_book.MAX_RECORDED_REPLY_BYTES:
            raise whole_book.LocalReplyFailure("oversized_reply", digest)
        if truncated:
            raise whole_book.LocalReplyFailure("incomplete_constrained_generation", digest)
        try:
            lessons = whole_book.validate_reply(whole_book.load_strict_reply(response), unit)
            constrained.validate_local_choice(lessons, request)
        except (ValueError, TypeError, RecursionError) as exc:
            raise whole_book.LocalReplyFailure("invalid_reply", digest) from exc
        return {
            "schema": whole_book.LOCAL_COMPLETION_SCHEMA,
            "request_sha256": request_sha256,
            "producer_fingerprint": producer_fingerprint,
            "raw_sha256": digest,
            "lessons": lessons,
        }

    def _read_local_unit(
        self,
        *,
        ref: whole_book.CandidateAnalysisRef,
        identity: StageIdentity,
        request_sha256: str,
        request: dict[str, object],
        unit: whole_book.WorkUnit,
        producer_fingerprint: str,
        stored: object,
    ) -> tuple[dict[str, object], str]:
        """Fail closed if raw immutable completion and validated DAG output diverge."""
        if (
            not isinstance(stored, dict)
            or set(stored)
            != {"schema", "request_sha256", "producer_fingerprint", "raw_sha256", "lessons"}
            or stored["schema"] != whole_book.LOCAL_COMPLETION_SCHEMA
            or stored["request_sha256"] != request_sha256
            or stored["producer_fingerprint"] != producer_fingerprint
            or not isinstance(stored["raw_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", stored["raw_sha256"]) is None
        ):
            raise BookRunError("cached local model output identity mismatch")
        digest = stored["raw_sha256"]
        path = (
            self.root
            / "book-candidates"
            / "analysis"
            / "completions"
            / ref.book_id
            / identity.cache_key
            / f"{digest}.txt"
        )
        private_root = self.root / "book-candidates"
        current = path.parent
        while current != private_root:
            if current.is_symlink():
                raise BookRunError("cached local model evidence is a symlink")
            if not current.is_dir():
                raise BookRunError("cached local model evidence directory missing")
            current = current.parent
        if private_root.is_symlink() or path.is_symlink():
            raise BookRunError("cached local model evidence is a symlink")
        if not path.is_file():
            raise BookRunError("cached local model raw evidence missing")
        # A cache PASS must not bypass private permissions. Tighten only
        # application-owned ancestors and the reused leaf, never shared roots.
        acquisition._private_directory(path.parent, private_root=private_root)
        try:
            path.chmod(0o600)
            raw = acquisition._read_bounded_file(path, whole_book.MAX_RECORDED_REPLY_BYTES + 1)
        except FileNotFoundError as exc:
            raise BookRunError("cached local model raw evidence missing") from exc
        if (
            len(raw) > whole_book.MAX_RECORDED_REPLY_BYTES
            or hashlib.sha256(raw).hexdigest() != digest
        ):
            raise BookRunError("cached local model raw evidence changed")
        try:
            value: object = whole_book.load_strict_reply(raw)
            lessons = whole_book.validate_reply(value, unit)
            if content_fingerprint(request) != request_sha256:
                raise constrained.ConstraintError("cached canonical request changed")
            constrained.validate_local_choice(lessons, request)
        except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
            raise BookRunError("cached local model response is invalid") from exc
        if not isinstance(value, dict) or lessons != stored["lessons"]:
            raise BookRunError("cached local model validation evidence mismatch")
        return value, digest

    def _record_local_failure(
        self,
        *,
        ref: whole_book.CandidateAnalysisRef,
        identity: StageIdentity,
        request_sha256: str,
        producer_fingerprint: str,
        failure: whole_book.LocalReplyFailure,
    ) -> None:
        """Preserve sanitized per-attempt failure provenance without prompt/log leakage."""
        event = {
            "schema": "clipper-book-local-failure-v1",
            "book_id": ref.book_id,
            "generation_stage_key": identity.cache_key,
            "request_sha256": request_sha256,
            "producer_fingerprint": producer_fingerprint,
            "attempt": self.dag.attempt_count(identity),
            "category": failure.category,
            "raw_sha256": failure.raw_sha256,
        }
        event_key = content_fingerprint(event)
        failure_path = (
            self.root
            / "book-candidates"
            / "analysis"
            / "failures"
            / ref.book_id
            / identity.cache_key
            / f"{event_key}.json"
        )
        acquisition._private_write(
            failure_path,
            (json.dumps(event, sort_keys=True) + "\n").encode(),
            private_root=self.root / "book-candidates",
        )
        # Earlier attempts in this same private evidence chain may have had
        # their modes relaxed. Tighten them without changing immutable bytes.
        for previous in failure_path.parent.glob("*.json"):
            if previous.is_symlink() or not previous.is_file():
                raise BookRunError("private local failure evidence is invalid")
            previous.chmod(0o600)

    def mine_candidate(
        self,
        ref_path: Path,
        *,
        responses: dict[str, object] | None = None,
        response_file_sha256: str | None = None,
        book_id: str | None = None,
        local_cpu: bool = False,
        max_new_units: int | None = None,
        local_completion: Callable[[dict[str, object]], str] | None = None,
        section_ids: tuple[str, ...] | None = None,
        section_manifest_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Process faithful original work units, optionally scoped to exact parent sections."""
        producer_provenance = (
            "INJECTED_TEST_PRODUCER"
            if local_cpu and local_completion is not None
            else "ACTUAL_CPU_QWEN"
            if local_cpu
            else "RECORDED_RESPONSE_UNVERIFIED"
        )
        ref, book = self.load_candidate_for_analysis(ref_path, book_id=book_id)
        if (section_ids is None) != (section_manifest_sha256 is None) or (
            section_manifest_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", section_manifest_sha256) is None
        ):
            raise BookRunError("scoped mining requires a pinned section manifest digest")
        if local_cpu:
            if responses is not None or response_file_sha256 is not None:
                raise BookRunError("local CPU and recorded response modes are exclusive")
            budget = 8 if max_new_units is None else max_new_units
            if (
                not isinstance(budget, int)
                or isinstance(budget, bool)
                or not 1 <= budget <= whole_book.MAX_PLAN_UNITS
            ):
                raise BookRunError("local CPU max-new-units must be between 1 and 20000")
            producer_material = {
                **whole_book.local_producer_material(),
                "execution_provenance": producer_provenance,
            }
            producer_fingerprint = content_fingerprint(producer_material)
            producer = local_completion or scripting._local_book_completion
            generation_contract = StageContract(
                "books.mining.local-completion.v1", producer_material
            )
            self.dag.protect_stage(generation_contract.name)
        else:
            if max_new_units is not None or local_completion is not None:
                raise BookRunError("local generation options require explicit local CPU mode")
            if (
                not isinstance(responses, dict)
                or not isinstance(response_file_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", response_file_sha256) is None
            ):
                raise BookRunError(
                    "candidate analysis requires a recorded response mapping and digest"
                )
            producer_fingerprint = ""
        work = whole_book.plan_for(book, section_ids=section_ids)
        # Plan identity binds selected original units and exact section scope, if any.
        book_fingerprint = work.book_fingerprint
        plan_fingerprint = work.fingerprint
        self.dag.protect_stage("books.mining.unit.v1")
        unit_contract = StageContract(
            "books.mining.unit.v1", {"schema": whole_book.RESULT_SCHEMA, "unit": "v1"}
        )
        progress: list[dict[str, Any]] = []
        lessons: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        local_attempts = 0
        for window_index, offset in enumerate(range(0, len(work.units), whole_book.WINDOW_UNITS)):
            window = work.units[offset : offset + whole_book.WINDOW_UNITS]
            for unit in window:
                unit_fingerprint = unit.fingerprint
                request = whole_book._request_for_identity(book, unit, book_fingerprint)
                request_key = content_fingerprint(request)
                status = "unresolved"
                count = 0
                response_digest: str | None = None
                raw_sha256: str | None = None
                failure_category: str | None = None
                generation_key: str | None = None
                answer: object | None = None
                has_answer = False
                if local_cpu:
                    identity_local = stage_identity(
                        generation_contract,
                        source_hash=content_fingerprint(
                            {
                                "book": book_fingerprint,
                                "projection": plan_fingerprint,
                                "unit": unit_fingerprint,
                                "request": request_key,
                                "producer": producer_fingerprint,
                            }
                        ),
                        dependency_output_hashes=(book_fingerprint,),
                        model_revision=scripting.SCRIPT_REVISION,
                        decoding_parameters=cast(
                            dict[str, object], producer_material["generation"]
                        ),
                    )
                    generation_key = identity_local.cache_key
                    tried = 0

                    def generate(
                        current_identity: StageIdentity = identity_local,
                        current_request: dict[str, object] = request,
                        current_request_key: str = request_key,
                        current_unit: whole_book.WorkUnit = unit,
                    ) -> object:
                        nonlocal tried
                        tried += 1
                        return self._produce_local_unit(
                            ref=ref,
                            identity=current_identity,
                            request=current_request,
                            request_sha256=current_request_key,
                            unit=current_unit,
                            producer_fingerprint=producer_fingerprint,
                            completion=producer,
                        )

                    try:
                        if local_attempts >= budget:
                            hit, stored_local = self.dag._cache_hit(identity_local)
                            if not hit:
                                stored_local = None
                        else:
                            stored_local, _reused = self.dag.execute(identity_local, generate)
                            hit = True
                        if hit:
                            answer, raw_sha256 = self._read_local_unit(
                                ref=ref,
                                identity=identity_local,
                                request_sha256=request_key,
                                request=request,
                                unit=unit,
                                producer_fingerprint=producer_fingerprint,
                                stored=stored_local,
                            )
                            response_digest = content_fingerprint(answer)
                            has_answer = True
                    except whole_book.LocalReplyFailure as exc:
                        failure_category = exc.category
                        raw_sha256 = exc.raw_sha256
                        status = "failed"
                        self._record_local_failure(
                            ref=ref,
                            identity=identity_local,
                            request_sha256=request_key,
                            producer_fingerprint=producer_fingerprint,
                            failure=exc,
                        )
                    finally:
                        local_attempts += tried
                else:
                    if responses is None:
                        raise BookRunError("recorded response mapping missing")
                    answer = whole_book.recorded_answer(responses, request)
                    has_answer = request_key in responses
                if has_answer:
                    try:
                        serialized = json.dumps(answer, ensure_ascii=False, sort_keys=True)
                        if len(serialized.encode("utf-8")) > whole_book.MAX_RECORDED_REPLY_BYTES:
                            raise mining.BookMiningError("recorded response exceeds per-unit size")
                        response_digest = content_fingerprint(answer)
                        identity = stage_identity(
                            unit_contract,
                            source_hash=content_fingerprint(
                                {
                                    "book": book_fingerprint,
                                    "projection": plan_fingerprint,
                                    "unit": unit_fingerprint,
                                    "request": request_key,
                                    "used_completion": response_digest,
                                    **(
                                        {
                                            "completion_origin": producer_provenance,
                                            "generation_stage_key": generation_key,
                                        }
                                        if local_cpu
                                        else {}
                                    ),
                                }
                            ),
                            dependency_output_hashes=(book_fingerprint,),
                            model_revision=scripting.SCRIPT_REVISION,
                            decoding_parameters={
                                "seed": 42,
                                "max_new_tokens": 960,
                                "chunk_words": mining.CHUNK_WORDS,
                                "evidence_field": unit.evidence_field,
                                **(
                                    cast(dict[str, object], producer_material["generation"])
                                    if local_cpu
                                    else {}
                                ),
                            },
                        )

                        def produce(
                            current_unit: whole_book.WorkUnit = unit,
                            current_unit_fingerprint: str = unit_fingerprint,
                            current_request: str = request_key,
                            current_response: str | None = response_digest,
                            current_answer: object = answer,
                        ) -> object:
                            return {
                                "unit": current_unit_fingerprint,
                                "request": current_request,
                                "completion": current_response,
                                "lessons": whole_book.validate_reply(current_answer, current_unit),
                            }

                        stored, _cached = self.dag.execute(identity, produce)
                        if (
                            not isinstance(stored, dict)
                            or set(stored) != {"unit", "request", "completion", "lessons"}
                            or stored["unit"] != unit_fingerprint
                            or stored["request"] != request_key
                            or stored["completion"] != response_digest
                        ):
                            raise mining.BookMiningError("cached analysis unit identity mismatch")
                        validated = whole_book.validate_reply({"lessons": stored["lessons"]}, unit)
                        for item in validated:
                            selected = (
                                constrained.selected_span(item, request) if local_cpu else None
                            )
                            candidate = whole_book.candidate_for(
                                book, unit, item, local_span=selected
                            )
                            if candidate["id"] in used_ids:
                                raise mining.BookMiningError("mined candidate identity collision")
                            used_ids.add(candidate["id"])
                            lessons.append(candidate)
                        count = len(validated)
                        status = "complete" if count else "zero_candidate"
                    except (TypeError, ValueError, RecursionError, OverflowError):
                        status = "failed"
                progress.append(
                    {
                        "window": window_index,
                        "unit": unit.ordinal,
                        "unit_fingerprint": unit_fingerprint,
                        "anchor": unit.anchor,
                        "source_path": unit.source_path,
                        "asset_sha256": unit.asset_sha256,
                        "origin": unit.origin,
                        "evidence_field": unit.evidence_field,
                        "review_status": unit.review_status,
                        "research_link_status": unit.research_link_status,
                        "text_sha256": unit.text_sha256,
                        "evidence_coverage": constrained.coverage_record(
                            cast(dict[str, object], request["constraints"])
                        ),
                        "request_sha256": request_key,
                        "used_completion_sha256": response_digest,
                        "status": status,
                        "candidate_count": count,
                        **(
                            {
                                "completion_origin": producer_provenance if has_answer else None,
                                "generation_stage_key": generation_key,
                                "raw_completion_sha256": raw_sha256,
                                "failure_category": failure_category,
                            }
                            if local_cpu
                            else {}
                        ),
                    }
                )
        counts = {
            key: sum(item["status"] == key for item in progress)
            for key in ("complete", "zero_candidate", "unresolved", "failed")
        }
        counts["skipped"] = len(work.skipped)
        outcome = "complete" if not counts["unresolved"] and not counts["failed"] else "incomplete"
        base = self.root / "book-candidates" / "analysis"
        ledger = {
            "schema": whole_book.RESULT_SCHEMA,
            "book_id": ref.book_id,
            "source_fingerprint": book.source_fingerprint,
            "book_fingerprint": book_fingerprint,
            "candidate_fingerprint": ref.candidate_fingerprint,
            "ingest_stage_key": ref.ingest_stage_key,
            "plan_fingerprint": plan_fingerprint,
            **(
                {
                    "section_scope": work.section_scope,
                    "section_manifest_sha256": section_manifest_sha256,
                    "parent_book_fingerprint": book.fingerprint,
                }
                if section_ids is not None
                else {}
            ),
            "window_size": whole_book.WINDOW_UNITS,
            "total_work_units": len(work.units),
            "response_file_sha256": response_file_sha256,
            "execution_provenance": producer_provenance,
            **(
                {
                    "completion_mode": "local_cpu",
                    "producer_fingerprint": producer_fingerprint,
                    "producer": producer_material,
                }
                if local_cpu
                else {}
            ),
            "recovery_status": book.recovery_status,
            "authorization_status": book.source.authorization_status,
            "parser_version": book.parser_version,
            "parser_config_sha256": book.parser_config_sha256,
            "excluded_pages": [asdict(p) for p in book.excluded_pages],
            "gaps": list(work.skipped),
            "progress": progress,
            "evidence_coverage_summary": constrained.coverage_summary(
                [cast(dict[str, object], item["evidence_coverage"]) for item in progress]
            ),
            "counts": counts,
            "candidate_count": len(lessons),
            "status": outcome,
            "review_status": "MINED_UNREVIEWED",
            "publish_eligible": False,
        }
        # Progress is immutable evidence of this *observed* outcome. A retry
        # with the same response file but a different failed/complete state
        # must publish a distinct report, never overwrite the earlier one.
        progress_bytes = (json.dumps(ledger, ensure_ascii=False, sort_keys=True) + "\n").encode()
        progress_sha256 = hashlib.sha256(progress_bytes).hexdigest()
        response_identity = producer_fingerprint if local_cpu else response_file_sha256
        artifact_name = f"{plan_fingerprint}-{response_identity}-{progress_sha256}.json"
        progress_path = base / "progress" / ref.book_id / artifact_name
        acquisition._private_write(
            progress_path, progress_bytes, private_root=self.root / "book-candidates"
        )
        export_path: Path | None = None
        if outcome == "complete":
            export = {
                "schema": "clipper-book-editorial-catalog-v1",
                "input_kind": "mined_separate_unreviewed_whole_book",
                "review_status": "MINED_UNREVIEWED",
                "publish_eligible": False,
                "analysis_plan_fingerprint": plan_fingerprint,
                "progress_sha256": progress_sha256,
                **(
                    {
                        "section_scope": work.section_scope,
                        "section_manifest_sha256": section_manifest_sha256,
                        "parent_full_book_complete": False,
                    }
                    if section_ids is not None
                    else {}
                ),
                "evidence_coverage_summary": ledger["evidence_coverage_summary"],
                "ingest_fingerprint": book_fingerprint,
                "recovery_status": book.recovery_status,
                "books": [
                    {
                        "id": ref.book_id,
                        "title": book.source.title,
                        "author": book.source.author,
                        "pillar": "unreviewed",
                        "source_url": book.source.source_uri,
                        "status": "source_verified_candidate",
                        "review_status": "MINED_UNREVIEWED",
                        "publish_eligible": False,
                        "source_fingerprint": book.source_fingerprint,
                        "authorization_status": "unverified",
                        "lessons": lessons,
                    }
                ],
            }
            # Do NOT call the curated selector: zero candidates is a valid
            # analysis outcome, never an approved fact or episode brief.
            export_path = base / "exports" / ref.book_id / artifact_name
            acquisition._private_write(
                export_path,
                (json.dumps(export, ensure_ascii=False, sort_keys=True) + "\n").encode(),
                private_root=self.root / "book-candidates",
            )
        return {
            "book_id": ref.book_id,
            "status": outcome,
            "recovery_status": book.recovery_status,
            "review_status": "MINED_UNREVIEWED",
            "publish_eligible": False,
            "candidate_count": len(lessons),
            "total_work_units": len(work.units),
            **(
                {
                    "section_scope": work.section_scope,
                    "section_manifest_sha256": section_manifest_sha256,
                    "parent_full_book_complete": False,
                }
                if section_ids is not None
                else {}
            ),
            "counts": counts,
            "progress_path": str(progress_path),
            "export_path": str(export_path) if export_path is not None else None,
        }

    def ingest(
        self, title: str, author: str, resolver: ingest.SourceResolver
    ) -> BookText | SourceUnresolved:
        """Resolve exactly one source; parse inside the content-addressed ingest stage."""
        matches = resolver.resolve(title, author)
        if isinstance(matches, SourceUnresolved):
            return matches
        if len(matches) != 1:
            raise ingest.BookIngestError(f"ambiguous source resolver results: {len(matches)}")
        source = matches[0]
        raw = resolver.read(source)
        sha256 = hashlib.sha256(raw).hexdigest()
        if source.source_kind == "public_domain":
            source = replace(source, content_sha256=sha256)
        elif source.content_sha256 != sha256:
            raise ingest.BookIngestError("source changed between lookup and read: SHA-256 mismatch")
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS["ingest"],
            source_hash=content_fingerprint(
                {"title": title, "author": author, "source": source.fingerprint, "bytes": sha256}
            ),
        )
        stored, _reused = self.dag.execute(identity, lambda: asdict(ingest.parse_book(source, raw)))
        book = _book_text_cached(stored)
        if book.content_sha256 != sha256:
            raise BookRunError("cached ingest content SHA-256 does not match source bytes")
        if book.source.authorization_status != "unverified":
            raise BookRunError("ingest must not upgrade source authorization")
        snapshot = json.dumps(asdict(book), ensure_ascii=False, sort_keys=True).encode("utf-8")
        scripting._write_atomic(self.root / "book-text" / f"{_book_slug(title)}.json", snapshot)
        return book

    def load_text(self, book_id: str) -> BookText:
        if not selection.IDENTIFIER.fullmatch(book_id):
            raise BookRunError("invalid book-id for source lookup")
        path = self.root / "book-text" / f"{book_id}.json"
        if not path.is_file():
            raise BookRunError(f"ingested BookText missing for {book_id}; run books ingest first")
        book = _book_text_cached(json.loads(path.read_text(encoding="utf-8")))
        if book.source_fingerprint != book.source.fingerprint:
            raise BookRunError("stored BookText source fingerprint mismatch")
        if (
            sum(len(p.split()) for section in book.sections for p in section.paragraphs)
            != book.total_word_count
        ):
            raise BookRunError("stored BookText word-count mismatch")
        if _book_slug(book.source.title) != book_id:
            raise BookRunError("stored BookText book-id does not match source title")
        return book

    def mine(
        self,
        book: BookText,
        *,
        completion: Callable[[dict[str, object]], str],
        response_sha256: str,
    ) -> dict[str, Any]:
        """Cache only explicitly recorded completion identities; never load model weights."""
        if not re.fullmatch(r"[0-9a-f]{64}", response_sha256):
            raise BookRunError("mining requires exact recorded-response SHA-256")
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS["mining"],
            source_hash=content_fingerprint(
                {"book": book.fingerprint, "response_sha256": response_sha256}
            ),
            dependency_output_hashes=(book.fingerprint,),
            model_revision=scripting.SCRIPT_REVISION,
            decoding_parameters={
                "seed": 42,
                "max_new_tokens": 960,
                "chunk_words": mining.CHUNK_WORDS,
            },
        )
        stored, _reused = self.dag.execute(identity, lambda: mining.mine_book(book, completion))
        if not isinstance(stored, dict) or stored.get("ingest_fingerprint") != book.fingerprint:
            raise BookRunError("cached mined catalog is inconsistent with the ingested book")
        if stored.get("review_status") != "MINED_UNREVIEWED":
            raise BookRunError("mined catalog review gate missing")
        selection._validate_catalog(stored)
        snapshot = json.dumps(stored, ensure_ascii=False, sort_keys=True).encode("utf-8")
        scripting._write_atomic(
            self.root / "mined" / f"{_book_slug(book.source.title)}.json", snapshot
        )
        return stored

    def plan(
        self,
        catalog: dict[str, Any],
        storyboard: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        *,
        catalog_dir: Path = selection.CATALOG_PATH.parent,
        storyboard_path: Path | None = None,
        book_id: str | None = None,
        lesson_id: str | None = None,
        angle_id: str | None = None,
        pillar: str | None = None,
    ) -> BookPlan:
        history = [] if history is None else history
        options = {
            "book_id": book_id,
            "lesson_id": lesson_id,
            "angle_id": angle_id,
            "pillar": pillar,
        }
        original = selection.choose_lesson(catalog, history, **options)
        if storyboard is None:
            board, provenance = storyboards.resolve_template(
                catalog,
                original,
                catalog_dir=catalog_dir,
                override=storyboard_path,
            )
        else:
            storyboards.declared_filename(catalog, original["book_id"])
            storyboards.validate_template(storyboard, original)
            board = storyboard
            provenance = StoryboardProvenance(
                book_id=original["book_id"],
                template_path=str(storyboard_path) if storyboard_path else "<in-memory>",
                template_sha256=content_fingerprint(storyboard),
                schema=storyboards.SCHEMA,
            )
        evidence = asdict(provenance)
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS["selection"],
            source_hash=content_fingerprint(
                {
                    "catalog": catalog,
                    "history": history,
                    "options": options,
                    "storyboard": evidence,
                }
            ),
        )

        def select() -> object:
            return {
                "selection": asdict(_selected_payload(original)),
                "storyboard": evidence,
            }

        stored, _reused = self.dag.execute(identity, select)
        if not isinstance(stored, dict) or stored.get("storyboard") != evidence:
            raise BookRunError("cached storyboard provenance differs from verified template")
        selected = _selected_cached(stored.get("selection"))
        if original["fingerprint"] != selected.legacy_fingerprint:
            raise BookRunError("cached selection disagrees with current verified catalog")
        niche = selection.to_niche_config(original)
        outline = selection.to_storyboard_outline(original, board)

        def encode(value: object) -> str:
            return json.dumps(value, indent=2, ensure_ascii=False) + "\n"

        return BookPlan(
            selected,
            encode(original),
            encode(niche),
            encode(outline),
            provenance,
        )

    def script(
        self,
        plan: BookPlan,
        *,
        source: Path | None = None,
        completion: Callable[[dict[str, object]], str] | None = None,
    ) -> EpisodeScript:
        config = _niche_config(plan)
        script_key = scripting._script_fingerprint(config)
        source_sha256 = (
            hashlib.sha256(source.read_bytes()).hexdigest() if source is not None else None
        )
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS["scripting"],
            source_hash=content_fingerprint(
                {
                    "selection": plan.selected.fingerprint,
                    "script_request": script_key,
                    "recorded_script_sha256": source_sha256,
                }
            ),
            dependency_output_hashes=(plan.selected.fingerprint,),
            model_revision=scripting.SCRIPT_REVISION,
            decoding_parameters={"seed": 42, "max_new_tokens": 960},
        )

        def compose() -> object:
            if source is not None:
                raw = json.loads(source.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "script" in raw:
                    if raw.get("episode_id") != plan.selected.episode_id:
                        raise BookRunError("recorded script episode does not match selection")
                    raw = raw["script"]
                episode = scripting.Episode.from_json(raw)
                cache_key = script_key
            else:
                episode, cache_key = scripting.generate_episode(
                    config, self.script_cache_root, completion=completion
                )
            return asdict(
                EpisodeScript(
                    episode_id=plan.selected.episode_id,
                    selection_fingerprint=plan.selected.legacy_fingerprint,
                    script_cache_key=cache_key,
                    title=episode.title,
                    scenes=tuple(
                        ScriptScene(scene.headline, scene.narration) for scene in episode.scenes
                    ),
                )
            )

        stored, _reused = self.dag.execute(identity, compose)
        return _script_cached(stored)

    def narrate(
        self,
        plan: BookPlan,
        script: EpisodeScript,
        *,
        recorded_dir: Path | None = None,
        producer: Callable[[str, str], bytes] | None = None,
    ) -> tuple[NarrationTrack, ...]:
        """Run the complete scene narration stage, or hash owner-recorded WAV fixtures."""
        if script.episode_id != plan.selected.episode_id:
            raise BookRunError("narration script episode does not match selection")
        if script.selection_fingerprint != plan.selected.legacy_fingerprint:
            raise BookRunError("narration script fingerprint does not match selection")
        config = _niche_config(plan)
        scenes = tuple(scripting.Scene(item.headline, item.narration) for item in script.scenes)
        recordings: tuple[bytes, ...] | None = None
        if recorded_dir is not None:
            if not recorded_dir.is_dir():
                raise BookRunError("recorded narration directory is missing")
            expected = {f"scene-{index:02d}.wav" for index in range(len(scenes))}
            found = {item.name for item in recorded_dir.glob("scene-*.wav")}
            if found != expected:
                raise BookRunError("recorded narration requires exactly one scene-NN.wav per scene")
            recordings = tuple(
                (recorded_dir / f"scene-{index:02d}.wav").read_bytes()
                for index in range(len(scenes))
            )
        recorded_hashes = (
            [hashlib.sha256(payload).hexdigest() for payload in recordings]
            if recordings is not None
            else None
        )
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS["narration"],
            source_hash=content_fingerprint(
                {
                    "script": script.fingerprint,
                    "requests": [narration.tts_request(scene, config) for scene in scenes],
                    "recorded_sha256": recorded_hashes,
                    "mode": "recorded_wav" if recordings is not None else "kokoro",
                }
            ),
            dependency_output_hashes=(script.fingerprint,),
            model_revision=scripting.VOICE_REVISION,
            decoding_parameters={
                "model": scripting.VOICE_MODEL,
                "voice": config.voice,
                "contract": scripting.VOICE_CONTRACT,
                "language": "en-us",
                "sample_rate": 24000,
                "speed": 1.0,
            },
        )

        def synthesize() -> object:
            tracks: list[dict[str, object]] = []
            for index, scene in enumerate(scenes):
                if recordings is not None:
                    payload = recordings[index]
                else:
                    path = narration.synthesize_scene(
                        scene, config, self.script_cache_root, producer=producer
                    )
                    payload = path.read_bytes()
                _channels, rate, _frames = narration.validated_wav(payload)
                track = NarrationTrack(
                    episode_id=script.episode_id,
                    voice_id=config.voice,
                    audio_sha256=hashlib.sha256(payload).hexdigest(),
                    sample_rate=rate,
                )
                tracks.append(asdict(track))
            return tracks

        stored, reused = self.dag.execute(identity, synthesize)
        tracks = _narration_cached(stored, len(scenes))
        if reused and recordings is None:
            for scene, track in zip(scenes, tracks, strict=True):
                path = narration.tts_cache_path(scene, config, self.script_cache_root)
                if not path.is_file():
                    raise BookRunError("cached narration WAV missing on DAG resume")
                audio = path.read_bytes()
                if hashlib.sha256(audio).hexdigest() != track.audio_sha256:
                    raise BookRunError("cached narration WAV hash mismatch on DAG resume")
                narration.validated_wav(audio, cached=True)
        return tracks

    def evaluate_candidate(
        self,
        ref_path: Path,
        *,
        progress_path: Path | None = None,
        section_manifest: Path | None = None,
        review_file: Path | None = None,
        options: evaluation.EvaluationOptions | None = None,
        recorded_responses: dict[str, object] | None = None,
        response_file_sha256: str | None = None,
        mine_completion: Callable[[dict[str, object]], str] | None = None,
        script_completion: Callable[[dict[str, object]], str] | None = None,
        direction_completion: Callable[[dict[str, object]], str] | None = None,
        narration_producer: Callable[[str, str], bytes] | None = None,
    ) -> dict[str, object]:
        """Private source-bound assisted previsual evaluation; never catalog or video."""
        return evaluation.evaluate_candidate(
            self,
            ref_path,
            progress_path=progress_path,
            section_manifest=section_manifest,
            review_file=review_file,
            options=options if options is not None else evaluation.EvaluationOptions(),
            recorded_responses=recorded_responses,
            response_file_sha256=response_file_sha256,
            mine_completion=mine_completion,
            script_completion=script_completion,
            direction_completion=direction_completion,
            narration_producer=narration_producer,
        )

    def require_unported(
        self, stage: str, *, source_hash: str, dependency_hashes: tuple[str, ...]
    ) -> NoReturn:
        if stage not in BOOK_STAGE_CONTRACTS or stage in {"selection", "scripting", "narration"}:
            raise BookRunError(f"unknown or implemented book stage: {stage}")
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS[stage],
            source_hash=source_hash,
            dependency_output_hashes=dependency_hashes,
        )

        def not_ported() -> NoReturn:
            raise NotPortedStage(f"NOT_PORTED: books.{stage}; no producer was invoked")

        self.dag.execute(identity, not_ported)
        raise AssertionError("unreachable: unported stage unexpectedly returned")

    def run(
        self,
        catalog: dict[str, Any],
        storyboard: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        *,
        catalog_dir: Path = selection.CATALOG_PATH.parent,
        storyboard_path: Path | None = None,
        source: Path | None = None,
        completion: Callable[[dict[str, object]], str] | None = None,
        narration_dir: Path | None = None,
        narration_producer: Callable[[str, str], bytes] | None = None,
        book_id: str | None = None,
        lesson_id: str | None = None,
        angle_id: str | None = None,
        pillar: str | None = None,
    ) -> NoReturn:
        plan = self.plan(
            catalog,
            storyboard,
            history,
            catalog_dir=catalog_dir,
            storyboard_path=storyboard_path,
            book_id=book_id,
            lesson_id=lesson_id,
            angle_id=angle_id,
            pillar=pillar,
        )
        script = self.script(plan, source=source, completion=completion)
        tracks = self.narrate(plan, script, recorded_dir=narration_dir, producer=narration_producer)
        narrated_hash = content_fingerprint([asdict(track) for track in tracks])
        self.require_unported(
            "direction",
            source_hash=narrated_hash,
            dependency_hashes=(narrated_hash,),
        )

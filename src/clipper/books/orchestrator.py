"""Book stage DAG: deterministic selection, pinned Qwen script, unported stages blocked."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, NoReturn, cast

from clipper.dag import DagStore
from clipper.stage_contracts import content_fingerprint, stage_identity

from . import ingest, mining, narration, scripting, selection, storyboards
from .contracts import (
    BOOK_STAGE_CONTRACTS,
    BookRef,
    BookSection,
    BookSource,
    BookText,
    EpisodeScript,
    LessonSelection,
    NarrationTrack,
    ScriptScene,
    SourceUnresolved,
)
from .storyboards import StoryboardProvenance


class BookRunError(ValueError):
    """Book run input or stored stage output is inconsistent."""


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
        self.dag = DagStore(artifact_root / "book-dag")
        self.script_cache_root = artifact_root / "book-script-cache"

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
        stored, _reused = self.dag.execute(
            identity, lambda: asdict(ingest.parse_book(source, raw))
        )
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
        if sum(
            len(p.split()) for section in book.sections for p in section.paragraphs
        ) != book.total_word_count:
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
        stored, _reused = self.dag.execute(
            identity, lambda: mining.mine_book(book, completion)
        )
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

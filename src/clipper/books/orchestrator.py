"""Book stage DAG: deterministic selection, pinned Qwen script, unported stages blocked."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from clipper.dag import DagStore
from clipper.stage_contracts import content_fingerprint, stage_identity

from . import scripting, selection
from .contracts import (
    BOOK_STAGE_CONTRACTS,
    BookRef,
    EpisodeScript,
    LessonSelection,
    ScriptScene,
)


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
    return LessonSelection(book=BookRef(**record["book"]), **{
        key: item for key, item in record.items() if key != "book"
    })


def _script_cached(value: object) -> EpisodeScript:
    if not isinstance(value, dict) or not isinstance(value.get("scenes"), list):
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
        self.dag = DagStore(artifact_root / "book-dag")
        self.script_cache_root = artifact_root / "book-script-cache"

    def plan(
        self,
        catalog: dict[str, Any],
        storyboard: dict[str, Any],
        history: list[dict[str, str]] | None = None,
        *,
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
        identity = stage_identity(
            BOOK_STAGE_CONTRACTS["selection"],
            source_hash=content_fingerprint(
                {"catalog": catalog, "history": history, "options": options}
            ),
        )

        def select() -> object:
            original = selection.choose_lesson(catalog, history, **options)
            return asdict(_selected_payload(original))

        stored, _reused = self.dag.execute(identity, select)
        selected = _selected_cached(stored)
        original = selection.choose_lesson(catalog, history, **options)
        if original["fingerprint"] != selected.legacy_fingerprint:
            raise BookRunError("cached selection disagrees with current verified catalog")
        niche = selection.to_niche_config(original)
        outline = selection.to_storyboard_outline(original, storyboard)
        def encode(value: object) -> str:
            return json.dumps(value, indent=2, ensure_ascii=False) + "\n"

        return BookPlan(selected, encode(original), encode(niche), encode(outline))

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

    def require_unported(
        self, stage: str, *, source_hash: str, dependency_hashes: tuple[str, ...]
    ) -> NoReturn:
        if stage not in BOOK_STAGE_CONTRACTS or stage in {"selection", "scripting"}:
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
        storyboard: dict[str, Any],
        history: list[dict[str, str]] | None = None,
        *,
        source: Path | None = None,
        completion: Callable[[dict[str, object]], str] | None = None,
        book_id: str | None = None,
        lesson_id: str | None = None,
        angle_id: str | None = None,
        pillar: str | None = None,
    ) -> NoReturn:
        plan = self.plan(
            catalog, storyboard, history,
            book_id=book_id, lesson_id=lesson_id, angle_id=angle_id, pillar=pillar,
        )
        script = self.script(plan, source=source, completion=completion)
        self.require_unported(
            "narration",
            source_hash=script.fingerprint,
            dependency_hashes=(script.fingerprint,),
        )

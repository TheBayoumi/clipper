from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    pipeline = "src/clipper/pipeline.py"
    replace_once(
        pipeline,
        '''_VISUAL_CHECKPOINT_COMMIT: ContextVar[Callable[[], None] | None] = ContextVar(
    "clipper_visual_checkpoint_commit", default=None
)
''',
        '''_VISUAL_CHECKPOINT_COMMIT: ContextVar[Callable[[], None] | None] = ContextVar(
    "clipper_visual_checkpoint_commit", default=None
)
_REQUIRED_VISUAL_ENTITIES: ContextVar[tuple[str, ...]] = ContextVar(
    "clipper_required_visual_entities", default=()
)
''',
    )
    replace_once(
        pipeline,
        '''    provider: VisionProvider,
    run_dir: Path,
    required_visual_entities: tuple[str, ...] = (),
) -> tuple[VisualTimeline, dict[str, object]]:
''',
        '''    provider: VisionProvider,
    run_dir: Path,
) -> tuple[VisualTimeline, dict[str, object]]:
''',
    )
    replace_once(
        pipeline,
        "        required_visual_entities=required_visual_entities,\n",
        "        required_visual_entities=_REQUIRED_VISUAL_ENTITIES.get(),\n",
    )
    replace_once(
        pipeline,
        '''            checkpoint_dir_token = _VISUAL_CHECKPOINT_DIR.set(cache_root / "source-policy-vision")
            checkpoint_commit_token = _VISUAL_CHECKPOINT_COMMIT.set(checkpoint_commit)
            try:
                visual, visual_meta = _visual_timeline(
                    media_path,
                    video,
                    timeline,
                    scout,
                    run_dir,
                    required_visual_entities=brief.acceptance_policy.visual_presence.require_any,
                )
            finally:
                _VISUAL_CHECKPOINT_COMMIT.reset(checkpoint_commit_token)
                _VISUAL_CHECKPOINT_DIR.reset(checkpoint_dir_token)
''',
        '''            checkpoint_dir_token = _VISUAL_CHECKPOINT_DIR.set(cache_root / "source-policy-vision")
            checkpoint_commit_token = _VISUAL_CHECKPOINT_COMMIT.set(checkpoint_commit)
            required_visual_entities_token = _REQUIRED_VISUAL_ENTITIES.set(
                brief.acceptance_policy.visual_presence.require_any
            )
            try:
                visual, visual_meta = _visual_timeline(
                    media_path,
                    video,
                    timeline,
                    scout,
                    run_dir,
                )
            finally:
                _REQUIRED_VISUAL_ENTITIES.reset(required_visual_entities_token)
                _VISUAL_CHECKPOINT_COMMIT.reset(checkpoint_commit_token)
                _VISUAL_CHECKPOINT_DIR.reset(checkpoint_dir_token)
''',
    )


if __name__ == "__main__":
    main()

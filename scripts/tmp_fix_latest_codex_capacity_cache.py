# ruff: noqa
from pathlib import Path


source_path = Path("src/clipper/visual_ai.py")
source = source_path.read_text(encoding="utf-8")
old = '''def _persist_capacity_state(
    cache: FileCache | None,
    key: str | None,
    *,
    largest_good: int,
    smallest_bad: int | None,
    observed_output_tokens_per_item: float | None,
    checkpoint_commit: Callable[[], None] | None,
) -> None:
    if cache is None or key is None:
        return
    cache.write(
        key,
        "capacity",
        {
            "largest_good": largest_good,
            "smallest_bad": smallest_bad,
            "observed_output_tokens_per_item": observed_output_tokens_per_item,
        },
    )
    _best_effort_checkpoint_commit(checkpoint_commit)
'''
new = '''def _persist_capacity_state(
    cache: FileCache | None,
    key: str | None,
    *,
    largest_good: int,
    smallest_bad: int | None,
    observed_output_tokens_per_item: float | None,
    checkpoint_commit: Callable[[], None] | None,
) -> None:
    if cache is None or key is None:
        return
    payload = {
        "largest_good": largest_good,
        "smallest_bad": smallest_bad,
        "observed_output_tokens_per_item": observed_output_tokens_per_item,
    }
    for attempt in range(1, 4):
        try:
            cache.write(key, "capacity", payload)
            break
        except Exception as exc:
            LOGGER.warning(
                "vision capacity cache write failed (attempt %d/3): %s: %s",
                attempt,
                type(exc).__name__,
                exc,
            )
            if attempt < 3:
                time.sleep(0.1 * attempt)
    else:
        return
    _best_effort_checkpoint_commit(checkpoint_commit)
'''
if old not in source:
    raise RuntimeError("capacity persistence block drifted")
source_path.write_text(source.replace(old, new, 1), encoding="utf-8")


test_path = Path("tests/test_visual_ai.py")
test_source = test_path.read_text(encoding="utf-8")
import_anchor = "import pytest\n\nfrom clipper.providers.base"
if import_anchor not in test_source:
    raise RuntimeError("visual-ai test import anchor drifted")
test_source = test_source.replace(
    import_anchor,
    "import pytest\n\nfrom clipper.cache import FileCache\nfrom clipper.providers.base",
    1,
)
regression = '''


def test_source_policy_capacity_cache_write_failure_does_not_discard_successful_inference(
    tmp_path: Path,
) -> None:
    provider = PolicyVision()
    attempts = {"count": 0}
    original_write = FileCache.write

    def failing_capacity_write(
        cache: FileCache,
        key: str,
        name: str,
        payload: object,
    ) -> Path:
        if name == "capacity":
            attempts["count"] += 1
            raise OSError("transient capacity cache write failure")
        return original_write(cache, key, name, payload)

    with (
        patch("clipper.visual_ai.media_duration_seconds", return_value=12.0),
        patch(
            "clipper.visual_ai.extract_video_frames",
            side_effect=_prepared_source_policy_frames,
        ),
        patch.object(FileCache, "write", new=failing_capacity_write),
        patch("clipper.visual_ai.time.sleep"),
    ):
        timeline, result = scout_visual_timeline(
            tmp_path / "source.mp4",
            provider,
            video_id="v",
            source_hash="h",
            duration=12.0,
            output_dir=tmp_path / "frames-capacity-write-best-effort",
            checkpoint_dir=tmp_path / "cache-capacity-write-best-effort",
        )

    assert timeline.events
    assert result.usage.input_units > 0
    assert attempts["count"] >= 3
'''
regression_name = (
    "test_source_policy_capacity_cache_write_failure_does_not_discard_successful_inference"
)
if regression_name in test_source:
    raise RuntimeError("capacity-cache regression test already exists")
test_path.write_text(test_source.rstrip() + regression + "\n", encoding="utf-8")

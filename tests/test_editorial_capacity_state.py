from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clipper.editorial_capacity_state import (
    load_editorial_capacity_state,
    merge_editorial_capacity_state,
    write_editorial_capacity_state,
)


def test_capacity_state_merge_preserves_conservative_independent_boundaries() -> None:
    current = {
        "source_hazards": {
            "largest_good_input_tokens": 10_000,
            "largest_dynamic_good_input_tokens": 9_000,
            "smallest_bad_input_tokens": 30_000,
            "smallest_dynamic_oom_input_tokens": 28_000,
            "output_tokens_per_input_token": 0.05,
            "cuda_memory_by_device": {"0": {"allocated": 1}},
        },
        "semantic_cores": {"largest_good_input_tokens": 8_000},
    }
    incoming = {
        "source_hazards": {
            "largest_good_input_tokens": 15_000,
            "largest_offloaded_good_input_tokens": 14_000,
            "smallest_bad_input_tokens": 25_000,
            "smallest_dynamic_oom_input_tokens": 29_000,
            "smallest_offloaded_oom_input_tokens": 27_000,
            "output_tokens_per_input_token": 0.07,
            "successful_cache_implementation": "offloaded",
        },
        "quality_windows": {"smallest_bad_input_tokens": 12_000},
    }

    merged = merge_editorial_capacity_state(current, incoming)

    hazards = merged["source_hazards"]
    assert hazards["largest_good_input_tokens"] == 15_000
    assert hazards["largest_dynamic_good_input_tokens"] == 9_000
    assert hazards["largest_offloaded_good_input_tokens"] == 14_000
    assert hazards["smallest_bad_input_tokens"] == 25_000
    assert hazards["smallest_dynamic_oom_input_tokens"] == 28_000
    assert hazards["smallest_offloaded_oom_input_tokens"] == 27_000
    assert hazards["output_tokens_per_input_token"] == 0.07
    assert hazards["successful_cache_implementation"] == "offloaded"
    assert merged["semantic_cores"]["largest_good_input_tokens"] == 8_000
    assert merged["quality_windows"]["smallest_bad_input_tokens"] == 12_000

    assert current["source_hazards"]["largest_good_input_tokens"] == 10_000
    assert "largest_offloaded_good_input_tokens" not in current["source_hazards"]


def test_capacity_state_atomic_writer_uses_unique_temps_under_contention(tmp_path: Path) -> None:
    path = tmp_path / "capacity.json"

    def write(index: int) -> None:
        write_editorial_capacity_state(
            path,
            {"source_hazards": {"largest_good_input_tokens": index + 1}},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(8)))

    loaded = load_editorial_capacity_state(path)
    assert isinstance(loaded["source_hazards"]["largest_good_input_tokens"], int)
    assert not list(tmp_path.glob(".*.tmp"))


def test_capacity_state_loader_fails_closed_on_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "capacity.json"
    path.write_text("{broken", encoding="utf-8")
    assert load_editorial_capacity_state(path) == {}

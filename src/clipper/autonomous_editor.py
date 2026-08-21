from __future__ import annotations

from collections.abc import Callable

from .cache import FileCache
from .providers.base import EditorialProvider, EmbeddingProvider


class AutonomousEditorialPlanner:
    """Temporary progress bridge for the authoritative autonomous quality graph.

    Historical editorial planning, quota selection, hook taxonomies, embedding-based
    deduplication, and versioned cache compatibility were removed. The production
    pipeline performs semantic discovery and quality selection exclusively through
    `plan_quality_batch` / `AutonomousQualityPlanner`.

    This class remains only until the pipeline's progress plumbing is moved directly
    onto the quality graph. It performs no editorial inference and must never affect
    source selection, clip yield, boundaries, or ranking.
    """

    def __init__(
        self,
        editorial: EditorialProvider,
        embeddings: EmbeddingProvider,
        cache: FileCache,
        *,
        max_words_per_chunk: int = 900,
        chunk_overlap_words: int = 120,
        semantic_duplicate_threshold: float = 0.9,
        hook_duplicate_threshold: float = 0.94,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        del embeddings, cache, semantic_duplicate_threshold, hook_duplicate_threshold
        if max_words_per_chunk < 200:
            raise ValueError("max_words_per_chunk must be at least 200")
        if not 0 <= chunk_overlap_words < max_words_per_chunk:
            raise ValueError("chunk_overlap_words must be smaller than chunk size")
        self.editorial = editorial
        self.max_words_per_chunk = max_words_per_chunk
        self.chunk_overlap_words = chunk_overlap_words
        self.progress_callback = progress_callback

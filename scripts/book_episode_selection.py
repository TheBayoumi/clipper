"""Compatibility CLI: canonical source-backed selection is in clipper.books.selection."""

from __future__ import annotations

from clipper.books.selection import (
    CATALOG_PATH,
    IDENTIFIER,
    RECENT_BOOK_WINDOW,
    RECENT_LESSON_WINDOW,
    SCHEMA,
    SCRIPT_VERSION,
    BookSelectionError,
    _angles,
    _digest,
    _source_url,
    _validate_catalog,
    choose_lesson,
    main,
    to_niche_config,
    to_storyboard_outline,
)

if __name__ == "__main__":
    main()

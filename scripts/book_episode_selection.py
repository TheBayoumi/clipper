"""Compatibility CLI: canonical source-backed selection is in clipper.books.selection."""

from __future__ import annotations

from clipper.books.selection import (
    CATALOG_PATH as CATALOG_PATH,
    IDENTIFIER as IDENTIFIER,
    RECENT_BOOK_WINDOW as RECENT_BOOK_WINDOW,
    RECENT_LESSON_WINDOW as RECENT_LESSON_WINDOW,
    SCHEMA as SCHEMA,
    SCRIPT_VERSION as SCRIPT_VERSION,
    BookSelectionError as BookSelectionError,
    _angles as _angles,
    _digest as _digest,
    _source_url as _source_url,
    _validate_catalog as _validate_catalog,
    choose_lesson as choose_lesson,
    main,
    to_niche_config as to_niche_config,
    to_storyboard_outline as to_storyboard_outline,
)

if __name__ == "__main__":
    main()

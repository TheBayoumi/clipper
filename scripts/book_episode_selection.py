"""Compatibility CLI: canonical book selection lives in clipper.books.selection."""

from __future__ import annotations

from clipper.books import selection as _selection

CATALOG_PATH = _selection.CATALOG_PATH
IDENTIFIER = _selection.IDENTIFIER
RECENT_BOOK_WINDOW = _selection.RECENT_BOOK_WINDOW
RECENT_LESSON_WINDOW = _selection.RECENT_LESSON_WINDOW
SCHEMA = _selection.SCHEMA
SCRIPT_VERSION = _selection.SCRIPT_VERSION
BookSelectionError = _selection.BookSelectionError
_angles = _selection._angles
_digest = _selection._digest
_source_url = _selection._source_url
_validate_catalog = _selection._validate_catalog
choose_lesson = _selection.choose_lesson
main = _selection.main
to_niche_config = _selection.to_niche_config
to_storyboard_outline = _selection.to_storyboard_outline

if __name__ == "__main__":
    main()

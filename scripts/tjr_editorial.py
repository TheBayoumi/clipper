"""Compatibility imports for the packaged Clipper editorial engine.

Production callers should import :mod:`clipper.editorial` directly.
"""

from clipper.editorial import (
    MAX_RENDERABLE_CLIPS,
    RUBRIC_VERSION,
    WEIGHTS,
    CriterionRating,
    EditorialPick,
    EditorialReview,
    candidate_gate_failures,
    evaluate_candidate,
    select_editorial_moments,
)

__all__ = [
    "MAX_RENDERABLE_CLIPS",
    "RUBRIC_VERSION",
    "WEIGHTS",
    "CriterionRating",
    "EditorialPick",
    "EditorialReview",
    "candidate_gate_failures",
    "evaluate_candidate",
    "select_editorial_moments",
]

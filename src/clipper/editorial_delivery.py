"""Post-editorial, source-only checks before rendering or independent QA.

These never mutate saved selector or model-review decisions. An unresolved teaser
is not a delivered payoff, even if a model has found syntactically valid spans.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .models import ClipCandidate, TranscriptSegment

_OPEN_PROMISE = re.compile(
    r"\b(?:i(?:'ll| will)|we(?:'ll| will)|let me|i(?:'m| am) going to)\s+"
    r"(?:tell|show|explain|reveal)\b|\b(?:coming up|stay tuned|after the break)\b",
    re.IGNORECASE,
)


def unresolved_teaser(clip: ClipCandidate, segments: Sequence[TranscriptSegment]) -> bool:
    """Block a source-backed future story promise that ends without a delivered story.

    Inspect only the speech that *actually appears* in the cut; excluded context
    cannot be credited as a payoff. Narrow to promises within the final seconds,
    not arbitrary mentions of storytelling earlier in a complete anecdote.
    """
    final = [
        item.text
        for item in segments
        if item.end > max(clip.start, clip.end - 6.5) and item.start < clip.end
    ]
    if not final:
        return False
    text = " ".join(final)
    return bool(_OPEN_PROMISE.search(text))

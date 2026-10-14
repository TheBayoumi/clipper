"""Source-confirmed caption corrections, leaving original transcript untouched.

No guessed words or timestamps: the canonical name must already be spoken
within the selected original clip. Rules are an inspectable campaign glossary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .models import ClipCandidate, TranscriptSegment, WordTiming


def correct_verified_caption_names(
    clip: ClipCandidate, segments: Sequence[TranscriptSegment], lexicon_file: Path
) -> list[TranscriptSegment]:
    raw = json.loads(lexicon_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "source-confirmed-caption-names-v1":
        raise ValueError("caption lexicon needs source-confirmation schema")
    items = raw.get("names")
    if not isinstance(items, list):
        raise ValueError("caption lexicon needs verified canonical-name entries")
    result = list(segments)
    selected = " ".join(x.text for x in segments if x.end > clip.start and x.start < clip.end)
    for rule in items:
        if not isinstance(rule, dict) or set(rule) != {"canonical", "asr_variants"}:
            raise ValueError("caption lexicon entry has an invalid contract")
        canonical, variants = rule["canonical"], rule["asr_variants"]
        if (
            not isinstance(canonical, str)
            or not isinstance(variants, list)
            or not canonical.strip()
            or any(not isinstance(v, str) or not v.strip() for v in variants)
            or len(set(v.casefold() for v in variants)) != len(variants)
        ):
            raise ValueError("caption lexicon entry has invalid names")
        if not re.search(r"\b" + re.escape(canonical) + r"\b", selected, re.IGNORECASE):
            continue
        tokens = canonical.split()
        for alias in variants:
            alias_tokens = alias.split()
            if len(alias_tokens) != len(tokens) or len(tokens) < 2:
                raise ValueError("caption aliases must retain source word-timing positions")
            expression = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
            for i, segment in enumerate(result):
                if segment.end <= clip.start or segment.start >= clip.end:
                    continue
                corrected = expression.sub(canonical, segment.text)
                if corrected == segment.text:
                    continue
                words = list(segment.words)
                aligned_match = False
                for index in range(len(words) - len(alias_tokens) + 1):
                    actual = [
                        re.sub(r"\W+", "", w.text).casefold()
                        for w in words[index : index + len(alias_tokens)]
                    ]
                    expected = [re.sub(r"\W+", "", w).casefold() for w in alias_tokens]
                    if actual == expected:
                        aligned_match = True
                        for j, token in enumerate(tokens):
                            words[index + j] = WordTiming(
                                words[index + j].start, words[index + j].end, token
                            )
                if words and not aligned_match:
                    raise ValueError("caption alias has inconsistent word alignment")
                result[i] = replace(segment, text=corrected, words=tuple(words))
    return result

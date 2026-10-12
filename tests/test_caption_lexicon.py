import json
from pathlib import Path

import pytest

from clipper.caption_lexicon import correct_verified_caption_names
from clipper.models import ClipCandidate, TranscriptSegment, WordTiming

LEXICON = Path(__file__).resolve().parents[1] / "campaigns/reach-double-coverage-caption-names.json"


def _speech(start: float, text: str) -> TranscriptSegment:
    words = text.split()
    return TranscriptSegment(
        start,
        start + len(words) * 0.25,
        text,
        tuple(
            WordTiming(start + index * 0.25, start + (index + 1) * 0.25, word)
            for index, word in enumerate(words)
        ),
    )


def test_archived_brock_purdy_variants_are_source_confirmed_and_timings_unchanged():
    lines = [
        _speech(2929.33, "Brock Birdie and the 49ers start."),
        _speech(2932.03, "It's I love Brock Bertie."),
        _speech(2934.47, "I'm a Brock Purdy guy."),
        _speech(2952.75, "I love Brock Purdy."),
        _speech(2964.0, "Brock Bertie is a different character."),
    ]
    clip = ClipCandidate("3S-fL9rNPp0", 2925.95, 2963.39, "Brock Purdy", 6)
    fixed = correct_verified_caption_names(clip, lines, LEXICON)
    assert "Brock Purdy" in fixed[0].text and "Birdie" not in fixed[0].text
    assert "Brock Purdy" in fixed[1].text and "Bertie" not in fixed[1].text
    assert fixed[-1] == lines[-1]  # Outside delivered media is never altered.
    assert lines[0].text.startswith("Brock Birdie")  # Original ASR is immutable.
    for before, after in zip(lines, fixed, strict=True):
        assert [(w.start, w.end) for w in before.words] == [(w.start, w.end) for w in after.words]


def test_no_confirmed_name_no_rewrite_and_alias_length_is_guarded(tmp_path: Path):
    clip = ClipCandidate("v", 0, 22, "topic", 2)
    wrong = [_speech(1, "Brock Birdie argued with the coach.")]
    assert correct_verified_caption_names(clip, wrong, LEXICON) == wrong
    file = tmp_path / "invalid.json"
    file.write_text(
        json.dumps(
            {
                "schema": "source-confirmed-caption-names-v1",
                "names": [{"canonical": "Brock Purdy", "asr_variants": ["Brock"]}],
            }
        )
    )
    speech = [_speech(1, "Brock Purdy made the play.")]
    with pytest.raises(ValueError, match="word-timing positions"):
        correct_verified_caption_names(clip, speech, file)


def test_source_name_lexicon_rejects_invalid_contract(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text('{"schema":"invalid","names":[]}')
    with pytest.raises(ValueError, match="source-confirmation"):
        correct_verified_caption_names(ClipCandidate("v", 0, 20, "", 0), [], path)


@pytest.mark.parametrize(
    "names, error",
    [
        ("invalid", "canonical-name entries"),
        ([{"canonical": "Brock Purdy"}], "invalid contract"),
        ([{"canonical": "", "asr_variants": []}], "invalid names"),
    ],
)
def test_invalid_lexicon_fails_closed(tmp_path: Path, names: object, error: str):
    file = tmp_path / "lexicon.json"
    file.write_text(json.dumps({"schema": "source-confirmed-caption-names-v1", "names": names}))
    with pytest.raises(ValueError, match=error):
        correct_verified_caption_names(ClipCandidate("v", 0, 20, "title", 1), [], file)


def test_caption_lexicon_rejects_alias_word_timing_disagreement(tmp_path: Path):
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "schema": "source-confirmed-caption-names-v1",
                "names": [{"canonical": "Brock Purdy", "asr_variants": ["Brock Birdie"]}],
            }
        )
    )
    segments = [
        _speech(1, "Brock Purdy plays well."),
        TranscriptSegment(
            4, 5, "Brock Birdie", (WordTiming(4.0, 4.4, "Brock"), WordTiming(4.4, 5.0, "Bertie"))
        ),
    ]
    with pytest.raises(ValueError, match="inconsistent word alignment"):
        correct_verified_caption_names(ClipCandidate("v", 0, 20, "title", 1), segments, lexicon)

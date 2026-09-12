from clipper.models import SourceSpan, TranscriptSegment, TranscriptWord
from clipper.wordstream import (
    build_clip_word_stream,
    find_phrase_anchor,
    first_complete_word,
    flatten_source_words,
    segment_source_words,
)


def test_wordstream_handles_empty_cleaned_tokens_duplicates_and_missing_anchors() -> None:
    assert segment_source_words(TranscriptSegment(0, 1, ">>")) == []
    duplicate_a = TranscriptSegment(0, 1, "hello", (TranscriptWord(0, 0.5, "hello"),))
    duplicate_b = TranscriptSegment(0, 1, "hello", (TranscriptWord(0, 0.5, "hello"),))
    assert len(flatten_source_words([duplicate_a, duplicate_b])) == 1
    assert first_complete_word([duplicate_a], 0.6, 1.0) is None
    assert find_phrase_anchor([duplicate_a], 0, 1, "") is None
    assert find_phrase_anchor([duplicate_a], 0, 1, "not present") is None


def test_wordstream_maps_multiple_source_spans_to_contiguous_local_time() -> None:
    segments = [
        TranscriptSegment(
            0, 2, "one two", (TranscriptWord(0, 0.5, "one"), TranscriptWord(1, 1.5, "two"))
        ),
        TranscriptSegment(
            10,
            12,
            "three four",
            (TranscriptWord(10, 10.5, "three"), TranscriptWord(11, 11.5, "four")),
        ),
    ]
    stream, dropped = build_clip_word_stream((SourceSpan(0, 2), SourceSpan(10, 12)), segments)
    assert dropped == 0
    assert [word.text for word in stream] == ["one", "two", "three", "four"]
    assert stream[2].local_start == 2.0

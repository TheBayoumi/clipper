"""Regression cases from independently inspected 2026-10-12 Double Coverage drafts."""

from clipper.editorial_delivery import unresolved_teaser
from clipper.models import ClipCandidate, TranscriptSegment


def test_archived_trailer_does_not_qualify_as_delivered_story():
    # Faithful source words and cut from production artifact #38206696943.
    lines = [
        TranscriptSegment(8.34, 11.60, "I've had a couple teams reach out and gauge my interest."),
        TranscriptSegment(11.78, 15.86, "The UFL's actually reached out as well."),
        TranscriptSegment(15.98, 19.82, "Michael Parsons on Twitter doing podcasts."),
        TranscriptSegment(
            21.14, 24.64, "Micah's not always great for the locker room. It's a distraction."
        ),
        TranscriptSegment(25.06, 26.70, "And I'll tell you this story."),
        TranscriptSegment(26.94, 29.20, "I've told it before, but I haven't told it in this way."),
        TranscriptSegment(29.20, 32.08, "The only person who didn't come was Andrew Luck."),
    ]
    teaser = ClipCandidate("3S-fL9rNPp0", 8.34, 29.20, "teaser", 4)
    assert unresolved_teaser(teaser, lines) is True
    # A later delivered resolution is not retrospectively part of the cut.
    resolved = ClipCandidate("3S-fL9rNPp0", 29.20, 32.08, "reveal", 4)
    assert unresolved_teaser(resolved, lines) is False


def test_promised_story_must_be_near_cut_to_reject_not_an_earlier_setup():
    delivered = [
        TranscriptSegment(0, 4, "I'll tell you what the team did."),
        TranscriptSegment(5, 13, "They changed the lineup and won the final game."),
        TranscriptSegment(14, 22, "That outcome was the turning point of the season."),
    ]
    assert not unresolved_teaser(ClipCandidate("v", 0, 22, "complete", 3), delivered)
    assert not unresolved_teaser(ClipCandidate("v", 0, 22, "complete", 3), [])


def test_future_promises_at_cut_without_delivery_are_unresolved():
    for promise in [
        "We'll reveal the secret after the break.",
        "Stay tuned for what happened.",
        "Let me explain what happened.",
    ]:
        segments = [
            TranscriptSegment(10, 18, "A specific premise was established."),
            TranscriptSegment(18, 21, promise),
        ]
        assert unresolved_teaser(ClipCandidate("v", 10, 21, promise, 3), segments)


def test_archived_five_source_endings_reject_only_the_promised_story_teaser():
    # Exact final 6.5 seconds of each selected window in real artifact 38206696943.
    endings = [
        (
            2999.01,
            3030.85,
            "Every quarterback is a system quarterback. have a system that works "
            "for them. Brock Purdy is the perfect guy to run Kyle Shanahan's "
            "system.",
        ),
        (
            2418.27,
            2439.79,
            "the team to ask me to talk to them about going to Baylor? Yes. I like "
            "to consider myself the closer. So that's my job.",
        ),
        (
            2925.95,
            2963.39,
            "They're not just elitist, they're heightest. Brock is short, but what "
            "Brock Purdy does so well is he plays with great rhythm and timing.",
        ),
        (
            8.34,
            29.20,
            "for the locker room. It's a distraction. And I'll tell you this story. "
            "And I've told it before, but I haven't told it in this way.",
        ),
        (
            2324.63,
            2346.37,
            "And sometimes in life, are not going to go your way and they're not "
            "going to be rosy. But you are in charge of how you respond.",
        ),
    ]
    verdicts = []
    for start, end, words in endings:
        segments = [TranscriptSegment(end - 6.5, end, words)]
        verdicts.append(
            unresolved_teaser(ClipCandidate("3S-fL9rNPp0", start, end, words, 5), segments)
        )
    assert verdicts == [False, False, False, True, False]

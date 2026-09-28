"""Safe, full-duration editorial headlines and real word-aligned TikTok captions.

Style B v2 creates a burn-in-ready ASS sidecar without inventing spoken-word
boundaries, adding source logos, or making unverifiable financial claims.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from pathlib import Path

from PIL import ImageFont

from .models import ClipCandidate, TranscriptSegment, WordTiming

# Explicit bounding boxes in an exact 1080x1920 composition. All positions are
# fixed in the libass design resolution; font measurements include headroom for
# the active-word pop and the opaque background plate.
_HOOK_SAFE_WIDTH = 800
_CAPTION_SAFE_WIDTH = 780
_FONT = "DejaVuSans-Bold.ttf"
_WORD = re.compile(r"\S+")


def _safe(text: str) -> str:
    """Strip ASS override delimiters and line breaks from untrusted speech."""
    return re.sub(r"\s+", " ", re.sub(r"[{}\\\r\n]", " ", text)).strip()


def _ass_time(seconds: float) -> str:
    ticks = max(0, round(seconds * 100))
    hours, ticks = divmod(ticks, 360000)
    minutes, ticks = divmod(ticks, 6000)
    whole, centiseconds = divmod(ticks, 100)
    return f"{hours}:{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def creative_hook_from_text(text: str) -> str:
    """Create a compact, evidence-grounded headline from the *entire* clip.

    These are question/topic headlines, not invented returns or claims about
    footage. A real financial claim is explicitly attributed to the speaker.
    """
    original = _safe(text)
    lowered = original.lower()
    if not original:
        return ""
    stopped_order_blocks = re.search(
        r"\b(?:i|he|we|they)\s+(?:have\s+|has\s+|had\s+)?"
        r"stopped\s+using\s+(?:the\s+)?order\s+blocks?\b",
        lowered,
    )
    no_reason_order_blocks = re.search(
        r"\b(?:there(?:'s| is)|i\s+(?:see|have)|we\s+(?:see|have)|"
        r"he\s+(?:sees|has))\s+no\s+reason\s+to\s+use\s+"
        r"(?:the\s+)?order\s+blocks?\s+anymore\b",
        lowered,
    )
    negated_order_block_stop = re.search(
        r"\b(?:not|never|didn't|didnt|haven't|hasn't|hadn't|can't|cannot)\b"
        r"(?:\s+\w+){0,4}\s+stopped?\s+using\s+(?:the\s+)?order\s+blocks?\b",
        lowered,
    )
    negated_no_reason = re.search(
        r"\b(?:wouldn't|wouldnt|don't|dont|didn't|didnt|can't|cant|cannot|not|never)\b"
        r"(?:\s+\w+){0,6}\s+no\s+reason\s+to\s+use\s+"
        r"(?:the\s+)?order\s+blocks?\s+anymore\b",
        lowered,
    )
    if (
        (stopped_order_blocks and not negated_order_block_stop)
        or (no_reason_order_blocks and not negated_no_reason)
    ):
        return "WHY HE STOPPED USING ORDER BLOCKS"
    if "copy trad" in lowered and any(x in lowered for x in ("blind", "never", "don't")):
        return "WHY HE WARNS ABOUT COPY TRADING"
    # Editorially meaningful questions require affirmative source-language cues.
    # Questions invite viewing without inventing outcomes or financial guarantees.
    if "market cap" in lowered and any(
        cue in lowered for cue in ("bullshit", "scam", "looks wrong", "suspicious")
    ):
        return "WHAT MAKES THIS MARKET-CAP SETUP SUSPICIOUS?"
    wallet_buying = re.search(r"\bwallets?\s+(?:are\s+)?buying\b", lowered)
    denied_buying = re.search(
        r"\b(?:no|not|never|without)\s+(?:\w+\s+){0,2}"
        r"wallets?\s+(?:are\s+)?buying\b",
        lowered,
    )
    if (
        ("no socials" in lowered or "don't see any socials" in lowered)
        and wallet_buying
        and not denied_buying
    ):
        return "WHY ARE WALLETS BUYING WITHOUT SOCIALS?"
    sell_half = re.search(
        r"\b(?:should|would|could)\s+we\s+(?:just\s+)?sell\s+"
        r"(?:like\s+)?(?:50\s*%|half)(?=\s|[?.!,]|$)",
        lowered,
    )
    if sell_half:
        return "WOULD YOU SELL HALF HERE?"
    if (
        ("on chain" in lowered or "on-chain" in lowered)
        and "volume" in lowered
        and re.search(r"\bbuys?\b|\bbuying\b", lowered)
    ):
        return "CAN ON-CHAIN BUYS CONFIRM THE MOVE?"
    if "fomo" in lowered and re.search(r"\bbuy(?:ing|s)?\b", lowered):
        return "IS FOMO DRIVING THESE BUYS?"
    if "tweet" in lowered and "coin" in lowered:
        return "CAN A SINGLE TWEET MOVE A COIN?"
    if "coin" in lowered and any(
        cue in lowered for cue in ("sold off", "selling off", "getting sold off")
    ):
        return "WHY IS THIS COIN GETTING SOLD OFF?"
    if "community" in lowered and "good" in lowered and "bad" in lowered:
        return "GOOD COIN OR BAD COIN: HOW DO YOU TELL?"
    if "fee" in lowered and "rug" in lowered:
        return "CAN FEES FILTER OUT RUG COINS?"
    if "trade" in lowered and re.search(r"\brisk\s*-\s*free\b", lowered):
        return "IS THIS 'RISK-FREE' TRADE REALLY SAFE?"
    if "market cap" in lowered and any(
        term in lowered for term in ("late", "early", "entry", "enter", "buying")
    ):
        return "WHEN IS THE MARKET-CAP ENTRY TOO LATE?"
    if "meme coin" in lowered and any(
        x in lowered for x in ("beginner", "first", "get started", "start trading")
    ):
        return "MEMECOIN TRADING: WHERE DO YOU START?"
    million_earnings = re.search(
        r"\b(?:made|earned|profited?|up\s+over)\b(?:\s+\w+){0,5}\s+\bmillions?\b"
        r"|\bmillions?\b(?:\s+\w+){0,5}\b(?:made|earned|profited?)\b",
        lowered,
    )
    negated_million_earnings = re.search(
        r"\b(?:not|never|no|didn't|didnt|haven't|hasn't|can't|cant|cannot)\b"
        r"(?:\s+\w+){0,4}\s+\b(?:made|earned|profit(?:ed)?)\b"
        r"(?:\s+\w+){0,5}\s+\bmillions?\b",
        lowered,
    )
    if million_earnings and not negated_million_earnings:
        return "A TRADER CLAIMS MILLIONS: HOW?"
    if "stop loss" in lowered and any(
        x in lowered for x in ("got hit", "hit", "stopped out", "loss")
    ):
        return "WHAT HAPPENS WHEN THE STOP GETS HIT?"
    # Require an affirmative waiting-to-enter construction, not mere keyword
    # co-occurrence; a persistent hook must never reverse the spoken meaning.
    waiting_to_enter = re.search(
        r"\bwait(?:ing)?\b(?:\s+\w+){0,7}\s+(?:enter|entry)\b",
        lowered,
    )
    negated_wait = re.search(
        r"\b(?:not|never|no|without|didn't|don't|doesn't|cannot|can't|"
        r"cant|won't|wont|wouldn't|couldn't|shouldn't)\b"
        r"(?:\s+\w+){0,2}\s+wait(?:ing)?\b",
        lowered,
    )
    if waiting_to_enter and not negated_wait:
        return "WHY HE'S WAITING TO ENTER THIS TRADE"
    selloff = any(x in lowered for x in ("massive sell off", "massive sell-off", "massive selloff"))
    declining_short = re.search(
        r"\b(?:(?:will|would|could|should|do|does|did|can|am|is|are|was|were)\s+"
        r"not\s+short(?:ing)?|(?:won't|wont|can't|cant|cannot|don't|didn't|"
        r"wouldn't|wouldnt)\s+short(?:ing)?|never\s+short(?:ing)?|"
        r"refuse(?:d)?\s+to\s+short|avoid(?:ed)?\s+shorting|"
        r"stayed\s+away\s+from\s+shorting|not\s+going\s+to\s+short)\b",
        lowered,
    )
    if selloff and declining_short:
        return "WHY HE'S NOT SHORTING THE SELLOFF"
    if selloff and re.search(r"\bshort(?:ed|ing)?\b", lowered):
        return "WHAT'S THE SHORT SETUP IN THIS SELLOFF?"
    affirmative_excess_risk = re.search(
        r"\b(?:why\s+do\s+)?traders?\s+(?:keep\s+)?risk(?:ing)?\s+too\s+much\b"
        r"|\b(?:you|they|we)\s+(?:are\s+|keep\s+)?risk(?:ing)?\s+too\s+much\b"
        r"|\brisk(?:ed|ing)\s+too\s+much\s+(?:money|capital)\b",
        lowered,
    )
    if affirmative_excess_risk:
        return "WHY TRADERS RISK TOO MUCH"
    if (
        "stop loss" in lowered
        and re.search(r"\bmov(?:e|ed|ing)\b", lowered)
        and any(cue in lowered for cue in ("mistake", "losing", "loss", "bigger"))
    ):
        return "THE STOP-LOSS MISTAKE THAT MAKES LOSSES WORSE"
    if any(x in lowered for x in ("reversal", "reverse", "retracement")):
        return "WHAT CHANGED IN THIS MARKET SETUP?"
    if "meme coin" in lowered:
        return "WHAT MATTERS IN A MEMECOIN TRADE?"
    if "order block" in lowered:
        return "DO ORDER BLOCKS REALLY MATTER HERE?"
    # Fallback is a truthful invitation, not an invented reaction, outcome or
    # market opinion. The 2-second spoken hook remains in the audio itself.
    return "WHAT'S THE REAL TAKEAWAY HERE?"


def distinct_hook_from_text(
    text: str,
    used_hooks: Collection[str] = (),
    *,
    allow_quote_fallback: bool = False,
) -> str:
    """Choose a distinct source-grounded headline without padding the batch.

    Production TJR drafts must earn a semantic hook. Exact-quote fallbacks are
    available only for explicit review tooling; they never fill a production
    quota by default.
    """
    original = _safe(text)
    if not original:
        return ""
    used = {hook.casefold() for hook in used_hooks}
    template = creative_hook_from_text(original)
    if template != "WHAT'S THE REAL TAKEAWAY HERE?" and template.casefold() not in used:
        return template
    if not allow_quote_fallback:
        return ""

    tokens = re.findall(r"[A-Za-z0-9$']+", original)
    noise = {"and", "then", "this", "that", "like", "really", "just", "because", "the"}
    choices: list[tuple[int, int, str]] = []
    for start in range(max(0, len(tokens) - 3)):
        phrase = tokens[start : start + 5]
        if len(phrase) < 4 or phrase[0].casefold() in noise:
            continue
        snippet = " ".join(phrase)
        if len(snippet) > 37:
            continue
        relevance = sum(
            token.casefold()
            in {
                "risk",
                "trade",
                "trading",
                "market",
                "price",
                "loss",
                "stop",
                "enter",
                "entry",
                "exit",
                "profit",
                "position",
                "reversal",
                "mistake",
                "plan",
                "money",
            }
            for token in phrase
        )
        choices.append((-relevance, start, f'THE MOMENT: "{snippet.upper()}"'))
    for _, _, headline in sorted(choices):
        if headline.casefold() in used:
            continue
        try:
            _fit_lines(headline, max_width=_HOOK_SAFE_WIDTH, max_size=70)
        except ValueError:
            continue
        return headline
    return ""


def _fit_lines(
    text: str, *, max_width: int, max_size: int, min_size: int = 42
) -> tuple[int, tuple[str, ...], float]:
    """Measure true DejaVu Bold glyph widths and fit at most two safe lines.

    libass is configured for the same font in the ASS style. A 12% pop/plate
    allowance prevents transient animation from crossing the safe-area edge.
    Fail closed on an unrenderable headline rather than clipping it.
    """
    words = _safe(text).upper().split()
    if not words:
        raise ValueError("empty headline cannot be laid out")
    for size in range(max_size, min_size - 1, -2):
        font = ImageFont.truetype(_FONT, size)
        choices: list[tuple[float, tuple[str, ...]]] = []
        for split in range(1, len(words) + 1):
            lines: tuple[str, ...] = (" ".join(words[:split]),)
            if split < len(words):
                lines += (" ".join(words[split:]),)
            width = max(float(font.getlength(line)) for line in lines)
            if width * 1.12 + 40 <= max_width:
                choices.append((width, lines))
        if choices:
            width, lines = min(
                choices, key=lambda x: (max(len(y) for y in x[1]) - min(len(y) for y in x[1]), x[0])
            )
            return size, lines, width
    raise ValueError("headline exceeds two-line portrait safe area; shorten editorial headline")


def _ass_header() -> str:
    return (
        "[Script Info]\nTitle: TJR Style B v2 review overlay\n"
        "ScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Hook,DejaVu Sans,70,&H00FFFFFF,&H00FFFFFF,&H00131620,&H54131620,"
        "-1,0,0,0,100,100,0,0,3,18,0,8,120,120,185,1\n"
        "Style: Caption,DejaVu Sans,64,&H00FFFFFF,&H00FFFFFF,&H00131620,&H58131620,"
        "-1,0,0,0,100,100,0,0,3,18,0,2,120,120,375,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )


def _timed_words(
    segments: Sequence[TranscriptSegment], clip: ClipCandidate
) -> tuple[list[WordTiming], list[TranscriptSegment]]:
    aligned: list[WordTiming] = []
    fallback: list[TranscriptSegment] = []
    for segment in segments:
        if segment.end <= clip.start or segment.start >= clip.end:
            continue
        if segment.words:
            aligned.extend(
                word
                for word in segment.words
                if word.end > clip.start and word.start < clip.end and _safe(word.text)
            )
        else:
            fallback.append(segment)
    return sorted(aligned, key=lambda word: (word.start, word.end)), fallback


def _phrases(words: Sequence[WordTiming]) -> list[list[WordTiming]]:
    phrases: list[list[WordTiming]] = []
    phrase: list[WordTiming] = []
    for word in words:
        proposed = [*phrase, word]
        # A phrase remains visible while each authentic word gets its own
        # active frame interval. Preserve the original word timing untouched.
        if phrase and (
            len(proposed) > 5
            or len(" ".join(_safe(w.text) for w in proposed)) > 28
            or word.end - phrase[0].start > 2.1
            or word.start - phrase[-1].end > 0.65
            or phrase[-1].text.rstrip().endswith((".", "!", "?"))
        ):
            phrases.append(phrase)
            phrase = [word]
        else:
            phrase = proposed
    if phrase:
        phrases.append(phrase)
    return phrases


def _caption_text(phrase: Sequence[WordTiming], active: int, size: int) -> str:
    tokens: list[str] = []
    for i, word in enumerate(phrase):
        token = _safe(word.text).upper()
        if i == active:
            # Emphasis is determined exclusively by the original audio's
            # measured timestamp, never by lexical/keyword scoring.
            tokens.append(
                f"{{\\c&H0059DEFF&\\b1\\fs{size}\\t(0,85,\\fscx106\\fscy106)}}"
                + token
                + f"{{\\rCaption\\fs{size}}}"
            )
        else:
            tokens.append(token)
    return " ".join(tokens)


def create_tiktok_ass(
    clip: ClipCandidate,
    segments: Sequence[TranscriptSegment],
    output_path: str | Path,
    *,
    hook_text: str,
) -> Path:
    """Create persistent, safe creative hook and measured word-synced captions.

    Unaligned Whisper fallback segments appear as *static* captions; guessing
    intra-sentence times to simulate word synchronization is prohibited.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    events: list[str] = []
    hook = _safe(hook_text)
    if hook and clip.duration >= 1:
        size, lines, _ = _fit_lines(hook, max_width=_HOOK_SAFE_WIDTH, max_size=70)
        # No fade-out: the hook remains visible on the first and last frames.
        events.append(
            "Dialogue: 5,"
            f"{_ass_time(0)},{_ass_time(clip.duration)},"
            f"Hook,,0,0,0,,{{\\an8\\pos(540,185)\\fs{size}\\q2}}" + r"\N".join(lines)
        )
    words, unaligned = _timed_words(segments, clip)
    for phrase in _phrases(words):
        full_text = " ".join(_safe(word.text).upper() for word in phrase)
        size, lines, _ = _fit_lines(
            full_text, max_width=_CAPTION_SAFE_WIDTH, max_size=64, min_size=42
        )
        # Preserve measured word boundaries when wrapping a phrase.
        split = len(lines[0].split()) if len(lines) > 1 else len(phrase)
        for index, word in enumerate(phrase):
            start = max(word.start, clip.start) - clip.start
            end = min(word.end, clip.end) - clip.start
            if end - start < 0.04:
                continue
            first = (
                _caption_text(phrase[:split], index, size)
                if index < split
                else _caption_text(phrase[:split], -1, size)
            )
            second = (
                _caption_text(phrase[split:], index - split, size)
                if index >= split
                else _caption_text(phrase[split:], -1, size)
            )
            rendered = first + (r"\N" + second if second else "")
            events.append(
                "Dialogue: 2,"
                f"{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
                f"{{\\an2\\pos(540,1540)\\fs{size}\\q2}}{rendered}"
            )
    for segment in unaligned:
        start = max(segment.start, clip.start) - clip.start
        end = min(segment.end, clip.end) - clip.start
        if end - start <= 0.1:
            continue
        # No invented active word for a segment without real word alignment.
        fallback = " ".join(_WORD.findall(_safe(segment.text).upper())[:5])
        if fallback:
            size, lines, _ = _fit_lines(fallback, max_width=_CAPTION_SAFE_WIDTH, max_size=64)
            events.append(
                "Dialogue: 1,"
                f"{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
                f"{{\\an2\\pos(540,1540)\\fs{size}}}" + r"\N".join(lines)
            )
    output.write_text(_ass_header() + "\n".join(events) + "\n", encoding="utf-8")
    return output


def audit_tiktok_ass(path: str | Path, *, clip_duration: float) -> dict[str, object]:
    """Fail closed unless the actual ASS sidecar meets Style B v2 invariants."""
    content = Path(path).read_text(encoding="utf-8")
    hook_events = [line for line in content.splitlines() if line.startswith("Dialogue: 5,")]
    spoken = [
        line
        for line in content.splitlines()
        if line.startswith("Dialogue: 2,") and r"\c&H0059DEFF&" in line
    ]
    fallback = [line for line in content.splitlines() if line.startswith("Dialogue: 1,")]
    if len(hook_events) != 1:
        raise ValueError("Style B v2 requires one full-duration creative headline")
    fields = hook_events[0].split(",", 9)
    if fields[1:4] != [_ass_time(0), _ass_time(clip_duration), "Hook"]:
        raise ValueError("creative headline must cover the entire clip")
    if r"\an8\pos(540,185)" not in hook_events[0]:
        raise ValueError("creative headline escaped the locked safe-area anchor")
    if "Style: Hook," not in content or "Style: Caption," not in content:
        raise ValueError("missing opaque hook or caption style")
    if not spoken:
        raise ValueError("no genuinely word-timed captions were rendered")
    return {
        "style": "B2",
        "persistent_hook_seconds": clip_duration,
        "spoken_word_highlight_events": len(spoken),
        "unaligned_static_fallback_events": len(fallback),
        "editorial_visual_approval": False,
    }

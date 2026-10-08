"""Production source-position editorial engine.

Span validation establishes provenance, not semantic accuracy. The current
reviewer remains unqualified; moving it here does not authorize rendering.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

SOURCE_EVIDENCE_VERSION = "source_unit_spans_v1"
_WORD = re.compile(r"[A-Za-z0-9$%'.-]+")


def _unit_span_schema(count: int) -> dict[str, Any]:
    """Models select positions; they never author the text used as evidence."""
    return {
        "type": "object",
        "properties": {
            key: {"type": "integer", "enum": [-1, *range(count)]}
            for key in ("first_unit", "last_unit")
        },
        "required": ["first_unit", "last_unit"],
        "additionalProperties": False,
    }


def _resolve_source_units(pointer: Any, units: list[str]) -> dict[str, Any] | None:
    """Extract one contiguous range from the specified namespace, including repeats."""
    if not isinstance(pointer, dict) or set(pointer) != {"first_unit", "last_unit"}:
        raise RuntimeError("evidence pointer must contain only first_unit and last_unit")
    first, last = pointer["first_unit"], pointer["last_unit"]
    if type(first) is not int or type(last) is not int:
        raise RuntimeError("evidence positions must be integers, not booleans")
    if first == last == -1:
        return None
    if not 0 <= first <= last < len(units):
        raise RuntimeError("evidence range is reversed, absent or outside its namespace")
    if any(not isinstance(unit, str) or not unit.strip() for unit in units[first : last + 1]):
        raise RuntimeError("evidence range includes empty source speech")
    return {"text": " ".join(units[first : last + 1]), "first_unit": first, "last_unit": last}


def _unit_span_valid(span: Any, units: list[str]) -> bool:
    if not isinstance(span, dict) or set(span) != {"text", "first_unit", "last_unit"}:
        return False
    try:
        return (
            _resolve_source_units({key: span[key] for key in ("first_unit", "last_unit")}, units)
            == span
        )
    except RuntimeError:
        return False


def _numbered_source(units: list[str]) -> list[dict[str, Any]]:
    return [{"id": i, "text": text} for i, text in enumerate(units)]


_ACKNOWLEDGEMENTS = {
    "yeah",
    "yeah for sure",
    "yes",
    "yep",
    "right",
    "okay",
    "ok",
    "sure",
    "for sure",
    "uh huh",
    "mm hmm",
    "mhm",
    "exactly",
}


def _final_substantive_unit_id(units: list[str]) -> int:
    """Ignore trailing backchannels, never an earlier developed point."""
    if not units:
        raise ValueError("final substantive unit needs delivered speech")
    for index in range(len(units) - 1, -1, -1):
        words = " ".join(re.findall(r"[A-Za-z]+", units[index].casefold()))
        if words not in _ACKNOWLEDGEMENTS:
            return index
    return len(units) - 1


def _evidence_excerpt(text: str) -> str:
    """Format bounded evidence only after full source-span verification."""
    words = list(_WORD.finditer(text))
    return text[: words[11].end()] if len(words) > 12 else text


class HeadlineGenerator(Protocol):
    def __call__(
        self,
        editor: Any,
        units: list[str],
        *,
        exchange_spans: dict[str, Any],
        factual_audit: Callable[[Any, str, list[str]], dict[str, Any]],
    ) -> dict[str, Any]: ...


def review_source_positions(
    editor: Any,
    context: dict[str, Any],
    *,
    factual_audit: Callable[[Any, str, list[str]], dict[str, Any]],
    headline_generator: HeadlineGenerator,
) -> dict[str, Any]:
    """Review actual delivered speech; excluded evidence cannot become its resolution."""
    selected = context["selected_units"]
    if not selected or any(not isinstance(unit, str) or not unit.strip() for unit in selected):
        raise ValueError("review requires nonempty delivered units")
    purpose = editor._review_completion(
        "Classify the speech act performed in delivered_units, not its topic. "
        "ad_read_present is 1 only when a host delivers a sponsor message to the audience; "
        "conversation about sponsors, earnings or business is not an ad read. "
        "show_intro_present is 1 only when a host introduces the show, episode, segment "
        "or guest to the audience; a person being named in a story is not an introduction. "
        "For each present act, select the shortest continuous delivered unit span that "
        "actually performs it. For an absent act set both span positions to -1, even if "
        "the topic resembles an ad or introduction. Do not copy source text. "
        "reason is at most 20 words. Return output_schema JSON.",
        {"delivered_units": _numbered_source(selected)},
        {
            "ad_read_present": {"type": "integer", "enum": [0, 1]},
            "ad_read_span": _unit_span_schema(len(selected)),
            "show_intro_present": {"type": "integer", "enum": [0, 1]},
            "show_intro_span": _unit_span_schema(len(selected)),
            "reason": {"type": "string"},
        },
        160,
    )
    purpose_spans = {
        key: _resolve_source_units(purpose.get(key), selected)
        for key in ("ad_read_span", "show_intro_span")
    }
    for label, key in (
        ("ad_read_present", "ad_read_span"),
        ("show_intro_present", "show_intro_span"),
    ):
        if type(purpose.get(label)) is not int or purpose[label] not in (0, 1):
            raise RuntimeError("speech-purpose review returned an invalid label")
        if (purpose[label] == 1) != (purpose_spans[key] is not None):
            raise RuntimeError("speech-purpose label and evidence positions disagree")
    promotion_ids = sorted(
        {
            i
            for span in purpose_spans.values()
            if span
            for i in range(span["first_unit"], span["last_unit"] + 1)
        }
    )
    final_id = _final_substantive_unit_id(selected)
    story = editor._review_completion(
        "Assess ONLY delivered_units as the finished clip. Select its central setup_span "
        "and delivered resolution_span as continuous unit positions. A resolution can be "
        "an answer, consequence, contrast, reaction or punchline. Both resolution positions "
        "-1 means absent. A delivered resolution must reach final_substantive_unit_id; "
        "do not substitute an earlier answer for a new unfinished final premise. "
        "opening_independent is 1 when a new viewer understands the subject; "
        "a first-person story need not name its visible speaker. last_thought_finished is "
        "1 only when the final substantive thought has delivered its point. Punctuation "
        "alone is not proof. Select positions, never rewrite evidence. reason is at most "
        "25 words. Return output_schema JSON.",
        {
            "delivered_units": _numbered_source(selected),
            "final_substantive_unit_id": final_id,
            "final_substantive_unit_text": selected[final_id],
        },
        {
            "setup_span": _unit_span_schema(len(selected)),
            "resolution_span": {
                "type": "object",
                "properties": {
                    "first_unit": {"type": "integer", "enum": [-1, *range(len(selected))]},
                    "last_unit": {"type": "integer", "enum": [-1, final_id]},
                },
                "required": ["first_unit", "last_unit"],
                "additionalProperties": False,
            },
            "opening_independent": {"type": "integer", "enum": [0, 1]},
            "last_thought_finished": {"type": "integer", "enum": [0, 1]},
            "reason": {"type": "string"},
        },
        160,
    )
    for key in ("opening_independent", "last_thought_finished"):
        if type(story.get(key)) is not int or story[key] not in (0, 1):
            raise RuntimeError("thought reviewer returned an invalid verdict")
    setup = _resolve_source_units(story.get("setup_span"), selected)
    resolution = _resolve_source_units(story.get("resolution_span"), selected)
    if resolution is not None and resolution["last_unit"] != final_id:
        raise RuntimeError("delivered resolution omitted the final substantive unit")
    ending = story["last_thought_finished"] == 1
    continuation = None
    after = context.get("after", [])
    # Purpose and completion are independent judgments. A mistaken purpose veto
    # must not hide whether the excluded continuation contains the real payoff.
    if ending and after:
        continuation = editor._review_completion(
            "Judge the CUT after delivered_units. Python has identified the final "
            "substantive delivered unit; judge excluded_after against that point in "
            "the full delivered context, not against an earlier completed premise. "
            "Classify its relation to excluded_after: "
            "missing_answer, missing_contrast, unfinished_clause, optional_elaboration, "
            "new_topic or uncertain. A related example after a delivered resolution is "
            "optional. final_span must equal final_substantive_unit_id; continuation_span "
            "uses excluded_after IDs. These are separate namespaces. Both spans must "
            "exist. Excluded speech can reveal a missing resolution but cannot count as "
            "delivered payoff. reason is at most 25 words. Return output_schema JSON.",
            {
                "delivered_units": _numbered_source(selected),
                "final_substantive_unit_id": final_id,
                "final_substantive_unit_text": selected[final_id],
                "excluded_after": _numbered_source(after),
            },
            {
                "final_span": {
                    "type": "object",
                    "properties": {
                        key: {"type": "integer", "enum": [final_id]}
                        for key in ("first_unit", "last_unit")
                    },
                    "required": ["first_unit", "last_unit"],
                    "additionalProperties": False,
                },
                "continuation_span": _unit_span_schema(len(after)),
                "relation": {
                    "type": "string",
                    "enum": [
                        "missing_answer",
                        "missing_contrast",
                        "unfinished_clause",
                        "optional_elaboration",
                        "new_topic",
                        "uncertain",
                    ],
                },
                "reason": {"type": "string"},
            },
            160,
        )
        for key, region in (("final_span", selected), ("continuation_span", after)):
            span = _resolve_source_units(continuation.get(key), region)
            if span is None:
                raise RuntimeError("continuation review requires evidence in both namespaces")
            if key == "final_span" and (
                span["first_unit"] != final_id or span["last_unit"] != final_id
            ):
                raise RuntimeError("continuation review ignored the final substantive unit")
            continuation[key + "_source"] = span
        if continuation.get("relation") not in {
            "missing_answer",
            "missing_contrast",
            "unfinished_clause",
            "optional_elaboration",
            "new_topic",
            "uncertain",
        }:
            raise RuntimeError("continuation review returned an invalid relation")
        ending = continuation["relation"] in {"optional_elaboration", "new_topic"}
    payoff = resolution is not None and ending
    opening = story["opening_independent"] == 1
    reason = (continuation or story).get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("review must explain its boundary decision")
    spans = {"setup_quote": setup, "resolution_quote": resolution}
    result = {
        "evidence_contract": SOURCE_EVIDENCE_VERSION,
        "speech_purpose_review": purpose,
        "speech_purpose_quote_spans": purpose_spans,
        "thought_completion_review": story,
        "continuation_review": continuation,
        "source_quote_spans": spans,
        "delivered_units": selected,
        "boundary_audit": {
            "reason": reason,
            "promotion_unit_ids": promotion_ids,
            "opening": "standalone" if opening else "dependent",
            "ending": "closed" if ending else "unresolved",
            "payoff_location": "selected" if payoff else "absent",
            "setup_unit_id": setup["first_unit"] if setup else -1,
            "setup_unit_last_id": setup["last_unit"] if setup else -1,
            "payoff_unit_id": resolution["first_unit"] if payoff and resolution is not None else -1,
            "payoff_unit_last_id": resolution["last_unit"]
            if payoff and resolution is not None
            else -1,
        },
        "opening_standalone": opening,
        "ending_complete": ending,
        "exchange_has_payoff": resolution is not None,
        "payoff_complete": payoff,
        "contains_promotion_or_intro": bool(promotion_ids),
        "headline": "",
        "setup_quote": _evidence_excerpt(setup["text"]) if setup else "",
        "payoff_quote": _evidence_excerpt(resolution["text"])
        if payoff and resolution is not None
        else "",
        "headline_supported": False,
        "headline_self_contained": False,
        "reason": reason,
    }
    if promotion_ids or not opening or not payoff or setup is None:
        return result
    generated = headline_generator(
        editor,
        selected,
        exchange_spans=spans,
        factual_audit=factual_audit,
    )
    if not isinstance(generated, dict):
        raise RuntimeError("headline generator must return a structured result")
    # The headline stage must never overwrite the immutable source review or
    # grant publication approval. Only headline-owned review outputs may change.
    headline_fields = {
        "headline",
        "headline_source_spans",
        "headline_supported",
        "headline_self_contained",
        "headline_audits",
        "hook_status",
        "source_excerpts",
        "source_bound",
        "candidate_model_audit",
        "production_approved",
    }
    if generated.keys() - headline_fields or generated.get("production_approved") is True:
        raise RuntimeError(
            "headline generator attempted to override editorial evidence or approval"
        )
    if "headline_supported" in generated and type(generated["headline_supported"]) is not bool:
        raise RuntimeError("headline supported verdict must be boolean")
    if (
        "headline_self_contained" in generated
        and type(generated["headline_self_contained"]) is not bool
    ):
        raise RuntimeError("headline readability verdict must be boolean")
    result["exchange_accepted"] = True
    result.update(generated)
    return result

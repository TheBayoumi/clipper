"""Production source-position editorial engine.

Span validation establishes provenance, not semantic accuracy. The current
reviewer remains unqualified; moving it here does not authorize rendering.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol


class SourceEvidenceError(RuntimeError):
    """Invalid model-owned evidence; the current clip must fail closed."""


class FactualEvidenceError(SourceEvidenceError):
    """An answer/status/quote mismatch from a source-fact model response."""


class ModelOutputError(SourceEvidenceError):
    """The model returned invalid structured output, not an infrastructure fault."""


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
        raise SourceEvidenceError("evidence pointer must contain only first_unit and last_unit")
    first, last = pointer["first_unit"], pointer["last_unit"]
    if type(first) is not int or type(last) is not int:
        raise SourceEvidenceError("evidence positions must be integers, not booleans")
    if first == last == -1:
        return None
    if not 0 <= first <= last < len(units):
        raise SourceEvidenceError("evidence range is reversed, absent or outside its namespace")
    if any(not isinstance(unit, str) or not unit.strip() for unit in units[first : last + 1]):
        raise SourceEvidenceError("evidence range includes empty source speech")
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


# A continuation veto can be overcome only by proving TWO separate facts:
# the selected clip actually delivers its promised payoff, and the omitted speech
# is unnecessary to that payoff. The model never authors the final approval bit.
_PAYOFF_STATES = ("delivered", "pending", "uncertain")
_EXCLUDED_EFFECTS = (
    "supplies_missing_payoff",
    "changes_or_qualifies_delivered_payoff",
    "restates_or_illustrates_delivered_payoff",
    "new_subject",
    "uncertain",
)
_OPTIONAL_EXCLUDED_EFFECTS = frozenset({"restates_or_illustrates_delivered_payoff", "new_subject"})


def _optional_contrast_proof_valid(
    proof: Any, selected: list[str], after: list[str], final_id: int
) -> bool:
    """Check two source-bound narrative-obligation and exclusion decisions.

    Production has the same strict fail-closed rule for cached and live model
    responses. JSON fields and pointers prove the model made a traceable claim,
    not that the claim is semantically true.
    """
    if proof is None:
        return False
    if not isinstance(proof, dict) or set(proof) != {"verdicts", "optional_verified"}:
        raise SourceEvidenceError("invalid independent continuation proof")
    verdicts = proof["verdicts"]
    if not isinstance(verdicts, list) or len(verdicts) != 2:
        raise SourceEvidenceError("independent continuation proof requires two verdicts")
    approved = True
    for verdict in verdicts:
        if not isinstance(verdict, dict) or set(verdict) != {"obligation", "dependency"}:
            raise SourceEvidenceError("narrative obligation and excluded dependency are required")
        obligation, dependency = verdict["obligation"], verdict["dependency"]
        if not isinstance(obligation, dict) or set(obligation) != {
            "setup_span",
            "payoff_span",
            "payoff_state",
            "reason",
        }:
            raise SourceEvidenceError("invalid delivered narrative-obligation record")
        if not isinstance(dependency, dict) or set(dependency) != {
            "excluded_span",
            "effect",
            "reason",
        }:
            raise SourceEvidenceError("invalid omitted-material dependency record")
        setup = _resolve_source_units(obligation["setup_span"], selected)
        payoff = _resolve_source_units(obligation["payoff_span"], selected)
        excluded = _resolve_source_units(dependency["excluded_span"], after)
        if (
            setup is None
            or excluded is None
            or obligation["payoff_state"] not in _PAYOFF_STATES
            or (
                obligation["payoff_state"] == "delivered"
                and (payoff is None or payoff["last_unit"] != final_id)
            )
            or (obligation["payoff_state"] != "delivered" and payoff is not None)
            or dependency["effect"] not in _EXCLUDED_EFFECTS
            or any(
                not isinstance(record["reason"], str)
                or not record["reason"].strip()
                or len(record["reason"]) > 300
                for record in (obligation, dependency)
            )
        ):
            raise SourceEvidenceError("narrative obligation or dependency lacks canonical proof")
        approved = (
            approved
            and obligation["payoff_state"] == "delivered"
            and dependency["effect"] in _OPTIONAL_EXCLUDED_EFFECTS
        )
    if type(proof["optional_verified"]) is not bool or proof["optional_verified"] != approved:
        raise SourceEvidenceError("independent continuation approval was altered")
    return approved


def review_optional_contrast(
    editor: Any, selected: list[str], after: list[str], final_id: int
) -> dict[str, Any]:
    """Two blinded evidence-first evaluations of one continuation boundary.

    Each evaluation first examines DELIVERED speech alone and identifies the
    obligation and delivered payoff. A separate subsequent model call evaluates
    excluded speech against that fixed, source-positioned obligation. The two
    evaluations see neither each other's results nor the initial veto label.
    """
    if not selected or not after or not 0 <= final_id < len(selected):
        raise ValueError("contrast review requires source speech and a valid final unit")
    obligation_properties = {
        "setup_span": _unit_span_schema(len(selected)),
        "payoff_span": _unit_span_schema(len(selected)),
        "payoff_state": {"type": "string", "enum": list(_PAYOFF_STATES)},
        "reason": {"type": "string", "minLength": 1, "maxLength": 300},
    }
    dependency_properties = {
        "excluded_span": _unit_span_schema(len(after)),
        "effect": {"type": "string", "enum": list(_EXCLUDED_EFFECTS)},
        "reason": {"type": "string", "minLength": 1, "maxLength": 300},
    }
    delivered_source = {
        "delivered_units": _numbered_source(selected),
        "final_substantive_unit_id": final_id,
    }
    # Explicit obligations prevent a related follow-up being treated as
    # necessary merely because it mentions the same athlete or team.
    obligation_prompts = (
        "Read ONLY delivered_units; no future transcript is available. Identify "
        "the CENTRAL story, claim, opinion or question introduced by the excerpt "
        "(setup_span), then decide whether its specific payoff, answer, stated "
        "conclusion or promised consequence is actually spoken by the final unit. "
        "If yes, payoff_state=delivered and payoff_span MUST end at "
        "final_substantive_unit_id. If a punchline, result, promised outcome or "
        "meaningful answer remains unstated, payoff_state=pending and payoff_span "
        "MUST be (-1,-1). If ambiguous use uncertain with (-1,-1). "
        "An ending with a period or preliminary physical action is not a completed "
        "story if its teased consequence is still missing. Describe the obligation "
        "and what is or is not delivered in <=24 words. Return output_schema JSON.",
        "Evaluate delivered_units ALONE as a short-form video. What setup is "
        "the viewer promised, and which SOURCE SPAN actually finishes its point? "
        "A complete conclusion or opinion counts as a payoff even when another "
        "example could be added. A joke, anecdote or confrontation without its "
        "anticipated punchline or consequential result is not complete. "
        "Choose setup_span in delivered_units; payoff_state=delivered ONLY when "
        "payoff_span reaches final_substantive_unit_id and completes the LIVE "
        "obligation at the end. Otherwise set pending or uncertain and "
        "payoff_span=(-1,-1). Explain your source-based decision in <=24 words. "
        "Return output_schema JSON.",
    )
    dependency_prompts = (
        "Compare the FIXED narrative obligation from delivered speech with "
        "excluded_after. Does excluded_after provide the actually MISSING "
        "answer, event result or punchline (supplies_missing_payoff), alter or "
        "qualify a finished conclusion (changes_or_qualifies_delivered_payoff), "
        "merely repeat or illustrate an ALREADY STATED conclusion "
        "(restates_or_illustrates_delivered_payoff), or begin a separate "
        "subject/question (new_subject)? Topic similarity is NOT narrative "
        "dependency. If the obligation cannot be determined, choose uncertain. "
        "Cite the exact excluded unit span. Explain the causal relevance in "
        "<=24 words. Return output_schema JSON.",
        "Perform a cut-dependency test using the supplied frozen delivered "
        "setup/payoff positions: what SPECIFIC semantic obligation would be "
        "lost by cutting excluded_after? A subsequent question about another "
        "team is a new_subject even if it mentions the same league. An example "
        "repeating a complete comparison is a restatement, not required contrast. "
        "An unrevealed hit, real-world outcome, answer or joke punchline teased "
        "by the delivered setup supplies_missing_payoff. A correction, reversal "
        "or necessary exception changes_or_qualifies_delivered_payoff. "
        "Choose uncertain if unclear, cite actual excluded unit IDs, and "
        "explain in <=24 words. Return output_schema JSON.",
    )
    verdicts: list[dict[str, Any]] = []
    for obligation_prompt, dependency_prompt in zip(
        obligation_prompts, dependency_prompts, strict=True
    ):
        obligation = editor._review_completion(
            obligation_prompt, delivered_source, obligation_properties, 512
        )
        # Fail before the next expensive inference on malformed source pointers.
        if not isinstance(obligation, dict) or not {
            "setup_span",
            "payoff_span",
            "payoff_state",
            "reason",
        } == set(obligation):
            raise SourceEvidenceError("model returned invalid delivered obligation")
        if _resolve_source_units(obligation["setup_span"], selected) is None:
            raise SourceEvidenceError("narrative setup is not in delivered speech")
        payoff = _resolve_source_units(obligation["payoff_span"], selected)
        if (
            obligation["payoff_state"] not in _PAYOFF_STATES
            or (obligation["payoff_state"] == "delivered")
            != (payoff is not None and payoff["last_unit"] == final_id)
            or (obligation["payoff_state"] != "delivered" and payoff is not None)
        ):
            raise SourceEvidenceError("model payoff-state disagrees with its source span")
        dependency = editor._review_completion(
            dependency_prompt,
            {
                **delivered_source,
                "excluded_after": _numbered_source(after),
                "frozen_delivered_obligation": obligation,
            },
            dependency_properties,
            512,
        )
        verdicts.append({"obligation": obligation, "dependency": dependency})
    proof = {
        "verdicts": verdicts,
        "optional_verified": all(
            item["obligation"]["payoff_state"] == "delivered"
            and isinstance(item["dependency"], dict)
            and item["dependency"].get("effect") in _OPTIONAL_EXCLUDED_EFFECTS
            for item in verdicts
        ),
    }
    _optional_contrast_proof_valid(proof, selected, after, final_id)
    return proof


class HeadlineGenerator(Protocol):
    def __call__(
        self,
        editor: Any,
        units: list[str],
        *,
        exchange_spans: dict[str, Any],
        factual_audit: Callable[[Any, str, list[str]], dict[str, Any]],
    ) -> dict[str, Any]: ...


def review_source_boundaries(
    editor: Any,
    context: dict[str, Any],
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
            raise SourceEvidenceError("speech-purpose label and evidence positions disagree")
    promotion_ids = sorted(
        {
            i
            for span in purpose_spans.values()
            if span
            for i in range(span["first_unit"], span["last_unit"] + 1)
        }
    )
    final_id = _final_substantive_unit_id(selected)
    story_prompt, story_payload, story_schema, story_budget = (
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
    # A model can return schema-valid but impossible mixed -1/range pointers.
    # Request one new source-grounded response, then reject this candidate only.
    for evidence_attempt in range(2):
        story = editor._review_completion(story_prompt, story_payload, story_schema, story_budget)
        try:
            for key in ("opening_independent", "last_thought_finished"):
                if type(story.get(key)) is not int or story[key] not in (0, 1):
                    raise RuntimeError("thought reviewer returned an invalid verdict")
            setup = _resolve_source_units(story.get("setup_span"), selected)
            resolution = _resolve_source_units(story.get("resolution_span"), selected)
            if resolution is not None and resolution["last_unit"] != final_id:
                raise SourceEvidenceError("delivered resolution omitted the final substantive unit")
        except SourceEvidenceError as error:
            if evidence_attempt:
                raise
            story_payload = {
                **story_payload,
                "invalid_model_positions": {
                    key: story.get(key) for key in ("setup_span", "resolution_span")
                },
                "evidence_validation_error": str(error),
            }
            story_prompt += (
                " The prior output was invalid and is NOT source evidence. Re-read "
                "the numbered delivered_units. Each span must use two valid ordered "
                "indices, or exactly (-1,-1) when absent. A present resolution must "
                "reach final_substantive_unit_id. Do not manufacture a resolution."
            )
            continue
        break
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
                raise SourceEvidenceError(
                    "continuation review requires evidence in both namespaces"
                )
            if key == "final_span" and (
                span["first_unit"] != final_id or span["last_unit"] != final_id
            ):
                raise SourceEvidenceError("continuation review ignored the final substantive unit")
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
        if (
            continuation["relation"] == "missing_contrast"
            and resolution is not None
            and not promotion_ids
        ):
            continuation["optional_contrast_proof"] = review_optional_contrast(
                editor, selected, after, final_id
            )
        ending = continuation["relation"] in {"optional_elaboration", "new_topic"} or (
            continuation["relation"] == "missing_contrast"
            and _optional_contrast_proof_valid(
                continuation.get("optional_contrast_proof"), selected, after, final_id
            )
        )
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
    return result


def _validate_cached_boundary(context: dict[str, Any], boundary: dict[str, Any]) -> None:
    """Re-derive acceptance flags from source-owned model positions, never cache prose."""
    selected = context["selected_units"]
    if (
        boundary.get("evidence_contract") != SOURCE_EVIDENCE_VERSION
        or boundary.get("delivered_units") != selected
        or not isinstance(boundary.get("source_quote_spans"), dict)
        or not isinstance(boundary.get("speech_purpose_review"), dict)
        or not isinstance(boundary.get("thought_completion_review"), dict)
    ):
        raise SourceEvidenceError("cached story evidence does not match selected source")
    spans = boundary["source_quote_spans"]
    if set(spans) != {"setup_quote", "resolution_quote"} or any(
        span is not None and not _unit_span_valid(span, selected) for span in spans.values()
    ):
        raise SourceEvidenceError("cached story evidence is not canonical source speech")
    purpose = boundary["speech_purpose_review"]
    purpose_spans: dict[str, Any] = {}
    for label, key in (
        ("ad_read_present", "ad_read_span"),
        ("show_intro_present", "show_intro_span"),
    ):
        if type(purpose.get(label)) is not int or purpose[label] not in (0, 1):
            raise SourceEvidenceError("cached story has invalid promotion labels")
        canonical = _resolve_source_units(purpose.get(key), selected)
        if (purpose[label] == 1) != (canonical is not None):
            raise SourceEvidenceError("cached promotion labels contradict source spans")
        purpose_spans[key] = canonical
    if purpose_spans != boundary.get("speech_purpose_quote_spans"):
        raise SourceEvidenceError("cached promotion evidence differs from source")
    story = boundary["thought_completion_review"]
    setup = _resolve_source_units(story.get("setup_span"), selected)
    resolution = _resolve_source_units(story.get("resolution_span"), selected)
    if spans != {"setup_quote": setup, "resolution_quote": resolution}:
        raise SourceEvidenceError("cached story positions changed since assessment")
    final_id = _final_substantive_unit_id(selected)
    if resolution is not None and resolution["last_unit"] != final_id:
        raise SourceEvidenceError("cached payoff skipped last substantive source unit")
    for key in ("opening_independent", "last_thought_finished"):
        if type(story.get(key)) is not int or story[key] not in (0, 1):
            raise SourceEvidenceError("cached story has invalid completion verdict")
    after = context.get("after", [])
    continuation = boundary.get("continuation_review")
    ending = story["last_thought_finished"] == 1
    if ending and after:
        if not isinstance(continuation, dict):
            raise SourceEvidenceError("cached continuation decision is missing")
        final_span = _resolve_source_units(continuation.get("final_span"), selected)
        after_span = _resolve_source_units(continuation.get("continuation_span"), after)
        if (
            final_span is None
            or final_span["first_unit"] != final_id
            or final_span["last_unit"] != final_id
            or after_span is None
            or continuation.get("final_span_source") != final_span
            or continuation.get("continuation_span_source") != after_span
            or continuation.get("relation")
            not in {
                "missing_answer",
                "missing_contrast",
                "unfinished_clause",
                "optional_elaboration",
                "new_topic",
                "uncertain",
            }
        ):
            raise SourceEvidenceError("cached continuation is not canonical source evidence")
        proof = continuation.get("optional_contrast_proof")
        if proof is not None and continuation["relation"] != "missing_contrast":
            raise SourceEvidenceError("independent proof cannot override another relation")
        ending = continuation["relation"] in {"optional_elaboration", "new_topic"} or (
            continuation["relation"] == "missing_contrast"
            and _optional_contrast_proof_valid(proof, selected, after, final_id)
        )
    elif continuation is not None:
        raise SourceEvidenceError("cached continuation should not exist")
    opening = story["opening_independent"] == 1
    payoff = resolution is not None and ending
    promoted = sorted(
        {
            index
            for span in purpose_spans.values()
            if span is not None
            for index in range(span["first_unit"], span["last_unit"] + 1)
        }
    )
    expected = {
        "opening_standalone": opening,
        "ending_complete": ending,
        "exchange_has_payoff": resolution is not None,
        "payoff_complete": payoff,
        "contains_promotion_or_intro": bool(promoted),
        "setup_quote": _evidence_excerpt(setup["text"]) if setup else "",
        "payoff_quote": _evidence_excerpt(resolution["text"]) if payoff and resolution else "",
        "reason": (continuation or story).get("reason"),
    }
    if any(boundary.get(key) != value for key, value in expected.items()):
        raise SourceEvidenceError("cached completion verdict differs from source review")
    audit = boundary.get("boundary_audit")
    if not isinstance(audit, dict) or any(
        audit.get(key) != value
        for key, value in {
            "promotion_unit_ids": promoted,
            "opening": "standalone" if opening else "dependent",
            "ending": "closed" if ending else "unresolved",
            "payoff_location": "selected" if payoff else "absent",
            "setup_unit_id": setup["first_unit"] if setup else -1,
            "setup_unit_last_id": setup["last_unit"] if setup else -1,
            "payoff_unit_id": resolution["first_unit"] if payoff and resolution else -1,
            "payoff_unit_last_id": resolution["last_unit"] if payoff and resolution else -1,
            "reason": expected["reason"],
        }.items()
    ):
        raise SourceEvidenceError("cached boundary audit conflicts with source positions")


def complete_source_review(
    editor: Any,
    context: dict[str, Any],
    boundary: dict[str, Any],
    *,
    factual_audit: Callable[[Any, str, list[str]], dict[str, Any]],
    headline_generator: HeadlineGenerator,
) -> dict[str, Any]:
    """Produce only headline-owned outputs after verified boundary evidence."""
    _validate_cached_boundary(context, boundary)
    selected = context["selected_units"]
    spans = boundary["source_quote_spans"]
    result = {
        key: value
        for key, value in boundary.items()
        if key
        not in {
            "headline_audits",
            "headline_source_spans",
            "hook_status",
            "source_excerpts",
            "source_bound",
            "candidate_model_audit",
            "production_approved",
            "exchange_accepted",
        }
    }
    result.update({"headline": "", "headline_supported": False, "headline_self_contained": False})
    if (
        result.get("contains_promotion_or_intro") is True
        or result.get("opening_standalone") is not True
        or result.get("payoff_complete") is not True
        or result.get("ending_complete") is not True
        or spans["setup_quote"] is None
        or spans["resolution_quote"] is None
    ):
        return result
    generated = headline_generator(
        editor,
        selected,
        exchange_spans=result["source_quote_spans"],
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


def review_source_positions(
    editor: Any,
    context: dict[str, Any],
    *,
    factual_audit: Callable[[Any, str, list[str]], dict[str, Any]],
    headline_generator: HeadlineGenerator,
) -> dict[str, Any]:
    """Boundary and headline stages with independent implementation identities."""
    boundary = review_source_boundaries(editor, context)
    return complete_source_review(
        editor,
        context,
        boundary,
        factual_audit=factual_audit,
        headline_generator=headline_generator,
    )

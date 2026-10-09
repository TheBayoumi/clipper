"""Parser-owned headline obligations for source-grounded factual review.

The parser identifies predicates, arguments and qualifying syntax before any
source speech is shown to a model. This is an inventory diagnostic, not an
entailment check or a production approval.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_CLAUSE_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "pcomp", "conj"}
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_ARGUMENT_DEPS = {"dobj", "iobj", "attr", "acomp", "oprd", "dative"}
_PREPOSITION_DEPS = {"prep", "agent"}
_QUALIFIER_DEPS = {"aux", "auxpass", "neg", "advmod", "mark", "prt"}
_LEXICAL_POS = {"VERB", "AUX"}
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "NUM"}


def inventory_headline_syntax(headline: str, parser: Callable[[str], Any]) -> dict[str, Any]:
    """Return every parsed clause's explicit roles, without choosing source facts."""
    if not isinstance(headline, str) or not 4 <= len(_WORD.findall(headline)) <= 14:
        raise ValueError("syntax inventory requires a 4-14 word headline")
    doc = parser(headline)
    if doc.text != headline or not doc.has_annotation("DEP"):
        raise ValueError("syntax inventory requires an exact dependency parse")
    frames: list[dict[str, Any]] = []
    covered: set[int] = set()
    warnings: list[dict[str, Any]] = []

    def span(token: Any) -> dict[str, Any]:
        subtree = list(token.subtree)
        if not subtree:
            raise ValueError("syntax parser returned an empty subtree")
        first, last = (
            min(item.idx for item in subtree),
            max(item.idx + len(item.text) for item in subtree),
        )
        if not 0 <= first < last <= len(headline):
            raise ValueError("syntax parser returned an invalid source span")
        covered.update(item.i for item in subtree)
        return {
            "head_token": token.i,
            "first_char": first,
            "last_char": last,
            "text": headline[first:last],
        }

    for token in doc:
        if token.pos_ not in _LEXICAL_POS or token.dep_ not in _CLAUSE_DEPS:
            continue
        if (
            type(token.idx) is not int
            or not 0 <= token.idx < token.idx + len(token.text) <= len(headline)
            or headline[token.idx : token.idx + len(token.text)] != token.text
        ):
            raise ValueError("syntax parser returned an invalid predicate span")
        covered.add(token.i)
        roles: list[dict[str, Any]] = []
        for child in token.children:
            if child.dep_ in _SUBJECT_DEPS:
                role = "subject"
            elif child.dep_ in _ARGUMENT_DEPS:
                role = "argument"
            elif child.dep_ in _PREPOSITION_DEPS:
                role = "prepositional_argument"
            elif child.dep_ in _QUALIFIER_DEPS:
                role = "qualifier"
            elif child.dep_ in _CLAUSE_DEPS:
                role = "embedded_clause"
            else:
                continue
            roles.append({"role": role, "dependency": child.dep_, **span(child)})
        if sum(role["dependency"] == "dobj" for role in roles) > 1:
            warnings.append({"predicate_token": token.i, "reason": "multiple_direct_objects"})
        if not any(role["role"] == "subject" for role in roles):
            warnings.append(
                {"predicate_token": token.i, "reason": "subject_not_explicit_in_clause"}
            )
        frames.append(
            {
                "predicate_token": token.i,
                "predicate": token.text,
                "predicate_first_char": token.idx,
                "predicate_last_char": token.idx + len(token.text),
                "dependency": token.dep_,
                "subject_explicit": any(role["role"] == "subject" for role in roles),
                "roles": roles,
            }
        )
    if not frames:
        raise ValueError("syntax inventory found no predicate")
    for token in doc:
        if token.pos_ == "AUX" and token.dep_ == "auxpass" and token.head.dep_ not in _CLAUSE_DEPS:
            warnings.append(
                {"predicate_token": token.i, "reason": "auxiliary_outside_parsed_clause"}
            )
    uncovered = [
        {"token": token.i, "text": token.text, "dependency": token.dep_}
        for token in doc
        if token.pos_ in _CONTENT_POS and token.i not in covered
    ]
    return {
        "headline": headline,
        "frames": frames,
        "uncovered_content_tokens": uncovered,
        "parse_warnings": warnings,
        "parser_semantically_qualified": False,
        "source_entailment_checked": False,
        "diagnostic_only": True,
        "production_approved": False,
    }

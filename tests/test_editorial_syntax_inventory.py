"""Parser-owned claim obligations remain source-blind and unapproved."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from clipper.editorial_syntax_inventory import inventory_headline_syntax


@dataclass
class Token:
    text: str
    i: int
    idx: int
    dep_: str
    pos_: str
    children: list[Token] = field(default_factory=list)
    head: Token | None = None

    @property
    def subtree(self):
        result = [self]
        for child in self.children:
            result.extend(child.subtree)
        return result


class Doc:
    def __init__(self, text: str, tokens: list[Token], parsed: bool = True):
        self.text = text
        self.tokens = tokens
        self.parsed = parsed

    def has_annotation(self, name: str) -> bool:
        return name == "DEP" and self.parsed

    def __iter__(self):
        return iter(self.tokens)


def parsed_opponent_headline():
    headline = "Bobby was the speaker's opponent"
    tokens = [
        Token("Bobby", 0, 0, "nsubj", "PROPN"),
        Token("was", 1, 6, "ROOT", "AUX"),
        Token("the", 2, 10, "det", "DET"),
        Token("speaker's", 3, 14, "poss", "NOUN"),
        Token("opponent", 4, 24, "attr", "NOUN"),
    ]
    tokens[1].children = [tokens[0], tokens[4]]
    tokens[4].children = [tokens[2], tokens[3]]
    for token in tokens:
        if token is not tokens[1]:
            token.head = tokens[4] if token in (tokens[2], tokens[3]) else tokens[1]
    return Doc(headline, tokens)


def test_syntax_inventory_owns_subject_and_role_without_a_model_question():
    result = inventory_headline_syntax(
        "Bobby was the speaker's opponent", lambda _: parsed_opponent_headline()
    )
    assert len(result["frames"]) == 1
    frame = result["frames"][0]
    assert frame["predicate"] == "was"
    assert [(role["role"], role["text"]) for role in frame["roles"]] == [
        ("subject", "Bobby"),
        ("argument", "the speaker's opponent"),
    ]
    assert result["uncovered_content_tokens"] == []
    assert result["parse_warnings"] == []
    assert result["source_entailment_checked"] is False
    assert result["production_approved"] is False


def test_syntax_inventory_rejects_a_parse_without_any_predicate():
    headline = "They are salaried independent contractors"
    tokens = [
        Token("They", 0, 0, "nsubj", "PRON"),
        Token("are", 1, 5, "auxpass", "AUX"),
        Token("salaried", 2, 9, "amod", "ADJ"),
        Token("independent", 3, 18, "amod", "ADJ"),
        Token("contractors", 4, 30, "ROOT", "NOUN"),
    ]
    tokens[4].children = [tokens[0], tokens[1], tokens[2], tokens[3]]
    tokens[1].head = tokens[4]
    with pytest.raises(ValueError, match="no predicate"):
        inventory_headline_syntax(headline, lambda _: Doc(headline, tokens))


def test_syntax_inventory_flags_auxiliary_outside_a_clause():
    headline = "They say fighters are salaried employees"
    tokens = [
        Token("They", 0, 0, "nsubj", "PRON"),
        Token("say", 1, 5, "ROOT", "VERB"),
        Token("fighters", 2, 9, "nsubj", "NOUN"),
        Token("are", 3, 18, "auxpass", "AUX"),
        Token("salaried", 4, 22, "amod", "VERB"),
        Token("employees", 5, 31, "dobj", "NOUN"),
    ]
    tokens[1].children = [tokens[0], tokens[5]]
    tokens[5].children = [tokens[4]]
    tokens[4].children = [tokens[3]]
    tokens[3].children = [tokens[2]]
    tokens[3].head = tokens[4]
    result = inventory_headline_syntax(headline, lambda _: Doc(headline, tokens))
    assert result["parse_warnings"] == [
        {"predicate_token": 3, "reason": "auxiliary_outside_parsed_clause"}
    ]
    assert result["production_approved"] is False


def test_syntax_inventory_rejects_missing_or_changed_parse():
    with pytest.raises(ValueError, match="4-14 word"):
        inventory_headline_syntax("Bobby paced", lambda _: parsed_opponent_headline())
    with pytest.raises(ValueError, match="exact dependency parse"):
        inventory_headline_syntax(
            "Bobby was the speaker's opponent", lambda _: Doc("changed", [], False)
        )

"""Source-first atomic relation pipeline, pending semantic qualification.

Every stage is package-owned and passes only the data it needs. The proposed
headline answer is withheld until after source answering and scope review.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from clipper.editorial_answer_comparison import compare_claim_answer
from clipper.editorial_source_answer import answer_source_question
from clipper.editorial_source_scope import Completion, assess_source_answer_scope


def verify_atomic_relation(
    *,
    question: str,
    proposed_answer: str,
    delivered_units: Sequence[str],
    source_completion: Completion,
    scope_completion: Completion,
    comparison_completion: Completion,
) -> dict[str, Any]:
    """Run three ordered checks; never authorize a headline or rendered clip."""
    if not isinstance(proposed_answer, str) or not proposed_answer.strip():
        raise ValueError("atomic relation needs a proposed answer")
    source_answer = answer_source_question(question, delivered_units, source_completion)
    scope_review = assess_source_answer_scope(
        question=question,
        source_answer=source_answer,
        delivered_units=delivered_units,
        completion=scope_completion,
    )
    comparison = None
    verdict = "uncertain"
    if scope_review["responsiveness"] == "answers_question" and scope_review["scope"] != "unknown":
        comparison = compare_claim_answer(
            question=question,
            proposed_answer=proposed_answer,
            source_answer=source_answer,
            scope_review=scope_review,
            delivered_units=delivered_units,
            completion=comparison_completion,
        )
        if comparison["relation"] == "equivalent":
            verdict = "supported_label_unqualified"
        elif comparison["relation"] == "different":
            verdict = "unsupported_label_unqualified"
    return {
        "question": question,
        "proposed_answer": proposed_answer,
        "source_answer": source_answer,
        "scope_review": scope_review,
        "answer_comparison": comparison,
        "verdict": verdict,
        "claim_inventory_semantically_qualified": False,
        "source_entailment_qualified": False,
        "diagnostic_only": True,
        "production_approved": False,
    }

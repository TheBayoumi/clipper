"""Pinned CPU diagnostic for predicate-argument source entailment.

This deliberately does not implement a production factual gate. It measures
whether a specialist model can distinguish previously annotated relations,
then preserves every input and raw probability for source review.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

from clipper.editorial_benchmark import FrozenRelation, load_frozen_relations

MODEL_REPO = "lytang/MiniCheck-Flan-T5-Large"
MODEL_REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
MODEL_WEIGHTS_SHA256 = "41291881e13c6235ed47149cec903bee9493e45d9d7325587a9fa2e266c526c0"
MODEL_MAX_INPUT_TOKENS = 2048
SOURCE_SHA256 = "2a7e07b37074f3073d71b65e10a3efb4019b3cdd4277bc2d3770a99dcbc55e0a"


class RelationScorer(Protocol):
    def score(self, document: str, claim: str) -> tuple[float, int]: ...


class MiniCheckCpuScorer:
    """Reproduce the official Flan-T5 binary scoring path with pinned files."""

    def __init__(self) -> None:
        torch = import_module("torch")
        snapshot_download = import_module("huggingface_hub").snapshot_download
        transformers = import_module("transformers")
        AutoModelForSeq2SeqLM = transformers.AutoModelForSeq2SeqLM
        AutoTokenizer = transformers.AutoTokenizer

        torch.set_num_threads(2)
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_REPO,
                revision=MODEL_REVISION,
                allow_patterns=[
                    "added_tokens.json",
                    "config.json",
                    "generation_config.json",
                    "pytorch_model.bin",
                    "special_tokens_map.json",
                    "spiece.model",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
            )
        )
        with (snapshot / "pytorch_model.bin").open("rb") as weights:
            digest = hashlib.file_digest(weights, "sha256").hexdigest()
        if digest != MODEL_WEIGHTS_SHA256:
            raise RuntimeError("pinned MiniCheck model weights differ from published SHA-256")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(snapshot, local_files_only=True)
        self.model.eval()
        if self.tokenizer.eos_token_id != 1 or self.model.config.decoder_start_token_id != 0:
            raise RuntimeError("pinned MiniCheck label or prompt tokenization changed")
        self.torch = torch

    def score(self, document: str, claim: str) -> tuple[float, int]:
        if len(document.split()) > 500:
            raise RuntimeError("MiniCheck source exceeds the upstream 500-word chunk size")
        text = "predict: " + document + self.tokenizer.eos_token + claim
        encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        token_count = int(encoded["input_ids"].shape[1])
        if token_count > MODEL_MAX_INPUT_TOKENS:
            raise RuntimeError("MiniCheck source/claim would be truncated")
        decoder = self.torch.zeros((1, 1), dtype=self.torch.long)
        with self.torch.inference_mode():
            logits = self.model(**encoded, decoder_input_ids=decoder).logits[0, 0, [3, 209]]
            probability = float(self.torch.softmax(logits, dim=-1)[1].item())
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("MiniCheck returned an invalid support probability")
        return probability, token_count


def run_relation_probe(
    fixture_path: Path,
    proof_path: Path,
    transcript_path: Path,
    provenance_path: Path,
    output_path: Path,
    *,
    scorer: RelationScorer | None = None,
) -> int:
    """Score frozen relations and original headlines without changing production."""
    relations = load_frozen_relations(fixture_path, proof_path, transcript_path, provenance_path)
    if len(relations) != 15 or len({row.fixture_index for row in relations}) != 12:
        raise ValueError("relation probe requires all fifteen frozen QA controls")
    proof: Any = json.loads(proof_path.read_text(encoding="utf-8"))
    frozen = proof["annotated_fixtures"]
    if any(type(item.get("expected_supported")) is not bool for item in frozen):
        raise ValueError("frozen headline labels are incomplete")
    source_hash = proof["model_profile"].get("source_sha256")
    if source_hash != SOURCE_SHA256:
        raise ValueError("relation probe requires the pinned original source")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "experiment": "frozen_predicate_argument_minicheck_diagnostic",
        "diagnostic_only": True,
        "production_approved": False,
        "experiment_complete": False,
        "semantic_pass": False,
        "annotation_status": "derived_from_frozen_transcript_controls_not_independent_audio_gold",
        "baseline_proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "source_sha256": source_hash,
        "transcript_sha256": proof["transcript_sha256"],
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "runtime": {
                name: version(name)
                for name in ("torch", "transformers", "huggingface-hub", "sentencepiece")
            }
            if scorer is None
            else "injected_test_scorer",
            "max_input_tokens": MODEL_MAX_INPUT_TOKENS,
            "decision_rule": "support_probability > 0.5 (upstream MiniCheck default)",
        },
        "relation_rows": [],
        "headline_rows": [],
    }

    def checkpoint() -> None:
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint()
    began = time.monotonic()
    backend = scorer if scorer is not None else MiniCheckCpuScorer()

    def measure(document: str, claim: str, expected: bool) -> dict[str, Any]:
        probability, token_count = backend.score(document, claim)
        if (
            type(probability) is not float
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            or type(token_count) is not int
            or not 0 < token_count <= MODEL_MAX_INPUT_TOKENS
        ):
            raise RuntimeError("relation scorer returned an invalid or truncated result")
        supported = probability > 0.5
        return {
            "document": document,
            "claim": claim,
            "input_tokens": token_count,
            "support_probability": probability,
            "actual_supported": supported,
            "expected_supported": expected,
            "passed": supported is expected,
        }

    for relation in relations:
        document = " ".join(relation.source_units)
        claim = f"{relation.question} {relation.answer}"
        row = {
            "case_id": relation.case_id,
            "fixture_index": relation.fixture_index,
            "headline": relation.headline,
            "dimension": relation.dimension,
            "question": relation.question,
            "answer": relation.answer,
            "annotation_reason": relation.annotation_reason,
            "evidence": {
                "first_unit": relation.evidence_first_unit,
                "last_unit": relation.evidence_last_unit,
                "quote": relation.evidence_quote,
            },
            **measure(document, claim, relation.expected_supported),
        }
        report["relation_rows"].append(row)
        checkpoint()
        print(f"RELATION_QA {relation.case_id} passed={row['passed']}", flush=True)
    by_index: dict[int, FrozenRelation] = {}
    for relation in relations:
        by_index.setdefault(relation.fixture_index, relation)
    for index, item in enumerate(frozen):
        row = {
            "fixture_index": index,
            **measure(
                " ".join(by_index[index].source_units),
                item["headline"],
                item["expected_supported"],
            ),
        }
        report["headline_rows"].append(row)
        checkpoint()
        print(f"WHOLE_HEADLINE {index} passed={row['passed']}", flush=True)
    relation_rows = report["relation_rows"]
    headline_rows = report["headline_rows"]
    report["scores"] = {
        name: {
            "correct": sum(row["passed"] for row in rows),
            "total": len(rows),
            "false_approvals": sum(
                row["actual_supported"] and not row["expected_supported"] for row in rows
            ),
            "false_rejections": sum(
                not row["actual_supported"] and row["expected_supported"] for row in rows
            ),
        }
        for name, rows in (("relations", relation_rows), ("whole_headlines", headline_rows))
    }
    report["experiment_complete"] = True
    report["semantic_pass"] = all(row["passed"] for row in (*relation_rows, *headline_rows))
    report["seconds"] = round(time.monotonic() - began, 3)
    report["qualification_rule"] = (
        "This frozen, transcript-derived diagnostic cannot qualify production, even at 27/27; "
        "independent audio-reviewed held-out controls and integrated editorial review "
        "remain required."
    )
    checkpoint()
    return int(not report["semantic_pass"])

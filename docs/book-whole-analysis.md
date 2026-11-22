# Whole-book candidate analysis — recorded or opt-in local CPU, resumable and unreviewed

This is a **front-half-only** contract in `src/clipper/books/whole_book.py`
and `BookOrchestrator.mine_candidate`. It connects the exact versioned
title-only extraction to an ordered provisional mining work plan. Recorded
mode never infers. The separate **explicit `--local-cpu`** mode invokes pinned
open-weight Qwen on CPU only. Neither mode performs acquisition, OCR,
creative review, episode planning, rendering or publishing.

## Exact input and compatibility

```bash
# Requires an earlier authorized, pinned candidate acquisition on disk.
clipper books ingest --title "Atomic Habits" --offline --artifact-root artifacts/books

# Use the exact returned analysis_ref_path and separately prepared local responses.
clipper books mine --candidate-ref \
  artifacts/books/book-candidates/analysis-refs/atomic-habits/EXACT_STAGE_KEY.json \
  --book-id atomic-habits --responses /local/recorded-responses.json \
  --artifact-root artifacts/books

# Explicit alternative: no response-map file; local CPU inference only.
python -m pip install -e ".[niche-models]"
clipper books mine --candidate-ref \
  artifacts/books/book-candidates/analysis-refs/atomic-habits/EXACT_STAGE_KEY.json \
  --book-id atomic-habits --local-cpu --max-new-units 8 \
  --artifact-root artifacts/books
```

`--responses` and `--local-cpu` are mutually exclusive, and one is required.
`--max-new-units` is valid only with `--local-cpu --candidate-ref`; default
**8**, allowed **1–20,000**, counting only uncached generation attempts.
Return code **0** means all eligible units completed, **2** means
incomplete progress, and **1** means a command or validation error.
The existing legacy owner-file `books mine --book-id ... --responses ...`
interface stays recorded-only. Local CPU mode uses precisely the pinned
`Qwen/Qwen3-1.7B` revision
`8d4744f9e13072f4920c326350fa81eedb74eae9`.
Install `.[niche-models]` first. At the **first uncached eligible unit only**,
the loader asks Hugging Face Hub for the pinned *local* snapshot and passes
its verified `.../models--Qwen--Qwen3-1.7B/snapshots/<exact-revision>`
directory (not a repo ID) to Transformers 4.57.3. This avoids its
tokenizer repository-metadata lookup in offline mode. The cached snapshot
must contain its tokenizer/config files and all indexed safetensors shards.
Missing or incomplete offline cache fails with a bounded
`pinned_model_cache_unavailable` unit failure, not another revision or a
fabricated zero-candidate PASS. Model loading still occurs only after
the generated-unit DAG misses; recorded mode and completed cache replay
do not resolve or initialize the model.

To **pre-cache public weights explicitly** on a network-enabled owner
machine, outside this repository's CI, use the exact revision:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-1.7B", revision="8d4744f9e13072f4920c326350fa81eedb74eae9")'
# Once that full snapshot is cached, run the --local-cpu command above with:
HF_HUB_OFFLINE=1 clipper books mine --candidate-ref /exact/private/analysis-ref.json \
  --book-id atomic-habits --local-cpu --max-new-units 8 --artifact-root artifacts/books
```

Without offline mode the explicitly opted-in local CPU path may download
missing pinned public weights. All inference remains local; weight download
is **not** a remote model inference service. The full pinned weights
require substantially more than ~2 GB available RAM to initialize (the
separate feasibility attempt failed at mmap of its first 3.44 GB shard).
Plan for approximately 8–16 GB RAM or more including transient allocations
and potentially multi-hour/day full-book inference; actual memory needs
must be measured. No real weights were loaded in the coding CI.

The versioned `CandidateAnalysisRef` is read in full with a 128 KiB
byte limit plus a one-byte overflow probe; the acquisition dossier,
manifest and ingest-stage JSON are likewise bounded at 2 MB. Oversize
or truncated-prefix files fail closed even if their initial JSON value
is valid. Exactly bounded JSON files are accepted when complete, and
neither malformed input nor historic immutable evidence is rewritten.

The versioned `CandidateAnalysisRef` contains selected candidate metadata,
candidate/acquisition/source fingerprints, pinned raw SHA, ingest stage key,
extracted/dossier byte hashes, book fingerprint, parser version/config and
recovery status. The reader reconstructs the expected paths from this handle,
validates actual files, acquisition metadata and the canonical ingest DAG PASS,
and rejects mismatched, changed, unsafe or cross-book references before mining.
Reviewed candidate identity uses the registry's explicit `book_id` (for example,
`psychology-of-money`), while parsed title and author must exactly match the
reviewed edition; no additional title-derived ID is required. It never resolves
by last-modified or "latest" edition. The ingest CLI prints
`analysis_ref_path` even if source recovery is incomplete and it exits 2.
References and all source-bearing outputs are immutable/private (0700
directories, including intermediate application-owned ancestors, and 0600
files on supported POSIX filesystems) under an owner-controlled artifact root.
Unrelated shared ancestor permissions are not changed. Previous raw, extraction, report and cache
bytes are never migrated.

Legacy `clipper books ingest --source ...` and `books mine --book-id ...`
remain separate interfaces; their original limits and safeguards are intact.

## Deterministic evidence plan and response shape

`faithful.reading_blocks` supplies ordered native text and
per-occurrence image/research evidence. Nonempty reading text is split
into **at most 350 words per unit**, with an explicit 20,000-unit bound,
and processed in **32-unit windows**. It does not truncate at 128 global
units or 64 total proposed lessons; each individual answer still has
**zero to three** bounded candidates. The per-unit response limit is
16,384 UTF-8 JSON bytes, the input response file is bounded at 64 MB,
and the immutable extracted projection reader is bounded at 32 MB.
Exceeding a bound is an explicit failure, not a partial success.

Each unit request is `whole_book.request_for(book, unit)`. The recorded
response file is a JSON object keyed by
`content_fingerprint(request)` (canonical sorted-key JSON SHA-256).
Its values must contain:

```json
{
  "lessons": [
    {
      "id": "invented-idea",
      "topic": "Synthetic topic",
      "claim": "Unreviewed hypothesis",
      "hook_angle": "An original demonstration",
      "visual_action": "A fabricated visual",
      "evidence": "exact substring of unit text",
      "evidence_field": "native_text"
    }
  ]
}
```

The `evidence_field` **must** equal the field named by that exact
request, and `evidence` must occur within its supplied text, not a
different original/normalized/OCR field. Evidence anchor records preserve
the original section, source/asset paths, occurrence anchor, word offset,
source/content fingerprints, review state, research link state and origin.
Native prose, original image text, normalized research and unreviewed OCR
are never relabeled as interchangeable original words. Empty/unrecovered
images have explicit `skipped` gap records, with source and review
provenance retained. `incomplete_recovery`, unverified rights, excluded
pages and other quality warnings remain distinct from analysis progress.
There is no semantic deduplication, ranking, target episode count or
automatic claim approval.

## Interrupted work and output

Every validated unit still uses the existing private
`book-dag/books-mining-unit-v1/` DAG with source/book/projection/request
fingerprints, model revision, decoding parameters and **used answer
fingerprint**. The original recorded identity material remains unchanged;
different recorded mappings can reuse identical replies. In recorded mode,
missing answers never trigger inference and remain unresolved.

The local producer first constructs an independent private
`book-dag/books-mining-local-completion-v1/` stage key from the exact
request, source/book/plan fingerprints, pinned model/revision, generation
parameters, runtime package versions and producer/validator contract.
This identity is knowable **before generation**; the cache is checked before
model initialization, and the pinned Qwen pipeline loads at most once per
process. A recorded empty-test reply never satisfies this local identity.
Raw decoded completion text is retained under the private
`analysis/completions/<book-id>/<generation-key>/<sha256>.txt` hierarchy
before validation. The maximum is 16,384 UTF-8 bytes; oversize returns
failed status and only the bounded prefix is retained with its original
full-response SHA-256 in sanitized error provenance. Malformed JSON,
invalid quotes/fields and runtime errors become **FAILED**, never
successful zero-candidate results. Immutable sanitized failure events
under `analysis/failures/` identify the generation stage, request hash,
attempt number, category and raw hash when available, not source text or
exception bodies. Failure stages can retry; valid returned
`{"lessons":[]}` alone creates a zero-candidate result. Completed
generated stages verify their original raw bytes/hash and the validator
output on every reuse; changed generation settings, runtime version or
source/request material get different keys. Storage errors and immutable
byte conflicts remain errors, not successful mining.

A private immutable progress report is published under
`book-candidates/analysis/progress/<book-id>/`; it accounts for **every
planned unit** by ordinal/window, with `complete`, `zero_candidate`,
`unresolved` and `failed` states plus each excluded `skipped` image
gap. A missing or invalid later answer produces
`status="incomplete"` and exit code 2; no partial candidate catalog is
exported. A successful pass across all eligible units produces a separate
private `book-candidates/analysis/exports/<book-id>/` provisional catalog,
even when there are **zero candidates**.

The immutable report filename binds the work-plan fingerprint, either
the **complete recorded-response file SHA-256** or the distinct **local
producer fingerprint**, and the SHA-256 of exact serialized progress
bytes (including all per-unit outcomes). Local progress also records
completion origin, generation stage key, raw completion hash and sanitized
failure category. It never fabricates a recorded response-file hash. A transient
failure followed by a successful retry with the **same** response-file
bytes therefore creates a *different* complete progress artifact without
overwriting or deleting the earlier incomplete report. A replay with the
same outcomes reuses byte-identical immutable progress and export files.
The complete export uses the corresponding outcome-qualified filename
and embeds `progress_sha256` of the actual report bytes. Publication
conflicts and storage errors are not treated as successful retries.
Response-file provenance lives on the progress report; each DAG unit
depends only on its own used response. Validated book and plan fingerprints
are computed once per invocation and reused throughout the per-unit loop
without changing the existing cache or request identity material.
IDs include an exact unit/idea SHA-256 and collisions are rejected,
never overwritten. All candidates and exports remain
`MINED_UNREVIEWED` with `publish_eligible=false`; the old selector's
`approved_facts` name is **not** an approval mechanism and the new path
does not invoke it.

The analysis status `complete` means every **eligible work unit** has
a valid recorded or generated response; it is **not** a semantic lesson
quality verdict. Exact quotation matching only checks structural grounding:
it does not prove factual truth, usefulness, uniqueness, suitable
storytelling, rights or human approval. This status does **not** upgrade
source recovery, authenticate an edition, authorize rights, or certify
original image words. `recovery_status=incomplete_recovery` and explicit
skipped gaps can coexist with analysis completion. The CPU budget affects
only *this invocation*; revisiting the exact command resumes all cached
units and marks remaining work unresolved, without a partial catalog.

## Gates and verification

The unchanged production book stages beyond selection/script/recorded
narration still stop at `books.direction` (`NOT_PORTED`).
No mined result automatically enters curated selection or its
storyboard requirements. Rights, human observation/creative review,
source fidelity and actual final MP4 acceptance are distinct gates.
Repository tests use fabricated text and a synthetic EPUB; real private
book images and research sidecars are absent and not certified.

Reproduce with the existing repository tools (no live services):

```bash
ruff check .
ruff format --check .
mypy
pytest
coverage report --fail-under=95
pytest -o addopts="" -q tests/test_books_whole_book.py
```

The 95% threshold is for the `clipper` package, separate from scripts.
See [candidate acquisition and source rights](book-candidate-acquisition.md),
[historical Phase 3 path](book-pipeline-phase3.md),
[DAG recovery](dag-execution.md) and
[creative review](book-creative-review.md).

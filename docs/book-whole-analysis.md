# Whole-book candidate analysis — recorded, resumable and unreviewed

This is a **front-half-only**, CPU-only contract in
`src/clipper/books/whole_book.py` and `BookOrchestrator.mine_candidate`.
It connects the exact versioned title-only extraction to a provisional
mining work plan. It does not perform acquisition, OCR, model inference,
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
```

The versioned `CandidateAnalysisRef` contains selected candidate metadata,
candidate/acquisition/source fingerprints, pinned raw SHA, ingest stage key,
extracted/dossier byte hashes, book fingerprint, parser version/config and
recovery status. The reader reconstructs the expected paths from this handle,
validates actual files, acquisition metadata and the canonical ingest DAG PASS,
and rejects mismatched, changed, unsafe or cross-book references before mining.
It never resolves by last-modified or "latest" edition. The ingest CLI prints
`analysis_ref_path` even if source recovery is incomplete and it exits 2.
References and all source-bearing outputs are immutable/private (0700
directories, 0600 files on supported POSIX filesystems) under an
owner-controlled artifact root. Previous raw, extraction, report and cache
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

A successful unit is stored under the existing private
`book-dag/books-mining-unit-v1/` DAG mechanism with a dedicated versioned
contract, source/book/projection/request fingerprint, model revision,
decoding settings and the **used answer fingerprint**. Different later
response-file mappings do not invalidate an earlier identical unit answer.
Changed source, projection, request, pinned model identity or used answer
changes that unit's key. No model is loaded and missing responses never
trigger inference. Failed units can be retried with corrected recorded
answers; a different answer gets a new key.

A private immutable progress report is published under
`book-candidates/analysis/progress/<book-id>/`; it accounts for **every
planned unit** by ordinal/window, with `complete`, `zero_candidate`,
`unresolved` and `failed` states plus each excluded `skipped` image
gap. A missing or invalid later answer produces
`status="incomplete"` and exit code 2; no partial candidate catalog is
exported. A successful pass across all eligible units produces a separate
private `book-candidates/analysis/exports/<book-id>/` provisional catalog,
even when there are **zero candidates**. Both output identities include
the work-plan fingerprint and the complete recorded-response file SHA.
Response-file provenance lives on the progress report; each DAG unit
depends only on its own used response. IDs include an exact unit/idea
SHA-256 and collisions are rejected, never overwritten. All candidates
and exports remain `MINED_UNREVIEWED` with
`publish_eligible=false`; the old selector's `approved_facts` name is
**not** an approval mechanism and the new path does not invoke it.

The analysis status `complete` means every **eligible work unit** was
accounted for. It does **not** upgrade source recovery, authenticate a
publisher edition, authorize rights or certify every image word.
`recovery_status=incomplete_recovery` and explicit skipped gaps can
coexist with that analysis status. Producer calls require a separately
provided complete recorded response mapping.

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

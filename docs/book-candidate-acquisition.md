# Reviewed candidate acquisition and faithful source-text ingest

## Scope and evidence status

This is a **bounded title-only acquisition and text-recovery entry point**, not
title-to-video production acceptance and not a book-source licensing decision.
The committed registry `niches/book-source-candidates-v1.json` contains three
explicitly reviewed candidate records. It does not authenticate a publisher
edition, establish full-text exactness, infer usage authorization, or solve
arbitrary internet title search. Every candidate stays
`authorization_status="unverified"`, even if the owner chooses to investigate
it. The original owner-file and injected Gutenberg resolver paths remain
available.

An unmatched or ambiguous title yields a typed `SourceUnresolved` with a
concrete owner decision and known options. No fuzzy substitution, other-edition
fallback, automatic upload request, in-copyright source recommendation,
login, private browsing, paywall, DRM bypass or unauthorized preview is
performed. Source text, XML metadata, links and OCR are data, not instructions.

## Commands

Run from a Clipper repository checkout with local Python 3.11+:

```bash
clipper books candidates --title "Atomic Habits"
clipper books acquire --title "Atomic Habits" --artifact-root artifacts/books
clipper books ingest --title "Atomic Habits" --artifact-root artifacts/books
```

`books ingest` combines discovery, anonymous bounded HTTPS acquisition, source
validation, ordered extraction, private artifact persistence and evidence
dossier creation. `--author` and `--edition` restrict discovery; neither
overwrites the registry provenance. `--registry` permits an explicit
schema-validated candidate registry. The owner-placed
`books ingest --title ... --author ... --source /path/book.epub` legacy form
is preserved and distinct from anonymous acquisition.

An unchanged normal acquisition reuses the pinned, verified private cache without a network
request. Use `--refresh` to explicitly revalidate the remote record and asset; a new
immutable refresh event is saved, while the first acquisition remains immutable.
Changed metadata, restricted listings and changed pinned bytes fail closed with an
immutable incident; no automatic edition substitution takes place. `--refresh` and
`--offline` cannot be combined.

For deterministic offline resume after a prior successful acquisition:

```bash
clipper books ingest --title "Atomic Habits" --offline \
  --artifact-root artifacts/books \
  --overlay /private/reviewed-atomic-images.json
```

An overlay is **optional evidence import**, not an instruction to recreate
private transcriptions. The existing private Atomic and Psychology research
sidecars are not in the connected GitHub repository and have not been
read, executed, altered or reconstructed. Until those sidecars are available
on the authorized execution machine, their actual image recovery coverage
remains **unverified**.

Any injected OCR producer now requires a deterministic `OCRIdentity` containing the
adapter/version/settings; extraction cache keys bind to its fingerprint. The optional
local Tesseract mode records its resolved executable, version-probe digest,
explicit English language and PSM 6. OCR outputs remain unreviewed, with a
separate output-evidence fingerprint in the private dossier. An unchanged identity
reuses the extraction cache; changing it invalidates the extraction.

The `--ocr tesseract` option runs only an explicitly installed local CPU
Tesseract executable; the default is no OCR. Machine text stays separately
labeled `machine_unreviewed` and never replaces source prose or reviewed
overlay transcriptions. A missing/failed OCR path produces
`incomplete_recovery`, never a successful full-text assertion.

## Private artifact boundaries

Artifacts are placed under `<artifact-root>/book-candidates` and should be
kept OUTSIDE version control and outside shared build outputs:

- `raw/<sha-prefix>/<raw-sha256>.<epub|pdf>`: immutable source bytes
- `manifests/<candidate-fingerprint>.json`: HTTP status, requested/final asset
  URLs, metadata digest, fetched-at time, observed byte count/hash, declared
  edition, source fingerprint and authorization uncertainty
- `extracted/<parser-version>/<stage-cache-key>.json`: separate immutable
  source-ordered text, chapter/page anchors, source blocks, image evidence
- `dossiers/<book-id>/<stage-cache-key>.json`: NON-CONTENT aggregate counts,
  warnings, image occurrences and unresolved reviews, exclusion page hashes,
  parser/OCR identities, raw and extracted artifact hashes
- `errors/<candidate-fingerprint>-<incident-sha>.json`: immutable failure
  reasons for different unavailable/restricted/changed sources

Source, parser, OCR mode, candidate identity and reviewed overlay digest are
part of ingest stage identity. The recorded raw-book SHA, asset SHA and
occurrence anchors bind overlay evidence. A changed source, parser or
sidecar invalidates the relevant cached stage. Offline mode verifies the
pinned source digest and does not invoke transport.

Neither raw books nor extracted text, page images, recovered full text,
temporary OCR files or private sidecars are committed by this change.
The checked-in synthetic tests deliberately use original fabricated text
and generated tiny PDFs/EPUBs.

## Selected edition candidates (not authenticated publisher files)

| Title | Reviewed candidate | Format | Raw byte count | SHA-256 |
|---|---|---|---:|---|
| Atomic Habits — James Clear | ISBN 9780735211308, Archive item `atomic-habits-by-james-clear_202601` | EPUB | 4,753,022 | `4f7249b4cecfcf400d36b461e8a49632b041b139a7ce560b40dede43018afdd3` |
| The Psychology of Money — Morgan Housel | ISBN 9780857197696, item `the-psychology-of-money_202512` | EPUB | 1,912,971 | `13f8143a3f9a75a0beceba653dac9da21ec91eab5bd41384bc3171990e962c0f` |
| Deep Work — Cal Newport | UK Piatkus ISBN 9780349411910, item `deep-work-rules-for-focused-success-in-a-distracted-world_202608` | PDF | 1,498,415 | `1df665c62d154777a4db06c3600a568930d7fc79157dcf44a63f9836342fef4b` |

The Deep Work registry expects 230 physical PDF pages and excludes **only
page 230**, the reviewed appended third-party distribution page. The raw PDF
remains immutable; exclusion appears with reason and observed page-text
digest. Front matter, notes and index are retained. This edition is never
silently blended with the US comparison. The source-visible
`proscribed/prescribed` wording and anomalous Atomic chart label
`40 10 60` are never automatically corrected.

Source rights, acquisition provenance, edition fidelity, mechanical
extraction, image/OCR coverage, human review and publication eligibility
remain independent status fields. A successful parser does not prove the
complete edition was recovered. The exit code for `incomplete_recovery`
is **2**, with private diagnostics preserved; no creative approval follows.

## Extraction and review controls

EPUB extraction validates the OPF title aliases, exact author, reviewed
ISBN and unique package identifier when present, then walks the declared
spine order and preserves text, inline spans/tails, lists, tables, captions,
notes and image references. Unique assets and EVERY occurrence are
tracked by hash and reading-order anchor. It does not insert alt text,
OCR words or a diagram interpretation into source prose twice.

PDF extraction uses optional CPU PyMuPDF `>=1.26.3,<1.28` for ordered
page/text/image blocks and sparse-page detection. Check PyMuPDF's
AGPL/commercial licensing requirements independently before any commercial
deployment. Both parsers preserve exact source spelling and punctuation.
Local Tesseract is a separately optional adapter, not a dependency or
provider. Missing source-visible image words, empty/raster pages,
unreviewed OCR, low word counts, unclear edition or truncated packaging
produce diagnostics instead of invented recovered text.

Archive transport accepts only explicit official HTTPS Archive endpoints
and narrowly allowlisted Archive CDN redirects. It caps response bytes,
redirects, page/ZIP expansion and elapsed request time, rejects unsafe
paths, HTML download substitutes, private addresses, restricted files,
ambiguous listings, changed pinned hashes and throttling, with no
unbounded retry or weaker-source fallback. Separately reported anonymous
HTTP verification at the predecessor SHA `93784f4188be0b741c295da28011fdcac7ab8f5a`
recovered all three pinned candidates. That independent validation is not
verification of this newer recovery contract or the private sidecar imports.

## Acceptance limitations

Synthetic tests validate mechanics, fingerprints, negative security behavior,
offline resume, overlay invalidation and Phase 1–3 regressions. Separate
independent CPU execution at predecessor SHA `93784f4` succeeded in anonymous
acquisition and ordered parsing of the three pinned editions: Atomic
78 sections / 2,509 blocks / 24 unique images / 47 occurrences;
Psychology 51 sections / 1,656 blocks / 37 unique images / 37 occurrences;
Deep UK 229 retained pages / 1,272 blocks / 4 unique images / 38 occurrences,
with physical page 230 excluded. Counts are diagnostics, not fidelity
certificates; **all three remained incomplete_recovery**. This round has not
inspected, converted or reviewed actual private research observations. The
prior private research image inventory was incomplete, including a missing
Psychology cover observation. No source authorization or publisher
authentication is established.
The old curated selection, claim/HTTPS citation gate, recorded-WAV
narration, direction `NOT_PORTED` boundary and creative-review controls
are unchanged. Candidate text does NOT enter automatic mining, and an
Archive URL must not be treated as independently verified evidence for
every claimed lesson. No production acceptance is claimed. Full
candidate fidelity, checked human image overlays, any live mining,
source authorization, and the unported direction, visuals, assembly and
review stages require separate evidence and authorization.


## Deterministic private research import (repair boundary)

After an authorized local acquisition/extraction, import locally exported image
observations with `clipper books overlay-import --title "Atomic Habits" --research
/private/observations.json --output /private/reviewed-overlay.json --artifact-root
artifacts/books`. This command is offline-only. The input schema
`clipper-book-research-observations-v1` binds the raw book SHA-256 and each
reviewed asset SHA-256 to the exact ordered list of `{anchor, source_path}`
occurrences. Observations retain original transcription, separate normalized
transcription, normalization notes, reviewer label, and an optional actual
review timestamp; null means unknown, not a fabricated date. Partial inventories
leave their unmatched assets/occurrences unreviewed. Output is an immutable
private overlay and a non-content source/path/hash/occurrence reconciliation
report. No private sidecar is committed. This is an interface test boundary:
actual private research files are not available in this environment.

The observed Psychology discrepancy is a missing cover.jpeg in the earlier
reviewed occurrence inventory (36 reviewed observations versus 37 extraction
occurrences); this report is computed from the exact source, never from a
hardcoded count. The Deep UK final physical page exclusion stays source-bound.


## Recovery provenance v2: legacy normalized observation import

The v1 reviewed overlay and v1 research-observation conversion remain readable
as **explicit v1** formats; a v1 overlay requires original transcription,
nonempty reviewer label, and explicit reviewed status as before. The new
`clipper-book-image-overlay-v2` and `clipper-book-research-observations-v2`
are versioned separately. Ordered source extraction now uses
`clipper-book-ordered-extract-v2`, so previous extraction-stage cache
identities are not reused after the evidence contract changed. A stale
v1/unknown overlay is never silently interpreted as v2.

The new research input has these exact top-level keys: `schema`,
`source_sha256`, `research_origin`, `research_author` (nullable),
`occurrence_inventory` and `observations`. The inventory is a list of
`{asset_sha256, anchor, source_path}` records originating from the exact
extracted source (not index-based guesses). For EPUB, the occurrence
`source_path` is the resolved image asset file; a reading `SourceBlock` has a
different `source_path` for its containing XHTML file. The ordered
`ReadingBlock` retains both (`source_path` and `asset_source_path`) and
joins by unique source anchor plus asset hash, without equating these paths. An observation repeats these
three identity fields and adds `observation_id`, nullable
`original_transcription`, nullable `normalized_transcription`,
`normalization_notes`, nullable `reviewed_by`, nullable `reviewed_at`,
and nullable `review_status` (either `reviewed` or `needs_review` when
known). A normalized-only record needs a nonblank normalization note,
even when its original character transcription remains unknown.
Review cannot be marked complete without both an actual original
transcription and an explicit human reviewer label. The research author
is **not** automatically a reviewer. No reviewer-label authentication is
implied; this remains the documented local single-operator trust boundary.

The conversion rejects unknown book/asset/anchor/path combinations,
duplicate inventory occurrences, contradictory observation IDs,
duplicate annotations at an occurrence, missing review provenance on
asserted reviewed observations and normalized text without a change note.
An identical asset may occur repeatedly at different source anchors:
each observation is linked only to its exact occurrence. A partial legacy
inventory is acceptable; unmatched covers/images remain unreviewed,
not rejected or filled in. Report counts and per-occurrence reasons
distinguish linked, normalized-only, reviewed and unresolved observations.
A rejected import emits a separate immutable non-content rejection dossier
with its reason and research-input hash; it never emits a usable overlay.
Linear image reading projections carry distinct `text_provenance`,
source-original vs normalized text and normalization notes; they do not
claim preserved diagram geometry or duplicate image alt-text as prose.

On an authorized local machine with an existing pinned acquisition cache:

```bash
clipper books overlay-import --title "The Psychology of Money" \
  --artifact-root /private/clipper-books \
  --research /private/research-export-v2.json \
  --output /private/psychology-overlay-v2.json
clipper books ingest --title "The Psychology of Money" --offline \
  --artifact-root /private/clipper-books \
  --overlay /private/psychology-overlay-v2.json
```

The converter works **offline** and persists an unchanged private copy of
the research JSON (by its SHA-256) plus a separate versioned derived overlay
and a non-content reconciliation report. Repeating an identical import
reuses the same immutable bytes. Changing even provenance or grouping
notes changes the derived overlay hash and therefore extraction stage key.
Normalized-only and unreviewed records leave recovery
`incomplete_recovery`; neither this conversion nor an offline ingest
establishes rights, source exactness, editorial approval or eligibility
to publish. Real retained private research observations were not available
for this implementation round. The separately supplied three-book CPU
validation applies to the prior exact SHA, not to imported private recovery.

### Evidence hardening (parser v3)

Both research and overlay JSON are parsed with duplicate-member rejection at every
object depth and Unicode validation. V1 overlays remain readable. V2 standalone
research digests are **unverified claims** unless the exact raw research exists in
the private source directory and deterministic conversion matches the overlay.
The raw research digest (or its absence) is part of the extraction cache identity.
Parser v3 invalidates v2 extraction identities; no old review state is promoted.

An explicit empty overlay fails before DAG reuse. Rejection dossiers are keyed by
candidate fingerprint and research digest (v2 rejection schema). Partial v2
coverage permits explicitly requested OCR once per asset, with machine evidence
projected only at unlinked occurrences. OCR remains unreviewed and diagram
geometry remains unverified. V1 empty normalized values project original text
with original provenance. Source rights and publication gates remain unchanged.

Candidate and legacy book-ingest share the existing `books-ingest` stage.
That stage opts into private DagStore persistence; its dedicated book-DAG root,
stage directory and cache-key directories are mode 0700 on POSIX. Private
stage JSON temporary files are opened exclusively with mode 0600, then atomically
renamed as before. Candidate ingest restricts prior stage directories and files
(including v2 outputs) without changing their bytes or stage identities. Other
DagStore consumers retain their existing behavior; the book-DAG root restriction
also prevents non-owner traversal of sibling book stages. These are local
filesystem permission controls, not protection against another owner-equivalent
process or a filesystem without POSIX permission enforcement.

### Immutable import reports and bounded evidence

The common evidence boundary in `faithful.MAX_EVIDENCE_BYTES` limits each
research export and image overlay to 2,000,000 bytes, including direct byte
APIs and cached research. Opened files are read up to the limit plus one byte;
oversized input is rejected before hashing, persistence or JSON decoding.
Oversized research receives a typed non-content diagnostic only: no truncated
raw copy, invented digest, rejection dossier keyed to partial bytes or overlay.
Within the limit, malformed JSON (including unsupported numeric tokens, duplicate
members and invalid Unicode) retains the exact raw research and records its
source-bound rejection as before. Valid v1/v2 evidence contains no numeric tokens.

Import reports now use `research-import/<book_id>/report-<identity>.json`.
The deterministic identity binds the import-report contract version,
reconciliation schema, extraction parser version, candidate fingerprint, source,
raw research and overlay digests. The report includes this identity and its
material; future semantic changes must advance the relevant version. Overlay
byte identity is independent: the same overlay can have distinct report
generations. Legacy overlay-hash-named reports are left untouched. Repeated
identical current imports reuse the same path and bytes; an actual same-identity
conflict remains an error. The CLI prints the actual report path. The import
root and evidence subdirectories retain owner-only access on POSIX.

Initial DAG claim/RUNNING publication failures release the owned claim while
holding the local write lock, or through the existing owner-bound distributed
release API. Operation execution cannot begin until initial publication succeeds.
No leases or retry timing are changed. These changes do not alter accepted
parser-v3 extraction identities or establish source fidelity or authorization.

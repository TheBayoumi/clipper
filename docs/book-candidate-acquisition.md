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

An unchanged normal acquisition reuses the pinned, verified private cache without a network\nrequest. Use `--refresh` to explicitly revalidate the remote record and asset; a new\nimmutable refresh event is saved, while the first acquisition remains immutable.\nChanged metadata, restricted listings and changed pinned bytes fail closed with an\nimmutable incident; no automatic edition substitution takes place. `--refresh` and\n`--offline` cannot be combined.\n\nFor deterministic offline resume after a prior successful acquisition:

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

Any injected OCR producer now requires a deterministic `OCRIdentity` containing the\nadapter/version/settings; extraction cache keys bind to its fingerprint. The optional\nlocal Tesseract mode records its resolved executable, version-probe digest,\nexplicit English language and PSM 6. OCR outputs remain unreviewed, with a\nseparate output-evidence fingerprint in the private dossier. An unchanged identity\nreuses the extraction cache; changing it invalidates the extraction.\n\nThe `--ocr tesseract` option runs only an explicitly installed local CPU
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
unbounded retry or weaker-source fallback. HTTP interoperability and
version-dependent metadata remain **unverified against live Archive**.

## Acceptance limitations

Fixture-only tests validate mechanics, exact fingerprints, negative
security behavior, offline resume, overlay invalidation and the preserved
Phase 1–3 regression suite. **No live candidate acquisition, real EPUB/PDF
page/asset recovery count, actual image transcription import, or private
CPU Tesseract availability was verified.** In particular the separately
observed Atomic 24 unique images/47 occurrences and Psychology 36 unique
images/36 occurrences (60 unique/83 total across both) are PRIVATE RESEARCH
BASELINES, not observed output from this implementation.

The old curated selection, claim/HTTPS citation gate, recorded-WAV
narration, direction `NOT_PORTED` boundary and creative-review controls
are unchanged. Candidate text does NOT enter automatic mining, and an
Archive URL must not be treated as independently verified evidence for
every claimed lesson. No production acceptance is claimed. Full
candidate fidelity, checked human image overlays, any live mining,
source authorization, and the unported direction, visuals, assembly and
review stages require separate evidence and authorization.

# Book Pipeline Phase 3 — Offline Source Ingest and Lesson Mining

## Pipeline and approval boundary

The new front-half stages are `books.ingest` and `books.mining`.
They join the existing book DAG, whose later stages remain selection,
scripting, narration, direction, visuals, assembly, and review.
The source-backed curated catalog remains unchanged; mined candidates
are written separately to `<artifact-root>/mined/<book-id>.json`.
The original `selection._validate_catalog` and `selection.choose_lesson`
remain the validation and selection functions for that separate input.
No mined result is auto-merged, automatically approved, published, or
presented as a source-verified editorial fact.

**Rights provenance is not rights authorization.** `BookSource` records
title, author, optional edition, URI, source kind, declarant and timestamp,
format and raw source SHA-256. `authorization_status` defaults to
`unverified` and neither resolver nor miner upgrades it. The optional
`authorization_basis=public_domain` is recorded only for matching
source-text metadata for the named Gutenberg edition; this is not a
legal determination. `SourceUnresolved` carries a concrete owner decision
instead of guessing or searching other sources.

`BookText` records ordered `BookSection` structures containing
per-chapter paragraphs, total word count, edition, source fingerprint
and raw-byte content SHA-256. Ingest stage identity includes the source
fingerprint and observed raw-byte digest. A changed source creates a
distinct stage identity rather than reusing a previous PASS.

## Resolver choices and limits

- `LocalFileResolver`: an explicitly owner-placed UTF-8 TXT or EPUB
  (stdlib ZIP/XML) file. Requires matching embedded title and author.
  With no `--source`, the ingest CLI returns `SourceUnresolved`.
  An owner file is not an automatically verified HTTPS claim citation.
- `GutenbergResolver`: exact title/author match in official Gutenberg
  search-index markup; ambiguous matches fail, no match returns
  `SourceUnresolved`. Fetch/lookup depend on an injectable transport.
  No real HTTP request was performed in this phase. The optional
  production HTTPS transport is unverified against live Gutenberg
  changes and must not be considered approved for production.
- All other sources, including in-copyright works without an owner
  decision, remain unresolved. No library, storefront, or license
  selection is performed automatically.

Completeness checks require at least 3 chapters, at least 100 words
per chapter, and 1,200–500,000 words overall. Errors state the measured
count and required range. Input is limited to 12 MB compressed/raw
and 12 MB aggregate EPUB uncompressed size. Title, author, content
hash, edition, chapter structure, UTF-8 and Gutenberg boilerplate
markers are checked; mismatches fail closed. EPUB DTD/entity declarations
and unsafe internal paths are rejected.

## Mining and CLI

`clipper books ingest --title TITLE --author AUTHOR --source /path/book.txt
--artifact-root artifacts/books` stores the validated BookText locally.
`clipper books mine --book-id BOOK-ID --responses /path/responses.json
--artifact-root artifacts/books` uses **recorded response objects only**.
The response file maps each canonical request fingerprint to a JSON
completion with a `lessons` array; missing responses fail closed.
The miner itself accepts an injected completion function and never
implicitly invokes the pinned Qwen model. Requests use the existing
pinned script model/revision, seed, and token settings.

Chunk order, request payloads and output validation are deterministic.
Claim evidence must appear verbatim in a specific source chunk; the
catalog preserves chapter/paragraph/word-offset anchors and source
SHA-256 for independent review. Duplicate or malformed candidates,
unsupported evidence, unexpected review state, and chunk/lesson
capacity excesses fail closed. Every output is
`MINED_UNREVIEWED` with `publish_eligible=false`. Pillar is
`unreviewed` until a human makes an editorial classification.

The legacy selection and `clipper books validate|plan|run` interfaces,
Kokoro WAV cache, `VOICE_REVISION` residual, direction fail-closed gate,
creative-review controls and publishing boundary remain unchanged.
Direction, visuals, assembly and review remain `NOT_PORTED`.

## Test evidence and explicit limitations

Tests use synthetic full-book TXT and EPUB fixtures plus a committed,
short public-domain Gutenberg excerpt. That excerpt deliberately fails
the full-book completeness gate; synthetic complete-book text is used
for the fixture-only end-to-end test. The test transport and recorded
completion never access the network or load inference weights.

A local owner file can be ingested, but its mined candidates do not
enter the existing validated catalog without an independently verified
HTTPS claim source. Do not invent or substitute an HTTPS URL to bypass
the selector. In-copyright title-only source discovery, live inference,
live Gutenberg reliability, actual legal permissions, human editorial
approval, and production acceptance are **not** established by Phase 3.

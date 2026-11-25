# Public synthetic book-chapter rehearsal — model-free, unqualified

**Scope:** preparation only. The test harness is `tests/book_public_rehearsal.py`;
its companion tests are `tests/test_book_public_rehearsal.py`. There is **no
Modal launcher, client initialization, deploy, workflow trigger, model call,
private source acquisition, upload, publication, spending, or host selection**.

## Intentional execution boundary

From an independently pinned checkout with the already-installed Clipper
CPU/test dependencies, a later independently authorized local or container
operator may explicitly run:

```bash
python tests/book_public_rehearsal.py \
  --fresh-root /tmp/clipper-public-rehearsal-UNUSED \
  --expected-sha EXACT_REVIEWED_40_HEX_SHA
```

The CLI requires `git rev-parse HEAD` to match the explicit full SHA before
creating any artifact. Direct Python `rehearse(..., expected_sha)` reports
that the caller's SHA still needs independent checkout verification. A fresh
**absolute, nonexistent** root is mandatory; only public synthetic fixtures
are created. All artifacts are kept local to that root, not published.
Use only a dedicated, unprivileged, isolated rehearsal process. This is
not authorization to invoke the command on Modal or spend credits.

The entrypoint is deliberately under `tests/` and imports existing
`test_books_chapter_hybrid` fixture helpers. These helpers construct a public
synthetic EPUB through the recorded in-memory transport and production
`BookOrchestrator.ingest_candidate()`, `selected_scope.export_chapter()`,
`import_chapter()`, and `whole_book.plan_for()`. The optional fixture
`section_index=1` selects the section containing a synthetic image/gap; the
existing default fixture section remains unchanged for all other tests.
The original imported unit IDs, Unicode/paragraph bytes, gap marker,
parent-bound scope, artifact SHA-256 and review identity are checked.
The synthetic fixture is **not real chapter analysis**. The test registry
may use an illustrative catalog identity; only generated synthetic prose
and synthetic image markers are ingested, with no external HTTP.

Two separate tracks exercise the **real production**
`chapter.evaluate_hybrid_chapter()`, `chapter_hybrid` validators and
`chapter._stage()` cache/evidence logic. The orchestrator and validators are
not substituted:

1. **Review pause:** no model/callback initially, then deterministic injected
   argument, per-window audit and reconciliation responses. Must stop at
   `AWAITING_HUMAN_CHAPTER_BEAT_REVIEW`, `INJECTED_TEST` and
   `publish_eligible=False`. No synthetic review is supplied on this track.
2. **Distinct test-review continuation:** a fresh synthetic source repeats
   the argument/audit/reconcile logic, stops at the same review gate, and
   only a separate explicitly labeled `SYNTHETIC_TEST_REVIEW_NOT_HUMAN_APPROVAL`
   is used to test adaptation, script, fake valid PCM WAV and direction.
   The expected final state is
   `TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED`, audio provenance
   `INJECTED_TEST_WAV`, and `publish_eligible=False`. This does not bypass
   a human gate in the actual production workflow or prove real reviewer
   approval.

Identical request replay must use the validated cache with zero additional
callbacks. Negative cases reject changed source bundle bytes, stale review
identity, changed validated stage bytes, malformed model responses and an
interrupted in-flight attempt, without automatic producer replay. The report
includes source SHA status, stage names, per-producer callback counts, gaps,
artifact hashes, elapsed monotonic wall time, platform-specific peak RSS,
exclusions, and zero-call guard counters. No private source text is reported.

## Preventing real calls

The harness installs a reversible process-local guard **before** importing
fixture/production modules. It denies imports of `modal`, `torch`,
`transformers`, `huggingface_hub`, `kokoro`, `gdown`, `yt_dlp`; denies
socket connect/create_connection, urllib urlopen, subprocess.Popen and
os.system; and refuses model completion, model setup, miner and Kokoro
provider entrypoints even when already imported. The recorded in-memory
acquisition transport is allowed but no real network transport is allowed.
A prohibited attempt fails closed rather than being counted as zero.
The CLI's local git SHA check runs **before** this in-process guard.
Callbacks return only original synthetic fixture JSON/WAV bytes.

This is an application-level test guard, **not** an OS-level proof against
native networking, arbitrary imports, all potential Python APIs, process
crash or malicious fixture code. A later CPU container must additionally
disable outbound network at its execution boundary and mount no account
credentials, production volumes, private books or model caches. No provider
access is implied from an offline test pass.

## Modal execution gap and bounded proposal

No remote execution entrypoint is implemented or called in this change.
A future separately approved **one CPU-only job**, 1 core / initially 2 GiB
memory, no GPU, no persistent volume, no model/media image dependencies,
would run the **same test-only entrypoint** in an exact source-pinned image.
A nominal 15-minute job timeout is a candidate, **not** an end-to-end
lifecycle or spending cap; image build, idle CPU/memory, storage, startup,
failed jobs and cleanup can incur cost outside it. Free/paid credit status,
which Modal workspace/account/client to use, image build feasibility,
authorization and complete cost cap remain unresolved and must be reviewed
before any provider interaction. Preserve the qualified offline
`modal_infra_qualification.py` and `modal_infra_driver.py`; their synthetic
64 KiB infrastructure check is separate from this chapter rehearsal.

The optional test runner requires a supported Python (3.11–3.13), the
project's installed CPU dependencies and pytest fixture imports. The
existing `pyproject.toml` declares runtime dependencies including numpy,
PyYAML, opencv-python-headless, gdown and yt-dlp; **this rehearsal must not
invoke download behavior**. Torch, Transformers, Kokoro and Modal client
dependencies must not be installed for rehearsal execution. Actual minimal
resolved dependency inventory, image build time, measured local peak RSS
and wall time remain *unverified until execution*. The CLI emits one JSON
report if successful; failed preflight prints only a bounded blocked status.

## What it does **not** prove

It does **not** validate a pinned real tokenizer/chat template or effective
context count, real raw token/EOS receipts, Qwen behavior/quality, real Kokoro
intelligibility, real private chapters 1/7 or their 102/46 scoped units,
human semantic approval, genuine source rights, output video/rendering,
distributed Modal campaign `DagStore`, cross-process persistence, Codespace
signed heartbeat/identity or `chapter_host.execute()`, provider execution,
cost, infrastructure deletion or host-crash recovery. The old
`scripts/modal_pipeline.py` and `run_modal_pipeline()` are media/campaign
entrypoints and are **never imported** here.

This candidate requires exact-SHA local test/Ruff/mypy verification, full
CI with the unchanged 95% coverage gate, and a fresh independent read-only
review before claiming **rehearsal harness qualification**. Even then it
would not authorize a live Modal run.

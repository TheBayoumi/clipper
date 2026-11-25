# Public synthetic book-chapter rehearsal — model-free, unqualified

**Scope:** preparation only. The test harness is `tests/book_public_rehearsal.py`;
its companion tests are `tests/test_book_public_rehearsal.py`. There is **no
Modal launcher, client initialization, deploy, workflow trigger, model call,
private source acquisition, upload, publication, spending, or host selection**.

## Intentional execution boundary

From an independently pinned, **clean** source checkout with Git available,
Python 3.11+ and PyYAML 6.0.3, a local operator may explicitly run:

```bash
PYTHONPATH=src:tests python tests/book_public_rehearsal.py \
  --fresh-root /tmp/clipper-public-rehearsal-UNUSED \
  --expected-sha EXACT_REVIEWED_40_HEX_SHA
```

The `PYTHONPATH` shown assumes the command runs at the repository root.
Alternatively, install the same checked-out Clipper source locally without
adding its unrelated media/model extras, while retaining the `tests` module
path. The CLI checks Git HEAD **relative to its own source checkout**, not
to an arbitrary shell working directory. It rejects tracked modifications
and untracked files before starting the synthetic run.

The CLI requires `git rev-parse HEAD` to match the explicit full SHA before
creating any artifact. Direct Python `rehearse(..., expected_sha)` reports
that the caller's SHA still needs independent checkout verification. A fresh
**absolute, nonexistent** root is mandatory; only public synthetic fixtures
are created. All artifacts are kept local to that root, not published.
Use only a dedicated, unprivileged, isolated rehearsal process. This is
not authorization to invoke the command on Modal or spend credits.

The entrypoint is deliberately under `tests/` and imports minimal shared
helpers from `tests/public_book_fixture_support.py` — **not** from
`test_books_chapter_hybrid`, `test_books_chapter_host`, `test_books_whole_book`,
the CLI or the media pipeline. These helpers construct a public
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
A **future proposal only**, subject to separate account, cost and execution
approval, is one restricted Modal Function with `block_network=True`,
`restrict_modal_access=True`, `single_use_containers=True`,
`max_containers=1`, zero retries, 1 CPU, an initial candidate 512 MiB RAM
and **no GPU, secrets, attached volumes, model assets or private input**.
These Function options and the 512 MiB selection are proposed deployment
constraints, not tested provider configuration or a verified minimum.
The image must contain the **exact reviewed, clean Git checkout**, the
Git binary and the minimal local dependencies, so the existing CLI's
source-pin checks continue to work. There is no launcher or deployment
implementation in this repository change.

Execution timeout, image build duration, startup and idle billing, temporary
storage, cancellation, cleanup and failed attempts require a separately
enforced total resource/time/cost boundary. A short execution timeout alone
is **not** a hard spending cap. The currently observed Starter workspace
does not expose the Team/Enterprise environment-budget controls; the
approved workspace/account identity, credential route, available allowance,
total cost limit and independent control/cleanup method are unresolved.
Do not default to a dashboard-observed workspace or assume free credits.
Preserve the qualified offline `modal_infra_qualification.py` and
`modal_infra_driver.py`; their synthetic 64 KiB infrastructure qualification
is separate from book-chapter orchestration.

**Measured independently on exact code SHA
`6162ca65333f67445fa0bb8003f8198fe95de376`:** the clean-checkout,
fresh-process standalone CLI completed on minimal Python 3.11 with
**PyYAML 6.0.3 only**, plus the standard library, local Clipper source,
the lightweight fixture support and Git. It required no pytest, media
libraries, Torch, Transformers, Kokoro or Modal dependencies in that run.
Elapsed time **0.22127 seconds**; Linux process peak RSS **32,308 KiB**
(about 31.6 MiB). Forbidden module roots preloaded: **none**;
guarded prohibited-operation attempts: **none**. These are **local
observations**, not a reproducible bound for image build, a Modal SDK
container's memory overhead, remote latency, native network isolation,
private-book cardinality, or production pipeline performance. The 512 MiB
proposal includes an unmeasured margin, not a validated remote requirement.

The CLI produces a machine-checkable JSON report on synthetic success;
failed preflight prints a bounded blocked status. Its zero-call evidence
covers the enumerated Python-level guards and is not a universal
no-network/no-execution security proof.

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

**Independent verification reported for SHA `6162ca65333f67445fa0bb8003f8198fe95de376`:**
12 new rehearsal tests PASS; Ruff lint PASS; Ruff format PASS across 285
files; mypy PASS across 77 files; fresh standalone clean-checkout CLI PASS
with the measurements above. These checks were run externally by the
project owner, **not** by the connector-only assistant environment.
CI workflow `45592640560` has reported both-version Ruff/format/mypy
success, but its tokenizer-only probe was still running at the reporting
checkpoint. **Full CI is not yet PASS** and the unchanged 95% coverage gate
must be confirmed on the completed exact-SHA run. Independent read-only
final review has been requested and remains pending. The rehearsal harness
is therefore **UNQUALIFIED**; no live Modal run is authorized.


## REACHED1142 candidate correction (UNQUALIFIED)

The no-callback `brief_cpu=True` path fails **before** stage reservation
with `ChapterStop("hybrid injected-test tokenizer counter is required")`.
The harness asserts this exact production preflight outcome, the absence of
any `hybrid_argument` attempt directory, and no injected callbacks rather
than expecting the later `ChapterHostBlocked` path. The production code is
unchanged.

The CLI resolves its checkout from `__file__` (not the caller's CWD),
verifies the explicit Git HEAD, and requires a wholly clean tracked/untracked
worktree before reporting `VERIFIED_CLEAN_LOCAL_GIT_HEAD`. The direct
Python API cannot independently prove its claimed source SHA and retains an
explicit unverified source-pin label. An independent-process CLI regression
uses the genuine checkout and emits a timing/RSS artifact report; this test
requires a clean CI checkout.

Zero-call fields are explicitly **scoped to the guarded Python APIs**, and
the report inventories import roots, network hooks and model/source provider
hooks which the process-local guard blocks. They are not assertions about
all kernel/native APIs. This is not a network-namespace firewall and never
authorizes provider use or a remote run.


## ISOLATION1157: standalone test-support import graph

The public synthetic EPUB/recorded transport, selected-scope bundle,
argument/audit/reconcile response, creative response and PCM WAV fixture
helpers now live together in `tests/public_book_fixture_support.py`.
The rehearsal imports only this lightweight support module, never
`test_books_chapter_hybrid`, `test_books_chapter_host`,
`test_books_whole_book` or `clipper.cli`. Those tests reuse the same
shared helpers; the existing default source fixture section is unchanged.
A fresh subprocess checks that the support import itself does not preload
broad test, CLI, pipeline or forbidden model/download modules.

The real CLI requires no forbidden roots already loaded into
`sys.modules` before scenario execution. An in-process test invocation
may already have imported unrelated packages; its reported
`preloaded_forbidden_module_roots` makes this limitation observable
rather than claiming clean-process coverage. In either case a subsequent
forbidden import or guarded call fails closed. This remains a Python
application guard, not kernel network isolation.


## REVIEWFIX1227 source and dynamic-import evidence boundaries (UNQUALIFIED)

The CLI first checks exact clean Git HEAD and rejects already-loaded
`clipper` package/submodules and shared fixture modules unless their real
`__file__` paths resolve inside the script's **own** checkout. It then
prepends that checkout's `src` and `tests` to the import path before
scenario execution and rechecks loaded production/fixture module origins
after import. A foreign ambient `PYTHONPATH` or preloaded foreign
`clipper` module cannot be credited to the checked-out SHA. This is an
evidence-binding check for cooperative Python execution, not a code-signing
or adversarial Python interpreter sandbox.

The process-local offline guard now rejects `importlib.import_module()`
for the forbidden roots, including modules already present in
`sys.modules`. A temporary first-position `MetaPathFinder` also denies
new forbidden-module loading before a loader executes, and the guard fails
if new forbidden roots appear in the module registry before normal exit.
Patches and finder are restored on both success and exception. The scope
remains documented Python import/network/provider hooks only, **not**
arbitrary native C networking or OS-level network isolation.

These corrections are not included in the independently qualified
`6162ca6` executable checks; exact new-SHA tests, Ruff, full unchanged
95% CI and fresh review remain required. No Modal execution approval is
implied.

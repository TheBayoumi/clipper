# Dormant Modal CPU book-rehearsal route — NO EXECUTION AUTHORITY

**Status: DORMANT IMPLEMENTED / NO EXECUTION AUTHORITY.** The repository
contains a real opt-in launcher at `tests/book_modal_cpu_launcher.py` and
inert fake-only tests at `tests/test_book_modal_cpu_launcher.py`. The default
`plan` mode does not import Modal, read credentials or allocate resources.
`--execute` is a separate strictly gated route and is NOT authorized today.
There is no deployed App, Actions workflow, run request or approval. No account, credential, resource,
image build, workflow or provider has been accessed. The original qualified
infrastructure owner/driver and the Codespace private chapter host are
unchanged and are **not** interchangeable with this public synthetic rehearsal.

## Scope and exact source

Only the already-implemented public synthetic entrypoint
`tests/book_public_rehearsal.py` may run as an isolated child process.
It exercises the actual selected-source/hybrid chapter logic using inert
synthetic EPUB, JSON completions and PCM WAV; track A must STOP at
`AWAITING_HUMAN_CHAPTER_BEAT_REVIEW`. Track B uses only the explicit
`SYNTHETIC_TEST_REVIEW_NOT_HUMAN_APPROVAL` fixture, retains
`INJECTED_TEST` provenance, and stays `publish_eligible=False`.
No real chapter, private source, tokenizer/model, GPU, sound synthesis,
media pipeline, publication, external acquisition, or source upload.

An executable image must contain the **exact separately approved Git commit**
and a **verified clean source checkout** (including a functioning Git binary
and complete local .git evidence). The child requires
`PYTHONPATH=src:tests`, Python 3.11, PyYAML 6.0.3 and the checked-out
`clipper` and test-support sources. It must call the same CLI with
`--expected-sha` set to that approved full commit and a unique fresh,
absolute, ephemeral output root. Do not `pip install .`: the project-wide
dependencies include unrelated media/download packages. Do not install
torch, transformers, Modal, gdown or the media/model extras **inside the
child image**. A standalone local smoke completed without them on earlier
SHA `6162ca6`; this does not establish remote limits.

### Unresolved source-image boundary

**No image build has been authorized or performed.** The implemented
candidate builds from the **public unauthenticated** GitHub HTTPS source
`https://github.com/TheBayoumi/clipper.git`, checks out the explicit exact
SHA, validates Git HEAD and a clean worktree and installs Python 3.11,
Git and PyYAML 6.0.3. It does NOT copy the local Git tree or credentials,
and sets `GIT_TERMINAL_PROMPT=0` with Git credential helpers disabled.
Public cloning has been independently reported available; a live Modal
image build and its costs remain **untested**. A source archive without
the real clean Git checkout cannot satisfy the child CLI.

## SDK options to validate offline against pinned Modal 1.5.5

The published Modal `App.function()` reference lists these options.
They are a **configuration target**, not an instantiated object:

| Function option | Candidate value | Evidence/constraint |
| --- | --- | --- |
| `cpu` | `(1,1)` | One requested and hard-limited physical CPU |
| `memory` | `(512,512)` | MiB request and hard limit; remote overhead and OOM risk unmeasured |
| `gpu` | `None` | No GPU |
| `block_network` | `True` | Child network disabled |
| `restrict_modal_access` | `True` | Child not allowed to access other Modal resources |
| `single_use_containers` | `True` | One input per container |
| `max_containers` | `1` | No extra simultaneous containers |
| `min_containers` / `buffer_containers` | `0` | No warm pool |
| `retries` | `0` | No input retry |
| `startup_timeout` | `60` | Seconds, separate from execution timeout |
| `timeout` | `30` | Seconds, one execution; not a hard total-spend cap |
| `secrets` / `volumes` / `network_file_systems` | Empty | No child credentials, persistent mounts or private input |
| `include_source` | `False` for App and Function | No ambient host-source mount; the Function module is resolved from the public pinned image clone (unverified live) |
| `scaledown_window` | `2` | Candidate short container scale-down, not an end-to-end cost cap |

Source references:
- https://modal.com/docs/sdk/py/latest/App
- https://modal.com/docs/sdk/py/releases (1.5.5 release listed)
- https://modal.com/docs/guide/restricted-access
- https://modal.com/docs/sdk/py/latest/FunctionCall

The public reference is currently **latest**, not an independently
executed pinned-1.5.5 signature verification. Confirm `modal==1.5.5`
API behavior offline and exact image-build semantics before any
live run. `block_network` applies inside the remote Function; it does
**not** remove network, authentication or billing from the image-build and
local control-plane processes. This is an application contract, not proof
of a universal OS security boundary.

## Control plane and account approval — STOP BEFORE ANY BUILD

A possible **future** host is the *existing primary repository's*
GitHub Actions identity, after independently proving which profile/workspace
the already-authorized credentials represent. **Do not select the
dashboard-observed workspace automatically.** The user has not approved a
workspace/account/client credential route or a cost limit. Do not create
new tokens or accounts, nor reuse production campaign/model deployment
workflows.

The mandatory preflight, before constructing a Modal App or Image, must
fail closed unless all hold:

1. Expected exact full source SHA; own checkout is clean and the source
   fixture/package imports resolve only from that checkout.
2. A **separate owner-approved** immutable run authorization exists,
   explicitly identifying run intent, expected workspace, environment,
   credential/profile route, amount and total cost ceiling, and validity
   window. No such approval artifact exists.
3. Read-only authenticated workspace identity is obtained from the
   *actual selected control-plane identity*, and exactly matches that
   authorization. The profile, token precedence, account and environment
   cannot be guessed from shell defaults or past dashboards.
4. Available credit/billing state, build+startup+execution+idle/storage
   cost exposure and the independent hard cap are verified and accepted
   **before** any image build, App creation, Function invocation or
   resource reservation. A `timeout` does not enforce a monetary cap.
5. A separate owner-controlled control-plane termination/observation
   procedure is established, with exact App/call identifiers, cancellation
   escalation and post-run absence verification. On failed confirmation,
   disposition is UNKNOWN/BLOCKED, not SUCCESS.

Modal documents `FunctionCall.cancel(terminate_containers=True)` and
ephemeral `App.run()`; **neither is, by itself, independently verified
deletion of every resource or hard spending containment**. A bounded local one-shot control-plane process and exact-ID stop/info
attempts are now implemented; independently enforceable monetary budget
containment and provider-confirmed deletion are not established. In particular, no
unapproved new daemon, supervisor or host architecture is introduced.

## Execution/cleanup boundaries if a later request is authorized

Proposed one-shot sequence: exact-SHA+manifest+identity+cost preflight
→ independently controlled image build and source attestation
→ one ephemeral restricted Function invocation (no deployment)
→ one sanitized JSON result and artifact-hash comparison
→ stop/cancel any retained invocation and observe final status
→ verify no retained App/Function/container allocations
→ record exact cost and source/provenance evidence.
Every transition is fail-closed. App startup/image build, control-plane
startup waits and cleanup are *outside* the 30-second Function execution
limit. No retry and no retained paid idle container may be assumed
without observing the provider state.

**No Actions workflow** is added. Neither push/PR/CI nor a manual dispatch
can activate the launcher automatically. A future operator would have
to run the opt-in command deliberately and provide an out-of-band
SHA-256-pinned manifest and independent-control receipt.

## What remains blocked

- Verified and approved source image construction without exposing
  repository credentials or private content.
- Exact Modal 1.5.5 offline SDK signature and build dependency verification.
- Approved workspace/account/profile/environment and authenticated
  read-only identity binding.
- Owner-approved complete cost allowance, including image build, failed
  startup, idle time, cleanup, and any storage.
- Independent lifecycle control with provider-confirmed stop/deletion.
- Explicit separate permission to allocate any Modal resource or spend.
- Current test-only deterministic second-create regression, exact-SHA CI
  and final review are independent of all of the above.

The executable launcher is **implemented but not qualified or executed**;
its default mode is a pure local plan. Its `--execute` mode requires an
external approval document, an independently supplied SHA-256 digest,
a matching control receipt, nonexpired narrow permission, exact source
and clean worktree, an explicit nonambiguous credential route, and
a client-bound workspace identity match **before** App/Image construction.
The manifest digest and receipt are checks for accidental disagreement,
NOT cryptographic proof of the approving person's identity, and SDK
resource cleanup is NOT an independent cost cap. In particular, no
approval manifest or control receipt is present in this repository.
Do not actually invoke `--execute`, construct clients, or allocate
resources without the separately approved control-plane lifecycle and
cost containment plan. The implementation remains **UNQUALIFIED**
until exact-SHA offline tests, Ruff/mypy, full unchanged 95% CI and review.


## Dormant launcher command and evidence

```bash
# From exact clean checkout. No Modal import, client, token or resource.
PYTHONPATH=src:tests python tests/book_modal_cpu_launcher.py \
  --expected-sha EXACT_40_HEX_REVIEWED_SHA
```

Output: `DORMANT_PLAN_ONLY` with fixed Function candidate limits.
There is **no default run**, no checked-in manifest and no workflow.
The launcher separately implements `--execute` with
`--approval-file`, `--approval-sha256`,
`--independent-control-receipt` and exact `--expected-sha`;
these switches are not a request to execute and are not authorized.
It imports the SDK only after local verification, then verifies the
actual selected client's workspace before constructing an App/Image.

The published SDK 1.5.5 entrypoints targeted are `App.function`,
`App.run(client=..., detach=False, environment_name=...)`,
`Workspace.from_context(client=...)` / `.hydrate()` / `.name`,
`Function.spawn` / `FunctionCall.get(timeout=...)` /
`FunctionCall.cancel(terminate_containers=True)`. The code requires
`modal==1.5.5` by installed package metadata. Local service access,
image initialization, `App.run` cleanup, provider removal and 512 MiB
headroom remain untested. The remote child uses `subprocess.run` with
an environment containing only PATH, HOME, PYTHONDONTWRITEBYTECODE and
the pinned source's PYTHONPATH; Modal tokens are not inherited by that
child process. The Function launcher source itself is included for
serialization; no project/media source is automatically mounted.

The bounded control-plane wrapper returns `provider_removal=UNKNOWN`
when exact-ID stop and terminal provider status are not confirmed. App
context exit and FunctionCall cancellation do not prove provider deletion.
The live CLI remains blocked before SDK import by the unresolved budget gate.
A separately staffed/independent exact-ID control-plane stop and cost
reconciliation are still operational prerequisites to **any live run**.


## LAUNCHER-FIX1314: bounded implementation, independent budget blocker

The launcher now **implements a bounded one-shot process**, not a receipt-string
substitute for lifecycle control. Once independently authorized (not currently),
the operator process creates a **separate control-plane worker process** and
monitors its entire image/App/build/startup/spawn/get/context-exit execution
within a finite absolute local deadline (candidate 120 seconds). It writes
`BUILD_BEGIN`, `APP_RUN_BEGIN`, exact `APP_ID`, `SPAWN_BEGIN`,
`CALL_ID`, and `GET_BEGIN` phases to a fsynced local journal before
subsequent blocking operations whenever those IDs are actually known. If
the build stalls before an App ID exists, the result remains
`APP_ID_UNKNOWN` and `BLOCKED`; it never invents a provider resource ID.

At the deadline the parent kills the local worker process group, and when
an exact App ID is known attempts the documented, explicit-environment
`modal app stop -y --env ENV APP_ID` and
`modal app info --json --env ENV APP_ID`, each with finite deadlines.
Only both successful calls plus unambiguous terminal `state=stopped`
are accepted as local stop confirmation. Missing ID, worker failure,
malformed journal, cancel failure, stop error, unverifiable info schema
or a timeout yield `status=BLOCKED`, with exact IDs and cleanup errors
retained, and `provider_removal=UNKNOWN` if the stop observation fails.
This is **not** universal provider deletion evidence. The CLI's default
mode still has no SDK import or resource action. The bounded implementation
is exercised only with fake controls in offline tests; no live run occurred.

Function resource requests use **hard CPU and memory limit pairs**,
`cpu=(1,1)`, `memory=(512,512)`, not merely requests that could burst
beyond the reservation. The 512 MiB hard limit is a candidate; SDK
runtime overhead has not been remotely verified. The Function has
`include_source=False` with the module supplied by the anonymous pinned
public clone in the image and a fixed `PYTHONPATH`; no automatic host
test/module folder mount is intended. Modal serialization/import behavior
on a live 1.5.5 Function remains unverified. The child uses a fresh
temporary root, passes a **nonexistent** artifact subdirectory to the
accepted rehearsal CLI, returns selected provenance/artifact evidence and
discards temporary files on exit.

**The unresolved cost gate is explicitly closed:** the current Starter
workspace lacks independently enforceable account/environment cost budget
authority verified for this route. A number in the manifest and a matching
receipt string are **not** such authority. Therefore `--execute` explicitly
fails before Modal client construction, even with matching approval strings.
The bounded worker implementation is available for isolated fake tests,
but cannot run live until a separately verified enforceable budget control
and a review-approved gate are implemented. No billed operation is
authorized.

Reference: https://modal.com/docs/guide/resources (tuple resource limits),
https://modal.com/docs/cli/latest/app (explicit-environment app stop/info).
These currently published docs are not substitute for offline pinned
Modal 1.5.5 compatibility tests. No GitHub workflow or launcher request
file is introduced.


**Additional direct-worker containment:** `--internal-worker` is **not**
an alternate permitted entrypoint. It is explicitly blocked even if a
caller supplies forged manifest/receipt/file arguments, until the same
independently enforceable spend authority is implemented and reviewed.
The private worker's actual SDK logic remains available for isolated
fake-unit verification, but **no CLI path currently permits resource
allocation**.


## LAUNCHER-CORRECT1326: offline lifecycle evidence correction

Cancellation now occurs **within** the live `app.run()` context
before its potentially blocking exit. Exact App/call identities and
phase transitions are appended to an fsynced journal before later
blocking operations when the SDK has actually returned those IDs.
App creation may begin during context entry before any App ID can be
observed. In that case `APP_RUN_BEGIN` and the worker failure/deadline
remain durable but provider identity/removal is **UNKNOWN**, never
guessed or reported successful.

The bounded parent **accumulates** earlier worker/journal/cancellation,
timeout and cleanup errors rather than replacing them with the
result of a later stop attempt. Worker stdout/stderr are captured in
temporary local files and represented in the result only by bounded
4 KiB-prefix SHA-256 hashes, total byte counts and truncation flags,
not raw log strings or credential-bearing provider output. These
temporary files are destroyed after producing the local report.

The independently inspected **Modal 1.5.5** CLI source has `modal app
stop` and `modal app list` but **no `modal app info`**, so the
unsupported info command was removed. The proposed conservative
terminal probe issues:

```text
modal app stop -y --env ENV EXACT_APP_ID
modal app list --json --limit 0 --env ENV
```

It only accepts a single row matching the exact App ID with
`state="stopped"`, `tasks="0"`, both CLI commands succeeding and
no earlier errors. Absent rows, mismatched IDs, duplicates,
malformed JSON, nonzero tasks, unsupported state fields and
stop/observation errors are **UNKNOWN/BLOCKED**. The JSON list
schema was cross-checked against public Modal client source for the
same command; **the precise pinned-1.5.5 JSON output must still
be independently verified** before any live use. Never treat
missing rows as deletion. The result label is
`STOPPED_CONFIRMED_BY_LIST`, meaning only this narrow CLI
observation—not absolute resource deletion or cost closure.

The accepted synthetic rehearsal and production infrastructure owner
remain unchanged. Both `--execute` and `--internal-worker` remain
explicitly disabled by the unresolved spending-authority gate. This
is an offline engineering implementation candidate only; no
provider, credentials, App, image, resource or spend action occurred.


## LAUNCHER-FINAL1333 pinned CLI/static correction

The pinned Modal 1.5.5 `modal/cli/app.py` registers `app list`
with `--env` and `--json`, but **not** `--limit`.
The terminal observation now uses the exact supported command
`modal app list --json --env ENV`; the stop command remains
`modal app stop -y --env ENV EXACT_APP_ID`. The offline
regression asserts both complete argument arrays rather than
accepting extra invented options. The JSON field names
`app_id`, `state` and `tasks` are grounded in the pinned
CLI's output formatting. An absent, duplicated or nonterminal
exact-ID row remains `UNKNOWN/BLOCKED`.

The same bounded correction removes an unused child import,
orders the SDK-version helper before `modal`, fixes the
test import ordering and binds the parameterized fake's
`rows` locally. No accepted chapter rehearsal or owner
production code changed, and no live path is enabled.

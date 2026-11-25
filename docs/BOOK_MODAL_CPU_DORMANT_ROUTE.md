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
deletion of every resource or hard spending containment**. A bounded local one-shot control-plane process and exact-ID list-first
stop/observation attempts are implemented; independently enforceable monetary budget
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


## Current dormant lifecycle controls

A bounded one-shot worker owns SDK workspace identity lookup, image setup,
App context entry/exit, Function spawn/get and cancellation. The parent owns
a finite process deadline and collects fsynced phase/App/call-ID evidence.
The actual workspace lookup executes **inside the bounded worker, before
any image or App creation**. Immediately after a successful workspace
match, the worker independently rechecks the UTC approval expiry
**before** entering any resource-constructing operation. An identity
lookup that consumes the remaining approval window fails closed. The owner-approved workspace, environment
and credential route remain unresolved; matching strings are not authority.

Before beginning allocation, the worker records `WORKSPACE_VERIFY_BEGIN`
and then `BUILD_BEGIN`. `APP_RUN_BEGIN` precedes potentially allocating
App context entry. A context entry can create an App before its ID is
returned; missing identity therefore means **UNKNOWN**, never "no App".
The only actionable App identifier is `ap-` followed by **exactly
22 ASCII alphanumeric characters**, verified on SDK capture, journal
recovery and before any provider stop. Anything else, including short
`ap-a` or deployment names, cannot reach the stop command.

The control-plane parent retains KeyboardInterrupt as an interrupted
outcome. It still attempts finite process cleanup and exact-ID provider
reconciliation, then raises `LaunchInterrupted` with the blocked
evidence attached. Journal parsing treats non-object JSON records such
as `[]`, `null` or `123` as `MALFORMED_RECORD`, retains any previously
validated exact App ID, and continues bounded cleanup without mistaking
the malformed evidence for a safe outcome. A timeout or other uncertain cleanup likewise remains
`BLOCKED`; the result is not represented as a successful deletion.
Worker cancellation occurs within `app.run()` before context exit, and
earlier cleanup, journal, worker and cancellation failures accumulate.
Worker stdout/stderr are represented by bounded prefix hashes and byte
counts, not exposed as raw logs.

Restricted Function parameters remain `cpu=(1,1)`,
`memory=(512,512)`, `include_source=False`, no model/GPU/secrets/
volumes, no network, zero retries and a pinned anonymous public checkout.
The 512 MiB envelope is unverified for a remote Modal SDK runtime.
`--execute` and `--internal-worker` are **both disabled** while
account/cost authority is resolved. The manifest's dollar number and
matching receipt are not enforceable spending controls. No workflow,
build, client or provider run is authorized.

## Exact provider terminal observation — pinned Modal 1.5.5

The supported CLI is `modal app list --json --env ENV` and
`modal app stop -y --env ENV EXACT_APP_ID`. Modal 1.5.5 has no
`app info` command and no `app list --limit` flag.

**List first** using the explicitly selected environment. Exactly one
row with the full App ID, `state="stopped"`, and `tasks="0"` means
the App is already terminal; **do not issue stop** (which exits nonzero
for already-stopped Apps in this SDK). Otherwise, including when the
initial list fails, is malformed, or omits the exact App ID, retain
the initial uncertainty but attempt a bounded stop **only against the
already validated exact App ID**, followed by another bounded list.
Cleanup can be reported confirmed only with positive post-stop terminal
evidence and **no retained initial or stop error**. Missing, duplicate,
malformed or ambiguous rows never authorize guessing a deployment
name, and all such failures remain `UNKNOWN/BLOCKED`.
A nonzero stop cannot be turned into clean success merely because the
later list shows stopped. This is narrowly labeled
`STOPPED_CONFIRMED_BY_LIST`, not verified universal provider deletion.

The pinned CLI's JSON `app_id`, `state` and `tasks` schema was
independently verified in `modal/cli/utils.py`. Inert tests check
command arguments, valid/invalid exact App IDs, skipped redundant stop,
nonterminal stop/recheck, interrupts, and stage ordering.

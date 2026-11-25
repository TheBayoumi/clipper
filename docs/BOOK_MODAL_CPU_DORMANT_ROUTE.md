# Dormant Modal CPU book-rehearsal route — NO EXECUTION AUTHORITY

**Status: BLOCKED / PREPARATION-ONLY.** This is a future integration
contract, NOT a runnable launcher, Modal client, deployment, App definition,
Actions workflow, run request, or approval. No account, credential, resource,
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

**No image build is authorized.** Before a build recipe is committed,
an independent reviewer must establish a credential-safe and provenance-safe
way to put the complete clean checkout in the image. If the repository
requires authentication, **do not embed a GitHub token, .git/config
credential, host key, private source, artifact, or CI secret in any image
layer** or copy a credential-bearing `.git` tree. An unauthenticated public
Git checkout at the pinned commit would be preferable if actually
accessible; public availability has **not** been established here. A
source archive without a real clean Git checkout cannot satisfy the
existing CLI pin preflight, and fabricating a Git HEAD is prohibited.

## SDK options to validate offline against pinned Modal 1.5.5

The published Modal `App.function()` reference lists these options.
They are a **configuration target**, not an instantiated object:

| Function option | Candidate value | Evidence/constraint |
| --- | --- | --- |
| `cpu` | `1` | One CPU-only function |
| `memory` | `512` | MiB request candidate, not a proven peak/limit; verify hard-limit semantics |
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
| `include_source` | `False` | No automatic extra host source copying; image proof must be explicit |

Source references:
- https://modal.com/docs/sdk/py/latest/App
- https://modal.com/docs/sdk/py/releases (1.5.5 release listed)
- https://modal.com/docs/guide/restricted-access
- https://modal.com/docs/sdk/py/latest/FunctionCall

The public reference is currently **latest**, not an independently
executed pinned-1.5.5 signature verification. Confirm `modal==1.5.5`
API behavior offline and exact image-build semantics before writing a
launcher. `block_network` applies inside the remote Function; it does
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
deletion of every resource or hard spending containment**. A bounded
control-plane supervisor and post-run reconciliation are therefore
operational prerequisites, not implemented here. In particular, no
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

**No Actions workflow** has been added: no `push`, `pull_request`,
`schedule`, `workflow_run`, `workflow_dispatch` or run-request file
is introduced by this preparation. This prevents incidental execution
during routine CI or PR activity. An eventual manual workflow still
requires an independently reviewed authorization and identity gate, not
just a `workflow_dispatch` boolean.

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

Until these gates are met, **do not implement an executable launcher or
workflow, instantiate a Modal client/App/Image, download/build an image,
or invoke any host**. This document makes the intended route reviewable
without changing operational authority.

# Detached private book phase (Volume transport, optional L4)

The private brief and script phases can run detached from the client. The client starts the
phase, exits, and later short `poll` invocations collect the result.

Commands (`python -m clipper.books.private_modal_detached`):

- `start`: verifies the reviewed manifest digest and the pinned checkout, creates a dedicated
  private Modal Volume named `clipper-private-<first 40 hex of the manifest digest>`, uploads
  the packet (and a prior checkpoint if any), then runs the phase with `app.run(detach=True)`
  and `spawn()`. The exact App ID, FunctionCall ID and Volume name go to a private
  `ATTACH.json` beside the journal. The client then exits and the App keeps running.
- `poll`: re-authenticates with the same workspace guard, re-attaches by FunctionCall ID with
  `get(timeout=0)` and returns `DETACHED_PENDING` while the phase runs. When the result is
  present it reads `checkpoint.bin` from the Volume, runs the same identity, size and digest
  checks as the synchronous launcher, persists the checkpoint privately, stops the exact App,
  deletes the Volume and verifies the deletion by exact-name inventory. A phase that passes
  its manifest timeout plus a 15 minute margin is reported as blocked and cleaned up. If
  cleanup is not verified the journal stays open so a later poll retries it.

Container: network blocked, no secrets, `retries=0`, one container, the run's own Volume
mounted at `/vol`. Only the small result envelope is returned. The checkpoint is written to
the Volume with a temporary file and rename. An explicit `Volume.commit()` is best effort:
in a restricted container it raises an authentication error, and a data-free probe showed
the written file was still persisted when the container exited and was readable by the client.

Accelerator: the manifest carries a required `accelerator` (`none` or `L4`) that must agree
with the owner approval marker (`EXPLICIT_PRIVATE_CPU_PHASE` with `none`,
`EXPLICIT_PRIVATE_L4_PHASE` with `L4`). The marker records the owner's explicit approval of
the resource class. It is not a cost limit. With `L4` the image installs the CUDA build of
torch, the Function requests one L4, and `CLIPPER_QWEN_DEVICE=cuda` makes the pinned Qwen
model load on the GPU (failing closed if CUDA is unavailable). Model, revision and prompts
are unchanged. There is no code-level cost cap; the manifest timeout is the runtime bound.

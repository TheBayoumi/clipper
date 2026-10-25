"""Small, explicit A100 access qualification; no models, weights or video renders.

Used before expensive production to prove the two backup workspaces can
actually allocate an 80 GB A100. The probe itself allocates one short GPU
container per account and may incur a small usage charge.
"""

from __future__ import annotations

import subprocess

import modal

app = modal.App("clipper-a100-access-qualification")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, gpu="A100-80GB", timeout=120)
def check_a100() -> str:
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    device = probe.stdout.strip()
    if "A100" not in device or not any(tag in device for tag in ("80GB", "80 GiB", "81920 MiB")):
        # This failure must not be interpreted as a production quota event.
        raise RuntimeError("Allocated device is not the requested A100-80GB")
    return "A100-80GB_AVAILABLE"


@app.local_entrypoint()
def main() -> None:
    assert check_a100.remote() == "A100-80GB_AVAILABLE"
    print("MODAL_A100_GPU_ACCESS_VERIFIED")

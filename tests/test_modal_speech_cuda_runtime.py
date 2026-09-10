from pathlib import Path


def test_speech_image_has_cuda12_runtime_and_smoke() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04" in source
    assert "def speech_cuda_smoke()" in source
    assert 'ctypes.CDLL("libcublas.so.12")' in source
    assert 'ctypes.CDLL("libcudnn.so.9")' in source

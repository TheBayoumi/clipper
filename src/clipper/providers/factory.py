from __future__ import annotations

import os

from .base import (
    EditorialProvider,
    EmbeddingProvider,
    ModelIdentity,
    VisionProvider,
    compute_profile,
)
from .local import LocalEditorialProvider, LocalEmbeddingProvider, LocalVisionProvider
from .modal import ModalEditorialProvider, ModalEmbeddingProvider, ModalVisionProvider


def editorial_and_embedding_providers(
    profile_name: str,
) -> tuple[EditorialProvider, EmbeddingProvider]:
    profile = compute_profile(profile_name)  # type: ignore[arg-type]
    if profile.editorial_location == "local":
        return LocalEditorialProvider(), LocalEmbeddingProvider()
    app = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
    editorial = ModalEditorialProvider(
        app_name=app,
        function_name="editorial",
        identity=ModelIdentity(
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            os.getenv("CLIPPER_EDITORIAL_MODEL_REVISION", "main"),
            os.getenv("CLIPPER_EDITORIAL_QUANTIZATION", "none"),
            "modal-transformers",
            "editor-v1",
            "editorial-json-v1",
        ),
    )
    embedding = ModalEmbeddingProvider(
        app_name=app,
        function_name="embedding",
        identity=ModelIdentity(
            "Qwen/Qwen3-Embedding-0.6B",
            os.getenv("CLIPPER_EMBEDDING_MODEL_REVISION", "main"),
            "none",
            "modal-sentence-transformers",
            "none",
            "embedding-v1",
        ),
    )
    return editorial, embedding


def vision_provider(profile_name: str, *, large: bool = False) -> VisionProvider:
    profile = compute_profile(profile_name)  # type: ignore[arg-type]
    if large and not profile.allow_large_vlm_escalation:
        raise ValueError("large VLM escalation is disabled for this compute profile")
    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct" if large else "Qwen/Qwen3-VL-8B-Instruct"
    if profile.vision_location == "local":
        if large:
            raise ValueError("large VLM is not enabled for local-lite")
        return LocalVisionProvider(model_id=model_id)
    return ModalVisionProvider(
        app_name=os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor"),
        function_name="vision_large" if large else "vision",
        identity=ModelIdentity(
            model_id,
            os.getenv("CLIPPER_VISION_MODEL_REVISION", "main"),
            os.getenv("CLIPPER_VISION_QUANTIZATION", "none"),
            "modal-transformers",
            "vision-v1",
            "vision-json-v1",
        ),
    )

from __future__ import annotations

import os

from ..canonical import CanonicalTimeline, CanonicalWord
from ..stage_contracts import structural_contract_fingerprint
from .base import (
    AlignmentProvider,
    DiarizationProvider,
    EditorialProvider,
    ModelIdentity,
    TranscriptionProvider,
    VisionProvider,
    compute_profile,
)
from .editorial_model_contract import (
    EDITORIAL_INFERENCE_ENGINE,
    EDITORIAL_MODEL_ID,
    EDITORIAL_MODEL_REVISION,
    EDITORIAL_PROMPT_VERSION,
    EDITORIAL_QUANTIZATION,
    EDITORIAL_SCHEMA_VERSION,
)
from .local import LocalEditorialProvider, LocalVisionProvider
from .modal import ModalEditorialProvider, ModalVisionProvider
from .modal_endpoint import ModalEndpointEditorialProvider
from .modal_speech import (
    ModalAlignmentProvider,
    ModalDiarizationProvider,
    ModalMediaBridge,
    ModalTranscriptionProvider,
)
from .speech import (
    FasterWhisperTranscriptionProvider,
    PassthroughDiarizationProvider,
    PyannoteDiarizationProvider,
    WhisperXAlignmentProvider,
)
from .speech_contract import (
    ALIGNMENT_INFERENCE_ENGINE,
    ALIGNMENT_MODEL_ID,
    ALIGNMENT_MODEL_REVISION,
    ALIGNMENT_QUANTIZATION,
    ASR_COMPUTE_TYPE,
    ASR_INFERENCE_ENGINE,
    ASR_MODEL_ID,
    ASR_MODEL_REVISION,
    DIARIZATION_INFERENCE_ENGINE,
    DIARIZATION_MODEL_ID,
    DIARIZATION_MODEL_REVISION,
    DIARIZATION_QUANTIZATION,
)
from .vision_contract import (
    VISION_INFERENCE_ENGINE,
    VISION_LARGE_MODEL_ID,
    VISION_LARGE_MODEL_REVISION,
    VISION_LARGE_QUANTIZATION,
    VISION_MODEL_ID,
    VISION_MODEL_REVISION,
    VISION_QUANTIZATION,
)


def _canonical_contract() -> str:
    return structural_contract_fingerprint(
        "canonical-timeline",
        CanonicalWord,
        CanonicalTimeline,
        exclude_fields=("contract_fingerprint",),
    )


def editorial_provider(profile_name: str) -> EditorialProvider:
    """Build the autonomous production editor without an embedding sidecar."""
    profile = compute_profile(profile_name)  # type: ignore[arg-type]
    if profile.editorial_location == "local":
        return LocalEditorialProvider()
    app = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
    backend = os.getenv("CLIPPER_MODAL_EDITORIAL_BACKEND", "function").strip().lower()
    if backend in {"function", "self-hosted", "modal-function"}:
        return ModalEditorialProvider(
            app_name=app,
            class_name="EditorialModel",
            method_name="complete",
            identity=ModelIdentity(
                EDITORIAL_MODEL_ID,
                EDITORIAL_MODEL_REVISION,
                EDITORIAL_QUANTIZATION,
                EDITORIAL_INFERENCE_ENGINE,
                EDITORIAL_PROMPT_VERSION,
                EDITORIAL_SCHEMA_VERSION,
            ),
        )
    if backend in {"managed", "endpoint"}:
        default_model = "Qwen/Qwen3.6-27B-FP8" if profile.name == "quality" else "Qwen/Qwen3.5-4B"
        return ModalEndpointEditorialProvider(
            endpoint_url=os.getenv("CLIPPER_MODAL_EDITORIAL_ENDPOINT_URL", ""),
            proxy_token_id=os.getenv("MODAL_PROXY_TOKEN_ID", ""),
            proxy_token_secret=os.getenv("MODAL_PROXY_TOKEN_SECRET", ""),
            identity=ModelIdentity(
                os.getenv("CLIPPER_EDITORIAL_MODEL_ID", default_model),
                os.getenv("CLIPPER_EDITORIAL_MODEL_REVISION", "").strip(),
                "modal-managed",
                "modal-managed-endpoint",
                EDITORIAL_PROMPT_VERSION,
                EDITORIAL_SCHEMA_VERSION,
            ),
        )
    raise ValueError(f"unsupported Modal editorial backend: {backend}")


def vision_provider(profile_name: str, *, large: bool = False) -> VisionProvider:
    profile = compute_profile(profile_name)  # type: ignore[arg-type]
    if large and not profile.allow_large_vlm_escalation:
        raise ValueError("large VLM escalation is disabled for this compute profile")
    model_id = VISION_LARGE_MODEL_ID if large else VISION_MODEL_ID
    if profile.vision_location == "local":
        if large:
            raise ValueError("large VLM is not enabled for local-lite")
        return LocalVisionProvider(
            model_id=model_id,
            revision=VISION_MODEL_REVISION,
            quantization=VISION_QUANTIZATION,
        )
    revision = VISION_LARGE_MODEL_REVISION if large else VISION_MODEL_REVISION
    quantization = VISION_LARGE_QUANTIZATION if large else VISION_QUANTIZATION
    return ModalVisionProvider(
        app_name=os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor"),
        class_name="VisionModelLarge" if large else "VisionModel",
        method_name="inspect",
        identity=ModelIdentity(
            model_id,
            revision,
            quantization,
            VISION_INFERENCE_ENGINE,
            "vision",
            "structured-json",
        ),
    )


def speech_providers(
    profile_name: str,
) -> tuple[TranscriptionProvider, AlignmentProvider, DiarizationProvider]:
    profile = compute_profile(profile_name)  # type: ignore[arg-type]
    diarization_mode = os.getenv("CLIPPER_DIARIZATION_MODE", "pyannote").strip().lower()
    degraded_diarization = diarization_mode in {"passthrough", "none", "disabled"}
    if not degraded_diarization and diarization_mode != "pyannote":
        raise ValueError(f"unsupported diarization mode: {diarization_mode}")
    canonical_contract = _canonical_contract()
    if profile.editorial_location == "local":
        return (
            FasterWhisperTranscriptionProvider(
                model_id=ASR_MODEL_ID,
                revision=ASR_MODEL_REVISION,
                compute_type=ASR_COMPUTE_TYPE,
                schema_version=canonical_contract,
            ),
            WhisperXAlignmentProvider(
                model_id=ALIGNMENT_MODEL_ID,
                revision=ALIGNMENT_MODEL_REVISION,
                schema_version=canonical_contract,
            ),
            PassthroughDiarizationProvider()
            if degraded_diarization
            else PyannoteDiarizationProvider(
                model_id=DIARIZATION_MODEL_ID,
                revision=DIARIZATION_MODEL_REVISION,
                schema_version=canonical_contract,
            ),
        )
    app = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
    bridge = ModalMediaBridge(os.getenv("CLIPPER_MODAL_MEDIA_VOLUME", "clipper-media-cache"))
    return (
        ModalTranscriptionProvider(
            app_name=app,
            function_name="transcribe",
            identity=ModelIdentity(
                ASR_MODEL_ID,
                ASR_MODEL_REVISION,
                ASR_COMPUTE_TYPE,
                ASR_INFERENCE_ENGINE,
                "none",
                canonical_contract,
            ),
            media_bridge=bridge,
        ),
        ModalAlignmentProvider(
            app_name=app,
            function_name="align",
            identity=ModelIdentity(
                ALIGNMENT_MODEL_ID,
                ALIGNMENT_MODEL_REVISION,
                ALIGNMENT_QUANTIZATION,
                ALIGNMENT_INFERENCE_ENGINE,
                "none",
                canonical_contract,
            ),
            media_bridge=bridge,
        ),
        PassthroughDiarizationProvider()
        if degraded_diarization
        else ModalDiarizationProvider(
            app_name=app,
            function_name="diarize",
            identity=ModelIdentity(
                DIARIZATION_MODEL_ID,
                DIARIZATION_MODEL_REVISION,
                DIARIZATION_QUANTIZATION,
                DIARIZATION_INFERENCE_ENGINE,
                "none",
                canonical_contract,
            ),
            media_bridge=bridge,
        ),
    )

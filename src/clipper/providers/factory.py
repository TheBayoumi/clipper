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
from .editorial_prompt import EDITORIAL_IDENTITY, EDITORIAL_SCHEMA_IDENTITY
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
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                os.getenv(
                    "CLIPPER_EDITORIAL_MODEL_REVISION",
                    "110954009be4a882781a90356c7d2b8a9e3428dc",
                ),
                os.getenv("CLIPPER_EDITORIAL_QUANTIZATION", "bnb-4bit-nf4"),
                "modal-transformers",
                EDITORIAL_IDENTITY,
                EDITORIAL_SCHEMA_IDENTITY,
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
                os.getenv("CLIPPER_EDITORIAL_MODEL_REVISION", "modal-managed"),
                "modal-managed",
                "modal-managed-endpoint",
                EDITORIAL_IDENTITY,
                EDITORIAL_SCHEMA_IDENTITY,
            ),
        )
    raise ValueError(f"unsupported Modal editorial backend: {backend}")


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
        class_name="VisionModel",
        method_name="inspect",
        class_parameters={"model_id": model_id},
        identity=ModelIdentity(
            model_id,
            os.getenv("CLIPPER_VISION_MODEL_REVISION", "main"),
            os.getenv("CLIPPER_VISION_QUANTIZATION", "none"),
            "modal-transformers",
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
    if profile.editorial_location == "local":
        return (
            FasterWhisperTranscriptionProvider(),
            WhisperXAlignmentProvider(),
            PassthroughDiarizationProvider()
            if degraded_diarization
            else PyannoteDiarizationProvider(),
        )
    app = os.getenv("CLIPPER_MODAL_APP", "clipper-open-editor")
    bridge = ModalMediaBridge(os.getenv("CLIPPER_MODAL_MEDIA_VOLUME", "clipper-media-cache"))
    canonical_contract = _canonical_contract()
    return (
        ModalTranscriptionProvider(
            app_name=app,
            function_name="transcribe",
            identity=ModelIdentity(
                "faster-whisper/large-v3-turbo",
                os.getenv("CLIPPER_ASR_MODEL_REVISION", "main"),
                "int8_float16",
                "modal-faster-whisper",
                "none",
                canonical_contract,
            ),
            media_bridge=bridge,
        ),
        ModalAlignmentProvider(
            app_name=app,
            function_name="align",
            identity=ModelIdentity(
                "whisperx-forced-alignment",
                os.getenv("CLIPPER_ALIGNMENT_MODEL_REVISION", "main"),
                "none",
                "modal-whisperx",
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
                "pyannote/speaker-diarization-community-1",
                os.getenv("CLIPPER_DIARIZATION_MODEL_REVISION", "main"),
                "none",
                "modal-pyannote",
                "none",
                canonical_contract,
            ),
            media_bridge=bridge,
        ),
    )

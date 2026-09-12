from __future__ import annotations

ASR_MODEL_ID = "dropbox-dash/faster-whisper-large-v3-turbo"
ASR_MODEL_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
ASR_COMPUTE_TYPE = "float16"
ASR_INFERENCE_ENGINE = "modal-faster-whisper"
ASR_TRANSCRIPT_SOURCE = "modal-faster-whisper-large-v3-turbo"

ALIGNMENT_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
ALIGNMENT_MODEL_REVISION = "569a6236e92bd5f7652a0420bfe9bb94c5664080"
ALIGNMENT_QUANTIZATION = "none"
ALIGNMENT_INFERENCE_ENGINE = "modal-whisperx"

DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
DIARIZATION_MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
DIARIZATION_QUANTIZATION = "none"
DIARIZATION_INFERENCE_ENGINE = "modal-pyannote"

"""Offline text-to-speech (Kokoro-82M, English + Hindi). See `storyreel.voice.tts`."""

from storyreel.voice.models import ModelDownloadError, ModelMissingError, VoiceError
from storyreel.voice.tts import (
    SpeechResult,
    SynthesisError,
    SynthesisMemoryError,
    SynthesisTimeoutError,
    available,
    ensure_models,
    synthesize,
    voices,
)

__all__ = [
    "ModelDownloadError",
    "ModelMissingError",
    "SpeechResult",
    "SynthesisError",
    "SynthesisMemoryError",
    "SynthesisTimeoutError",
    "VoiceError",
    "available",
    "ensure_models",
    "synthesize",
    "voices",
]

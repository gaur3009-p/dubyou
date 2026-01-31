import numpy as np
from services.voice_identity.config import (
    SAMPLE_RATE,
    MIN_AUDIO_SECONDS,
    MAX_AUDIO_SECONDS,
    MIN_RMS,
    MAX_CLIP_RATIO,
)

def validate_audio(audio_np, sr):
    if sr != SAMPLE_RATE:
        raise ValueError("Sample rate must be 16kHz")

    duration = len(audio_np) / sr
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError("Audio too short")

    if duration > MAX_AUDIO_SECONDS:
        raise ValueError("Audio too long")

    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < MIN_RMS:
        raise ValueError("Audio too quiet")

    clip_ratio = np.mean(np.abs(audio_np) > 0.99)
    if clip_ratio > MAX_CLIP_RATIO:
        raise ValueError("Audio clipping detected")

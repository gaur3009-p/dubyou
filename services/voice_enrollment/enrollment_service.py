# ============================================================
# enrollment_service.py
# Phase 0 Orchestrator (FINAL)
# ============================================================

import numpy as np
import torch
import torchaudio

from .vad import trim_silence
from .speaker_encoder import SpeakerEncoder
from .storage import save_profile

_encoder = SpeakerEncoder()


def _resample_if_needed(audio_np: np.ndarray, sr: int, target_sr: int = 16000):
    """
    Silero VAD only supports 8k / 16k.
    We standardize EVERYTHING to 16k.
    """
    if sr == target_sr:
        return audio_np, sr

    audio_tensor = torch.tensor(audio_np).unsqueeze(0)

    resampler = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=target_sr
    )

    with torch.no_grad():
        audio_16k = resampler(audio_tensor)

    return audio_16k.squeeze(0).numpy(), target_sr


def enroll_user(audio_np: np.ndarray, sr: int):
    # --------------------------------------------------------
    # 1. Resample to 16k (CRITICAL)
    # --------------------------------------------------------
    audio_np, sr = _resample_if_needed(audio_np, sr, 16000)

    # --------------------------------------------------------
    # 2. Silence trimming (Silero VAD)
    # --------------------------------------------------------
    clean_audio = trim_silence(audio_np, sr)

    # Require at least ~5 seconds of speech
    if len(clean_audio) < sr * 5:
        return None

    # --------------------------------------------------------
    # 3. Speaker embedding (Voice Identity)
    # --------------------------------------------------------
    embedding = _encoder.encode(clean_audio)

    # --------------------------------------------------------
    # 4. Persist profile
    # --------------------------------------------------------
    user_id = save_profile(embedding, clean_audio, sr)

    return user_id

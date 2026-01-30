# ============================================================
# audio_postprocess.py — TTS Audio Normalization
# ============================================================

import numpy as np
import soundfile as sf
import os
import tempfile


def normalize_audio(
    audio_np: np.ndarray,
    target_dbfs: float = -14.0,
    peak_limit: float = 0.99
) -> np.ndarray:
    """
    Loudness normalize audio to target dBFS
    """

    # Remove DC offset
    audio_np = audio_np - np.mean(audio_np)

    # Compute RMS
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 1e-6:
        return audio_np

    # Current loudness
    current_dbfs = 20 * np.log10(rms)

    # Gain needed
    gain = 10 ** ((target_dbfs - current_dbfs) / 20)
    audio_np = audio_np * gain

    # Peak limiting (prevent clipping)
    peak = np.max(np.abs(audio_np))
    if peak > peak_limit:
        audio_np = audio_np * (peak_limit / peak)

    return audio_np


def trim_silence(
    audio_np: np.ndarray,
    threshold: float = 0.01
) -> np.ndarray:
    """
    Trim leading and trailing silence
    """
    energy = np.abs(audio_np)
    mask = energy > threshold

    if not mask.any():
        return audio_np

    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])

    return audio_np[start:end]


def postprocess_wav(
    wav_path: str,
    target_sr: int = 16000,
    normalize: bool = True,
    trim: bool = True
) -> str:
    """
    Postprocess a WAV file in-place-safe manner
    Returns path to processed WAV
    """

    audio_np, sr = sf.read(wav_path)

    # Convert stereo to mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    if normalize:
        audio_np = normalize_audio(audio_np)

    if trim:
        audio_np = trim_silence(audio_np)

    # Write to temp file
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    sf.write(out_path, audio_np, sr if sr else target_sr)

    return out_path

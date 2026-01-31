import numpy as np

def normalize_audio(audio_np):
    audio_np = audio_np - np.mean(audio_np)
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        audio_np = audio_np / peak
    return audio_np.astype("float32")

import numpy as np

def prepare_audio(audio_np):
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    return audio_np.astype("float32")

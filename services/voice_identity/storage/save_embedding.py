import os
import numpy as np
import soundfile as sf
from services.voice_identity.config import VOICE_STORAGE_DIR, SAMPLE_RATE

def save_voice_identity(user_id, audio_np, embedding):
    os.makedirs(VOICE_STORAGE_DIR, exist_ok=True)

    np.save(
        os.path.join(VOICE_STORAGE_DIR, f"{user_id}_embedding.npy"),
        embedding
    )

    sf.write(
        os.path.join(VOICE_STORAGE_DIR, f"{user_id}_reference.wav"),
        audio_np,
        SAMPLE_RATE
    )

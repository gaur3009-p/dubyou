import os
import uuid
import torch
import soundfile as sf

BASE_DIR = "voice_profiles"
os.makedirs(BASE_DIR, exist_ok=True)


def save_profile(embedding, audio_np, sr=16000):
    user_id = str(uuid.uuid4())

    torch.save(
        embedding,
        os.path.join(BASE_DIR, f"{user_id}_embedding.pt")
    )

    sf.write(
        os.path.join(BASE_DIR, f"{user_id}_reference.wav"),
        audio_np,
        sr
    )

    return user_id

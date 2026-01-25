import torch
import os
import uuid

BASE_DIR = "voice_profiles"

def save_profile(embedding, audio, user_id=None):
    os.makedirs(BASE_DIR, exist_ok=True)
    user_id = user_id or str(uuid.uuid4())

    torch.save(
        embedding,
        f"{BASE_DIR}/{user_id}_embedding.pt"
    )

    import soundfile as sf
    sf.write(
        f"{BASE_DIR}/{user_id}_reference.wav",
        audio,
        16000
    )

    return user_id

import os
import numpy as np
from services.voice_identity.config import VOICE_STORAGE_DIR

def load_embedding(user_id):
    path = os.path.join(VOICE_STORAGE_DIR, f"{user_id}_embedding.npy")
    if not os.path.exists(path):
        raise FileNotFoundError("Voice identity not found")
    return np.load(path)

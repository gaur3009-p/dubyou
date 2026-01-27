from .vad import trim_silence
from .speaker_encoder import SpeakerEncoder
from .storage import save_profile

_encoder = SpeakerEncoder()


def enroll_user(audio_np, sr):
    # Silence trimming
    clean_audio = trim_silence(audio_np, sr)

    # Require at least ~5 seconds of speech
    if len(clean_audio) < sr * 5:
        return None

    # Speaker embedding
    embedding = _encoder.encode(clean_audio)

    # Save profile
    user_id = save_profile(embedding, clean_audio, sr)

    return user_id

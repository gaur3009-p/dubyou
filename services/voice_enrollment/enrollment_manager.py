from .recorder import record_voice
from .vad import trim_silence
from .speaker_encoder import SpeakerEncoder
from .storage import save_profile

def enroll_user():
    raw_path = record_voice(duration=30)
    clean_audio = trim_silence(raw_path)

    encoder = SpeakerEncoder()
    embedding = encoder.encode(clean_audio)

    user_id = save_profile(embedding, clean_audio)
    print("🎉 Voice enrolled successfully!")
    print("🆔 User ID:", user_id)

    return user_id

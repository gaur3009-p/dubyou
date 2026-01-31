from services.voice_identity.capture.quality_checks import validate_audio
from services.voice_identity.speaker_encoder.normalize import normalize_audio
from services.voice_identity.speaker_encoder.encoder import SpeakerEncoder
from services.voice_identity.storage.save_embedding import save_voice_identity


def enroll_voice(audio_np, sr, user_id):
    validate_audio(audio_np, sr)

    audio_np = normalize_audio(audio_np)

    encoder = SpeakerEncoder()
    embedding = encoder.encode(audio_np, sr)

    save_voice_identity(user_id, audio_np, embedding)

    return user_id

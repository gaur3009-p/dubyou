import torch
import numpy as np

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True
)
(get_speech_timestamps, _, _, _, _) = utils

def trim_silence(audio_path, sr=16000):
    import soundfile as sf
    audio, _ = sf.read(audio_path)
    audio = torch.from_numpy(audio).float()

    timestamps = get_speech_timestamps(audio, model, sampling_rate=sr)

    chunks = [
        audio[t["start"]:t["end"]] for t in timestamps
    ]
    clean_audio = torch.cat(chunks)
    return clean_audio.numpy()

import torch

_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True
        )
    return _vad_model, _vad_utils


def trim_silence(audio_np, sr=16000):
    if sr not in (8000, 16000):
        raise ValueError(
            f"Silero VAD requires 8k or 16k audio, got {sr}"
        )

    model, utils = _load_vad()
    (get_speech_timestamps, _, _, _, _) = utils

    audio_tensor = torch.from_numpy(audio_np)
    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sr
    )

    if not timestamps:
        return audio_np

    chunks = [
        audio_tensor[t["start"]:t["end"]]
        for t in timestamps
    ]

    return torch.cat(chunks).numpy()

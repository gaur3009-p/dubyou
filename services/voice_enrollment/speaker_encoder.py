import torchaudio

# ---- CRITICAL PATCH (must be FIRST) ----
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import torch
from speechbrain.pretrained import EncoderClassifier


class SpeakerEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device}
        )

    def encode(self, audio_np):
        waveform = torch.tensor(audio_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
        return embedding.squeeze().cpu()

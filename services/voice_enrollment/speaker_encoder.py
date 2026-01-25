from speechbrain.pretrained import EncoderClassifier
import torch

class SpeakerEncoder:
    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )

    def encode(self, audio_np):
        waveform = torch.tensor(audio_np).unsqueeze(0)
        embedding = self.model.encode_batch(waveform)
        return embedding.squeeze().detach()

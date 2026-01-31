import torch
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech


class SpeakerEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_vc"
        )
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained(
            "microsoft/speecht5_vc"
        ).to(self.device)

    def encode(self, audio_np, sample_rate=16000):
        inputs = self.processor(
            audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            emb = self.model.get_speaker_embeddings(**inputs)

        emb = emb.squeeze().cpu().numpy()
        emb = emb / np.linalg.norm(emb)

        return emb

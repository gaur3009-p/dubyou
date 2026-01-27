import torch
import torchaudio
import numpy as np


class AudioBuffer:
    def __init__(self, target_sr=16000, max_seconds=5):
        self.target_sr = target_sr
        self.max_len = target_sr * max_seconds
        self.buffer = torch.zeros(0)

    def add(self, audio_np, sr):
        # Convert to torch
        audio = torch.tensor(audio_np)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sr
            )
            audio = resampler(audio.unsqueeze(0)).squeeze(0)

        # Append
        self.buffer = torch.cat([self.buffer, audio])

        # Trim buffer
        if len(self.buffer) > self.max_len:
            self.buffer = self.buffer[-self.max_len:]

        return self.buffer.numpy()

import numpy as np


class AudioBuffer:
    def __init__(self, max_seconds=5, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.buffer = np.zeros(0, dtype=np.float32)

    def add(self, chunk: np.ndarray, sr: int):
        self.buffer = np.concatenate([self.buffer, chunk])
        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]
        return self.buffer

    def get_recent(self, seconds: float):
        samples = int(seconds * self.sample_rate)
        return self.buffer[-samples:]

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)

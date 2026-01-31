import numpy as np


class AudioBuffer:
    def __init__(self, max_seconds=10, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.buffer = np.zeros(0, dtype=np.float32)

    def add(self, chunk: np.ndarray, sr: int):
        """
        Add audio chunk to buffer (assumes sr already correct)
        """
        self.buffer = np.concatenate([self.buffer, chunk])

        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]

        return self.buffer

    def get_recent(self, seconds: float):
        """
        Get last N seconds of audio (for live ASR only)
        """
        samples = int(seconds * self.sample_rate)
        return self.buffer[-samples:]

    def flush(self):
        """
        Return full utterance and reset buffer
        """
        audio = self.buffer.copy()
        self.buffer = np.zeros(0, dtype=np.float32)
        return audio

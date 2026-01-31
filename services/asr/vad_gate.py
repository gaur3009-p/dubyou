# ============================================================
# vad_gate.py — Simple, robust energy-based VAD
# ============================================================

import numpy as np
import time


class VadGate:
    def __init__(self, threshold=0.015, silence_time=0.8):
        self.threshold = threshold
        self.silence_time = silence_time
        self.last_voice_time = time.time()

    def is_speech(self, chunk: np.ndarray) -> bool:
        energy = np.sqrt(np.mean(chunk ** 2))
        if energy > self.threshold:
            self.last_voice_time = time.time()
            return True
        return False

    def is_silence_long(self) -> bool:
        return (time.time() - self.last_voice_time) > self.silence_time

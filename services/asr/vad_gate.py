import numpy as np
import time


class VadGate:
    def __init__(self, energy_threshold=0.015, silence_time=0.6):
        self.energy_threshold = energy_threshold
        self.silence_time = silence_time
        self.last_voice_time = time.time()

    def is_speech(self, chunk):
        energy = np.sqrt(np.mean(chunk ** 2))
        if energy > self.energy_threshold:
            self.last_voice_time = time.time()
            return True
        return False

    def is_silence_long(self):
        return (time.time() - self.last_voice_time) >= self.silence_time

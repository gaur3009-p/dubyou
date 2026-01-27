from faster_whisper import WhisperModel
import numpy as np


class StreamingASR:
    def __init__(self, model_size="large-v3"):
        self.model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16"
        )

    def transcribe(self, audio_np, language="en"):
        segments, _ = self.model.transcribe(
            audio_np,
            language=language,
            vad_filter=False,
            beam_size=1
        )
        return " ".join(seg.text.strip() for seg in segments)

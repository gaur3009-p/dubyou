from faster_whisper import WhisperModel


class StreamingASR:
    def __init__(self):
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )

    def transcribe(self, audio_np):
        if audio_np is None or len(audio_np) < 1600:
            return ""

        segments, _ = self.model.transcribe(
            audio_np,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
            beam_size=1
        )

        return " ".join(seg.text.strip() for seg in segments)

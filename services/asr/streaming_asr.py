from faster_whisper import WhisperModel
class StreamingASR:
    def __init__(self, window_sec=3):
        self.window_samples = window_sec * 16000
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )

    def transcribe(self, audio_np):
        if audio_np is None or len(audio_np) < 1600:
            return ""

        audio_np = audio_np[-self.window_samples:]

        segments, _ = self.model.transcribe(
            audio_np,
            language="en",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False
        )

        return " ".join(
            seg.text.strip()
            for seg in segments
            if seg.avg_logprob > -1.2
        )

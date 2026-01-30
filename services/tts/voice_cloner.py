# ============================================================
# voice_cloner.py — Phase 3 Voice Cloning TTS
# ============================================================

import torch
import soundfile as sf
import tempfile
import os

from TTS.api import TTS


class VoiceCloner:
    def __init__(self):
        """
        XTTS-v2 multilingual voice cloning
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=torch.cuda.is_available()
        )

    def synthesize(
        self,
        text: str,
        reference_wav: str,
        language: str = "en"
    ) -> str:
        """
        text: translated text
        reference_wav: path to speaker reference audio
        language: target language code (en, hi, etc.)
        """

        if not text.strip():
            return None

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        self.tts.tts_to_file(
            text=text,
            file_path=tmp_path,
            speaker_wav=reference_wav,
            language=language
        )

        return tmp_path

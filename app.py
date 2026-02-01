# ============================================================
# app.py — Unified App (Phase 0 + 1 + 2 + 3)
# Python 3.12 SAFE
# ============================================================

import os
import sys
import uuid
import numpy as np
import gradio as gr

# ------------------------------------------------------------
# Fix project root
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# Phase 0 — Voice Identity
# ------------------------------------------------------------
from services.voice_identity.capture.prompts import VOICE_PROMPTS
from services.voice_identity.interface import enroll_voice

# ------------------------------------------------------------
# Phase 1 — Streaming ASR
# ------------------------------------------------------------
from services.asr.audio_buffer import AudioBuffer
from services.asr.vad_gate import VadGate
from services.asr.streaming_asr import StreamingASR
from services.asr.phrase_committer import PhraseCommitter

# ------------------------------------------------------------
# Phase 2 — Streaming Translation
# ------------------------------------------------------------
from services.translation.translator import StreamingTranslator
from services.translation.translation_buffer import TranslationBuffer

# ------------------------------------------------------------
# Phase 3 — Speaker-conditioned TTS
# ------------------------------------------------------------
from services.tts.speecht5_tts import SpeechT5TTS
from services.voice_identity.storage.load_embedding import load_embedding


# ============================================================
# GLOBAL STATE (SAFE + EXPLICIT)
# ============================================================
audio_buffer = AudioBuffer()
vad = VadGate()
asr = StreamingASR()
committer = PhraseCommitter(min_words=5)

translator = StreamingTranslator()
translation_buffer = TranslationBuffer()

tts_engine = SpeechT5TTS()

final_asr_history: list[str] = []
final_translation_history: list[str] = []


# ============================================================
# Phase 0 — Voice Enrollment Logic
# ============================================================
def phase0_enroll(audio):
    if audio is None:
        return "❌ No audio received."

    sr, audio_np = audio

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    audio_np = audio_np.astype("float32")

    user_id = str(uuid.uuid4())[:8]

    try:
        enroll_voice(audio_np, sr, user_id)
    except Exception as e:
        return f"❌ Enrollment failed:\n{str(e)}"

    return (
        "✅ Voice identity enrolled successfully!\n\n"
        f"🆔 USER ID:\n{user_id}\n\n"
        "Save this ID — it will be used for voice cloning in Phase 3."
    )


# ============================================================
# Phase 1 + 2 — Streaming ASR + Translation
# ============================================================
def phase1_and_2_pipeline(audio, src_lang, tgt_lang):
    global final_asr_history, final_translation_history

    if audio is None:
        return "", " ".join(final_asr_history), " ".join(final_translation_history)

    sr, chunk = audio
    chunk = chunk.astype("float32")

    # Voice activity detection
    speaking = vad.is_speech(chunk)

    # Add audio to rolling buffer
    buffered_audio = audio_buffer.add(chunk, sr)

    live_text = ""

    if speaking and buffered_audio is not None:
        # Live unstable transcription
        live_text = asr.transcribe(buffered_audio)

        committed = committer.process(live_text)
        if committed:
            final_asr_history.append(committed)

            delta = translation_buffer.get_delta(
                " ".join(final_asr_history)
            )

            if delta:
                translated = translator.translate(
                    delta,
                    src_lang,
                    tgt_lang
                )
                final_translation_history.append(translated)

    # Reset buffer after long silence
    if vad.is_silence_long():
        audio_buffer.reset()

    return (
        live_text,
        " ".join(final_asr_history),
        " ".join(final_translation_history),
    )


# ============================================================
# Phase 3 — Voice-Cloned TTS
# ============================================================
def phase3_tts(user_id, text):
    if not user_id or not text.strip():
        return None

    try:
        speaker_embedding = load_embedding(user_id)
    except Exception:
        return None

    return tts_engine.speak(text, speaker_embedding)


# ============================================================
# UI — Gradio App
# ============================================================
with gr.Blocks(title="DubYou — Multilingual Voice Platform") as demo:
    gr.Markdown(
        """
        # 🌍 DubYou — Real-Time Multilingual Voice Platform  
        Speak naturally. Be understood instantly.
        """
    )

    with gr.Tabs():

        # -----------------------------------------------------
        # Phase 0 Tab
        # -----------------------------------------------------
        with gr.Tab("Phase 0 — Voice Identity"):
            gr.Markdown("### 🎙️ Voice Enrollment")

            for p in VOICE_PROMPTS:
                gr.Markdown(f"- {p}")

            mic0 = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record 30–60 seconds of clean speech"
            )

            enroll_out = gr.Textbox(
                label="Enrollment Result",
                lines=8
            )

            gr.Button("Enroll Voice").click(
                phase0_enroll,
                mic0,
                enroll_out
            )

        # -----------------------------------------------------
        # Phase 1 + 2 Tab
        # -----------------------------------------------------
        with gr.Tab("Phase 1 & 2 — Live Translation"):
            gr.Markdown("### 🎧 Interpreter Mode")

            with gr.Row():
                src = gr.Dropdown(
                    ["eng_Latn", "hin_Deva"],
                    value="eng_Latn",
                    label="Speaker Language"
                )
                tgt = gr.Dropdown(
                    ["hin_Deva", "eng_Latn"],
                    value="hin_Deva",
                    label="Listener Language"
                )

            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Speak"
            )

            live_txt = gr.Textbox(label="Live ASR (Unstable)", lines=4)
            final_asr = gr.Textbox(label="Final Transcription", lines=6)
            final_trans = gr.Textbox(label="Translated Text", lines=8)

            mic.stream(
                phase1_and_2_pipeline,
                inputs=[mic, src, tgt],
                outputs=[live_txt, final_asr, final_trans]
            )

        # -----------------------------------------------------
        # Phase 3 Tab
        # -----------------------------------------------------
        with gr.Tab("Phase 3 — Voice-Cloned TTS"):
            gr.Markdown("### 🔊 Speak in Your Own Voice")

            user_id_input = gr.Textbox(
                label="User ID (from Phase 0)"
            )

            tts_text = gr.Textbox(
                label="Text to Speak",
                value=lambda: " ".join(final_translation_history),
                lines=4
            )

            tts_audio = gr.Audio(label="Generated Speech")

            gr.Button("Generate Speech").click(
                phase3_tts,
                inputs=[user_id_input, tts_text],
                outputs=tts_audio
            )

demo.launch(share=True)

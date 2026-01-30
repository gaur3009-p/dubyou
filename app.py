# ============================================================
# app.py — Unified App (Phase 0 + Phase 1 + Phase 2 + Phase 3)
# Python 3.12 SAFE — Piper TTS
# ============================================================

import os
import sys
import numpy as np
import gradio as gr

# ------------------------------------------------------------
# Fix project root
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# Phase 0 — Voice Enrollment
# ------------------------------------------------------------
from services.voice_enrollment.prompts import VOICE_PROMPTS
from services.voice_enrollment.enrollment_service import enroll_user

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
# Phase 3 — TTS (Piper, Python 3.12 safe)
# ------------------------------------------------------------
from services.tts.piper_tts import PiperTTS


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

    user_id = enroll_user(audio_np, sr)
    if user_id is None:
        return "❌ Please speak clearly for at least 30 seconds."

    return (
        "✅ Voice enrolled successfully!\n\n"
        f"🆔 USER ID:\n{user_id}\n\n"
        "This voice identity will be reused in later phases."
    )


# ============================================================
# Phase 1 + Phase 2 — Shared State
# ============================================================
buffer = AudioBuffer()
vad = VadGate()
asr = StreamingASR()
committer = PhraseCommitter(min_words=5)

translator = StreamingTranslator()
translation_buffer = TranslationBuffer()

final_asr_history = []
final_translation_history = []


# ============================================================
# Phase 1 + Phase 2 — Streaming Pipeline
# ============================================================
def phase1_and_2_pipeline(audio, src_lang, tgt_lang):
    global final_asr_history, final_translation_history

    if audio is None:
        return "", " ".join(final_asr_history), " ".join(final_translation_history)

    sr, chunk = audio
    chunk = chunk.astype("float32")

    is_speech = vad.is_speech(chunk)
    audio_np = buffer.add(chunk, sr)

    live_text = ""

    if is_speech:
        live_text = asr.transcribe(audio_np)

        committed = committer.process(live_text)
        if committed:
            final_asr_history.append(committed)

            delta = translation_buffer.get_delta(" ".join(final_asr_history))
            if delta:
                translated = translator.translate(delta, src_lang, tgt_lang)
                final_translation_history.append(translated)

    if vad.is_silence_long():
        buffer.buffer = buffer.buffer[:0]

    return (
        live_text,
        " ".join(final_asr_history),
        " ".join(final_translation_history),
    )


# ============================================================
# Phase 3 — TTS (Piper)
# ============================================================
# NOTE: Replace model paths with ones you download
piper_en = PiperTTS("models/en_US-amy-medium.onnx")
piper_hi = PiperTTS("models/hi_IN-voices.onnx")


def phase3_tts(text, lang):
    if not text:
        return None

    if lang == "hi":
        return piper_hi.speak(text)
    else:
        return piper_en.speak(text)


# ============================================================
# 🖥️ Unified Gradio UI
# ============================================================
with gr.Blocks(title="DubYou — Multilingual Voice Platform") as demo:
    gr.Markdown(
        """
        # 🌍 DubYou — Multilingual Voice Platform  
        **Understand. Translate. Speak back.**
        """
    )

    with gr.Tabs():

        # =====================================================
        # Phase 0 — Voice Enrollment
        # =====================================================
        with gr.Tab("Phase 0 — Voice Enrollment"):
            gr.Markdown("### 🎙️ Create Your Voice Identity")

            for p in VOICE_PROMPTS:
                gr.Markdown(f"- **{p}**")

            mic0 = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record your voice (30–45 seconds)"
            )

            enroll_out = gr.Textbox(label="Enrollment Status", lines=7)

            gr.Button("Enroll Voice").click(
                phase0_enroll,
                mic0,
                enroll_out
            )

        # =====================================================
        # Phase 1 + Phase 2 — Live Translation
        # =====================================================
        with gr.Tab("Phase 1 & 2 — Live Translation"):
            gr.Markdown("### 🎧 Interpreter Mode")

            with gr.Row():
                src = gr.Dropdown(
                    ["eng_Latn", "hin_Deva"],
                    value="eng_Latn",
                    label="🗣️ Speaker Language"
                )
                tgt = gr.Dropdown(
                    ["hin_Deva", "eng_Latn"],
                    value="hin_Deva",
                    label="👂 Listener Language"
                )

            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                format="wav",
                label="🎙️ Speak"
            )

            with gr.Row():
                with gr.Column():
                    live_txt = gr.Textbox(label="Live ASR (Unstable)", lines=4)
                    final_asr = gr.Textbox(label="Final ASR", lines=6)

                with gr.Column():
                    final_trans = gr.Textbox(label="Translated Text", lines=10)

            mic.stream(
                phase1_and_2_pipeline,
                inputs=[mic, src, tgt],
                outputs=[live_txt, final_asr, final_trans]
            )

        # =====================================================
        # Phase 3 — Text to Speech
        # =====================================================
        with gr.Tab("Phase 3 — Text to Speech"):
            gr.Markdown("### 🔊 Speak the Translation")

            tts_lang = gr.Dropdown(
                ["en", "hi"],
                value="hi",
                label="Output Language"
            )

            tts_text = gr.Textbox(
                label="Text to Speak",
                value=lambda: " ".join(final_translation_history),
                lines=4
            )

            tts_audio = gr.Audio(label="Generated Speech")

            gr.Button("Generate Speech").click(
                phase3_tts,
                inputs=[tts_text, tts_lang],
                outputs=tts_audio
            )

demo.launch(share=True)

# ============================================================
# app.py — Unified App (Phase 0 + Phase 1)
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
# Phase 0 imports — Voice Enrollment
# ------------------------------------------------------------
from services.voice_enrollment.prompts import VOICE_PROMPTS
from services.voice_enrollment.enrollment_service import enroll_user

# ------------------------------------------------------------
# Phase 1 imports — Streaming ASR
# ------------------------------------------------------------
from services.asr.audio_buffer import AudioBuffer
from services.asr.vad_gate import VadGate
from services.asr.streaming_asr import StreamingASR
from services.asr.phrase_committer import PhraseCommitter


# =========================
# Phase 0 Logic
# =========================
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
        "This ID will be reused in all next phases."
    )


# =========================
# Phase 1 Logic
# =========================
buffer = AudioBuffer()
vad = VadGate()
asr = StreamingASR()
committer = PhraseCommitter(min_words=5)

final_history = []


def phase1_streaming_asr(audio):
    global final_history

    if audio is None:
        return "", " ".join(final_history)

    sr, chunk = audio
    chunk = chunk.astype("float32")

    is_speech = vad.is_speech(chunk)
    audio_np = buffer.add(chunk, sr)

    live_text = ""
    if is_speech:
        live_text = asr.transcribe(audio_np)
        committed = committer.process(live_text)
        if committed:
            final_history.append(committed)

    if vad.is_silence_long():
        buffer.buffer = buffer.buffer[:0]

    return live_text, " ".join(final_history)


# ============================================================
# 🖥️ Unified Gradio UI
# ============================================================
with gr.Blocks(title="DubYou — Multilingual Voice Platform") as demo:
    gr.Markdown(
        """
        # 🌍 DubYou — Multilingual Voice Platform
        Progressive system from **voice identity → real-time speech translation**
        """
    )

    with gr.Tabs():

        # =========================
        # Phase 0 TAB
        # =========================
        with gr.Tab("Phase 0 — Voice Enrollment"):
            gr.Markdown("### 🎙️ Create Your Voice Identity")

            for p in VOICE_PROMPTS:
                gr.Markdown(f"- **{p}**")

            mic0 = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record your voice (30–45 seconds)"
            )

            enroll_btn = gr.Button("Enroll Voice")
            enroll_out = gr.Textbox(lines=8)

            enroll_btn.click(
                phase0_enroll,
                inputs=mic0,
                outputs=enroll_out
            )

        # =========================
        # Phase 1 TAB
        # =========================
        with gr.Tab("Phase 1 — Streaming ASR"):
            gr.Markdown("### 🎧 Live Speech → Text")

            mic1 = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Speak"
            )

            live_txt = gr.Textbox(label="Live (Unstable)")
            final_txt = gr.Textbox(label="Final (Committed)")

            mic1.stream(
                phase1_streaming_asr,
                inputs=mic1,
                outputs=[live_txt, final_txt]
            )

        # =========================
        # Phase 2 PLACEHOLDER
        # =========================
        with gr.Tab("Phase 2 — Streaming Translation"):
            gr.Markdown(
                """
                🚧 **Coming next**

                - Streaming text-to-text translation  
                - Context-aware buffering  
                - Multilingual support (NLLB / mBART)
                """
            )

        # =========================
        # Phase 3 PLACEHOLDER
        # =========================
        with gr.Tab("Phase 3 — Voice Cloning TTS"):
            gr.Markdown(
                """
                🚧 **Coming next**

                - Same voice, different language  
                - XTTS / VALL-E style cloning  
                - Speaker embedding reuse
                """
            )

        # =========================
        # Phase 4 PLACEHOLDER
        # =========================
        with gr.Tab("Phase 4 — Speech ↔ Speech"):
            gr.Markdown(
                """
                🚧 **Coming next**

                - Person A ↔ Person B  
                - Bidirectional real-time conversation  
                - Full S2S pipeline
                """
            )

demo.launch(share=True)

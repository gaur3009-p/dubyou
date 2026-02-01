# ============================================================
# app.py — Unified App (Phase 0 + Phase 1 + Phase 2 + Phase 3)
# UPDATED — Streaming, Emotion-Preserved, Cross-Lingual
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

# ============================================================
# Phase 0 — Voice Enrollment
# ============================================================
from services.voice_identity.capture.prompts import VOICE_PROMPTS
from services.voice_identity.interface import enroll_voice

# ============================================================
# Phase 1–3 — Core Services
# ============================================================
from services.asr.audio_buffer import AudioBuffer
from services.asr.vad_gate import VadGate
from services.asr.streaming_asr import StreamingASR
from services.asr.phrase_committer import PhraseCommitter

from services.translation.emotion import EmotionDetector
from services.translation.translator import EmotionAwareTranslator

from services.tts.voice_cloner import VoiceCloner

# ============================================================
# SESSION STORE (per user)
# ============================================================
SESSIONS = {}


def get_session(user_id: str):
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {
            "buffer": AudioBuffer(max_seconds=5),
            "vad": VadGate(),
            "asr": StreamingASR(),
            "committer": PhraseCommitter(min_words=4),
            "emotion": EmotionDetector(),
            "translator": EmotionAwareTranslator(),
            "tts": VoiceCloner(user_id),

            # ---- Streaming state ----
            "last_live_asr": "",
            "last_translation": ""
        }
    return SESSIONS[user_id]


# ============================================================
# Phase 0 — Enrollment Callback
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
        "✅ Voice enrolled successfully!\n\n"
        f"🆔 USER ID: {user_id}\n\n"
        "Your English voice will now be used to speak Hindi\n"
        "with emotion preserved."
    )


# ============================================================
# Phase 1–3 — STREAMING PIPELINE (THIS IS THE CODE YOU ASKED ABOUT)
# ============================================================
def streaming_pipeline(audio, user_id):
    """
    This function CONTAINS the UPDATED STREAMING PIPELINE.
    It is called repeatedly by gr.Audio(streaming=True)
    """

    if not user_id or audio is None:
        return "", "", None

    sr, chunk = audio
    chunk = chunk.astype("float32")

    session = get_session(user_id)

    # 1️⃣ Always buffer audio
    session["buffer"].add(chunk, sr)

    # 2️⃣ Voice activity detection
    if session["vad"].is_speech(chunk):

        # Sliding window ASR (last 3 seconds)
        live_text = session["asr"].transcribe(
            session["buffer"].get_recent(3)
        )

        session["last_live_asr"] = live_text

        # Commit stable phrase
        phrase = session["committer"].process(live_text)

        if phrase:
            # Emotion detection
            emotion = session["emotion"].detect(phrase)

            # Emotion-aware translation (EN → HI)
            hindi_text = session["translator"].translate(
                phrase,
                src_lang="eng_Latn",
                tgt_lang="hin_Deva",
                emotion=emotion
            )

            session["last_translation"] = hindi_text

            # Streaming TTS (Hindi spoken in YOUR English voice)
            audio_chunk = session["tts"].speak_chunk(
                hindi_text,
                emotion=emotion
            )

            return (
                session["last_live_asr"],
                session["last_translation"],
                audio_chunk
            )

    # 3️⃣ Handle silence → flush buffer
    if session["vad"].should_flush():
        session["buffer"].reset()

    return (
        session["last_live_asr"],
        session["last_translation"],
        None
    )


# ============================================================
# UI — Gradio App
# ============================================================
with gr.Blocks(title="DubYou — Real-Time Voice Translation") as demo:

    gr.Markdown(
        """
        # 🌍 DubYou — Real-Time Voice Translation  
        Speak English → Hear Hindi in **your own voice**
        """
    )

    with gr.Tabs():

        # ---------------- Phase 0 ----------------
        with gr.Tab("Phase 0 — Voice Enrollment"):
            gr.Markdown("### 🎙️ Record 30–60 seconds of clean English speech")

            for p in VOICE_PROMPTS:
                gr.Markdown(f"- {p}")

            mic0 = gr.Audio(
                sources=["microphone"],
                type="numpy"
            )

            enroll_out = gr.Textbox(lines=8)

            gr.Button("Enroll Voice").click(
                phase0_enroll,
                mic0,
                enroll_out
            )

        # ---------------- Phase 1–3 ----------------
        with gr.Tab("Phase 1–3 — Live Translation"):
            user_id_input = gr.Textbox(
                label="User ID (from Phase 0)"
            )

            mic = gr.Audio(
                sources=["microphone"],
                type="numpy",
                streaming=True,
                label="Speak English"
            )

            live_asr = gr.Textbox(
                label="Live ASR (English)",
                lines=3
            )

            translated_txt = gr.Textbox(
                label="Translated Text (Hindi)",
                lines=4
            )

            tts_audio = gr.Audio(
                label="Hindi Speech (Your Voice)",
                autoplay=True
            )

            mic.stream(
                streaming_pipeline,
                inputs=[mic, user_id_input],
                outputs=[live_asr, translated_txt, tts_audio]
            )

demo.launch(share=True)

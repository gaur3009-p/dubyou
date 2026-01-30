# ============================================================
# app.py — Unified App (Phase 0 + Phase 1 + Phase 2 + Phase 3)
# STABLE Gradio Streaming Version
# ============================================================

import os
import sys
import numpy as np
import gradio as gr

# ------------------------------------------------------------
# Fix project root (Colab / Docker / Local safe)
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
# Phase 3 — Voice Cloning TTS
# ------------------------------------------------------------
from services.tts.voice_cloner import VoiceCloner
from services.tts.audio_postprocess import postprocess_wav


# ============================================================
# Phase 0 — Voice Enrollment Logic
# ============================================================
def phase0_enroll(audio):
    if audio is None:
        return "❌ No audio received."

    sr, audio_np = audio

    # Convert to mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    audio_np = audio_np.astype("float32")

    user_id = enroll_user(audio_np, sr)
    if user_id is None:
        return "❌ Please speak clearly for at least 30 seconds."

    return (
        "✅ Voice enrolled successfully!\n\n"
        f"🆔 USER ID:\n{user_id}\n\n"
        "Save this ID — it will be reused for voice cloning."
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

voice_cloner = VoiceCloner()

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
# Phase 3 — Voice Cloning TTS Logic
# ============================================================
def phase3_tts(user_id, text, lang):
    if not user_id or not text:
        return None

    ref_path = f"voice_profiles/{user_id}_reference.wav"
    if not os.path.exists(ref_path):
        return None

    raw_audio = voice_cloner.synthesize(
        text=text,
        reference_wav=ref_path,
        language=lang
    )

    if raw_audio is None:
        return None

    final_audio = postprocess_wav(
        raw_audio,
        normalize=True,
        trim=True
    )

    return final_audio


# ============================================================
# 🖥️ Unified Gradio UI
# ============================================================
with gr.Blocks(title="DubYou — Multilingual Voice Platform") as demo:
    gr.Markdown(
        """
        # 🌍 DubYou — Real-Time Multilingual Voice Platform  
        **Speak naturally. Be understood instantly. Speak back in your own voice.**
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
                label="🎙️ Speak here"
            )

            with gr.Row():
                with gr.Column():
                    live_txt = gr.Textbox(label="📝 Live ASR (Unstable)", lines=4)
                    final_asr = gr.Textbox(label="✅ Final Transcription", lines=6)

                with gr.Column():
                    final_trans = gr.Textbox(label="🌍 Translated Text", lines=10)

            mic.stream(
                phase1_and_2_pipeline,
                inputs=[mic, src, tgt],
                outputs=[live_txt, final_asr, final_trans]
            )

        # =====================================================
        # Phase 3 — Voice Cloning TTS
        # =====================================================
        with gr.Tab("Phase 3 — Voice Cloning TTS"):
            gr.Markdown("### 🔊 Same Voice, Different Language")

            user_id_input = gr.Textbox(
                label="🆔 User ID (from Phase 0)",
                placeholder="Paste your enrolled user ID"
            )

            tts_lang = gr.Dropdown(
                ["en", "hi"],
                value="hi",
                label="🎯 Output Language"
            )

            tts_text = gr.Textbox(
                label="📝 Text to Speak",
                value=lambda: " ".join(final_translation_history),
                lines=4
            )

            tts_audio = gr.Audio(label="🔊 Cloned Voice Output")

            gr.Button("Generate Voice").click(
                phase3_tts,
                inputs=[user_id_input, tts_text, tts_lang],
                outputs=tts_audio
            )

demo.launch(share=True)

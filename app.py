"""
DubYou — Real-Time Voice Translation Application
Unified App (Phase 0 + Phase 1 + Phase 2 + Phase 3)
UPDATED — Streaming, Emotion-Preserved, Cross-Lingual
Python 3.12 Compatible
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Optional
from pathlib import Path

import numpy as np
import gradio as gr
from numpy.typing import NDArray

# Fix project root
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Phase 0 — Voice Enrollment
from services.voice_identity.capture.prompts import VOICE_PROMPTS
from services.voice_identity.interface import enroll_voice

# Phase 1–3 — Core Services
from services.asr.audio_buffer import AudioBuffer
from services.asr.vad_gate import VadGate
from services.asr.streaming_asr import StreamingASR
from services.asr.phrase_committer import PhraseCommitter
from services.translation.emotion import EmotionDetector
from services.translation.translator import EmotionAwareTranslator
from services.tts.voice_cloner import VoiceCloner


# Type alias for audio data
AudioTuple = tuple[int, NDArray[np.float32]]


class SessionState:
    """Container for user session state."""
    
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.buffer = AudioBuffer(max_seconds=5)
        self.vad = VadGate()
        self.asr = StreamingASR()
        self.committer = PhraseCommitter(min_words=4)
        self.emotion = EmotionDetector()
        self.translator = EmotionAwareTranslator()
        self.tts = VoiceCloner()
        
        # Streaming state
        self.last_live_asr: str = ""
        self.last_translation: str = ""


# SESSION STORE (per user)
SESSIONS: dict[str, SessionState] = {}


def get_session(user_id: str) -> SessionState:
    """Get or create a session for the given user ID."""
    if user_id not in SESSIONS:
        SESSIONS[user_id] = SessionState(user_id)
    return SESSIONS[user_id]


def phase0_enroll(audio: Optional[AudioTuple]) -> str:
    """
    Phase 0 — Voice enrollment callback.
    
    Args:
        audio: Tuple of (sample_rate, audio_array) or None
        
    Returns:
        Status message for the user
    """
    if audio is None:
        return "❌ No audio received."

    sr, audio_np = audio

    # Convert stereo to mono if needed
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    audio_np = audio_np.astype(np.float32)
    user_id = str(uuid.uuid4())[:8]

    try:
        enroll_voice(audio_np, sr, user_id)
    except ValueError as e:
        return f"❌ Enrollment validation failed:\n{e}"
    except Exception as e:
        return f"❌ Enrollment failed:\n{e}"

    return (
        "✅ Voice enrolled successfully!\n\n"
        f"🆔 USER ID: {user_id}\n\n"
        "Your English voice will now be used to speak Hindi\n"
        "with emotion preserved."
    )


def streaming_pipeline(
    audio: Optional[AudioTuple], 
    user_id: str
) -> tuple[str, str, Optional[tuple[int, NDArray[np.float32]]]]:
    """
    Phase 1–3 — Streaming translation pipeline.
    
    This function is called repeatedly by gr.Audio(streaming=True).
    
    Args:
        audio: Tuple of (sample_rate, audio_chunk) or None
        user_id: User identifier from enrollment
        
    Returns:
        Tuple of (live_asr_text, translated_text, audio_output)
    """
    if not user_id or audio is None:
        return "", "", None

    sr, chunk = audio
    chunk = chunk.astype(np.float32)

    try:
        session = get_session(user_id)
    except Exception as e:
        print(f"Error getting session: {e}")
        return "", "", None

    # 1️⃣ Always buffer audio
    session.buffer.add(chunk, sr)

    # 2️⃣ Voice activity detection
    if session.vad.is_speech(chunk):
        try:
            # Sliding window ASR (last 3 seconds)
            live_text = session.asr.transcribe(
                session.buffer.get_recent(3)
            )

            session.last_live_asr = live_text

            # Commit stable phrase
            phrase = session.committer.process(live_text)

            if phrase:
                # Emotion detection
                emotion = session.emotion.detect(phrase)

                # Emotion-aware translation (EN → HI)
                hindi_text = session.translator.translate(
                    phrase,
                    src_lang="eng_Latn",
                    tgt_lang="hin_Deva",
                    emotion=emotion
                )

                session.last_translation = hindi_text

                # Streaming TTS (Hindi spoken in YOUR English voice)
                audio_chunk = session.tts.speak_chunk(
                    hindi_text,
                    emotion=emotion
                )

                return (
                    session.last_live_asr,
                    session.last_translation,
                    audio_chunk
                )
        except Exception as e:
            print(f"Error in pipeline processing: {e}")
            return session.last_live_asr, session.last_translation, None

    # 3️⃣ Handle silence → flush buffer
    if session.vad.should_flush():
        session.buffer.reset()

    return (
        session.last_live_asr,
        session.last_translation,
        None
    )


# UI — Gradio App
def create_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    
    with gr.Blocks(
        title="DubYou — Real-Time Voice Translation",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown(
            """
            # 🌍 DubYou — Real-Time Voice Translation  
            Speak English → Hear Hindi in **your own voice**
            
            Built with Python 3.12 | Emotion-Aware Translation | Voice Cloning
            """
        )

        with gr.Tabs():

            # Phase 0 — Voice Enrollment
            with gr.Tab("📝 Phase 0 — Voice Enrollment"):
                gr.Markdown("### 🎙️ Record 30–60 seconds of clean English speech")
                
                gr.Markdown("**Read these prompts clearly:**")
                for idx, prompt in enumerate(VOICE_PROMPTS, 1):
                    gr.Markdown(f"{idx}. {prompt}")

                mic0 = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Voice Enrollment Recording"
                )

                enroll_out = gr.Textbox(
                    lines=8,
                    label="Enrollment Status",
                    interactive=False
                )

                enroll_btn = gr.Button("🎯 Enroll Voice", variant="primary")
                enroll_btn.click(
                    fn=phase0_enroll,
                    inputs=mic0,
                    outputs=enroll_out
                )

            # Phase 1–3 — Live Translation
            with gr.Tab("🎤 Phase 1–3 — Live Translation"):
                gr.Markdown(
                    """
                    ### Start Translation
                    Enter your User ID from enrollment and start speaking!
                    """
                )
                
                user_id_input = gr.Textbox(
                    label="🆔 User ID (from Phase 0)",
                    placeholder="Enter your 8-character user ID"
                )

                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="🎙️ Speak English"
                )

                live_asr = gr.Textbox(
                    label="📝 Live ASR (English)",
                    lines=3,
                    interactive=False
                )

                translated_txt = gr.Textbox(
                    label="🔄 Translated Text (Hindi)",
                    lines=4,
                    interactive=False
                )

                tts_audio = gr.Audio(
                    label="🔊 Hindi Speech (Your Voice)",
                    autoplay=True,
                    interactive=False
                )

                mic.stream(
                    fn=streaming_pipeline,
                    inputs=[mic, user_id_input],
                    outputs=[live_asr, translated_txt, tts_audio]
                )

        gr.Markdown(
            """
            ---
            **Note:** This application uses advanced AI models. 
            Ensure you have a CUDA-compatible GPU for optimal performance.
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

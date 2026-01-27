import os
import sys
import numpy as np
import gradio as gr

# Fix project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.voice_enrollment.prompts import VOICE_PROMPTS
from services.voice_enrollment.enrollment_service import enroll_user


def enroll_voice(audio):
    if audio is None:
        return "❌ No audio received."

    sr, audio_np = audio

    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    audio_np = audio_np.astype("float32")

    user_id = enroll_user(audio_np, sr)

    if user_id is None:
        return "❌ Audio too short or invalid."

    return (
        "✅ Voice enrolled successfully!\n\n"
        f"🆔 USER ID:\n{user_id}\n\n"
        "Save this ID — it will be used in all future phases."
    )


with gr.Blocks(title="DubYou — Voice Enrollment (Phase 0)") as demo:
    gr.Markdown("# 🎙️ Voice Enrollment (Phase 0)")
    gr.Markdown("### Please read the following sentences clearly:")

    for p in VOICE_PROMPTS:
        gr.Markdown(f"- **{p}**")

    mic = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="Record your voice (30–45 seconds recommended)"
    )

    btn = gr.Button("🚀 Enroll Voice")
    out = gr.Textbox(lines=7)

    btn.click(enroll_voice, mic, out)

demo.launch(share=True)

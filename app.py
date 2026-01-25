import gradio as gr
import numpy as np
import torch

from services.voice_enrollment.prompts import VOICE_PROMPTS
from services.voice_enrollment.vad import trim_silence
from services.voice_enrollment.speaker_encoder import SpeakerEncoder
from services.voice_enrollment.storage import save_profile


encoder = SpeakerEncoder()


def enroll_voice(audio):
    """
    audio: (sample_rate, numpy_array)
    """
    if audio is None:
        return "❌ No audio received"

    sr, audio_np = audio

    # mono + float32
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    audio_np = audio_np.astype("float32")

    # --- Silence trimming ---
    clean_audio = trim_silence_from_np(audio_np, sr)

    if len(clean_audio) < sr * 5:
        return "❌ Audio too short. Please speak clearly."

    # --- Speaker embedding ---
    embedding = encoder.encode(clean_audio)

    # --- Save profile ---
    user_id = save_profile(embedding, clean_audio)

    return f"✅ Voice enrolled successfully!\n\n🆔 User ID:\n{user_id}"


# --- Helper: NP-based VAD wrapper ---
def trim_silence_from_np(audio_np, sr):
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    (get_speech_timestamps, _, _, _, _) = utils

    audio_tensor = torch.from_numpy(audio_np)
    timestamps = get_speech_timestamps(
        audio_tensor, model, sampling_rate=sr
    )

    if not timestamps:
        return audio_np

    chunks = [
        audio_tensor[t["start"]:t["end"]]
        for t in timestamps
    ]
    return torch.cat(chunks).numpy()


# ---------------- UI ----------------

with gr.Blocks(title="Voice Enrollment") as demo:
    gr.Markdown(
        """
        # 🎙️ Voice Enrollment (Phase 0)

        Please **read the following sentences clearly**.
        This will be used to **clone your voice later**.
        """
    )

    for p in VOICE_PROMPTS:
        gr.Markdown(f"• **{p}**")

    mic = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="Record your voice (30–45 seconds)"
    )

    enroll_btn = gr.Button("🚀 Enroll My Voice")

    output = gr.Textbox(
        label="Enrollment Status",
        lines=5
    )

    enroll_btn.click(
        enroll_voice,
        inputs=mic,
        outputs=output
    )


demo.launch(share=True)

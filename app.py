import os
import sys
import gradio as gr
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.asr.audio_buffer import AudioBuffer
from services.asr.vad_gate import VadGate
from services.asr.streaming_asr import StreamingASR
from services.asr.phrase_committer import PhraseCommitter

buffer = AudioBuffer()
vad = VadGate()
asr = StreamingASR()
committer = PhraseCommitter()

final_history = []


def live_asr(audio):
    global final_history

    if audio is None:
        return "", " ".join(final_history)

    sr, chunk = audio
    chunk = chunk.astype("float32")

    speech = vad.is_speech(chunk)
    audio_np = buffer.add(chunk, sr)

    live_text = ""
    if speech:
        live_text = asr.transcribe(audio_np)

        committed = committer.process(live_text)
        if committed:
            final_history.append(committed)

    if vad.is_silence_long():
        buffer.buffer = buffer.buffer[:0]

    return live_text, " ".join(final_history)


with gr.Blocks(title="Phase 1 — Streaming ASR") as demo:
    gr.Markdown("# 🎙️ Phase 1 — Streaming ASR")

    mic = gr.Audio(
        sources=["microphone"],
        type="numpy",
        streaming=True,
        label="Speak"
    )

    live_txt = gr.Textbox(label="Live (Unstable)")
    final_txt = gr.Textbox(label="Final (Committed)")

    mic.stream(
        live_asr,
        inputs=mic,
        outputs=[live_txt, final_txt]
    )

demo.launch(share=True)

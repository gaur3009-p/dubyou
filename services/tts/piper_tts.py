import subprocess
import tempfile
import os


class PiperTTS:
    def __init__(self, model_path):
        self.model_path = model_path

    def speak(self, text: str) -> str:
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        cmd = [
            "piper",
            "--model", self.model_path,
            "--output_file", out_path
        ]

        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        p.stdin.write(text)
        p.stdin.close()
        p.wait()

        return out_path

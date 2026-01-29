class TranslationBuffer:
    def __init__(self):
        self.last_source = ""

    def get_delta(self, new_text: str) -> str | None:
        if not new_text.startswith(self.last_source):
            # ASR rewrite or reset
            self.last_source = ""
            return None

        delta = new_text[len(self.last_source):].strip()
        if not delta:
            return None

        self.last_source = new_text
        return delta

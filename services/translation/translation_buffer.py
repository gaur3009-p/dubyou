class TranslationBuffer:
    def __init__(self):
        self.last_text = ""

    def get_delta(self, new_text):
        if not new_text.startswith(self.last_text):
            self.last_text = ""
            return None

        delta = new_text[len(self.last_text):].strip()
        if not delta:
            return None

        self.last_text = new_text
        return delta

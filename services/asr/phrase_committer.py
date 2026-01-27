class PhraseCommitter:
    def __init__(self, min_words=5):
        self.min_words = min_words
        self.last_committed = ""

    def process(self, live_text: str):
        # Whisper may rewrite earlier tokens
        if not live_text.startswith(self.last_committed):
            self.last_committed = ""
            return None

        delta = live_text[len(self.last_committed):].strip()
        words = delta.split()

        if len(words) >= self.min_words:
            phrase = " ".join(words)
            self.last_committed = live_text
            return phrase

        return None

class PhraseCommitter:
    def __init__(self, min_words=5):
        self.last_committed = ""

    def process(self, live_text):
        if not live_text.startswith(self.last_committed):
            # Whisper rewrote history → reset safely
            self.last_committed = ""
            return None

        delta = live_text[len(self.last_committed):].strip()
        words = delta.split()

        if len(words) >= min_words:
            phrase = " ".join(words)
            self.last_committed = live_text
            return phrase

        return None

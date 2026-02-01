class PhraseCommitter:
    def __init__(self, min_words=4):
        self.min_words = min_words
        self.last_tokens = []

    def process(self, live_text: str):
        tokens = live_text.strip().split()

        if len(tokens) <= len(self.last_tokens):
            return None

        delta = tokens[len(self.last_tokens):]

        if len(delta) >= self.min_words:
            self.last_tokens = tokens
            return " ".join(delta)

        return None

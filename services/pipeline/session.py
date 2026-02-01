class SessionState:
    def __init__(self, user_id):
        self.buffer = AudioBuffer(max_seconds=5)
        self.vad = VadGate()
        self.asr = StreamingASR()
        self.committer = PhraseCommitter(min_words=4)

        self.emotion = EmotionDetector()
        self.translator = EmotionAwareTranslator()
        self.tts = StreamingXTTS(user_id)

    def process_audio(self, chunk):
        speaking = self.vad.is_speech(chunk)
        self.buffer.add(chunk, 16000)

        if not speaking:
            return None

        text = self.asr.transcribe(
            self.buffer.get_recent(3)
        )

        phrase = self.committer.process(text)
        if not phrase:
            return None

        emotion = self.emotion.detect(phrase)

        hindi_text = self.translator.translate(
            phrase,
            src_lang="eng_Latn",
            tgt_lang="hin_Deva",
            emotion=emotion
        )

        return self.tts.stream(
            hindi_text,
            emotion=emotion
        )

from transformers import pipeline


class EmotionDetector:
    def __init__(self):
        self.model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )

    def detect(self, text: str) -> str:
        result = self.model(text)[0]
        return result["label"]

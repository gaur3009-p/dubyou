import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class EmotionAwareTranslator:
    def __init__(self, model_name="facebook/m2m100_418M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def translate(self, text: str, src_lang: str, tgt_lang: str, emotion: str) -> str:
        if not text.strip():
            return ""

        # Emotion-preserving prefix (VERY IMPORTANT)
        emotion_prefix = {
            "joy": "खुशी के साथ: ",
            "anger": "गुस्से में: ",
            "sadness": "उदासी के साथ: ",
            "fear": "डर के साथ: ",
            "surprise": "हैरानी से: ",
            "neutral": ""
        }.get(emotion, "")

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)

        tgt_id = self.tokenizer.get_lang_id(tgt_lang)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=128,
                num_beams=1
            )

        translated = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
        return emotion_prefix + translated

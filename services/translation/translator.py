import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class StreamingTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text.strip():
            return ""

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        output = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_id,
            max_length=256
        )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

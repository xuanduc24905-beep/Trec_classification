"""Vietnamese -> English translator dùng vinai-translate-vi2en-v2.

Chất lượng cao hơn Google Translate cho cặp Việt-Anh,
output đầy đủ ("Who is" thay vì "Who's") → khớp distribution train của TREC.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

_MODEL_NAME = "vinai/vinai-translate-vi2en-v2"


class Translator:
    def __init__(self, model_name: str = _MODEL_NAME, device: str = None):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="vi_VN")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        tok = self.tokenizer
        self.en_id = (
            tok.lang_code_to_id["en_XX"] if hasattr(tok, "lang_code_to_id")
            else tok.convert_tokens_to_ids("en_XX")
        )

    @functools.lru_cache(maxsize=512)
    def translate(self, text: str, target_lang: str = "en") -> str:
        if not text.strip():
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                decoder_start_token_id=self.en_id,
                num_beams=5, max_length=64, early_stopping=True,
            )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]


_singleton: Optional[Translator] = None


def get_translator() -> Translator:
    global _singleton
    if _singleton is None:
        _singleton = Translator()
    return _singleton

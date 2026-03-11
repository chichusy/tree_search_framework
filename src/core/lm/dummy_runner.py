# src/core/lm/dummy_runner.py
from __future__ import annotations

import hashlib
from typing import Optional

from .results import LMResult


class DummyLMRunner:
    """
    可控假模型：不做真实推理，只返回固定长度输出。
    """
    def __init__(self, fixed_output_tokens: int = 64):
        self.fixed_output_tokens = fixed_output_tokens

    def generate(self, prompt_text: str, max_new_tokens: Optional[int] = None, **kwargs) -> LMResult:
        out_len = self.fixed_output_tokens if max_new_tokens is None else min(self.fixed_output_tokens, max_new_tokens)
        h = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:8]
        output_text = f"<dummy:{h}> " + ("x" * out_len)
        return LMResult(output_text=output_text, output_tokens_len=out_len)
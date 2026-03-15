from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LMResult:
    output_text: str
    output_tokens_len: int

    # vLLM prefix caching stats (optional)
    num_cached_tokens: int = 0
    prompt_tokens_len: Optional[int] = None

    # optional exact token ids from runner
    output_token_ids: Optional[List[int]] = None
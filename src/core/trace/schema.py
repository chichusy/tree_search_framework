# src/core/trace/schema.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class CallRecord:
    # identity / structure
    call_id: int
    rollout_id: int
    node_id: int
    parent_node_id: Optional[int]
    purpose: str  # "expand" | "simulate" | "select" | "reward" | "final" | "prm"

    # token-level prompt/decoding sizes
    input_len: int = 0          # prompt token length (prefill length)
    output_len: int = 0         # generated tokens length (decode length)

    # lightweight identity for prompt tokens
    input_ids_digest: str = ""  # sha1 digest of input_ids

    # locality vs LAST call (time locality)
    lcp_last: int = 0
    reuse_last: int = 0
    miss_last: int = 0

    # locality vs PARENT node's prompt (tree locality)
    lcp_parent: int = 0
    reuse_parent: int = 0
    miss_parent: int = 0

    # Optional timing
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None

    # Extra info (keep flexible)
    meta: Optional[Dict[str, Any]] = None

    # Optional: save raw texts for debugging (can be large)
    prompt_text: Optional[str] = None
    output_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
# src/core/terminal/presets.py
from __future__ import annotations

import re
from typing import List, Pattern


def _c(p: str) -> Pattern[str]:
    return re.compile(p, re.IGNORECASE)


# Generic patterns (cross-dataset)
GENERIC_FINAL_PATTERNS: List[Pattern[str]] = [
    # GSM8K-like final marker: "#### 18"
    _c(r"####\s*([-+]?\d+(?:\.\d+)?)"),

    # If generation stops right after "####"
    _c(r"####\s*$"),

    # Common textual final line formats
    _c(r"(?:final\s*answer|answer)\s*[:：]\s*(.+)"),

    # Math style: \boxed{...}
    _c(r"\\boxed\{([^}]*)\}"),
]


# GSM8K stricter preference
GSM8K_FINAL_PATTERNS: List[Pattern[str]] = [
    # Standard GSM8K final format
    _c(r"####\s*([-+]?\d+(?:\.\d+)?)"),

    # Sometimes model only outputs ####
    _c(r"####\s*$"),

    # Some prompts produce "Answer:"
    _c(r"(?:final\s*answer|answer)\s*[:：]\s*(.+)"),

    # Math-style boxed answer (Qwen often outputs this)
    _c(r"\\boxed\{([^}]*)\}"),

    # Robust fallback: \boxed{18  (missing closing brace due to stop token)
    _c(r"\\boxed\{([^}]*)$"),
]
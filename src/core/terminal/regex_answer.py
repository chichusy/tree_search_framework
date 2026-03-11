# src/core/terminal/regex_answer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Pattern

from .base import TerminalChecker, TerminalDecision


def _generated_part(text: str, root_text: str) -> str:
    """Return only the generated continuation (exclude the original question),
    to avoid false positives when the question contains 'Answer:' / '####' examples.

    If text doesn't start with root_text, fall back to full text.
    """
    if not root_text:
        return text
    if text.startswith(root_text):
        return text[len(root_text):]
    return text


def _compile_patterns(patterns: Sequence[str | Pattern[str]]) -> List[Pattern[str]]:
    compiled: List[Pattern[str]] = []
    for p in patterns:
        if isinstance(p, re.Pattern):
            compiled.append(p)
        else:
            compiled.append(re.compile(p, re.IGNORECASE))
    return compiled


@dataclass
class RegexAnswerTerminal(TerminalChecker):
    """Terminal if the generated continuation matches any answer marker regex.

    If your regex has a capture group (group 1), we store it in info["capture"].
    """
    patterns: List[Pattern[str]]
    name: str = "regex_answer"

    @classmethod
    def from_patterns(cls, patterns: Sequence[str | Pattern[str]], name: str = "regex_answer") -> "RegexAnswerTerminal":
        return cls(patterns=_compile_patterns(patterns), name=name)

    def decide(self, text: str, depth: int, tokens_len: int, root_text: str) -> TerminalDecision:
        haystack = _generated_part(text, root_text)

        for pat in self.patterns:
            m = pat.search(haystack)
            if not m:
                continue

            info: Dict[str, Any] = {
                "pattern": pat.pattern,
            }

            # Optional: record capture group 1 for debugging / later analysis
            try:
                if m.lastindex and m.lastindex >= 1:
                    info["capture"] = m.group(1)
            except Exception:
                # Don't let capture bugs break terminal detection
                pass

            return TerminalDecision(True, reason="final_answer", info=info)

        return TerminalDecision(False)
# src/core/terminal/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class TerminalDecision:
    """Decision returned by a terminal checker."""
    is_terminal: bool
    reason: str = ""
    info: Dict[str, Any] = field(default_factory=dict)

    def to_meta(self, name: str) -> Dict[str, Any]:
        """Convenient method to write terminal info into trace meta."""
        return {
            "term_name": name,
            "term_reason": self.reason,
            "term_info": self.info,
        }


class TerminalChecker(Protocol):
    """Pluggable terminal criterion.

    Inputs:
      - text: current node prefix text (question + generated continuation)
      - depth: node depth (root=0)
      - tokens_len: prefix token length (if tokenizer exists: strict; else estimate)
      - root_text: the original root question text (used to avoid matching on question)
    """

    name: str

    def decide(self, text: str, depth: int, tokens_len: int, root_text: str) -> TerminalDecision:
        ...
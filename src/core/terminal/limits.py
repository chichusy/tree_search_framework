# src/core/terminal/limits.py
from __future__ import annotations

from dataclasses import dataclass

from .base import TerminalChecker, TerminalDecision


@dataclass
class MaxTokensTerminal(TerminalChecker):
    max_prefix_tokens: int
    name: str = "max_tokens"

    def decide(self, text: str, depth: int, tokens_len: int, root_text: str) -> TerminalDecision:
        if not self.max_prefix_tokens or int(self.max_prefix_tokens) <= 0:
            return TerminalDecision(False)

        if tokens_len >= int(self.max_prefix_tokens):
            return TerminalDecision(
                True,
                reason="max_tokens",
                info={"max_prefix_tokens": int(self.max_prefix_tokens), "tokens_len": int(tokens_len)},
            )
        return TerminalDecision(False)
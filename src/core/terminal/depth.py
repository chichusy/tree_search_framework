# src/core/terminal/depth.py
from __future__ import annotations

from dataclasses import dataclass

from .base import TerminalChecker, TerminalDecision


@dataclass
class MaxDepthTerminal(TerminalChecker):
    max_depth: int
    name: str = "max_depth"

    def decide(self, text: str, depth: int, tokens_len: int, root_text: str) -> TerminalDecision:
        if self.max_depth is None:
            return TerminalDecision(False)
        if depth >= int(self.max_depth):
            return TerminalDecision(
                True,
                reason="max_depth",
                info={"max_depth": int(self.max_depth), "depth": int(depth)},
            )
        return TerminalDecision(False)
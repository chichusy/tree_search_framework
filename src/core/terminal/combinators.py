# src/core/terminal/combinators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import TerminalChecker, TerminalDecision


@dataclass
class AnyOfTerminal(TerminalChecker):
    checkers: List[TerminalChecker]
    name: str = "any_of"

    def decide(self, text: str, depth: int, tokens_len: int, root_text: str) -> TerminalDecision:
        for c in self.checkers:
            d = c.decide(text=text, depth=depth, tokens_len=tokens_len, root_text=root_text)
            if d.is_terminal:
                # annotate which checker fired
                info = dict(d.info or {})
                info["fired_checker"] = getattr(c, "name", c.__class__.__name__)
                return TerminalDecision(True, reason=d.reason, info=info)
        return TerminalDecision(False)


@dataclass
class AllOfTerminal(TerminalChecker):
    checkers: List[TerminalChecker]
    name: str = "all_of"

    def decide(self, text: str, depth: int, tokens_len: int, root_text: str) -> TerminalDecision:
        fired = []
        last_reason = ""
        merged_info = {}
        for c in self.checkers:
            d = c.decide(text=text, depth=depth, tokens_len=tokens_len, root_text=root_text)
            if not d.is_terminal:
                return TerminalDecision(False)
            fired.append(getattr(c, "name", c.__class__.__name__))
            last_reason = d.reason or last_reason
            if d.info:
                merged_info.update(d.info)

        merged_info["fired_checkers"] = fired
        return TerminalDecision(True, reason=last_reason or "all_of", info=merged_info)
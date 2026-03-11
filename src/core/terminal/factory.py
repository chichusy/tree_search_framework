# src/core/terminal/factory.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .base import TerminalChecker
from .combinators import AnyOfTerminal
from .depth import MaxDepthTerminal
from .limits import MaxTokensTerminal
from .presets import GENERIC_FINAL_PATTERNS, GSM8K_FINAL_PATTERNS
from .regex_answer import RegexAnswerTerminal


@dataclass
class TerminalBuildConfig:
    terminal_mode: str = "answer_or_depth"
    max_depth: int = 6
    max_prefix_tokens: int = 0  # 0 => disabled
    dataset_name: str = ""
    custom_regex: List[str] = field(default_factory=list)


def build_terminal_checker(cfg: TerminalBuildConfig) -> TerminalChecker:
    mode = (cfg.terminal_mode or "answer_or_depth").strip().lower()

    dataset = (cfg.dataset_name or "").strip().lower()
    if dataset in ("gsm8k", "gsm8k_test", "gsm8k_train"):
        preset_patterns = GSM8K_FINAL_PATTERNS
    else:
        preset_patterns = GENERIC_FINAL_PATTERNS

    # Build main answer checker
    if cfg.custom_regex:
        answer_checker = RegexAnswerTerminal.from_patterns(cfg.custom_regex, name="custom_regex_answer")
    else:
        answer_checker = RegexAnswerTerminal(patterns=preset_patterns, name="preset_regex_answer")

    depth_checker = MaxDepthTerminal(max_depth=int(cfg.max_depth))
    tokens_checker = MaxTokensTerminal(max_prefix_tokens=int(cfg.max_prefix_tokens or 0))

    if mode == "depth":
        # only depth
        return depth_checker

    if mode == "answer":
        # only answer marker
        return answer_checker

    if mode == "answer_or_depth":
        # answer first, then safety
        parts: List[TerminalChecker] = [answer_checker, depth_checker]
        if int(cfg.max_prefix_tokens or 0) > 0:
            parts.insert(1, tokens_checker)  # answer -> tokens -> depth
        return AnyOfTerminal(parts)

    if mode == "custom_regex_or_depth":
        # same as answer_or_depth but requires custom_regex
        # (we already used custom_regex if provided; if not provided, falls back to preset)
        parts = [answer_checker, depth_checker]
        if int(cfg.max_prefix_tokens or 0) > 0:
            parts.insert(1, tokens_checker)
        return AnyOfTerminal(parts)

    raise ValueError(f"Unknown terminal_mode: {cfg.terminal_mode}")
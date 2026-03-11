# src/core/reward/none.py
from __future__ import annotations
from dataclasses import dataclass
from .base import RewardScorer, RewardResult

@dataclass
class NoneScorer:
    mode: str = "none"

    def score_terminal(self, question: str, trajectory_text: str) -> RewardResult:
        return RewardResult(reward=0.0, mode="none", raw_score=None, extra=None)

    def score_step(self, question: str, partial_text: str, step_idx: int) -> RewardResult:
        return RewardResult(reward=0.0, mode="none", raw_score=None, extra=None)
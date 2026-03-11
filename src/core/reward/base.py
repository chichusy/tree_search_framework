# src/core/reward/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Literal, runtime_checkable


RewardMode = Literal["none", "orm", "prm"]


@dataclass
class RewardResult:
    """
    Reward scorer 的统一返回结构。
    - reward: 交给搜索算法（MCTS/UCT/etc）的最终标量
    - raw_score: scorer 原始输出（比如 0/1、logit、概率等），可选
    - extra: 附加信息（prompt_hash、reward_text、token_count、错误原因等）
    """
    reward: float
    mode: RewardMode
    raw_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RewardScorer(Protocol):
    """
    可插拔 reward 模块的最小接口。
    搜索算法只依赖这个接口，不关心内部是 LLM 还是规则还是别的模型。
    """

    mode: RewardMode

    def score_terminal(self, question: str, trajectory_text: str) -> RewardResult:
        """终局打分（ORM/PRM 最终都要落到一个 terminal reward）"""
        ...

    def score_step(self, question: str, partial_text: str, step_idx: int) -> RewardResult:
        """过程打分（PRM 用）；ORM 可直接返回 score_terminal 或者固定值"""
        ...
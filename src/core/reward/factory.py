# src/core/reward/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Any

from .base import RewardScorer
from .none import NoneScorer

from .outcome_verifier import BinaryOutcomeVerifier, BinaryOutcomeVerifierConfig
from .orm_yesno import ORMYESNOSCorer

# Rule-based GSM8K ORM
from .orm_rule_gsm8k import ORMGSM8KRuleScorer

# Step-wise PRM (LLM-as-a-PRM)
from .prm_stepwise import StepwisePRMScorer, StepwisePRMConfig


RewardType = Literal[
    "none",
    "orm_yesno",
    "orm_rule_gsm8k",
    "prm_stepwise",
]


@dataclass
class RewardBuildConfig:
    reward_type: RewardType = "none"

    # LLM-based scorer configs (orm_yesno / prm_stepwise)
    max_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 1.0

    # Rule-based configs (orm_rule_gsm8k)
    gt_answer: Optional[str] = None


def build_reward_scorer(reward_type: RewardType, reward_lm: Any, cfg: RewardBuildConfig) -> RewardScorer:
    if reward_type == "none":
        return NoneScorer()

    if reward_type == "orm_yesno":
        verifier = BinaryOutcomeVerifier(
            reward_lm,
            cfg=BinaryOutcomeVerifierConfig(
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            ),
        )
        return ORMYESNOSCorer(verifier=verifier)

    if reward_type == "orm_rule_gsm8k":
        if not cfg.gt_answer:
            raise ValueError("orm_rule_gsm8k requires cfg.gt_answer (a normalized GT number string).")
        return ORMGSM8KRuleScorer(gt_answer=cfg.gt_answer)

    if reward_type == "prm_stepwise":
        prm_cfg = StepwisePRMConfig(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        return StepwisePRMScorer(reward_lm=reward_lm, cfg=prm_cfg)

    raise ValueError(f"Unknown reward_type: {reward_type}")
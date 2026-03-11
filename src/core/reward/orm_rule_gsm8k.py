# src/core/reward/orm_rule_gsm8k.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .base import RewardResult


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_gsm8k_gt_from_answer_field(answer_text: str) -> Optional[str]:
    """
    GSM8K 的 answer 字段通常包含：
      ...
      #### 18

    返回 "18" 这样的字符串（不带多余空白）。
    """
    if not answer_text:
        return None
    m = re.search(r"####\s*([^\n\r]+)", answer_text)
    if not m:
        return None
    gt = m.group(1).strip()
    # 有的会是 "18" 或 "18.0" 或 "$18"；这里尽量抽数字
    nums = _NUM_RE.findall(gt)
    if not nums:
        return None
    return nums[-1].strip()


def extract_final_number_from_trajectory(text: str) -> Optional[str]:
    """
    从模型的 trajectory_text 中抽取最终答案数字。
    规则（从强到弱）：
      1) 优先找 "#### xxx"
      2) 再找 "Answer: xxx" / "Final answer: xxx"
      3) 最后退化为：全文最后一个数字
    返回数字字符串（如 "18" / "18.0" / "1e3"）。
    """
    if not text:
        return None

    # 1) "####"
    m = re.search(r"####\s*([^\n\r]+)", text)
    if m:
        part = m.group(1)
        nums = _NUM_RE.findall(part)
        if nums:
            return nums[-1].strip()

    # 2) Answer / Final answer
    m = re.search(r"(?:final\s*answer|answer)\s*[:：]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        part = m.group(1)
        nums = _NUM_RE.findall(part)
        if nums:
            return nums[-1].strip()

    # 3) fallback: last number in whole text
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    return nums[-1].strip()


def normalize_num_str(s: str) -> str:
    """
    规范化数字字符串，尽量把 "18.0" 归一成 "18"。
    不做复杂容错，只做轻量清洗。
    """
    s = s.strip()
    # 去掉逗号分隔
    s = s.replace(",", "")
    try:
        # 尽量转 float 再判断是否整数
        x = float(s)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        # 保留有限的小数表示
        return str(x)
    except Exception:
        return s


@dataclass
class ORMGSM8KRuleScorer:
    """
    规则型 ORM：用 GT 与模型最终答案对比，输出 +1 / -1。
    """
    gt_answer: str  # 规范化后的 gt 数字字符串
    mode: str = "orm"
    reward_correct: float = 1.0
    reward_wrong: float = -1.0

    def score_terminal(self, question: str, trajectory_text: str) -> RewardResult:
        pred = extract_final_number_from_trajectory(trajectory_text)
        pred_norm = normalize_num_str(pred) if pred is not None else None

        ok = (pred_norm is not None) and (pred_norm == self.gt_answer)
        reward = self.reward_correct if ok else self.reward_wrong

        extra: Dict[str, Any] = {
            "rule_gt": self.gt_answer,
            "rule_pred": pred_norm,
            "rule_hit": bool(ok),
        }
        return RewardResult(reward=float(reward), mode="orm", raw_score=float(reward), extra=extra)

    def score_step(self, question: str, partial_text: str, step_idx: int) -> RewardResult:
        # ORM 不做 step-level；返回 0，不影响树搜索
        return RewardResult(reward=0.0, mode="orm", raw_score=None, extra={"rule_step_unused": True})
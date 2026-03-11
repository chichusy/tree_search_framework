from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

from .base import RewardResult


def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_scores(text: str) -> List[float]:
    if not text:
        return []
    vals: List[float] = []
    for m in _FLOAT_RE.finditer(text):
        try:
            vals.append(float(m.group(0)))
        except Exception:
            continue
    return vals


def _normalize_score_0_1(v: float) -> Optional[float]:
    if v is None:
        return None

    if v > 1.0:
        if v <= 100.0:
            v = v / 100.0
        else:
            return None

    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0

    return v


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _looks_incomplete_number(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.fullmatch(r"[-+]?\d+\.", s))


def _parse_score_robust(text: str) -> Tuple[Optional[float], int]:
    """
    Returns:
      (score, num_candidates)

    Strategy:
    1. first non-empty line first
    2. prefer FIRST valid score, not last
    3. fallback to full text
    """
    if not text:
        return None, 0

    text = text.strip()

    first_line = _first_nonempty_line(text)
    if first_line:
        vals = _extract_scores(first_line)
        for v in vals:
            s = _normalize_score_0_1(v)
            if s is not None:
                return s, len(vals)

    vals = _extract_scores(text)
    for v in vals:
        s = _normalize_score_0_1(v)
        if s is not None:
            return s, len(vals)

    return None, 0


@dataclass
class StepwisePRMConfig:
    max_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[List[str]] = None


class StepwisePRMScorer:
    mode: str = "prm"

    def __init__(self, reward_lm: Any, cfg: StepwisePRMConfig):
        self.reward_lm = reward_lm
        self.cfg = cfg
        if self.cfg.stop is None:
            self.cfg.stop = ["\n"]

    def _build_step_prompt(self, question: str, partial_text: str, step_idx: int) -> str:
        return (
            "You are a Process Reward Model (PRM) for mathematical reasoning.\n"
            "Given a QUESTION and the PARTIAL SOLUTION SO FAR, output a single number between 0 and 1.\n"
            "Output format must be like 0.73 (two decimals).\n"
            "The number is the probability that continuing from this partial solution will reach the correct final answer.\n"
            "Strict rules:\n"
            "- Output ONLY ONE number.\n"
            "- NO words.\n"
            "- NO extra punctuation.\n"
            "- NO extra lines.\n"
            "- The number MUST be in [0, 1].\n\n"
            f"QUESTION:\n{question}\n\n"
            f"PARTIAL_SOLUTION_STEP_{step_idx}:\n{partial_text}\n\n"
            "SCORE:"
        )

    def _build_terminal_prompt(self, question: str, trajectory_text: str) -> str:
        return (
            "You are a Process Reward Model (PRM) for mathematical reasoning.\n"
            "Given a QUESTION and the COMPLETE SOLUTION, output a single number between 0 and 1.\n"
            "Output format must be like 0.73 (two decimals).\n"
            "The number is the probability that this complete solution reaches the correct final answer.\n"
            "Strict rules:\n"
            "- Output ONLY ONE number.\n"
            "- NO words.\n"
            "- NO extra punctuation.\n"
            "- NO extra lines.\n"
            "- The number MUST be in [0, 1].\n\n"
            f"QUESTION:\n{question}\n\n"
            f"COMPLETE_SOLUTION:\n{trajectory_text}\n\n"
            "SCORE:"
        )

    def _gen_once(self, prompt: str, max_tokens: int):
        try:
            return self.reward_lm.generate(
                prompt_text=prompt,
                max_new_tokens=max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                stop=self.cfg.stop,
            )
        except TypeError:
            return self.reward_lm.generate(
                prompt_text=prompt,
                max_new_tokens=max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )

    def _unwrap_text(self, res: Any) -> str:
        t = getattr(res, "output_text", None)
        if isinstance(t, str):
            return t.strip()

        t = getattr(res, "text", None)
        if isinstance(t, str):
            return t.strip()

        if isinstance(res, str):
            return res.strip()

        return str(res).strip()

    def _judge(self, prompt: str) -> Tuple[Optional[float], str, int]:
        first_max = max(4, int(self.cfg.max_tokens))

        res = self._gen_once(prompt, first_max)
        out = self._unwrap_text(res)

        if _looks_incomplete_number(out):
            res2 = self._gen_once(prompt, first_max + 8)
            out2 = self._unwrap_text(res2)
            if out2 and not _looks_incomplete_number(out2):
                out = out2

        score, num_candidates = _parse_score_robust(out)
        return score, out, num_candidates

    def score_step(self, question: str, partial_text: str, step_idx: int) -> RewardResult:
        prompt = self._build_step_prompt(question, partial_text, step_idx)
        score, out, num_candidates = self._judge(prompt)

        if score is None:
            score = 0.0
            parsed_ok = False
        else:
            parsed_ok = True

        return RewardResult(
            reward=float(score),
            mode="prm",
            raw_score=float(score),
            extra={
                "prm_step_idx": int(step_idx),
                "prm_prompt_hash": _hash(prompt),
                "prm_parsed_ok": bool(parsed_ok),
                "prm_num_candidates": int(num_candidates),
                "prm_out_text": (out[:200] if out else ""),
                "prm_out_first_line": _first_nonempty_line(out)[:100] if out else "",
            },
        )

    def score_terminal(self, question: str, trajectory_text: str) -> RewardResult:
        prompt = self._build_terminal_prompt(question, trajectory_text)
        score, out, num_candidates = self._judge(prompt)

        if score is None:
            score = 0.0
            parsed_ok = False
        else:
            parsed_ok = True

        return RewardResult(
            reward=float(score),
            mode="prm",
            raw_score=float(score),
            extra={
                "prm_step_idx": -1,
                "prm_prompt_hash": _hash(prompt),
                "prm_parsed_ok": bool(parsed_ok),
                "prm_num_candidates": int(num_candidates),
                "prm_out_text": (out[:200] if out else ""),
                "prm_out_first_line": _first_nonempty_line(out)[:100] if out else "",
            },
        )
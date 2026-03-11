from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .base import RewardResult
from .outcome_verifier import BinaryOutcomeVerifier


def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    t = getattr(x, "output_text", None)
    if isinstance(t, str):
        return t
    t = getattr(x, "text", None)
    if isinstance(t, str):
        return t
    return str(x)


def _first_nonempty_line(s: str) -> str:
    for line in (s or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


@dataclass
class ORMYESNOSCorer:
    verifier: BinaryOutcomeVerifier
    mode: str = "orm"

    def score_terminal(self, question: str, trajectory_text: str) -> RewardResult:
        score, prompt, out = self.verifier.score_with_io(question, trajectory_text)
        out_text = _safe_text(out).strip()
        first_line = _first_nonempty_line(out_text)

        if score > 0:
            verdict = "YES"
        elif score < 0:
            verdict = "NO"
        else:
            verdict = "UNKNOWN"

        return RewardResult(
            reward=float(score),
            mode="orm",
            raw_score=float(score),
            extra={
                "reward_prompt_hash": _hash(prompt),
                "reward_out_text": out_text[:200],
                "reward_out_first_line": first_line[:100],
                "reward_verdict": verdict,
            },
        )

    def score_step(self, question: str, partial_text: str, step_idx: int) -> RewardResult:
        return self.score_terminal(question, partial_text)
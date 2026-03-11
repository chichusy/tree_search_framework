from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Tuple, Any, Optional


# -----------------------------
# Cleaning + Parsing helpers
# -----------------------------
_DEPTH_TAG_RE = re.compile(r"\[DEPTH=\d+\]\s*")
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\\\((.*?)\\\)", flags=re.DOTALL)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_YES_RE = re.compile(r"(?<![A-Za-z])YES(?![A-Za-z])", flags=re.IGNORECASE)
_NO_RE = re.compile(r"(?<![A-Za-z])NO(?![A-Za-z])", flags=re.IGNORECASE)


def _clean_reasoning(reasoning: str) -> str:
    """
    Clean reasoning text before sending to YES/NO verifier.

    Important:
    - Do NOT remove \\boxed{...}, because it may contain the final answer.
    """
    if not reasoning:
        return reasoning

    s = reasoning
    s = _DEPTH_TAG_RE.sub("", s)
    s = _FENCED_CODE_BLOCK_RE.sub("", s)
    s = _LATEX_INLINE_RE.sub(lambda m: m.group(1), s)
    s = s.replace("\\", "")
    s = _MULTI_SPACE_RE.sub(" ", s)
    s = _MULTI_NEWLINE_RE.sub("\n\n", s)
    return s.strip()


def _extract_first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _parse_yes_no(out_text: str) -> float:
    """
    Parse YES/NO robustly.

    Priority:
    1. first non-empty line
    2. first token in full text
    3. fallback regex search
    """
    if out_text is None:
        return -1.0

    s = out_text.strip()
    if not s:
        return -1.0

    first_line = _extract_first_nonempty_line(s)
    if first_line:
        first_token = first_line.split()[0].strip(" .,:;!?()[]{}\"'")
        up = first_token.upper()
        if up == "YES":
            return 1.0
        if up == "NO":
            return -1.0

    full_tokens = s.split()
    if full_tokens:
        first_token = full_tokens[0].strip(" .,:;!?()[]{}\"'")
        up = first_token.upper()
        if up == "YES":
            return 1.0
        if up == "NO":
            return -1.0

    m_yes = _YES_RE.search(s)
    m_no = _NO_RE.search(s)

    if m_yes and m_no:
        return 1.0 if m_yes.start() < m_no.start() else -1.0
    if m_yes:
        return 1.0
    if m_no:
        return -1.0
    return -1.0


def _unwrap_generate_output(out: Any) -> str:
    """
    Convert runner output into plain text.

    Supported:
    - str
    - dict with text/output_text/generated_text/content
    - list[...] where first item is str/dict/object
    - object with .output_text or .text
    - fallback to str(out)
    """
    if out is None:
        return ""

    if isinstance(out, str):
        return out

    if isinstance(out, list):
        if not out:
            return ""
        first = out[0]

        if isinstance(first, str):
            return first

        if isinstance(first, dict):
            for k in ("output_text", "text", "generated_text", "content"):
                if k in first and first[k] is not None:
                    return str(first[k])
            return str(first)

        t = getattr(first, "output_text", None)
        if t is not None:
            return str(t)

        t = getattr(first, "text", None)
        if t is not None:
            return str(t)

        return str(first)

    if isinstance(out, dict):
        for k in ("output_text", "text", "generated_text", "content"):
            if k in out and out[k] is not None:
                return str(out[k])
        return str(out)

    t = getattr(out, "output_text", None)
    if t is not None:
        return str(t)

    t = getattr(out, "text", None)
    if t is not None:
        return str(t)

    return str(out)


@dataclass
class BinaryOutcomeVerifierConfig:
    max_tokens: int = 4
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Optional[list[str]] = None


class BinaryOutcomeVerifier:
    """
    Outcome verifier:
    YES -> +1.0, NO -> -1.0
    """

    def __init__(self, lm, cfg: BinaryOutcomeVerifierConfig | None = None):
        self.lm = lm
        self.cfg = cfg or BinaryOutcomeVerifierConfig()
        if self.cfg.stop is None:
            self.cfg.stop = ["\n"]

    def build_prompt(self, question: str, reasoning: str) -> str:
        return (
            "You are a strict reasoning verifier.\n"
            "Given a QUESTION and a COMPLETE SOLUTION, determine whether the final answer is correct.\n"
            "Answer with exactly ONE token: YES or NO.\n"
            "Do NOT output explanation.\n"
            "Do NOT output punctuation.\n\n"
            f"Question:\n{question}\n\n"
            f"Complete solution:\n{reasoning}\n\n"
            "Answer:"
        )

    def _generate_text(self, prompt: str) -> str:
        try:
            out = self.lm.generate(
                prompt_text=prompt,
                max_new_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                stop=self.cfg.stop,
            )
        except TypeError:
            out = self.lm.generate(
                prompt,
                max_new_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )

        text = _unwrap_generate_output(out).strip()

        if text == "":
            try:
                out2 = self.lm.generate(
                    prompt_text=prompt,
                    max_new_tokens=max(8, self.cfg.max_tokens),
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    stop=self.cfg.stop,
                )
            except TypeError:
                out2 = self.lm.generate(
                    prompt,
                    max_new_tokens=max(8, self.cfg.max_tokens),
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                )
            text = _unwrap_generate_output(out2).strip()

        return text

    def score(self, question: str, reasoning: str) -> float:
        reasoning = _clean_reasoning(reasoning)
        prompt = self.build_prompt(question, reasoning)
        out_text = self._generate_text(prompt)
        return _parse_yes_no(out_text)

    def score_with_io(self, question: str, reasoning: str) -> Tuple[float, str, str]:
        reasoning = _clean_reasoning(reasoning)
        prompt = self.build_prompt(question, reasoning)
        out_text = self._generate_text(prompt)
        score = _parse_yes_no(out_text)
        return score, prompt, out_text
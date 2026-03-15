from __future__ import annotations

from typing import Optional

from .results import LMResult


class VLLMRunner:
    """
    真实 vLLM 推理 runner
    """

    def __init__(
        self,
        model_ckpt: str,
        tensor_parallel_size: int = 1,
        seed: int = 0,
        max_num_seqs: int = 16,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.8,
        enable_prefix_caching: bool = True,
    ):
        from vllm import LLM  # lazy import
        from transformers import AutoTokenizer

        self.model_ckpt = model_ckpt
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = LLM(
            model=model_ckpt,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
        )

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: Optional[int] = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> LMResult:
        from vllm import SamplingParams  # lazy import

        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        outputs = self.llm.generate([prompt_text], sp, use_tqdm=False)
        req0 = outputs[0]
        out0 = req0.outputs[0]

        num_cached = getattr(req0, "num_cached_tokens", 0)
        out_token_ids = list(getattr(out0, "token_ids", []) or [])

        return LMResult(
            output_text=out0.text,
            output_tokens_len=len(out_token_ids),
            num_cached_tokens=int(num_cached) if num_cached is not None else 0,
            prompt_tokens_len=len(self.tokenizer.encode(prompt_text)) if self.tokenizer is not None else None,
            output_token_ids=out_token_ids,
        )
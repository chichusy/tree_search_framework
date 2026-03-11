from __future__ import annotations

import argparse
from pathlib import Path

from src.core.dataset import load_dataset, extract_question
from src.core.lm import DummyLMRunner, VLLMRunner
from src.core.reward.factory import build_reward_scorer, RewardBuildConfig
from src.core.trace.writer import TraceLogger
from src.core.terminal.factory import build_terminal_checker, TerminalBuildConfig
from src.search.mcts_legacy import MCTSWorkload


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--dataset_format", type=str, default="jsonl")
    ap.add_argument("--max_samples", type=int, default=1)

    ap.add_argument("--runner", type=str, default="dummy", choices=["dummy", "vllm"])
    ap.add_argument("--model_ckpt", type=str, default="")
    ap.add_argument("--tp", type=int, default=1)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_rollouts", type=int, default=10)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--branch_factor", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--cache_key_mode", type=str, default="node", choices=["node", "prompt"])

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)

    # reward mode
    ap.add_argument("--reward_mode", type=str, default="none", choices=["none", "orm", "prm"])

    # ---- ORM (Outcome Reward Model / terminal verifier) ----
    ap.add_argument(
        "--orm_type",
        type=str,
        default="orm_yesno",
        choices=["orm_yesno", "orm_rule_gsm8k"],
        help="When reward_mode=orm: choose LLM yes/no judge or GSM8K GT rule judge.",
    )
    ap.add_argument("--orm_max_tokens", type=int, default=4)
    ap.add_argument("--orm_temperature", type=float, default=0.0)
    ap.add_argument("--orm_top_p", type=float, default=1.0)

    # ---- PRM (Process Reward Model / step-wise scoring) ----
    ap.add_argument(
        "--prm_type",
        type=str,
        default="prm_stepwise",
        choices=["prm_stepwise"],
        help="When reward_mode=prm: step-wise PRM scorer.",
    )
    ap.add_argument("--prm_step_max_tokens", type=int, default=8)
    ap.add_argument("--prm_step_temperature", type=float, default=0.0)
    ap.add_argument("--prm_step_top_p", type=float, default=1.0)
    ap.add_argument(
        "--prm_agg",
        type=str,
        default="mean",
        choices=["mean", "min", "prod", "last", "terminal"],
        help="How to aggregate step scores into a rollout reward.",
    )

    # ---- terminal mode ----
    ap.add_argument(
        "--terminal_mode",
        type=str,
        default="answer_or_depth",
        choices=["depth", "answer", "answer_or_depth", "custom_regex_or_depth"],
        help="Terminal condition: depth only / answer only / answer or depth / custom regex or depth.",
    )
    ap.add_argument(
        "--max_prefix_tokens",
        type=int,
        default=0,
        help="Optional prefix token limit. 0 means disabled.",
    )
    ap.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="Used by terminal factory to select preset answer regex patterns.",
    )
    ap.add_argument(
        "--custom_regex",
        type=str,
        nargs="*",
        default=[],
        help="Optional custom regex patterns used only when terminal_mode=custom_regex_or_depth.",
    )

    ap.add_argument("--save_prompt_text", action="store_true")
    ap.add_argument("--save_output_text", action="store_true")
    ap.add_argument("--save_prompt_max_chars", type=int, default=0)
    ap.add_argument("--save_output_max_chars", type=int, default=0)

    args = ap.parse_args()

    samples = load_dataset(args.dataset_path, args.dataset_format)[: args.max_samples]

    # build policy runner
    if args.runner == "dummy":
        lm = DummyLMRunner(fixed_output_tokens=min(64, args.max_new_tokens))
        tokenizer = None
    else:
        if not args.model_ckpt:
            raise ValueError("--model_ckpt is required when --runner vllm")
        lm = VLLMRunner(model_ckpt=args.model_ckpt, tensor_parallel_size=args.tp)
        tokenizer = getattr(lm, "tokenizer", None)

    reward_mode = args.reward_mode
    orm_max_tokens = args.orm_max_tokens
    orm_temperature = args.orm_temperature
    orm_top_p = args.orm_top_p

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        q = extract_question(sample)
        sample_id = f"{i:05d}"
        sample_dir = out_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        trace_path = sample_dir / "trace_calls.jsonl"
        trace_logger = TraceLogger(str(trace_path), mode="w")

        # ---- extract GT for GSM8K rule scorer (per-sample) ----
        gt_answer = None
        if isinstance(sample, dict) and "answer" in sample:
            from src.core.reward.orm_rule_gsm8k import (
                extract_gsm8k_gt_from_answer_field,
                normalize_num_str,
            )

            gt = extract_gsm8k_gt_from_answer_field(sample.get("answer", ""))
            gt_answer = normalize_num_str(gt) if gt is not None else None

        # ---- build reward scorer per-sample ----
        if reward_mode == "none":
            reward_scorer = build_reward_scorer(
                reward_type="none",
                reward_lm=lm,
                cfg=RewardBuildConfig(reward_type="none"),
            )

        elif reward_mode == "orm":
            if args.orm_type == "orm_yesno":
                reward_scorer = build_reward_scorer(
                    reward_type="orm_yesno",
                    reward_lm=lm,
                    cfg=RewardBuildConfig(
                        reward_type="orm_yesno",
                        max_tokens=orm_max_tokens,
                        temperature=orm_temperature,
                        top_p=orm_top_p,
                    ),
                )
            else:
                # orm_rule_gsm8k
                if gt_answer is None:
                    raise ValueError(
                        f"[sample {sample_id}] GT answer not found/parsed; cannot use orm_rule_gsm8k."
                    )
                reward_scorer = build_reward_scorer(
                    reward_type="orm_rule_gsm8k",
                    reward_lm=lm,  # not used by rule scorer
                    cfg=RewardBuildConfig(
                        reward_type="orm_rule_gsm8k",
                        gt_answer=gt_answer,
                    ),
                )

        elif reward_mode == "prm":
            if args.prm_type != "prm_stepwise":
                raise ValueError(f"Unsupported prm_type: {args.prm_type}")

            reward_scorer = build_reward_scorer(
                reward_type="prm_stepwise",
                reward_lm=lm,
                cfg=RewardBuildConfig(
                    reward_type="prm_stepwise",
                    max_tokens=args.prm_step_max_tokens,
                    temperature=args.prm_step_temperature,
                    top_p=args.prm_step_top_p,
                ),
            )

        else:
            raise ValueError(f"Unsupported reward mode: {reward_mode}")

        # ---- build terminal checker per-sample ----
        terminal_checker = build_terminal_checker(
            TerminalBuildConfig(
                terminal_mode=args.terminal_mode,
                max_depth=args.max_depth,
                max_prefix_tokens=args.max_prefix_tokens,
                dataset_name=args.dataset_name,
                custom_regex=args.custom_regex,
            )
        )

        try:
            workload = MCTSWorkload(
                lm=lm,
                trace_logger=trace_logger,
                tokenizer=tokenizer,
                max_depth=args.max_depth,
                branch_factor=args.branch_factor,
                num_rollouts=args.num_rollouts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                reward_scorer=reward_scorer,
                terminal_checker=terminal_checker,
                prm_agg=args.prm_agg,
                cache_key_mode=args.cache_key_mode,
                save_prompt_text=args.save_prompt_text,
                save_output_text=args.save_output_text,
                save_prompt_max_chars=args.save_prompt_max_chars,
                save_output_max_chars=args.save_output_max_chars,
            )

            root_text = q
            if tokenizer is not None:
                root_tokens_len = len(tokenizer.encode(root_text))
            else:
                root_tokens_len = max(1, len(root_text.split()))

            workload.run(root_text, root_tokens_len, args.num_rollouts)
            print(
                f"[OK] {sample_id} -> {sample_dir} "
                f"(reward_mode={reward_mode}, orm_type={args.orm_type}, prm_type={args.prm_type}, terminal_mode={args.terminal_mode})"
            )
        finally:
            trace_logger.close()

    print("[DONE]")


if __name__ == "__main__":
    main()
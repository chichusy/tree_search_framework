import os
import json
import re
import argparse
import random
from typing import Optional, List, Dict

from tqdm import tqdm
from vllm import LLM, SamplingParams


DATA_PATH = "/home/suyu/projects/tree_search_framework/data/test.jsonl"
OUT_DIR = "/home/suyu/projects/tree_search_framework/data"
MODEL_PATH = "/data/suyu/models/Qwen2.5-7B-Instruct"


def extract_gt(answer: str) -> Optional[str]:
    """
    从 GSM8K 标准答案中提取 GT:
    ... #### 18
    """
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer)
    if m:
        return m.group(1)
    return None


def normalize_num_str(s: Optional[str]) -> Optional[str]:
    """
    统一数值格式:
    '18.0' -> '18'
    '018'  -> '18'
    """
    if s is None:
        return None
    s = s.strip().replace(",", "")
    try:
        x = float(s)
        if x.is_integer():
            return str(int(x))
        return str(x)
    except Exception:
        return s


def extract_answer(text: str):
    """
    更稳的答案提取：
    优先匹配 boxed / answer is / final answer 等强信号；
    最后只允许“最后一行是单独数字”这种保守 fallback。
    """
    if not text:
        return None

    text = text.strip()
    lower_text = text.lower()

    patterns = [
        r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}",
        r"####\s*(-?\d+(?:\.\d+)?)",
        r"(?:the\s+)?answer\s+is\s+(-?\d+(?:\.\d+)?)",
        r"final\s+answer\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
        r"therefore[, ]+(?:the answer is )?(-?\d+(?:\.\d+)?)",
        r"thus[, ]+(?:the answer is )?(-?\d+(?:\.\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, lower_text, flags=re.IGNORECASE)
        if m:
            return normalize_num_str(m.group(1))

    # 更保守：只匹配“最后一行是单独数字”
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1].replace(",", "")
        m = re.fullmatch(r"-?\d+(?:\.\d+)?", last_line)
        if m:
            return normalize_num_str(m.group(0))

    return None


def build_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem.\n\n"
        f"Question: {question}\n\n"
        "Let's think step by step, and give the final answer as a number.\n"
    )


def load_data(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def sample_subset(data: List[Dict], n: int, seed: int) -> List[Dict]:
    """
    从全量数据中随机抽样 n 道。
    若 n<=0 或 n>=len(data)，则返回全量。
    """
    if n <= 0 or n >= len(data):
        return data
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    indices = indices[:n]
    return [data[i] for i in indices]


def run_probe(
    llm: LLM,
    data: List[Dict],
    n_questions: int,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    out_path: str,
    seed: int,
) -> None:
    """
    先随机抽几题，看看模型最常见输出形式。
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    subset = sample_subset(data, n_questions, seed)
    rows = []

    outer_pbar = tqdm(subset, desc="Probe questions", dynamic_ncols=True)

    for idx, sample in enumerate(outer_pbar):
        q = sample["question"]
        gt = normalize_num_str(extract_gt(sample["answer"]))
        prompt = build_prompt(q)

        print("=" * 80)
        print(f"[PROBE] local_sample_id={idx}")
        print("Question:", q)
        print("GT:", gt)
        print("-" * 80)

        # 同一道题批量生成 n_samples 次
        prompts = [prompt] * n_samples
        outputs = llm.generate(prompts, sampling_params)

        for k, out in enumerate(outputs):
            text = out.outputs[0].text.strip()
            pred = extract_answer(text)
            hit = (pred == gt)

            print(f"[trial {k}] pred={pred}, hit={hit}")
            print(text[:1200])
            print("-" * 80)

            rows.append({
                "sample_id": idx,
                "trial_id": k,
                "question": q,
                "gt": gt,
                "raw_output": text,
                "pred": pred,
                "hit": hit,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Probe saved] {out_path}")


def run_full(
    llm: LLM,
    data: List[Dict],
    n_eval: int,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    out_dir: str,
    seed: int,
    batch_size: int,
) -> None:
    """
    正式 difficulty split。
    提速策略：
    - 先从全量随机抽样
    - 再把多题 * 多次采样 展平成大批量 prompts 一次 generate
    - 用 tqdm 展示进度
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # 关键修改 3：从全量中随机抽 n_eval 道，而不是取前 n_eval 道
    data = sample_subset(data, n_eval, seed)

    print(f"[Full mode] Using {len(data)} randomly sampled questions.")

    # 预构造样本信息
    items = []
    for sample in data:
        q = sample["question"]
        gt = normalize_num_str(extract_gt(sample["answer"]))
        prompt = build_prompt(q)
        items.append({
            "question": q,
            "answer": sample["answer"],
            "gt": gt,
            "prompt": prompt,
        })

    # 展平为多次采样任务
    jobs = []
    for i, item in enumerate(items):
        for trial_id in range(n_samples):
            jobs.append({
                "item_id": i,
                "trial_id": trial_id,
                "prompt": item["prompt"],
            })

    print(f"[Full mode] Total generations = {len(items)} questions × {n_samples} samples = {len(jobs)}")

    # 存每题的预测
    per_item_preds = [[] for _ in range(len(items))]

    pbar = tqdm(total=len(jobs), desc="Generating answers", dynamic_ncols=True)

    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        prompts = [x["prompt"] for x in chunk]

        outputs = llm.generate(prompts, sampling_params)

        for job, out in zip(chunk, outputs):
            text = out.outputs[0].text.strip()
            pred = extract_answer(text)
            per_item_preds[job["item_id"]].append(pred)

        pbar.update(len(chunk))

    pbar.close()

    results = []
    score_pbar = tqdm(total=len(items), desc="Scoring / bucketing", dynamic_ncols=True)

    for i, item in enumerate(items):
        preds = per_item_preds[i]
        correct = sum(1 for p in preds if p == item["gt"])
        acc = correct / n_samples

        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "gt": item["gt"],
            "correct": correct,
            "acc": acc,
            "preds": preds,
        })
        score_pbar.update(1)

    score_pbar.close()

    easy, medium, hard = [], [], []

    for r in results:
        # 关键修改 1：5/5 才算 easy；2/5,3/5,4/5 算 medium；0/5,1/5 算 hard
        if r["correct"] == n_samples:
            easy.append(r)
        elif r["correct"] >= 2:
            medium.append(r)
        else:
            hard.append(r)

    os.makedirs(out_dir, exist_ok=True)

    def dump(path: str, rows: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    dump(os.path.join(out_dir, "gsm8k_easy.jsonl"), easy)
    dump(os.path.join(out_dir, "gsm8k_medium.jsonl"), medium)
    dump(os.path.join(out_dir, "gsm8k_hard.jsonl"), hard)
    dump(os.path.join(out_dir, "gsm8k_difficulty_all.jsonl"), results)

    print("easy:", len(easy))
    print("medium:", len(medium))
    print("hard:", len(hard))
    print("total:", len(results))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["probe", "full"], default="probe")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)

    parser.add_argument("--n_questions", type=int, default=5,
                        help="probe 模式下随机抽多少题")
    parser.add_argument("--n_eval", type=int, default=60,
                        help="full 模式下从全量中随机抽多少题；0 表示全量")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="每题采样多少次")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                        help="随机抽题种子")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="full 模式下批量生成的 batch size，A100 可尝试 32/64")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
    )

    data = load_data(args.data_path)
    print(f"Loaded {len(data)} samples from {args.data_path}")

    if args.mode == "probe":
        probe_path = os.path.join(args.out_dir, "gsm8k_probe_outputs.jsonl")
        run_probe(
            llm=llm,
            data=data,
            n_questions=args.n_questions,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            out_path=probe_path,
            seed=args.seed,
        )
    else:
        run_full(
            llm=llm,
            data=data,
            n_eval=args.n_eval,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            out_dir=args.out_dir,
            seed=args.seed,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
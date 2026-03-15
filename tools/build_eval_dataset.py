import json
import random
import argparse


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--easy_path", type=str, default="data/gsm8k_easy.jsonl")
    parser.add_argument("--medium_path", type=str, default="data/gsm8k_medium.jsonl")
    parser.add_argument("--hard_path", type=str, default="data/gsm8k_hard.jsonl")

    parser.add_argument("--n_easy", type=int, default=30)
    parser.add_argument("--n_medium", type=int, default=30)
    parser.add_argument("--n_hard", type=int, default=30)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", type=str, default="data/gsm8k_difficulty_eval_90.jsonl")

    args = parser.parse_args()

    random.seed(args.seed)

    easy = load_jsonl(args.easy_path)
    medium = load_jsonl(args.medium_path)
    hard = load_jsonl(args.hard_path)

    print("easy pool:", len(easy))
    print("medium pool:", len(medium))
    print("hard pool:", len(hard))

    easy_sample = random.sample(easy, args.n_easy)
    medium_sample = random.sample(medium, args.n_medium)
    hard_sample = random.sample(hard, args.n_hard)

    dataset = []

    for x in easy_sample:
        dataset.append({
            "question": x["question"],
            "answer": x["answer"],
            "difficulty": "easy"
        })

    for x in medium_sample:
        dataset.append({
            "question": x["question"],
            "answer": x["answer"],
            "difficulty": "medium"
        })

    for x in hard_sample:
        dataset.append({
            "question": x["question"],
            "answer": x["answer"],
            "difficulty": "hard"
        })

    # 打乱顺序（可选但推荐）
    random.shuffle(dataset)

    # 加 id
    for i, x in enumerate(dataset):
        x["id"] = i

    with open(args.out_path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved:", args.out_path)
    print("Total samples:", len(dataset))


if __name__ == "__main__":
    main()
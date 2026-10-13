#!/usr/bin/env python3
"""生成离线冒烟测试用的伪语料（词汤 + 少量"魔法数字"句式），无需网络。

用法:
  python scripts/make_dummy_data.py --out_dir data/dummy
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYL = ["ka", "ro", "mi", "ta", "lu", "sen", "dor", "vil", "pen", "tor",
       "nal", "bes", "qui", "mon", "fel", "gar", "hil", "jas", "ker", "lum"]
KEYS = ["lyre", "orchid", "wolf", "raven", "ember", "cinder", "maple", "slate"]
FILLERS = ["The grass is green. The sky is blue. The sun is yellow.",
           "Here we go. There and back again.",
           "A quiet river runs through the old town.",
           "Some books are to be tasted, others swallowed, and some few chewed."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/dummy")
    ap.add_argument("--train_words", type=int, default=260000)
    ap.add_argument("--val_words", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = set()
    while len(vocab) < 800:
        vocab.add("".join(rng.choice(SYL) for _ in range(rng.randint(2, 3))))
    vocab = sorted(vocab)

    def make_text(n_words: int) -> str:
        parts = []
        used = 0
        while used < n_words:
            r = rng.random()
            if r < 0.04:  # 4% 魔法数字句（给检索一点点可学的信号）
                k = rng.choice(KEYS)
                v = rng.randint(10 ** 6, 10 ** 7 - 1)
                sent = f"One of the special magic numbers for {k} is: {v}."
                n = 10
            elif r < 0.18:  # 重复句式（模拟 RULER 合成干扰）
                sent = ("The grass is green. The sky is blue. The sun is yellow. "
                        "Here we go. There and back again.")
                n = 18
            else:
                n = rng.randint(6, 16)
                sent = " ".join(rng.choice(vocab) for _ in range(n)) + "."
            parts.append(sent)
            used += n
            if len(parts) % rng.randint(3, 7) == 0:
                parts.append("\n")
        return " ".join(parts) + "\n"

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write(make_text(args.train_words))
    with open(os.path.join(args.out_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write(make_text(args.val_words))

    from lega.data import prepare_bins
    meta = prepare_bins(args.out_dir, kind="auto", force=True)
    print(f"[dummy] 完成: {meta}")


if __name__ == "__main__":
    main()

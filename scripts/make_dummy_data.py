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
    ap.add_argument("--test_words", type=int, default=30000,
                    help="held-out test 语料词数；<=0 则不生成 test（评测回退 val）")
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
    # 冒烟同样走 held-out test 口径：评测默认 test，与 best-checkpoint 选择集（val）分离
    test_txt = os.path.join(args.out_dir, "test.txt")
    if args.test_words > 0:
        with open(test_txt, "w", encoding="utf-8") as f:
            f.write(make_text(args.test_words))
    elif os.path.exists(test_txt):
        # 明确要求不生成 test 时删除上次残留的 test.txt，
        # 否则旧 split 会被 force 重建回 test.bin，与“回退 val”口径矛盾
        os.remove(test_txt)

    from lega.data import prepare_bins
    meta = prepare_bins(
        args.out_dir, kind="auto", force=True,
        extra_report={
            "source": "dummy_synthetic",
            # 生成器版本信息进 data_report：数据内容指纹之外，种子/规模也可追溯
            "generator": {
                "script": "scripts/make_dummy_data.py",
                "seed": int(args.seed),
                "train_words": int(args.train_words),
                "val_words": int(args.val_words),
                "test_words": int(args.test_words),
            },
            "split_protocol": {
                "mode": "independent_generation",
                "order": "train/val/test 各自独立生成（共享分布、无共享样本）",
                "shuffled": False,
            },
        })
    print(f"[dummy] 完成: {meta}")


if __name__ == "__main__":
    main()

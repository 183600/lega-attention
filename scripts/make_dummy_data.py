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
    # 三个 split 的词汤必须互不相同：共享词表会让 64-gram 在 split 间偶然重合，
    # 与“无共享样本”的协议声明相矛盾，也会触发防泄漏闸门（跨 split 复制告警）。
    # 每个 split 用不相交的词表分片 + 不相交的样板句，保证任意长 n-gram 都不重。
    vocab = set()
    while len(vocab) < 2400:
        vocab.add("".join(rng.choice(SYL) for _ in range(rng.randint(2, 3))))
    vocab = sorted(vocab)
    per = len(vocab) // 3
    split_vocab = {"train": vocab[:per],
                   "val": vocab[per:2 * per],
                   "test": vocab[2 * per:]}

    def make_text(n_words: int, split: str) -> str:
        # split 专属标识词前缀：即使词表分片意外相交，文本层面也不共享任何长串。
        sv = split_vocab[split]
        parts = []
        used = 0
        while used < n_words:
            r = rng.random()
            if r < 0.04:  # 4% 魔法数字句（给检索一点点可学的信号）
                k = rng.choice(KEYS)
                v = rng.randint(10 ** 6, 10 ** 7 - 1)
                sent = f"One of the special magic numbers for {split} {k} is: {v}."
                n = 11
            elif r < 0.18:  # 重复句式（模拟 RULER 合成干扰），带 split 标识避免跨 split 复制
                sent = (f"The {split} grass is green. The {split} sky is blue. "
                        f"The {split} sun is yellow. Here we go {split}. "
                        f"There and back again {split}.")
                n = 22
            else:
                n = rng.randint(6, 16)
                sent = " ".join(rng.choice(sv) for _ in range(n)) + "."
            parts.append(sent)
            used += n
            if len(parts) % rng.randint(3, 7) == 0:
                parts.append("\n")
        return " ".join(parts) + "\n"

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write(make_text(args.train_words, "train"))
    with open(os.path.join(args.out_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write(make_text(args.val_words, "val"))
    # 冒烟同样走 held-out test 口径：评测默认 test，与 best-checkpoint 选择集（val）分离
    test_txt = os.path.join(args.out_dir, "test.txt")
    if args.test_words > 0:
        with open(test_txt, "w", encoding="utf-8") as f:
            f.write(make_text(args.test_words, "test"))
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

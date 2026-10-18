"""简单统计基线：一元/二元平滑 n-gram 的 held-out PPL。

科研方法问题：没有简单基线时，“神经模型 PPL=xx”无法回答“数据本身
有多难、复杂架构是否真的比无参数的统计模型更好”。本模块提供一个
零训练、零调参空间（除平滑系数在验证集上选优）的 n-gram 基线：

  * 一元：P(w) = (c(w) + α) / (N + α·V)，即加 α 平滑
  * 二元：P(w_t | w_{t-1}) = (c(w_{t-1}, w_t) + α) / (c(w_{t-1}) + α·V)

平滑系数 α 只在**验证集**上选优（与神经模型的 checkpoint 选择同口径），
再在指定 held-out split 上报告 PPL；选择记录完整落盘，供审阅者核对
“没有偷偷用测试集调参”。测试集 split 需显式 `--allow_test` 并写入
数据目录的访问账本（与 lega.evaluate 同一账本），保持尘封协议一致。

约定：训练统计只来自 `train.bin`；评测 split 内容绝不参与拟合或选参
（split="val" 时 α 选择与报告同集，属轻微乐观，已在输出中显式注明）。
"""
import argparse
import json
import math
import os

import numpy as np

from .data import load_bin, load_meta
from .datacheck import append_test_access

# α 候选：跨 3 个数量级，覆盖稀疏/密集两种极端；只数 3 档，避免把
# “选平滑系数”变成又一次大规模调参。
ALPHA_GRID = (0.01, 0.1, 1.0)


def _pair_keys(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """token 序列 → 相邻对的唯一整数键 prev*V + cur（int64 防溢出）。"""
    return (ids[:-1].astype(np.int64) * int(vocab_size)
            + ids[1:].astype(np.int64))


def fit_counts(ids: np.ndarray, vocab_size: int):
    """在训练流上统计一元/二元频次；返回 (uni_counts, pair_keys, pair_counts)。

    一元用 bincount（V=50257 量级可接受）；二元的 V² 稀疏不可枚举，
    用 np.unique 收集出现的对并计数（内存与 train_tokens 同阶）。
    """
    ids = np.asarray(ids, dtype=np.int64)
    uni = np.bincount(ids, minlength=int(vocab_size)).astype(np.float64)
    if len(ids) < 2:
        return uni, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    keys = _pair_keys(ids, vocab_size)
    pair_keys, pair_counts = np.unique(keys, return_counts=True)
    return uni, pair_keys, pair_counts.astype(np.float64)


def _bigram_counts_for(pair_keys: np.ndarray, pair_counts: np.ndarray,
                       keys: np.ndarray) -> np.ndarray:
    """查 eval 二元组的训练频次：排序键上二分，未出现即 0。"""
    if len(pair_keys) == 0 or len(keys) == 0:
        return np.zeros(len(keys), dtype=np.float64)
    pos = np.searchsorted(pair_keys, keys)
    pos_clip = np.clip(pos, 0, len(pair_keys) - 1)
    hit = pair_keys[pos_clip] == keys
    return np.where(hit, pair_counts[pos_clip], 0.0)


def ppl_unigram(eval_ids: np.ndarray, uni: np.ndarray, alpha: float) -> dict:
    """加 α 平滑一元模型的 per-token 平均负对数似然与 PPL。"""
    vocab = int(len(uni))
    n = int(len(eval_ids))
    if n == 0:
        return {"ppl": None, "mean_nll": None, "n_tokens": 0}
    total = float(uni.sum())
    c = uni[np.asarray(eval_ids, dtype=np.int64)]
    probs = (c + alpha) / (total + alpha * vocab)
    mean_nll = float(-np.log(probs).mean())
    return {"ppl": float(math.exp(min(20.0, mean_nll))),
            "mean_nll": mean_nll, "n_tokens": n,
            "ppl_clamped": bool(mean_nll > 20.0)}


def ppl_bigram(eval_ids: np.ndarray, uni: np.ndarray, pair_keys: np.ndarray,
               pair_counts: np.ndarray, alpha: float, vocab_size: int) -> dict:
    """加 α 平滑二元模型；第一个 token 无前文，从第二个 token 起计分。"""
    n = int(len(eval_ids))
    if n < 2:
        return {"ppl": None, "mean_nll": None, "n_tokens": 0}
    keys = _pair_keys(np.asarray(eval_ids, dtype=np.int64), vocab_size)
    cur = np.asarray(eval_ids, dtype=np.int64)[1:]
    prev = np.asarray(eval_ids, dtype=np.int64)[:-1]
    c_pair = _bigram_counts_for(pair_keys, pair_counts, keys)
    c_prev = uni[prev]
    vocab = int(vocab_size)
    probs = (c_pair + alpha) / (c_prev + alpha * vocab)
    mean_nll = float(-np.log(probs).mean())
    return {"ppl": float(math.exp(min(20.0, mean_nll))),
            "mean_nll": mean_nll, "n_tokens": int(len(cur)),
            "ppl_clamped": bool(mean_nll > 20.0)}


def evaluate(data_dir: str, split: str = "val",
             alphas=ALPHA_GRID, allow_test: bool = False) -> dict:
    """拟合并评测 n-gram 基线，返回完整报告 dict（不写文件）。

    选参协议：α 在 **val** 上选优（val 不存在或过短时报错），随后在
    `split` 上按选定 α 报告；split == "val" 时选参与报告同集，报告中
    以 `alpha_selected_on` 与 `selection_eval_same_split` 显式披露。
    """
    meta = load_meta(data_dir)
    vocab_size = int(meta["vocab_size"])
    train_ids = np.asarray(load_bin(data_dir, "train"), dtype=np.int64)
    if split == "test":
        if not allow_test:
            raise PermissionError(
                "test 集已尘封：评测 n-gram 基线需显式 allow_test=True，"
                "并会写入访问账本")
        eval_ids = np.asarray(load_bin(data_dir, "test", allow_test=True),
                              dtype=np.int64)
        sel_ids = np.asarray(load_bin(data_dir, "val"), dtype=np.int64)
    else:
        eval_ids = np.asarray(load_bin(data_dir, split), dtype=np.int64)
        sel_ids = eval_ids
    uni, pair_keys, pair_counts = fit_counts(train_ids, vocab_size)
    # α 选择：val 上按 mean_nll（等价于 PPL 排序，避免 exp 截断干扰选择）
    best = {"unigram": None, "bigram": None}
    for order in ("unigram", "bigram"):
        best_nll = None
        for a in alphas:
            if order == "unigram":
                m = ppl_unigram(sel_ids, uni, a)
            else:
                m = ppl_bigram(sel_ids, uni, pair_keys, pair_counts, a,
                               vocab_size)
            if m["mean_nll"] is None:
                continue
            if best_nll is None or m["mean_nll"] < best_nll:
                best_nll = m["mean_nll"]
                best[order] = float(a)
    report = {
        "baseline": "smoothed n-gram (add-α), no gradient training",
        "split": split,
        "allow_test": bool(allow_test),
        "tokenizer": meta.get("tokenizer"),
        "vocab_size": vocab_size,
        "train_tokens": int(len(train_ids)),
        "eval_tokens": int(len(eval_ids)),
        "alpha_grid": [float(a) for a in alphas],
        "selected_alpha": best,
        "alpha_selected_on": "val (mean NLL，与 model_best 选择同口径)",
        "selection_eval_same_split": bool(split == "val"),
        "ppl": {},
        "ppl_by_alpha": {},
    }
    if split == "test":
        report["test_access"] = append_test_access(data_dir, {
            "split": "test", "kind": "ngram_baseline",
            "ckpt": "ngram-baseline",
            "note": "简单统计基线评测（α 在 val 上选择）"})
    for order in ("unigram", "bigram"):
        a = best[order]
        if a is None:
            report["ppl"][order] = None
            continue
        if order == "unigram":
            m = ppl_unigram(eval_ids, uni, a)
        else:
            m = ppl_bigram(eval_ids, uni, pair_keys, pair_counts, a,
                           vocab_size)
        report["ppl"][order] = m["ppl"]
        # test 上只消费一次：仅报告 val 选定的 α，不额外扫网格
        # （避免"一次评测"被多次读数的灰色空间）。
        if split != "test":
            by_alpha = {}
            for a2 in alphas:
                if order == "unigram":
                    m2 = ppl_unigram(eval_ids, uni, a2)
                else:
                    m2 = ppl_bigram(eval_ids, uni, pair_keys, pair_counts, a2,
                                    vocab_size)
                by_alpha[str(a2)] = m2["ppl"]
            report["ppl_by_alpha"][order] = by_alpha
    return report


def main():
    p = argparse.ArgumentParser(description="简单统计基线：平滑 n-gram PPL")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default=None, help="JSON 输出路径（默认打印）")
    p.add_argument("--alphas", default=None,
                   help="逗号分隔的 α 网格，默认 0.01,0.1,1.0")
    p.add_argument("--allow_test", action="store_true",
                   help="允许在 test 上评测（写访问账本）")
    args = p.parse_args()
    alphas = (tuple(float(x) for x in args.alphas.split(","))
              if args.alphas else ALPHA_GRID)
    report = evaluate(args.data_dir, split=args.split, alphas=alphas,
                      allow_test=args.allow_test)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    print(f"[ngram] {args.split} PPL: unigram={report['ppl'].get('unigram')} "
          f"bigram={report['ppl'].get('bigram')}")


if __name__ == "__main__":
    main()
